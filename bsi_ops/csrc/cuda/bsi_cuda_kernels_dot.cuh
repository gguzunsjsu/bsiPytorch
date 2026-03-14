#pragma once

// Multi-query fused version: process Q queries and multiple keys per block; tiles both axes to shrink grid
#include <stdint.h>
#include <stdio.h>

#if defined(__CUDACC__)
#include <cuda/barrier>
#include <cuda/ptx>
#endif

#if !defined(__CUDA_ARCH__)
#include <cuda.h>

#include <mutex>
#include <unordered_map>
#endif

// Indirection arrays are optional in the "packed batch" fast path.
// When the pointer is null, treat indices as identity (i -> i).
__device__ __forceinline__ long long bsi_load_index_or_identity(const long long* idx, int i) {
#if defined(__CUDA_ARCH__)
    return idx ? __ldg(idx + i) : static_cast<long long>(i);
#else
    return idx ? idx[i] : static_cast<long long>(i);
#endif
}

// Hopper/Ampere async global->shared copies.
__device__ __forceinline__ void bsi_cp_async_cg_16B(void* dst_shared, const void* src_global) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    const unsigned int dst = __cvta_generic_to_shared(dst_shared);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src_global));
#else
    auto* dst64 = reinterpret_cast<unsigned long long*>(dst_shared);
    const auto* src64 = reinterpret_cast<const unsigned long long*>(src_global);
    dst64[0] = src64[0];
    dst64[1] = src64[1];
#endif
}

__device__ __forceinline__ void bsi_cp_async_commit_group() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void bsi_cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

extern "C" __global__
void popcount_weighted_keys_literal_fused_multiq_kernel(
    const unsigned long long* __restrict__ A,
    const float* __restrict__ Aw,
    int Sa,
    int W,
    const unsigned long long* __restrict__ B,
    const float* __restrict__ Bw,
    int Sb,
    int R,
    int Q,
    int q_tile,
    int r_tile,
    const long long* __restrict__ key_indices,
    const long long* __restrict__ query_indices,
    float scale_inv,
    int R_total,
    float* __restrict__ out_global)
{
    extern __shared__ unsigned char shmem[];
    int r_block = blockIdx.x;
    int q_block = blockIdx.y;
    const int tile_q = (q_tile > 0) ? q_tile : 1;
    const int tile_r = (r_tile > 0) ? r_tile : 1;
    const int q_start = q_block * tile_q;
    const int r_start = r_block * tile_r;
    if (q_start >= Q) return;

    __shared__ float warp_buf[32];
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

    const int pairs = Sa * Sb;
    const int total = pairs * W;
    const int w_mask = W - 1;
    const bool w_pow2 = (W & w_mask) == 0;
    const int w_shift = w_pow2 ? (__ffs(W) - 1) : 0;

    unsigned long long* A_sh = reinterpret_cast<unsigned long long*>(shmem);
    unsigned long long* B_sh = A_sh + (size_t)Sa * (size_t)W;
    float* coeff = reinterpret_cast<float*>(B_sh + (size_t)Sb * (size_t)W);
    int* pair_i = reinterpret_cast<int*>(coeff + pairs);
    int* pair_j = pair_i + pairs;
    float* Aw_sh = reinterpret_cast<float*>(pair_j + pairs);
    float* Bw_sh = Aw_sh + Sa;

    for (int pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
        int i = pair / Sb;
        int j = pair - i * Sb;
        pair_i[pair] = i;
        pair_j[pair] = j;
    }
    __syncthreads();

    int q_end = q_start + tile_q;
    if (q_end > Q) q_end = Q;
    for (int q = q_start; q < q_end; ++q) {

        long long global_q = bsi_load_index_or_identity(query_indices, q);
        const unsigned long long* A_base = A + ((size_t)q * Sa * W);
        const float* Aw_base = Aw + ((size_t)q * Sa);

        for (int idx = threadIdx.x; idx < Sa * W; idx += blockDim.x) {
            A_sh[idx] = __ldg(&A_base[idx]);
        }
        for (int idx = threadIdx.x; idx < Sa; idx += blockDim.x) {
            Aw_sh[idx] = __ldg(&Aw_base[idx]);
        }
        __syncthreads();

        for (int tr = 0; tr < tile_r; ++tr) {
            int r = r_start + tr;
            if (r >= R) break;

            long long global_r = bsi_load_index_or_identity(key_indices, r);
            const unsigned long long* B_base = B + ((size_t)r * Sb * W);
            const float* Bw_base = Bw + ((size_t)r * Sb);

            for (int idx = threadIdx.x; idx < Sb * W; idx += blockDim.x) {
                B_sh[idx] = __ldg(&B_base[idx]);
            }
            for (int idx = threadIdx.x; idx < Sb; idx += blockDim.x) {
                Bw_sh[idx] = __ldg(&Bw_base[idx]);
            }
            __syncthreads();

            for (int pair = threadIdx.x; pair < pairs; pair += blockDim.x) {
                coeff[pair] = Aw_sh[pair_i[pair]] * Bw_sh[pair_j[pair]];
            }
            __syncthreads();

            float local = 0.0f;
            for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
                int pair = 0;
                int w = 0;
                if (w_pow2) {
                    pair = idx >> w_shift;
                    w = idx & w_mask;
                } else {
                    pair = idx / W;
                    w = idx - pair * W;
                }
                const int i = pair_i[pair];
                const int j = pair_j[pair];
                unsigned long long a_val = A_sh[(size_t)i * (size_t)W + (size_t)w];
                unsigned long long b_val = B_sh[(size_t)j * (size_t)W + (size_t)w];
                int cnt = __popcll(a_val & b_val);

                local += (float)cnt * coeff[pair];
            }

            local = warp_reduce_sum_float(local);

            if (lane == 0) {
                warp_buf[warp_id] = local;
            }
            __syncthreads();

            if (warp_id == 0) {
                float val = (lane < num_warps) ? warp_buf[lane] : 0.0f;
                val = warp_reduce_sum_float(val);

                if (lane == 0) {
                    out_global[((size_t)global_q * (size_t)R_total) + (size_t)global_r] = val * scale_inv;
                }
            }
            __syncthreads();
        }
    }
}

template <int SB>
__device__ __forceinline__ void bsi_bmma_tm32_accum_sa7_sb_hot(
    const uint32_t* __restrict__ A_bits,
    const float* __restrict__ Aw_tile,
    const uint32_t* __restrict__ b_col_base,
    const float* __restrict__ bw_col0,
    const float* __restrict__ bw_col1,
    int threadID,
    int m0,
    int m1,
    bool use_chunk_scale,
    float qscale_m0,
    float qscale_m1,
    float& acc00,
    float& acc01,
    float& acc10,
    float& acc11);

extern "C" __global__ __launch_bounds__(256)
void popcount_weighted_keys_literal_fused_bmma_tc_kernel_tm32(
    const unsigned long long* __restrict__ A,    
    const float* __restrict__ Aw,                
    const float* __restrict__ A_chunk_scales,    
    int A_scale_stride,                          
    int Sa,
    int W64,
    const unsigned long long* __restrict__ B,    
    const float* __restrict__ Bw,                
    int Sb,
    int R,
    int Q,
    const long long* __restrict__ key_indices,   
    const long long* __restrict__ query_indices, 
    float scale_inv,
    int R_total,
    float* __restrict__ out_global,
    int use_cpasync)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    constexpr int TM_TOTAL = 32;
    constexpr int TM = 16;
    constexpr int TN = 32;
    constexpr int WARPS_PER_QTILE = 4;
    constexpr int QTILES = TM_TOTAL / TM; 
    constexpr int SB_MAX = 16;
    constexpr int K_BITS = 256;
    constexpr int K_WORDS64 = K_BITS / 64;   
    constexpr int K_WORDS32 = K_BITS / 32;   
    constexpr int K_STRIDE32 = K_WORDS32 + 4; 

    if (blockDim.x != (WARPS_PER_QTILE * QTILES * 32)) return; 
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;          
    const int q_tile_id = warp_id >> 2;            
    const int warp_in_tile = warp_id & (WARPS_PER_QTILE - 1); 

    const int q0 = blockIdx.y * TM_TOTAL;
    const int r0 = blockIdx.x * TN;
    const bool full_q_tile = (q0 + TM_TOTAL) <= Q;
    const bool full_r_tile = (r0 + TN) <= R;

    extern __shared__ unsigned char smem_raw[];
    uintptr_t p = reinterpret_cast<uintptr_t>(smem_raw);
    p = (p + 15u) & ~uintptr_t(15u);

    const int stages = (use_cpasync != 0) ? 2 : 1;
    const size_t A_words = (size_t)Sa * (size_t)TM_TOTAL * (size_t)K_STRIDE32;
    const size_t B_words = (size_t)Sb * (size_t)TN * (size_t)K_STRIDE32;

    auto* A_bits0 = reinterpret_cast<uint32_t*>(p); 
    p += (size_t)stages * A_words * sizeof(uint32_t);
    auto* B_bits0 = reinterpret_cast<uint32_t*>(p); 
    p += (size_t)stages * B_words * sizeof(uint32_t);
    auto* Aw_tile = reinterpret_cast<float*>(p);   
    p += (size_t)TM_TOTAL * (size_t)Sa * sizeof(float);
    auto* Bw_tile = reinterpret_cast<float*>(p);   
    p += (size_t)TN * (size_t)Sb * sizeof(float);
    (void)p;

    if (full_q_tile) {
        for (int idx = threadIdx.x; idx < TM_TOTAL * Sa; idx += blockDim.x) {
            const int m = idx / Sa;
            const int i = idx - m * Sa;
            const int q = q0 + m;
            Aw_tile[(size_t)m * (size_t)Sa + (size_t)i] = __ldg(&Aw[(size_t)q * (size_t)Sa + (size_t)i]);
        }
    } else {
        for (int idx = threadIdx.x; idx < TM_TOTAL * Sa; idx += blockDim.x) {
            const int m = idx / Sa;
            const int i = idx - m * Sa;
            const int q = q0 + m;
            Aw_tile[(size_t)m * (size_t)Sa + (size_t)i] =
                (q < Q) ? __ldg(&Aw[(size_t)q * (size_t)Sa + (size_t)i]) : 0.0f;
        }
    }
    if (full_r_tile) {
        for (int idx = threadIdx.x; idx < TN * Sb; idx += blockDim.x) {
            const int n = idx / Sb;
            const int j = idx - n * Sb;
            const int r = r0 + n;
            Bw_tile[(size_t)n * (size_t)Sb + (size_t)j] = __ldg(&Bw[(size_t)r * (size_t)Sb + (size_t)j]);
        }
    } else {
        for (int idx = threadIdx.x; idx < TN * Sb; idx += blockDim.x) {
            const int n = idx / Sb;
            const int j = idx - n * Sb;
            const int r = r0 + n;
            Bw_tile[(size_t)n * (size_t)Sb + (size_t)j] =
                (r < R) ? __ldg(&Bw[(size_t)r * (size_t)Sb + (size_t)j]) : 0.0f;
        }
    }
    __syncthreads();

    const int groupID = lane >> 2;            
    const int threadID = lane & 3;            
    const int row0 = groupID;                 
    const int row1 = groupID + 8;             
    const int col_base = warp_in_tile * 8;    
    const int col0 = col_base + threadID * 2; 
    const int col1 = col0 + 1;                
    const int m0 = q_tile_id * TM + row0;     
    const int m1 = q_tile_id * TM + row1;     

    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    const bool cache_sb = (Sb <= SB_MAX);
    const float* bw_col0 = Bw_tile + (size_t)col0 * (size_t)Sb;
    const float* bw_col1 = Bw_tile + (size_t)col1 * (size_t)Sb;

    const int chunks = W64 / K_WORDS64;
    const bool can_cpasync = (use_cpasync != 0) && full_q_tile && full_r_tile && (chunks > 1);

    if (can_cpasync) {
        uint32_t* A_bits = A_bits0;
        uint32_t* B_bits = B_bits0;
        constexpr int K_WORDS64_16B = K_WORDS64 / 2;
        for (int idx = threadIdx.x; idx < TM_TOTAL * Sa * K_WORDS64_16B; idx += blockDim.x) {
            int t = idx;
            const int w64_pair = t & (K_WORDS64_16B - 1);
            t >>= 1;
            const int m = t & (TM_TOTAL - 1);
            const int i = t >> 5; 

            const int q = q0 + m;
            const unsigned long long* a_slice = A + ((size_t)q * (size_t)Sa + (size_t)i) * (size_t)W64;
            const int w64_i = w64_pair << 1;
            const int base = ((i * TM_TOTAL + m) * K_STRIDE32) + (w64_i << 1);
            bsi_cp_async_cg_16B(A_bits + base, &a_slice[(size_t)0 * (size_t)K_WORDS64 + (size_t)w64_i]);
        }
        for (int idx = threadIdx.x; idx < TN * Sb * K_WORDS64_16B; idx += blockDim.x) {
            int t = idx;
            const int w64_pair = t & (K_WORDS64_16B - 1);
            t >>= 1;
            const int n = t & (TN - 1);
            const int j = t >> 5; 

            const int r = r0 + n;
            const unsigned long long* b_slice = B + ((size_t)r * (size_t)Sb + (size_t)j) * (size_t)W64;
            const int w64_i = w64_pair << 1; 
            const int base = ((j * TN + n) * K_STRIDE32) + (w64_i << 1);
            bsi_cp_async_cg_16B(B_bits + base, &b_slice[(size_t)0 * (size_t)K_WORDS64 + (size_t)w64_i]);
        }
        bsi_cp_async_commit_group();
        bsi_cp_async_wait_all();
        __syncthreads();
    }

    int stage = 0;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        uint32_t* A_bits = (stage == 0) ? A_bits0 : (A_bits0 + A_words);
        uint32_t* B_bits = (stage == 0) ? B_bits0 : (B_bits0 + B_words);

        if (!can_cpasync) {
            if (full_q_tile) {
                for (int idx = threadIdx.x; idx < TM_TOTAL * Sa * K_WORDS64; idx += blockDim.x) {
                    int t = idx;
                    const int w64_i = t & (K_WORDS64 - 1);
                    t >>= 2;
                    const int m = t & (TM_TOTAL - 1);
                    const int i = t >> 5; 

                    const int q = q0 + m;
                    const unsigned long long* a_slice = A + ((size_t)q * (size_t)Sa + (size_t)i) * (size_t)W64;
                    const unsigned long long w64 =
                        __ldg(&a_slice[(size_t)chunk * (size_t)K_WORDS64 + (size_t)w64_i]);
                    const uint32_t lo = static_cast<uint32_t>(w64);
                    const uint32_t hi = static_cast<uint32_t>(w64 >> 32);
                    const int base = ((i * TM_TOTAL + m) * K_STRIDE32) + (w64_i << 1);
                    A_bits[base] = lo;
                    A_bits[base + 1] = hi;
                }
            } else {
                for (int idx = threadIdx.x; idx < TM_TOTAL * Sa * K_WORDS64; idx += blockDim.x) {
                    int t = idx;
                    const int w64_i = t & (K_WORDS64 - 1);
                    t >>= 2;
                    const int m = t & (TM_TOTAL - 1);
                    const int i = t >> 5; 

                    uint32_t lo = 0u, hi = 0u;
                    const int q = q0 + m;
                    if (q < Q) {
                        const unsigned long long* a_slice = A + ((size_t)q * (size_t)Sa + (size_t)i) * (size_t)W64;
                        const unsigned long long w64 =
                            __ldg(&a_slice[(size_t)chunk * (size_t)K_WORDS64 + (size_t)w64_i]);
                        lo = static_cast<uint32_t>(w64);
                        hi = static_cast<uint32_t>(w64 >> 32);
                    }
                    const int base = ((i * TM_TOTAL + m) * K_STRIDE32) + (w64_i << 1);
                    A_bits[base] = lo;
                    A_bits[base + 1] = hi;
                }
            }

            if (full_r_tile) {
                for (int idx = threadIdx.x; idx < TN * Sb * K_WORDS64; idx += blockDim.x) {
                    int t = idx;
                    const int w64_i = t & (K_WORDS64 - 1);
                    t >>= 2;
                    const int n = t & (TN - 1);
                    const int j = t >> 5; 

                    const int r = r0 + n;
                    const unsigned long long* b_slice = B + ((size_t)r * (size_t)Sb + (size_t)j) * (size_t)W64;
                    const unsigned long long w64 =
                        __ldg(&b_slice[(size_t)chunk * (size_t)K_WORDS64 + (size_t)w64_i]);
                    const uint32_t lo = static_cast<uint32_t>(w64);
                    const uint32_t hi = static_cast<uint32_t>(w64 >> 32);
                    const int base = ((j * TN + n) * K_STRIDE32) + (w64_i << 1);
                    B_bits[base] = lo;
                    B_bits[base + 1] = hi;
                }
            } else {
                for (int idx = threadIdx.x; idx < TN * Sb * K_WORDS64; idx += blockDim.x) {
                    int t = idx;
                    const int w64_i = t & (K_WORDS64 - 1);
                    t >>= 2;
                    const int n = t & (TN - 1);
                    const int j = t >> 5;

                    uint32_t lo = 0u, hi = 0u;
                    const int r = r0 + n;
                    if (r < R) {
                        const unsigned long long* b_slice = B + ((size_t)r * (size_t)Sb + (size_t)j) * (size_t)W64;
                        const unsigned long long w64 =
                            __ldg(&b_slice[(size_t)chunk * (size_t)K_WORDS64 + (size_t)w64_i]);
                        lo = static_cast<uint32_t>(w64);
                        hi = static_cast<uint32_t>(w64 >> 32);
                    }
                    const int base = ((j * TN + n) * K_STRIDE32) + (w64_i << 1);
                    B_bits[base] = lo;
                    B_bits[base + 1] = hi;
                }
            }
            __syncthreads();
        } else {
            if (chunk + 1 < chunks) {
                const int next_stage = stage ^ 1;
                uint32_t* A_bits_next = (next_stage == 0) ? A_bits0 : (A_bits0 + A_words);
                uint32_t* B_bits_next = (next_stage == 0) ? B_bits0 : (B_bits0 + B_words);

                const int next_chunk = chunk + 1;
                constexpr int K_WORDS64_16B = K_WORDS64 / 2;
                for (int idx = threadIdx.x; idx < TM_TOTAL * Sa * K_WORDS64_16B; idx += blockDim.x) {
                    int t = idx;
                    const int w64_pair = t & (K_WORDS64_16B - 1); 
                    t >>= 1;
                    const int m = t & (TM_TOTAL - 1);
                    const int i = t >> 5; 

                    const int q = q0 + m;
                    const unsigned long long* a_slice = A + ((size_t)q * (size_t)Sa + (size_t)i) * (size_t)W64;
                    const int w64_i = w64_pair << 1;
                    const int base = ((i * TM_TOTAL + m) * K_STRIDE32) + (w64_i << 1);
                    bsi_cp_async_cg_16B(
                        A_bits_next + base,
                        &a_slice[(size_t)next_chunk * (size_t)K_WORDS64 + (size_t)w64_i]);
                }
                for (int idx = threadIdx.x; idx < TN * Sb * K_WORDS64_16B; idx += blockDim.x) {
                    int t = idx;
                    const int w64_pair = t & (K_WORDS64_16B - 1); 
                    t >>= 1;
                    const int n = t & (TN - 1);
                    const int j = t >> 5; 

                    const int r = r0 + n;
                    const unsigned long long* b_slice = B + ((size_t)r * (size_t)Sb + (size_t)j) * (size_t)W64;
                    const int w64_i = w64_pair << 1; 
                    const int base = ((j * TN + n) * K_STRIDE32) + (w64_i << 1);
                    bsi_cp_async_cg_16B(
                        B_bits_next + base,
                        &b_slice[(size_t)next_chunk * (size_t)K_WORDS64 + (size_t)w64_i]);
                }
                bsi_cp_async_commit_group();
            }
        }

        const bool use_chunk_scale = (A_chunk_scales != nullptr) && (A_scale_stride > 0);
        float qscale_m0 = 1.0f;
        float qscale_m1 = 1.0f;
        if (use_chunk_scale) {
            if (threadID == 0) {
                const int q_m0 = q0 + m0;
                const int q_m1 = q0 + m1;
                if (full_q_tile) {
                    qscale_m0 = __ldg(&A_chunk_scales[(size_t)q_m0 * (size_t)A_scale_stride + (size_t)chunk]);
                    qscale_m1 = __ldg(&A_chunk_scales[(size_t)q_m1 * (size_t)A_scale_stride + (size_t)chunk]);
                } else {
                    qscale_m0 = (q_m0 < Q)
                        ? __ldg(&A_chunk_scales[(size_t)q_m0 * (size_t)A_scale_stride + (size_t)chunk])
                        : 0.0f;
                    qscale_m1 = (q_m1 < Q)
                        ? __ldg(&A_chunk_scales[(size_t)q_m1 * (size_t)A_scale_stride + (size_t)chunk])
                        : 0.0f;
                }
            }
            qscale_m0 = __shfl_sync(0xffffffff, qscale_m0, lane & ~3);
            qscale_m1 = __shfl_sync(0xffffffff, qscale_m1, lane & ~3);
        }

        if (cache_sb) {
            constexpr int JBLOCK = 4;
            const int b_slice_stride = TN * K_STRIDE32;
            const uint32_t* b_col_base = B_bits + (col_base + groupID) * K_STRIDE32;
            
            if (Sa == 7 && Sb == 7) {
                bsi_bmma_tm32_accum_sa7_sb_hot<7>(
                    A_bits, Aw_tile, b_col_base, bw_col0, bw_col1, threadID, m0, m1,
                    use_chunk_scale, qscale_m0, qscale_m1, acc00, acc01, acc10, acc11);
            } else if (Sa == 7 && Sb == 6) {
                bsi_bmma_tm32_accum_sa7_sb_hot<6>(
                    A_bits, Aw_tile, b_col_base, bw_col0, bw_col1, threadID, m0, m1,
                    use_chunk_scale, qscale_m0, qscale_m1, acc00, acc01, acc10, acc11);
            } else if (Sb == 7) {
                // Legacy non-hot paths logic preserved
                {
                    uint32_t b0_cache[JBLOCK];
                    uint32_t b1_cache[JBLOCK];
                    float bw0_cache[JBLOCK];
                    float bw1_cache[JBLOCK];

#pragma unroll
                    for (int jj = 0; jj < JBLOCK; ++jj) {
                        const uint32_t* b_col = b_col_base + jj * b_slice_stride;
                        b0_cache[jj] = b_col[threadID];
                        b1_cache[jj] = b_col[threadID + 4];
                        bw0_cache[jj] = bw_col0[jj];
                        bw1_cache[jj] = bw_col1[jj];
                    }

                    for (int i = 0; i < Sa; ++i) {
                        const float aw0 = Aw_tile[(size_t)m0 * (size_t)Sa + (size_t)i] * qscale_m0;
                        const float aw1 = Aw_tile[(size_t)m1 * (size_t)Sa + (size_t)i] * qscale_m1;

                        const uint32_t* A_i = A_bits + (size_t)i * (size_t)TM_TOTAL * (size_t)K_STRIDE32;
                        const uint32_t a0 = A_i[(size_t)m0 * (size_t)K_STRIDE32 + (size_t)threadID];
                        const uint32_t a1 = A_i[(size_t)m1 * (size_t)K_STRIDE32 + (size_t)threadID];
                        const uint32_t a2 = A_i[(size_t)m0 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];
                        const uint32_t a3 = A_i[(size_t)m1 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];

                        float sum00 = 0.0f, sum01 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;
                        {
                            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                            asm volatile(
                                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_cache[0]), "r"(b1_cache[0]));
                            asm volatile(
                                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_cache[1]), "r"(b1_cache[1]));
                            sum00 = __fmaf_rn(static_cast<float>(c0), bw0_cache[0], sum00);
                            sum01 = __fmaf_rn(static_cast<float>(c1), bw1_cache[0], sum01);
                            sum10 = __fmaf_rn(static_cast<float>(c2), bw0_cache[0], sum10);
                            sum11 = __fmaf_rn(static_cast<float>(c3), bw1_cache[0], sum11);
                            sum00 = __fmaf_rn(static_cast<float>(d0), bw0_cache[1], sum00);
                            sum01 = __fmaf_rn(static_cast<float>(d1), bw1_cache[1], sum01);
                            sum10 = __fmaf_rn(static_cast<float>(d2), bw0_cache[1], sum10);
                            sum11 = __fmaf_rn(static_cast<float>(d3), bw1_cache[1], sum11);
                        }
                        {
                            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                            asm volatile(
                                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_cache[2]), "r"(b1_cache[2]));
                            asm volatile(
                                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_cache[3]), "r"(b1_cache[3]));
                            sum00 = __fmaf_rn(static_cast<float>(c0), bw0_cache[2], sum00);
                            sum01 = __fmaf_rn(static_cast<float>(c1), bw1_cache[2], sum01);
                            sum10 = __fmaf_rn(static_cast<float>(c2), bw0_cache[2], sum10);
                            sum11 = __fmaf_rn(static_cast<float>(c3), bw1_cache[2], sum11);
                            sum00 = __fmaf_rn(static_cast<float>(d0), bw0_cache[3], sum00);
                            sum01 = __fmaf_rn(static_cast<float>(d1), bw1_cache[3], sum01);
                            sum10 = __fmaf_rn(static_cast<float>(d2), bw0_cache[3], sum10);
                            sum11 = __fmaf_rn(static_cast<float>(d3), bw1_cache[3], sum11);
                        }
                        acc00 = __fmaf_rn(aw0, sum00, acc00);
                        acc01 = __fmaf_rn(aw0, sum01, acc01);
                        acc10 = __fmaf_rn(aw1, sum10, acc10);
                        acc11 = __fmaf_rn(aw1, sum11, acc11);
                    }
                }
                {
                    uint32_t b0_cache[3];
                    uint32_t b1_cache[3];
                    float bw0_cache[3];
                    float bw1_cache[3];

#pragma unroll
                    for (int jj = 0; jj < 3; ++jj) {
                        const uint32_t* b_col = b_col_base + (4 + jj) * b_slice_stride;
                        b0_cache[jj] = b_col[threadID];
                        b1_cache[jj] = b_col[threadID + 4];
                        bw0_cache[jj] = bw_col0[4 + jj];
                        bw1_cache[jj] = bw_col1[4 + jj];
                    }

                    for (int i = 0; i < Sa; ++i) {
                        const float aw0 = Aw_tile[(size_t)m0 * (size_t)Sa + (size_t)i] * qscale_m0;
                        const float aw1 = Aw_tile[(size_t)m1 * (size_t)Sa + (size_t)i] * qscale_m1;

                        const uint32_t* A_i = A_bits + (size_t)i * (size_t)TM_TOTAL * (size_t)K_STRIDE32;
                        const uint32_t a0 = A_i[(size_t)m0 * (size_t)K_STRIDE32 + (size_t)threadID];
                        const uint32_t a1 = A_i[(size_t)m1 * (size_t)K_STRIDE32 + (size_t)threadID];
                        const uint32_t a2 = A_i[(size_t)m0 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];
                        const uint32_t a3 = A_i[(size_t)m1 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];

                        float sum00 = 0.0f, sum01 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;
                        {
                            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                            asm volatile(
                                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_cache[0]), "r"(b1_cache[0]));
                            asm volatile(
                                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_cache[1]), "r"(b1_cache[1]));
                            sum00 = __fmaf_rn(static_cast<float>(c0), bw0_cache[0], sum00);
                            sum01 = __fmaf_rn(static_cast<float>(c1), bw1_cache[0], sum01);
                            sum10 = __fmaf_rn(static_cast<float>(c2), bw0_cache[0], sum10);
                            sum11 = __fmaf_rn(static_cast<float>(c3), bw1_cache[0], sum11);
                            sum00 = __fmaf_rn(static_cast<float>(d0), bw0_cache[1], sum00);
                            sum01 = __fmaf_rn(static_cast<float>(d1), bw1_cache[1], sum01);
                            sum10 = __fmaf_rn(static_cast<float>(d2), bw0_cache[1], sum10);
                            sum11 = __fmaf_rn(static_cast<float>(d3), bw1_cache[1], sum11);
                        }
                        {
                            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                            asm volatile(
                                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_cache[2]), "r"(b1_cache[2]));
                            sum00 = __fmaf_rn(static_cast<float>(c0), bw0_cache[2], sum00);
                            sum01 = __fmaf_rn(static_cast<float>(c1), bw1_cache[2], sum01);
                            sum10 = __fmaf_rn(static_cast<float>(c2), bw0_cache[2], sum10);
                            sum11 = __fmaf_rn(static_cast<float>(c3), bw1_cache[2], sum11);
                        }
                        acc00 = __fmaf_rn(aw0, sum00, acc00);
                        acc01 = __fmaf_rn(aw0, sum01, acc01);
                        acc10 = __fmaf_rn(aw1, sum10, acc10);
                        acc11 = __fmaf_rn(aw1, sum11, acc11);
                    }
                }
            } else if (Sb == 6) {
                uint32_t b0_cache0[JBLOCK];
                uint32_t b1_cache0[JBLOCK];
                float bw0_cache0[JBLOCK];
                float bw1_cache0[JBLOCK];
#pragma unroll
                for (int jj = 0; jj < JBLOCK; ++jj) {
                    const uint32_t* b_col = b_col_base + jj * b_slice_stride;
                    b0_cache0[jj] = b_col[threadID];
                    b1_cache0[jj] = b_col[threadID + 4];
                    bw0_cache0[jj] = bw_col0[jj];
                    bw1_cache0[jj] = bw_col1[jj];
                }

                uint32_t b0_cache1[2];
                uint32_t b1_cache1[2];
                float bw0_cache1[2];
                float bw1_cache1[2];
#pragma unroll
                for (int jj = 0; jj < 2; ++jj) {
                    const uint32_t* b_col = b_col_base + (4 + jj) * b_slice_stride;
                    b0_cache1[jj] = b_col[threadID];
                    b1_cache1[jj] = b_col[threadID + 4];
                    bw0_cache1[jj] = bw_col0[4 + jj];
                    bw1_cache1[jj] = bw_col1[4 + jj];
                }

                for (int i = 0; i < Sa; ++i) {
                    const float aw0 = Aw_tile[(size_t)m0 * (size_t)Sa + (size_t)i] * qscale_m0;
                    const float aw1 = Aw_tile[(size_t)m1 * (size_t)Sa + (size_t)i] * qscale_m1;

                    const uint32_t* A_i = A_bits + (size_t)i * (size_t)TM_TOTAL * (size_t)K_STRIDE32;
                    const uint32_t a0 = A_i[(size_t)m0 * (size_t)K_STRIDE32 + (size_t)threadID];
                    const uint32_t a1 = A_i[(size_t)m1 * (size_t)K_STRIDE32 + (size_t)threadID];
                    const uint32_t a2 = A_i[(size_t)m0 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];
                    const uint32_t a3 = A_i[(size_t)m1 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];

                    float sum00 = 0.0f, sum01 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;
                    {
                        int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                        asm volatile(
                            "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_cache0[0]), "r"(b1_cache0[0]));
                        asm volatile(
                            "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_cache0[1]), "r"(b1_cache0[1]));

                        sum00 = __fmaf_rn(static_cast<float>(c0), bw0_cache0[0], sum00);
                        sum01 = __fmaf_rn(static_cast<float>(c1), bw1_cache0[0], sum01);
                        sum10 = __fmaf_rn(static_cast<float>(c2), bw0_cache0[0], sum10);
                        sum11 = __fmaf_rn(static_cast<float>(c3), bw1_cache0[0], sum11);
                        sum00 = __fmaf_rn(static_cast<float>(d0), bw0_cache0[1], sum00);
                        sum01 = __fmaf_rn(static_cast<float>(d1), bw1_cache0[1], sum01);
                        sum10 = __fmaf_rn(static_cast<float>(d2), bw0_cache0[1], sum10);
                        sum11 = __fmaf_rn(static_cast<float>(d3), bw1_cache0[1], sum11);
                    }
                    {
                        int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                        asm volatile(
                            "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_cache0[2]), "r"(b1_cache0[2]));
                        asm volatile(
                            "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_cache0[3]), "r"(b1_cache0[3]));

                        sum00 = __fmaf_rn(static_cast<float>(c0), bw0_cache0[2], sum00);
                        sum01 = __fmaf_rn(static_cast<float>(c1), bw1_cache0[2], sum01);
                        sum10 = __fmaf_rn(static_cast<float>(c2), bw0_cache0[2], sum10);
                        sum11 = __fmaf_rn(static_cast<float>(c3), bw1_cache0[2], sum11);
                        sum00 = __fmaf_rn(static_cast<float>(d0), bw0_cache0[3], sum00);
                        sum01 = __fmaf_rn(static_cast<float>(d1), bw1_cache0[3], sum01);
                        sum10 = __fmaf_rn(static_cast<float>(d2), bw0_cache0[3], sum10);
                        sum11 = __fmaf_rn(static_cast<float>(d3), bw1_cache0[3], sum11);
                    }
                    {
                        int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                        asm volatile(
                            "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_cache1[0]), "r"(b1_cache1[0]));
                        asm volatile(
                            "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_cache1[1]), "r"(b1_cache1[1]));

                        sum00 = __fmaf_rn(static_cast<float>(c0), bw0_cache1[0], sum00);
                        sum01 = __fmaf_rn(static_cast<float>(c1), bw1_cache1[0], sum01);
                        sum10 = __fmaf_rn(static_cast<float>(c2), bw0_cache1[0], sum10);
                        sum11 = __fmaf_rn(static_cast<float>(c3), bw1_cache1[0], sum11);
                        sum00 = __fmaf_rn(static_cast<float>(d0), bw0_cache1[1], sum00);
                        sum01 = __fmaf_rn(static_cast<float>(d1), bw1_cache1[1], sum01);
                        sum10 = __fmaf_rn(static_cast<float>(d2), bw0_cache1[1], sum10);
                        sum11 = __fmaf_rn(static_cast<float>(d3), bw1_cache1[1], sum11);
                    }
                    acc00 = __fmaf_rn(aw0, sum00, acc00);
                    acc01 = __fmaf_rn(aw0, sum01, acc01);
                    acc10 = __fmaf_rn(aw1, sum10, acc10);
                    acc11 = __fmaf_rn(aw1, sum11, acc11);
                }
            } else {
                const int Sb_full = Sb & ~(JBLOCK - 1);
                for (int j0 = 0; j0 < Sb_full; j0 += JBLOCK) {
                    uint32_t b0_cache[JBLOCK];
                    uint32_t b1_cache[JBLOCK];
                    float bw0_cache[JBLOCK];
                    float bw1_cache[JBLOCK];

#pragma unroll
                    for (int jj = 0; jj < JBLOCK; ++jj) {
                        const uint32_t* b_col = b_col_base + (j0 + jj) * b_slice_stride;
                        b0_cache[jj] = b_col[threadID];
                        b1_cache[jj] = b_col[threadID + 4];
                        bw0_cache[jj] = bw_col0[j0 + jj];
                        bw1_cache[jj] = bw_col1[j0 + jj];
                    }

                    for (int i = 0; i < Sa; ++i) {
                        const float aw0 = Aw_tile[(size_t)m0 * (size_t)Sa + (size_t)i] * qscale_m0;
                        const float aw1 = Aw_tile[(size_t)m1 * (size_t)Sa + (size_t)i] * qscale_m1;

                        const uint32_t* A_i = A_bits + (size_t)i * (size_t)TM_TOTAL * (size_t)K_STRIDE32;
                        const uint32_t a0 = A_i[(size_t)m0 * (size_t)K_STRIDE32 + (size_t)threadID];
                        const uint32_t a1 = A_i[(size_t)m1 * (size_t)K_STRIDE32 + (size_t)threadID];
                        const uint32_t a2 = A_i[(size_t)m0 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];
                        const uint32_t a3 = A_i[(size_t)m1 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];

                        float sum00 = 0.0f, sum01 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;
                        {
                            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                            asm volatile(
                                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_cache[0]), "r"(b1_cache[0]));
                            asm volatile(
                                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_cache[1]), "r"(b1_cache[1]));

                            sum00 = __fmaf_rn(static_cast<float>(c0), bw0_cache[0], sum00);
                            sum01 = __fmaf_rn(static_cast<float>(c1), bw1_cache[0], sum01);
                            sum10 = __fmaf_rn(static_cast<float>(c2), bw0_cache[0], sum10);
                            sum11 = __fmaf_rn(static_cast<float>(c3), bw1_cache[0], sum11);
                            sum00 = __fmaf_rn(static_cast<float>(d0), bw0_cache[1], sum00);
                            sum01 = __fmaf_rn(static_cast<float>(d1), bw1_cache[1], sum01);
                            sum10 = __fmaf_rn(static_cast<float>(d2), bw0_cache[1], sum10);
                            sum11 = __fmaf_rn(static_cast<float>(d3), bw1_cache[1], sum11);
                        }
                        {
                            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                            asm volatile(
                                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_cache[2]), "r"(b1_cache[2]));
                            asm volatile(
                                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_cache[3]), "r"(b1_cache[3]));

                            sum00 = __fmaf_rn(static_cast<float>(c0), bw0_cache[2], sum00);
                            sum01 = __fmaf_rn(static_cast<float>(c1), bw1_cache[2], sum01);
                            sum10 = __fmaf_rn(static_cast<float>(c2), bw0_cache[2], sum10);
                            sum11 = __fmaf_rn(static_cast<float>(c3), bw1_cache[2], sum11);
                            sum00 = __fmaf_rn(static_cast<float>(d0), bw0_cache[3], sum00);
                            sum01 = __fmaf_rn(static_cast<float>(d1), bw1_cache[3], sum01);
                            sum10 = __fmaf_rn(static_cast<float>(d2), bw0_cache[3], sum10);
                            sum11 = __fmaf_rn(static_cast<float>(d3), bw1_cache[3], sum11);
                        }
                        acc00 = __fmaf_rn(aw0, sum00, acc00);
                        acc01 = __fmaf_rn(aw0, sum01, acc01);
                        acc10 = __fmaf_rn(aw1, sum10, acc10);
                        acc11 = __fmaf_rn(aw1, sum11, acc11);
                    }
                }
                const int tail = Sb - Sb_full;
                if (tail) {
                    const int j0 = Sb_full;
                    uint32_t b0_tail[3];
                    uint32_t b1_tail[3];
                    float bw0_tail[3];
                    float bw1_tail[3];

#pragma unroll
                    for (int t = 0; t < 3; ++t) {
                        if (t < tail) {
                            const int j = j0 + t;
                            const uint32_t* b_col = b_col_base + j * b_slice_stride;
                            b0_tail[t] = b_col[threadID];
                            b1_tail[t] = b_col[threadID + 4];
                            bw0_tail[t] = bw_col0[j];
                            bw1_tail[t] = bw_col1[j];
                        }
                    }

                    if (tail == 1) {
                        for (int i = 0; i < Sa; ++i) {
                            const float aw0 = Aw_tile[(size_t)m0 * (size_t)Sa + (size_t)i] * qscale_m0;
                            const float aw1 = Aw_tile[(size_t)m1 * (size_t)Sa + (size_t)i] * qscale_m1;

                            const uint32_t* A_i = A_bits + (size_t)i * (size_t)TM_TOTAL * (size_t)K_STRIDE32;
                            const uint32_t a0 = A_i[(size_t)m0 * (size_t)K_STRIDE32 + (size_t)threadID];
                            const uint32_t a1 = A_i[(size_t)m1 * (size_t)K_STRIDE32 + (size_t)threadID];
                            const uint32_t a2 = A_i[(size_t)m0 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];
                            const uint32_t a3 = A_i[(size_t)m1 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];

                            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                            asm volatile(
                                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_tail[0]), "r"(b1_tail[0]));

                            acc00 = __fmaf_rn(aw0, static_cast<float>(c0) * bw0_tail[0], acc00);
                            acc01 = __fmaf_rn(aw0, static_cast<float>(c1) * bw1_tail[0], acc01);
                            acc10 = __fmaf_rn(aw1, static_cast<float>(c2) * bw0_tail[0], acc10);
                            acc11 = __fmaf_rn(aw1, static_cast<float>(c3) * bw1_tail[0], acc11);
                        }
                    } else if (tail == 2) {
                        for (int i = 0; i < Sa; ++i) {
                            const float aw0 = Aw_tile[(size_t)m0 * (size_t)Sa + (size_t)i] * qscale_m0;
                            const float aw1 = Aw_tile[(size_t)m1 * (size_t)Sa + (size_t)i] * qscale_m1;

                            const uint32_t* A_i = A_bits + (size_t)i * (size_t)TM_TOTAL * (size_t)K_STRIDE32;
                            const uint32_t a0 = A_i[(size_t)m0 * (size_t)K_STRIDE32 + (size_t)threadID];
                            const uint32_t a1 = A_i[(size_t)m1 * (size_t)K_STRIDE32 + (size_t)threadID];
                            const uint32_t a2 = A_i[(size_t)m0 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];
                            const uint32_t a3 = A_i[(size_t)m1 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];

                            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                            asm volatile(
                                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_tail[0]), "r"(b1_tail[0]));
                            asm volatile(
                                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_tail[1]), "r"(b1_tail[1]));

                            acc00 = __fmaf_rn(aw0, static_cast<float>(c0) * bw0_tail[0] + static_cast<float>(d0) * bw0_tail[1], acc00);
                            acc01 = __fmaf_rn(aw0, static_cast<float>(c1) * bw1_tail[0] + static_cast<float>(d1) * bw1_tail[1], acc01);
                            acc10 = __fmaf_rn(aw1, static_cast<float>(c2) * bw0_tail[0] + static_cast<float>(d2) * bw0_tail[1], acc10);
                            acc11 = __fmaf_rn(aw1, static_cast<float>(c3) * bw1_tail[0] + static_cast<float>(d3) * bw1_tail[1], acc11);
                        }
                    } else { 
                        for (int i = 0; i < Sa; ++i) {
                            const float aw0 = Aw_tile[(size_t)m0 * (size_t)Sa + (size_t)i] * qscale_m0;
                            const float aw1 = Aw_tile[(size_t)m1 * (size_t)Sa + (size_t)i] * qscale_m1;

                            const uint32_t* A_i = A_bits + (size_t)i * (size_t)TM_TOTAL * (size_t)K_STRIDE32;
                            const uint32_t a0 = A_i[(size_t)m0 * (size_t)K_STRIDE32 + (size_t)threadID];
                            const uint32_t a1 = A_i[(size_t)m1 * (size_t)K_STRIDE32 + (size_t)threadID];
                            const uint32_t a2 = A_i[(size_t)m0 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];
                            const uint32_t a3 = A_i[(size_t)m1 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];

                            float sum00 = 0.0f, sum01 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;
                            {
                                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                                int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                                asm volatile(
                                    "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                    : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_tail[0]), "r"(b1_tail[0]));
                                asm volatile(
                                    "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                    : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_tail[1]), "r"(b1_tail[1]));

                                sum00 = __fmaf_rn(static_cast<float>(c0), bw0_tail[0], sum00);
                                sum01 = __fmaf_rn(static_cast<float>(c1), bw1_tail[0], sum01);
                                sum10 = __fmaf_rn(static_cast<float>(c2), bw0_tail[0], sum10);
                                sum11 = __fmaf_rn(static_cast<float>(c3), bw1_tail[0], sum11);
                                sum00 = __fmaf_rn(static_cast<float>(d0), bw0_tail[1], sum00);
                                sum01 = __fmaf_rn(static_cast<float>(d1), bw1_tail[1], sum01);
                                sum10 = __fmaf_rn(static_cast<float>(d2), bw0_tail[1], sum10);
                                sum11 = __fmaf_rn(static_cast<float>(d3), bw1_tail[1], sum11);
                            }
                            {
                                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                                asm volatile(
                                    "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                    : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_tail[2]), "r"(b1_tail[2]));

                                sum00 = __fmaf_rn(static_cast<float>(c0), bw0_tail[2], sum00);
                                sum01 = __fmaf_rn(static_cast<float>(c1), bw1_tail[2], sum01);
                                sum10 = __fmaf_rn(static_cast<float>(c2), bw0_tail[2], sum10);
                                sum11 = __fmaf_rn(static_cast<float>(c3), bw1_tail[2], sum11);
                            }

                            acc00 = __fmaf_rn(aw0, sum00, acc00);
                            acc01 = __fmaf_rn(aw0, sum01, acc01);
                            acc10 = __fmaf_rn(aw1, sum10, acc10);
                            acc11 = __fmaf_rn(aw1, sum11, acc11);
                        }
                    }
                }
            }
        } else {
            for (int i = 0; i < Sa; ++i) {
                const float aw0 = Aw_tile[(size_t)m0 * (size_t)Sa + (size_t)i] * qscale_m0;
                const float aw1 = Aw_tile[(size_t)m1 * (size_t)Sa + (size_t)i] * qscale_m1;

                const uint32_t* A_i = A_bits + (size_t)i * (size_t)TM_TOTAL * (size_t)K_STRIDE32;
                const uint32_t a0 = A_i[(size_t)m0 * (size_t)K_STRIDE32 + (size_t)threadID];
                const uint32_t a1 = A_i[(size_t)m1 * (size_t)K_STRIDE32 + (size_t)threadID];
                const uint32_t a2 = A_i[(size_t)m0 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];
                const uint32_t a3 = A_i[(size_t)m1 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];

                float sum00 = 0.0f, sum01 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;
                for (int j = 0; j < Sb; ++j) {
                    const float bw0 = Bw_tile[(size_t)col0 * (size_t)Sb + (size_t)j];
                    const float bw1 = Bw_tile[(size_t)col1 * (size_t)Sb + (size_t)j];

                    const uint32_t* B_j = B_bits + (size_t)j * (size_t)TN * (size_t)K_STRIDE32;
                    const uint32_t* B_col = B_j + (size_t)(col_base + groupID) * (size_t)K_STRIDE32;
                    const uint32_t b0 = B_col[(size_t)threadID];
                    const uint32_t b1 = B_col[(size_t)(threadID + 4)];

                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                    asm volatile(
                        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));

                    sum00 = __fmaf_rn(static_cast<float>(c0), bw0, sum00);
                    sum01 = __fmaf_rn(static_cast<float>(c1), bw1, sum01);
                    sum10 = __fmaf_rn(static_cast<float>(c2), bw0, sum10);
                    sum11 = __fmaf_rn(static_cast<float>(c3), bw1, sum11);
                }
                acc00 = __fmaf_rn(aw0, sum00, acc00);
                acc01 = __fmaf_rn(aw0, sum01, acc01);
                acc10 = __fmaf_rn(aw1, sum10, acc10);
                acc11 = __fmaf_rn(aw1, sum11, acc11);
            }
        }

        if (can_cpasync && (chunk + 1 < chunks)) {
            bsi_cp_async_wait_all();
        }
        __syncthreads();
        if (can_cpasync) stage ^= 1;
    }

    const int q_out0 = q0 + m0;
    const int q_out1 = q0 + m1;
    const int r_out0 = r0 + col0;
    const int r_out1 = r0 + col1;

    const bool identity_q = (query_indices == nullptr);
    const bool identity_r = (key_indices == nullptr);
    if (full_q_tile && full_r_tile && identity_q && identity_r) {
        out_global[(size_t)q_out0 * (size_t)R_total + (size_t)r_out0] = acc00 * scale_inv;
        out_global[(size_t)q_out0 * (size_t)R_total + (size_t)r_out1] = acc01 * scale_inv;
        out_global[(size_t)q_out1 * (size_t)R_total + (size_t)r_out0] = acc10 * scale_inv;
        out_global[(size_t)q_out1 * (size_t)R_total + (size_t)r_out1] = acc11 * scale_inv;
    } else if (full_q_tile && full_r_tile) {
        long long gq0 = 0, gq1 = 0;
        if (threadID == 0) {
            gq0 = bsi_load_index_or_identity(query_indices, q_out0);
            gq1 = bsi_load_index_or_identity(query_indices, q_out1);
        }
        gq0 = __shfl_sync(0xffffffff, gq0, lane & ~3);
        gq1 = __shfl_sync(0xffffffff, gq1, lane & ~3);

        long long gr0 = 0, gr1 = 0;
        if (groupID == 0) {
            gr0 = bsi_load_index_or_identity(key_indices, r_out0);
            gr1 = bsi_load_index_or_identity(key_indices, r_out1);
        }
        gr0 = __shfl_sync(0xffffffff, gr0, threadID);
        gr1 = __shfl_sync(0xffffffff, gr1, threadID);

        out_global[(size_t)gq0 * (size_t)R_total + (size_t)gr0] = acc00 * scale_inv;
        out_global[(size_t)gq0 * (size_t)R_total + (size_t)gr1] = acc01 * scale_inv;
        out_global[(size_t)gq1 * (size_t)R_total + (size_t)gr0] = acc10 * scale_inv;
        out_global[(size_t)gq1 * (size_t)R_total + (size_t)gr1] = acc11 * scale_inv;
    } else if (identity_q && identity_r) {
        if (q_out0 < Q && r_out0 < R) {
            out_global[(size_t)q_out0 * (size_t)R_total + (size_t)r_out0] = acc00 * scale_inv;
        }
        if (q_out0 < Q && r_out1 < R) {
            out_global[(size_t)q_out0 * (size_t)R_total + (size_t)r_out1] = acc01 * scale_inv;
        }
        if (q_out1 < Q && r_out0 < R) {
            out_global[(size_t)q_out1 * (size_t)R_total + (size_t)r_out0] = acc10 * scale_inv;
        }
        if (q_out1 < Q && r_out1 < R) {
            out_global[(size_t)q_out1 * (size_t)R_total + (size_t)r_out1] = acc11 * scale_inv;
        }
    } else {
        long long gq0 = 0, gq1 = 0;
        if (threadID == 0) {
            if (q_out0 < Q) gq0 = bsi_load_index_or_identity(query_indices, q_out0);
            if (q_out1 < Q) gq1 = bsi_load_index_or_identity(query_indices, q_out1);
        }
        gq0 = __shfl_sync(0xffffffff, gq0, lane & ~3);
        gq1 = __shfl_sync(0xffffffff, gq1, lane & ~3);

        long long gr0 = 0, gr1 = 0;
        if (groupID == 0) {
            if (r_out0 < R) gr0 = bsi_load_index_or_identity(key_indices, r_out0);
            if (r_out1 < R) gr1 = bsi_load_index_or_identity(key_indices, r_out1);
        }
        gr0 = __shfl_sync(0xffffffff, gr0, threadID);
        gr1 = __shfl_sync(0xffffffff, gr1, threadID);

        if (q_out0 < Q && r_out0 < R) {
            out_global[(size_t)gq0 * (size_t)R_total + (size_t)gr0] = acc00 * scale_inv;
        }
        if (q_out0 < Q && r_out1 < R) {
            out_global[(size_t)gq0 * (size_t)R_total + (size_t)gr1] = acc01 * scale_inv;
        }
        if (q_out1 < Q && r_out0 < R) {
            out_global[(size_t)gq1 * (size_t)R_total + (size_t)gr0] = acc10 * scale_inv;
        }
        if (q_out1 < Q && r_out1 < R) {
            out_global[(size_t)gq1 * (size_t)R_total + (size_t)gr1] = acc11 * scale_inv;
        }
    }
#else
    (void)A;
    (void)Aw;
    (void)Sa;
    (void)W64;
    (void)B;
    (void)Bw;
    (void)Sb;
    (void)R;
    (void)Q;
    (void)key_indices;
    (void)query_indices;
    (void)scale_inv;
    (void)R_total;
    (void)out_global;
    (void)use_cpasync;
#endif
}

extern "C" __global__ __launch_bounds__(256, 2)
void popcount_weighted_keys_literal_fused_bmma_tc_kernel_tm32_fixed76_chunkscale(
    const unsigned long long* __restrict__ A, 
    const float* __restrict__ A_chunk_scales, 
    int A_scale_stride,
    int W64,
    const unsigned long long* __restrict__ B, 
    const float* __restrict__ Bw,             
    int R,
    int Q,
    float scale_inv,
    int R_total,
    float* __restrict__ out_global)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    constexpr int SA = 7;
    constexpr int SB = 6;
    constexpr int TM_TOTAL = 32;
    constexpr int TM = 16;
    constexpr int TN = 32;
    constexpr int WARPS_PER_QTILE = 4;
    constexpr int QTILES = TM_TOTAL / TM;
    constexpr int K_BITS = 256;
    constexpr int K_WORDS64 = K_BITS / 64;
    constexpr int K_WORDS32 = K_BITS / 32;
    constexpr int K_STRIDE32 = K_WORDS32 + 4;

    if (blockDim.x != (WARPS_PER_QTILE * QTILES * 32)) return;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int q_tile_id = warp_id >> 2;
    const int warp_in_tile = warp_id & (WARPS_PER_QTILE - 1);

    const int q0 = blockIdx.y * TM_TOTAL;
    const int r0 = blockIdx.x * TN;

    extern __shared__ unsigned char smem_raw[];
    uintptr_t p = reinterpret_cast<uintptr_t>(smem_raw);
    p = (p + 15u) & ~uintptr_t(15u);

    constexpr int stages = 2;
    constexpr size_t A_words = (size_t)SA * (size_t)TM_TOTAL * (size_t)K_STRIDE32;
    constexpr size_t B_words = (size_t)SB * (size_t)TN * (size_t)K_STRIDE32;
    auto* A_bits0 = reinterpret_cast<uint32_t*>(p);
    p += (size_t)stages * A_words * sizeof(uint32_t);
    auto* B_bits0 = reinterpret_cast<uint32_t*>(p);
    p += (size_t)stages * B_words * sizeof(uint32_t);
    (void)p;

    const int groupID = lane >> 2;
    const int threadID = lane & 3;
    const int row0 = groupID;
    const int row1 = groupID + 8;
    const int col_base = warp_in_tile * 8;
    const int col0 = col_base + threadID * 2;
    const int col1 = col0 + 1;
    const int m0 = q_tile_id * TM + row0;
    const int m1 = q_tile_id * TM + row1;

    const float bscale0 = __ldg(&Bw[((size_t)(r0 + col0) * (size_t)SB) + 0]);
    const float bscale1 = __ldg(&Bw[((size_t)(r0 + col1) * (size_t)SB) + 0]);

    const int chunks = W64 / K_WORDS64;

    {
        uint32_t* A_bits = A_bits0;
        uint32_t* B_bits = B_bits0;
        constexpr int K_WORDS64_16B = K_WORDS64 / 2;
        for (int idx = threadIdx.x; idx < TM_TOTAL * SA * K_WORDS64_16B; idx += blockDim.x) {
            int t = idx;
            const int w64_pair = t & (K_WORDS64_16B - 1);
            t >>= 1;
            const int m = t & (TM_TOTAL - 1);
            const int i = t >> 5;
            const int q = q0 + m;
            const unsigned long long* a_slice = A + ((size_t)q * (size_t)SA + (size_t)i) * (size_t)W64;
            const int w64_i = w64_pair << 1;
            const int base = ((i * TM_TOTAL + m) * K_STRIDE32) + (w64_i << 1);
            bsi_cp_async_cg_16B(A_bits + base, &a_slice[(size_t)0 * (size_t)K_WORDS64 + (size_t)w64_i]);
        }
        for (int idx = threadIdx.x; idx < TN * SB * K_WORDS64_16B; idx += blockDim.x) {
            int t = idx;
            const int w64_pair = t & (K_WORDS64_16B - 1);
            t >>= 1;
            const int n = t & (TN - 1);
            const int j = t >> 5;
            const int r = r0 + n;
            const unsigned long long* b_slice = B + ((size_t)r * (size_t)SB + (size_t)j) * (size_t)W64;
            const int w64_i = w64_pair << 1;
            const int base = ((j * TN + n) * K_STRIDE32) + (w64_i << 1);
            bsi_cp_async_cg_16B(B_bits + base, &b_slice[(size_t)0 * (size_t)K_WORDS64 + (size_t)w64_i]);
        }
        bsi_cp_async_commit_group();
        bsi_cp_async_wait_all();
        __syncthreads();
    }

    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;
    int stage = 0;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        uint32_t* A_bits = (stage == 0) ? A_bits0 : (A_bits0 + A_words);
        uint32_t* B_bits = (stage == 0) ? B_bits0 : (B_bits0 + B_words);

        if (chunk + 1 < chunks) {
            const int next_stage = stage ^ 1;
            uint32_t* A_bits_next = (next_stage == 0) ? A_bits0 : (A_bits0 + A_words);
            uint32_t* B_bits_next = (next_stage == 0) ? B_bits0 : (B_bits0 + B_words);
            const int next_chunk = chunk + 1;
            constexpr int K_WORDS64_16B = K_WORDS64 / 2;
            for (int idx = threadIdx.x; idx < TM_TOTAL * SA * K_WORDS64_16B; idx += blockDim.x) {
                int t = idx;
                const int w64_pair = t & (K_WORDS64_16B - 1);
                t >>= 1;
                const int m = t & (TM_TOTAL - 1);
                const int i = t >> 5;
                const int q = q0 + m;
                const unsigned long long* a_slice = A + ((size_t)q * (size_t)SA + (size_t)i) * (size_t)W64;
                const int w64_i = w64_pair << 1;
                const int base = ((i * TM_TOTAL + m) * K_STRIDE32) + (w64_i << 1);
                bsi_cp_async_cg_16B(
                    A_bits_next + base,
                    &a_slice[(size_t)next_chunk * (size_t)K_WORDS64 + (size_t)w64_i]);
            }
            for (int idx = threadIdx.x; idx < TN * SB * K_WORDS64_16B; idx += blockDim.x) {
                int t = idx;
                const int w64_pair = t & (K_WORDS64_16B - 1);
                t >>= 1;
                const int n = t & (TN - 1);
                const int j = t >> 5;
                const int r = r0 + n;
                const unsigned long long* b_slice = B + ((size_t)r * (size_t)SB + (size_t)j) * (size_t)W64;
                const int w64_i = w64_pair << 1;
                const int base = ((j * TN + n) * K_STRIDE32) + (w64_i << 1);
                bsi_cp_async_cg_16B(
                    B_bits_next + base,
                    &b_slice[(size_t)next_chunk * (size_t)K_WORDS64 + (size_t)w64_i]);
            }
            bsi_cp_async_commit_group();
        }

        float qscale_m0 = 1.0f;
        float qscale_m1 = 1.0f;
        if (threadID == 0) {
            const int q_m0 = q0 + m0;
            const int q_m1 = q0 + m1;
            qscale_m0 = __ldg(&A_chunk_scales[(size_t)q_m0 * (size_t)A_scale_stride + (size_t)chunk]);
            qscale_m1 = __ldg(&A_chunk_scales[(size_t)q_m1 * (size_t)A_scale_stride + (size_t)chunk]);
        }
        qscale_m0 = __shfl_sync(0xffffffff, qscale_m0, lane & ~3);
        qscale_m1 = __shfl_sync(0xffffffff, qscale_m1, lane & ~3);

        uint32_t b0_cache[SB];
        uint32_t b1_cache[SB];
        const int b_slice_stride = TN * K_STRIDE32;
        const uint32_t* b_col_base = B_bits + (col_base + groupID) * K_STRIDE32;
#pragma unroll
        for (int j = 0; j < SB; ++j) {
            const uint32_t* b_col = b_col_base + j * b_slice_stride;
            b0_cache[j] = b_col[threadID];
            b1_cache[j] = b_col[threadID + 4];
        }

        int chunk00 = 0, chunk01 = 0, chunk10 = 0, chunk11 = 0;
#pragma unroll
        for (int i = 0; i < SA; ++i) {
            const uint32_t* A_i = A_bits + (size_t)i * (size_t)TM_TOTAL * (size_t)K_STRIDE32;
            const uint32_t a0 = A_i[(size_t)m0 * (size_t)K_STRIDE32 + (size_t)threadID];
            const uint32_t a1 = A_i[(size_t)m1 * (size_t)K_STRIDE32 + (size_t)threadID];
            const uint32_t a2 = A_i[(size_t)m0 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];
            const uint32_t a3 = A_i[(size_t)m1 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];

#pragma unroll
            for (int j = 0; j < SB; j += 2) {
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                asm volatile(
                    "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_cache[j]), "r"(b1_cache[j]));

                int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                asm volatile(
                    "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                    : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_cache[j + 1]), "r"(b1_cache[j + 1]));

                const int shift0 = i + j;
                const int shift1 = shift0 + 1;
                const bool neg0 = ((i == (SA - 1)) ^ (j == (SB - 1)));
                const bool neg1 = ((i == (SA - 1)) ^ ((j + 1) == (SB - 1)));

                const int v0 = c0 << shift0;
                const int v1 = c1 << shift0;
                const int v2 = c2 << shift0;
                const int v3 = c3 << shift0;
                if (neg0) {
                    chunk00 -= v0; chunk01 -= v1; chunk10 -= v2; chunk11 -= v3;
                } else {
                    chunk00 += v0; chunk01 += v1; chunk10 += v2; chunk11 += v3;
                }

                const int w0 = d0 << shift1;
                const int w1 = d1 << shift1;
                const int w2 = d2 << shift1;
                const int w3 = d3 << shift1;
                if (neg1) {
                    chunk00 -= w0; chunk01 -= w1; chunk10 -= w2; chunk11 -= w3;
                } else {
                    chunk00 += w0; chunk01 += w1; chunk10 += w2; chunk11 += w3;
                }
            }
        }

        const float s00 = qscale_m0 * bscale0;
        const float s01 = qscale_m0 * bscale1;
        const float s10 = qscale_m1 * bscale0;
        const float s11 = qscale_m1 * bscale1;
        acc00 = __fmaf_rn(static_cast<float>(chunk00), s00, acc00);
        acc01 = __fmaf_rn(static_cast<float>(chunk01), s01, acc01);
        acc10 = __fmaf_rn(static_cast<float>(chunk10), s10, acc10);
        acc11 = __fmaf_rn(static_cast<float>(chunk11), s11, acc11);

        if (chunk + 1 < chunks) {
            bsi_cp_async_wait_all();
        }
        __syncthreads();
        stage ^= 1;
    }

    const int q_out0 = q0 + m0;
    const int q_out1 = q0 + m1;
    const int r_out0 = r0 + col0;
    const int r_out1 = r0 + col1;
    out_global[(size_t)q_out0 * (size_t)R_total + (size_t)r_out0] = acc00 * scale_inv;
    out_global[(size_t)q_out0 * (size_t)R_total + (size_t)r_out1] = acc01 * scale_inv;
    out_global[(size_t)q_out1 * (size_t)R_total + (size_t)r_out0] = acc10 * scale_inv;
    out_global[(size_t)q_out1 * (size_t)R_total + (size_t)r_out1] = acc11 * scale_inv;
#else
    (void)A;
    (void)A_chunk_scales;
    (void)A_scale_stride;
    (void)W64;
    (void)B;
    (void)Bw;
    (void)R;
    (void)Q;
    (void)scale_inv;
    (void)R_total;
    (void)out_global;
#endif
}

template <int R_SWEEP>
__device__ __forceinline__ void bsi_fixed76_tm32_chunkscale_rsweep_body(
    unsigned char* __restrict__ smem_raw,
    const unsigned long long* __restrict__ A, 
    const float* __restrict__ A_chunk_scales, 
    int A_scale_stride,
    int W64,
    const unsigned long long* __restrict__ B, 
    const float* __restrict__ Bw,             
    int R,
    int Q,
    float scale_inv,
    int R_total,
    float* __restrict__ out_global)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    static_assert(R_SWEEP == 2 || R_SWEEP == 4, "R_SWEEP must be 2 or 4");
    constexpr int SA = 7;
    constexpr int SB = 6;
    constexpr int TM_TOTAL = 32;
    constexpr int TM = 16;
    constexpr int TN = 32;
    constexpr int WARPS_PER_QTILE = 4;
    constexpr int QTILES = TM_TOTAL / TM;
    constexpr int K_BITS = 256;
    constexpr int K_WORDS64 = K_BITS / 64;
    constexpr int K_WORDS32 = K_BITS / 32;
    constexpr int K_STRIDE32 = K_WORDS32 + 4;

    if (blockDim.x != (WARPS_PER_QTILE * QTILES * 32)) return;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int q_tile_id = warp_id >> 2;
    const int warp_in_tile = warp_id & (WARPS_PER_QTILE - 1);

    const int q0 = blockIdx.y * TM_TOTAL;
    const int r_base = blockIdx.x * (TN * R_SWEEP);

    uintptr_t p = reinterpret_cast<uintptr_t>(smem_raw);
    p = (p + 15u) & ~uintptr_t(15u);

    constexpr int stages = 2;
    constexpr size_t A_words = (size_t)SA * (size_t)TM_TOTAL * (size_t)K_STRIDE32;
    constexpr size_t B_words = (size_t)SB * (size_t)TN * (size_t)K_STRIDE32;
    constexpr size_t B_words_sweep = (size_t)R_SWEEP * B_words;
    auto* A_bits0 = reinterpret_cast<uint32_t*>(p);
    p += (size_t)stages * A_words * sizeof(uint32_t);
    auto* B_bits0 = reinterpret_cast<uint32_t*>(p);
    p += (size_t)stages * B_words_sweep * sizeof(uint32_t);
    auto* acc0 = reinterpret_cast<float4*>(p);
    p += (size_t)R_SWEEP * (size_t)blockDim.x * sizeof(float4);
    (void)p;

    const int groupID = lane >> 2;
    const int threadID = lane & 3;
    const int row0 = groupID;
    const int row1 = groupID + 8;
    const int col_base = warp_in_tile * 8;
    const int col0 = col_base + threadID * 2;
    const int col1 = col0 + 1;
    const int m0 = q_tile_id * TM + row0;
    const int m1 = q_tile_id * TM + row1;

    float bscale0[R_SWEEP];
    float bscale1[R_SWEEP];
#pragma unroll
    for (int t = 0; t < R_SWEEP; ++t) {
        const int r0 = r_base + t * TN;
        bscale0[t] = __ldg(&Bw[((size_t)(r0 + col0) * (size_t)SB) + 0]);
        bscale1[t] = __ldg(&Bw[((size_t)(r0 + col1) * (size_t)SB) + 0]);
    }

#pragma unroll
    for (int t = 0; t < R_SWEEP; ++t) {
        acc0[(size_t)t * (size_t)blockDim.x + (size_t)threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    const int chunks = W64 / K_WORDS64;

    {
        uint32_t* A_bits = A_bits0;
        uint32_t* B_bits = B_bits0;
        constexpr int K_WORDS64_16B = K_WORDS64 / 2;
        for (int idx = threadIdx.x; idx < TM_TOTAL * SA * K_WORDS64_16B; idx += blockDim.x) {
            int t = idx;
            const int w64_pair = t & (K_WORDS64_16B - 1);
            t >>= 1;
            const int m = t & (TM_TOTAL - 1);
            const int i = t >> 5;
            const int q = q0 + m;
            const unsigned long long* a_slice = A + ((size_t)q * (size_t)SA + (size_t)i) * (size_t)W64;
            const int w64_i = w64_pair << 1;
            const int base = ((i * TM_TOTAL + m) * K_STRIDE32) + (w64_i << 1);
            bsi_cp_async_cg_16B(A_bits + base, &a_slice[(size_t)0 * (size_t)K_WORDS64 + (size_t)w64_i]);
        }
        constexpr int B_CHUNK_ELEMS = TN * SB * K_WORDS64_16B;
#pragma unroll
        for (int t = 0; t < R_SWEEP; ++t) {
            for (int idx = threadIdx.x; idx < B_CHUNK_ELEMS; idx += blockDim.x) {
                int u = idx;
                const int w64_pair = u & (K_WORDS64_16B - 1);
                u >>= 1;
                const int n = u & (TN - 1);
                const int j = u >> 5;
                const int r = r_base + t * TN + n;
                const unsigned long long* b_slice = B + ((size_t)r * (size_t)SB + (size_t)j) * (size_t)W64;
                const int w64_i = w64_pair << 1;
                const size_t base = (size_t)t * B_words + ((size_t)j * (size_t)TN + (size_t)n) * (size_t)K_STRIDE32 + (size_t)(w64_i << 1);
                bsi_cp_async_cg_16B(B_bits + base, &b_slice[(size_t)0 * (size_t)K_WORDS64 + (size_t)w64_i]);
            }
        }
        bsi_cp_async_commit_group();
        bsi_cp_async_wait_all();
        __syncthreads();
    }

    int stage = 0;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        uint32_t* A_bits = (stage == 0) ? A_bits0 : (A_bits0 + A_words);
        uint32_t* B_bits = (stage == 0) ? B_bits0 : (B_bits0 + B_words_sweep);

        if (chunk + 1 < chunks) {
            const int next_stage = stage ^ 1;
            uint32_t* A_bits_next = (next_stage == 0) ? A_bits0 : (A_bits0 + A_words);
            uint32_t* B_bits_next = (next_stage == 0) ? B_bits0 : (B_bits0 + B_words_sweep);
            const int next_chunk = chunk + 1;
            constexpr int K_WORDS64_16B = K_WORDS64 / 2;
            for (int idx = threadIdx.x; idx < TM_TOTAL * SA * K_WORDS64_16B; idx += blockDim.x) {
                int t = idx;
                const int w64_pair = t & (K_WORDS64_16B - 1);
                t >>= 1;
                const int m = t & (TM_TOTAL - 1);
                const int i = t >> 5;
                const int q = q0 + m;
                const unsigned long long* a_slice = A + ((size_t)q * (size_t)SA + (size_t)i) * (size_t)W64;
                const int w64_i = w64_pair << 1;
                const int base = ((i * TM_TOTAL + m) * K_STRIDE32) + (w64_i << 1);
                bsi_cp_async_cg_16B(
                    A_bits_next + base,
                    &a_slice[(size_t)next_chunk * (size_t)K_WORDS64 + (size_t)w64_i]);
            }
            constexpr int B_CHUNK_ELEMS = TN * SB * K_WORDS64_16B;
#pragma unroll
            for (int t = 0; t < R_SWEEP; ++t) {
                for (int idx = threadIdx.x; idx < B_CHUNK_ELEMS; idx += blockDim.x) {
                    int u = idx;
                    const int w64_pair = u & (K_WORDS64_16B - 1);
                    u >>= 1;
                    const int n = u & (TN - 1);
                    const int j = u >> 5;
                    const int r = r_base + t * TN + n;
                    const unsigned long long* b_slice = B + ((size_t)r * (size_t)SB + (size_t)j) * (size_t)W64;
                    const int w64_i = w64_pair << 1;
                    const size_t base = (size_t)t * B_words + ((size_t)j * (size_t)TN + (size_t)n) * (size_t)K_STRIDE32 + (size_t)(w64_i << 1);
                    bsi_cp_async_cg_16B(
                        B_bits_next + base,
                        &b_slice[(size_t)next_chunk * (size_t)K_WORDS64 + (size_t)w64_i]);
                }
            }
            bsi_cp_async_commit_group();
        }

        float qscale_m0 = 1.0f;
        float qscale_m1 = 1.0f;
        if (threadID == 0) {
            const int q_m0 = q0 + m0;
            const int q_m1 = q0 + m1;
            qscale_m0 = __ldg(&A_chunk_scales[(size_t)q_m0 * (size_t)A_scale_stride + (size_t)chunk]);
            qscale_m1 = __ldg(&A_chunk_scales[(size_t)q_m1 * (size_t)A_scale_stride + (size_t)chunk]);
        }
        qscale_m0 = __shfl_sync(0xffffffff, qscale_m0, lane & ~3);
        qscale_m1 = __shfl_sync(0xffffffff, qscale_m1, lane & ~3);

#pragma unroll
        for (int t_pair = 0; t_pair < R_SWEEP; t_pair += 2) {
            const int t0 = t_pair;
            const int t1 = t_pair + 1;
            const uint32_t* B0 = B_bits + (size_t)t0 * B_words;
            const uint32_t* B1 = B_bits + (size_t)t1 * B_words;

            uint32_t b0_0[SB];
            uint32_t b1_0[SB];
            uint32_t b0_1[SB];
            uint32_t b1_1[SB];

            const int b_slice_stride = TN * K_STRIDE32;
            const uint32_t* b_col_base0 = B0 + (col_base + groupID) * K_STRIDE32;
            const uint32_t* b_col_base1 = B1 + (col_base + groupID) * K_STRIDE32;
#pragma unroll
            for (int j = 0; j < SB; ++j) {
                const uint32_t* b_col0 = b_col_base0 + j * b_slice_stride;
                const uint32_t* b_col1 = b_col_base1 + j * b_slice_stride;
                b0_0[j] = b_col0[threadID];
                b1_0[j] = b_col0[threadID + 4];
                b0_1[j] = b_col1[threadID];
                b1_1[j] = b_col1[threadID + 4];
            }

            int chunk00_0 = 0, chunk01_0 = 0, chunk10_0 = 0, chunk11_0 = 0;
            int chunk00_1 = 0, chunk01_1 = 0, chunk10_1 = 0, chunk11_1 = 0;
#pragma unroll
            for (int i = 0; i < SA; ++i) {
                const uint32_t* A_i = A_bits + (size_t)i * (size_t)TM_TOTAL * (size_t)K_STRIDE32;
                const uint32_t a0 = A_i[(size_t)m0 * (size_t)K_STRIDE32 + (size_t)threadID];
                const uint32_t a1 = A_i[(size_t)m1 * (size_t)K_STRIDE32 + (size_t)threadID];
                const uint32_t a2 = A_i[(size_t)m0 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];
                const uint32_t a3 = A_i[(size_t)m1 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];

#pragma unroll
                for (int j = 0; j < SB; ++j) {
                    const int shift = i + j;
                    const bool neg = ((i == (SA - 1)) ^ (j == (SB - 1)));

                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                    asm volatile(
                        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_0[j]), "r"(b1_0[j]));

                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    asm volatile(
                        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_1[j]), "r"(b1_1[j]));

                    const int v0 = c0 << shift; const int v1 = c1 << shift; const int v2 = c2 << shift; const int v3 = c3 << shift;
                    const int w0 = d0 << shift; const int w1 = d1 << shift; const int w2 = d2 << shift; const int w3 = d3 << shift;
                    if (neg) {
                        chunk00_0 -= v0; chunk01_0 -= v1; chunk10_0 -= v2; chunk11_0 -= v3;
                        chunk00_1 -= w0; chunk01_1 -= w1; chunk10_1 -= w2; chunk11_1 -= w3;
                    } else {
                        chunk00_0 += v0; chunk01_0 += v1; chunk10_0 += v2; chunk11_0 += v3;
                        chunk00_1 += w0; chunk01_1 += w1; chunk10_1 += w2; chunk11_1 += w3;
                    }
                }
            }

            {
                const float s00 = qscale_m0 * bscale0[t0];
                const float s01 = qscale_m0 * bscale1[t0];
                const float s10 = qscale_m1 * bscale0[t0];
                const float s11 = qscale_m1 * bscale1[t0];
                float4 acc = acc0[(size_t)t0 * (size_t)blockDim.x + (size_t)threadIdx.x];
                acc.x = __fmaf_rn(static_cast<float>(chunk00_0), s00, acc.x);
                acc.y = __fmaf_rn(static_cast<float>(chunk01_0), s01, acc.y);
                acc.z = __fmaf_rn(static_cast<float>(chunk10_0), s10, acc.z);
                acc.w = __fmaf_rn(static_cast<float>(chunk11_0), s11, acc.w);
                acc0[(size_t)t0 * (size_t)blockDim.x + (size_t)threadIdx.x] = acc;
            }
            {
                const float s00 = qscale_m0 * bscale0[t1];
                const float s01 = qscale_m0 * bscale1[t1];
                const float s10 = qscale_m1 * bscale0[t1];
                const float s11 = qscale_m1 * bscale1[t1];
                float4 acc = acc0[(size_t)t1 * (size_t)blockDim.x + (size_t)threadIdx.x];
                acc.x = __fmaf_rn(static_cast<float>(chunk00_1), s00, acc.x);
                acc.y = __fmaf_rn(static_cast<float>(chunk01_1), s01, acc.y);
                acc.z = __fmaf_rn(static_cast<float>(chunk10_1), s10, acc.z);
                acc.w = __fmaf_rn(static_cast<float>(chunk11_1), s11, acc.w);
                acc0[(size_t)t1 * (size_t)blockDim.x + (size_t)threadIdx.x] = acc;
            }
        }

        if (chunk + 1 < chunks) {
            bsi_cp_async_wait_all();
        }
        __syncthreads();
        stage ^= 1;
    }

    const int q_out0 = q0 + m0;
    const int q_out1 = q0 + m1;
#pragma unroll
    for (int t = 0; t < R_SWEEP; ++t) {
        const int r0 = r_base + t * TN;
        const int r_out0 = r0 + col0;
        const int r_out1 = r0 + col1;
        const float4 acc = acc0[(size_t)t * (size_t)blockDim.x + (size_t)threadIdx.x];
        out_global[(size_t)q_out0 * (size_t)R_total + (size_t)r_out0] = acc.x * scale_inv;
        out_global[(size_t)q_out0 * (size_t)R_total + (size_t)r_out1] = acc.y * scale_inv;
        out_global[(size_t)q_out1 * (size_t)R_total + (size_t)r_out0] = acc.z * scale_inv;
        out_global[(size_t)q_out1 * (size_t)R_total + (size_t)r_out1] = acc.w * scale_inv;
    }
#else
    (void)A;
    (void)A_chunk_scales;
    (void)A_scale_stride;
    (void)W64;
    (void)B;
    (void)Bw;
    (void)R;
    (void)Q;
    (void)scale_inv;
    (void)R_total;
    (void)out_global;
#endif
}

extern "C" __global__ __launch_bounds__(256, 2)
void popcount_weighted_keys_literal_fused_bmma_tc_kernel_tm32_fixed76_chunkscale_rsweep2(
    const unsigned long long* __restrict__ A,
    const float* __restrict__ A_chunk_scales,
    int A_scale_stride,
    int W64,
    const unsigned long long* __restrict__ B,
    const float* __restrict__ Bw,
    int R,
    int Q,
    float scale_inv,
    int R_total,
    float* __restrict__ out_global)
{
    extern __shared__ unsigned char smem_raw[];
    bsi_fixed76_tm32_chunkscale_rsweep_body<2>(
        smem_raw, A, A_chunk_scales, A_scale_stride, W64, B, Bw, R, Q, scale_inv, R_total, out_global);
}

extern "C" __global__ __launch_bounds__(256, 2)
void popcount_weighted_keys_literal_fused_bmma_tc_kernel_tm32_fixed76_chunkscale_rsweep4(
    const unsigned long long* __restrict__ A,
    const float* __restrict__ A_chunk_scales,
    int A_scale_stride,
    int W64,
    const unsigned long long* __restrict__ B,
    const float* __restrict__ Bw,
    int R,
    int Q,
    float scale_inv,
    int R_total,
    float* __restrict__ out_global)
{
    extern __shared__ unsigned char smem_raw[];
    bsi_fixed76_tm32_chunkscale_rsweep_body<4>(
        smem_raw, A, A_chunk_scales, A_scale_stride, W64, B, Bw, R, Q, scale_inv, R_total, out_global);
}

template <int R_SWEEP>
__device__ __forceinline__ void bsi_fixed76_tm32_chunkscale_rsweep_body_tma_tensorB(
    unsigned char* __restrict__ smem_raw,
    const unsigned long long* __restrict__ A, 
    const float* __restrict__ A_chunk_scales, 
    int A_scale_stride,
    int W64,
    const unsigned long long* __restrict__ B, 
    const float* __restrict__ Bw,             
    const void* __restrict__ B_tensor_map,
    int R,
    int Q,
    float scale_inv,
    int R_total,
    float* __restrict__ out_global)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) && defined(__cccl_lib_local_barrier_arrive_tx)
    static_assert(R_SWEEP == 2 || R_SWEEP == 4, "R_SWEEP must be 2 or 4");
    namespace ptx = cuda::ptx;
    using barrier_t = cuda::barrier<cuda::thread_scope_block>;

    constexpr int SA = 7;
    constexpr int SB = 6;
    constexpr int TM_TOTAL = 32;
    constexpr int TM = 16;
    constexpr int TN = 32;
    constexpr int WARPS_PER_QTILE = 4;
    constexpr int QTILES = TM_TOTAL / TM;
    constexpr int K_BITS = 256;
    constexpr int K_WORDS64 = K_BITS / 64;
    constexpr int K_WORDS32 = K_BITS / 32;
    constexpr int K_STRIDE32 = K_WORDS32; // TMA path stores tight 32B rows (no padding)

    if (blockDim.x != (WARPS_PER_QTILE * QTILES * 32)) return;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int q_tile_id = warp_id >> 2;
    const int warp_in_tile = warp_id & (WARPS_PER_QTILE - 1);

    const int q0 = blockIdx.y * TM_TOTAL;
    const int r_base = blockIdx.x * (TN * R_SWEEP);
    const int tile_base = blockIdx.x * R_SWEEP;

    uintptr_t p = reinterpret_cast<uintptr_t>(smem_raw);
    p = (p + 127u) & ~uintptr_t(127u);

    constexpr int stages = 2;
    constexpr size_t A_words = (size_t)SA * (size_t)TM_TOTAL * (size_t)K_STRIDE32;
    constexpr size_t B_words = (size_t)SB * (size_t)TN * (size_t)K_STRIDE32;
    constexpr size_t B_words_sweep = (size_t)R_SWEEP * B_words;
    auto* A_bits0 = reinterpret_cast<uint32_t*>(p);
    p += (size_t)stages * A_words * sizeof(uint32_t);
    auto* B_bits0 = reinterpret_cast<uint32_t*>(p);
    p += (size_t)stages * B_words_sweep * sizeof(uint32_t);
    auto* acc0 = reinterpret_cast<float4*>(p);
    p += (size_t)R_SWEEP * (size_t)blockDim.x * sizeof(float4);
    (void)p;

    const int groupID = lane >> 2;
    const int threadID = lane & 3;
    const int row0 = groupID;
    const int row1 = groupID + 8;
    const int col_base = warp_in_tile * 8;
    const int col0 = col_base + threadID * 2;
    const int col1 = col0 + 1;
    const int m0 = q_tile_id * TM + row0;
    const int m1 = q_tile_id * TM + row1;

    float bscale0[R_SWEEP];
    float bscale1[R_SWEEP];
#pragma unroll
    for (int t = 0; t < R_SWEEP; ++t) {
        const int r0 = r_base + t * TN;
        bscale0[t] = __ldg(&Bw[((size_t)(r0 + col0) * (size_t)SB) + 0]);
        bscale1[t] = __ldg(&Bw[((size_t)(r0 + col1) * (size_t)SB) + 0]);
    }

#pragma unroll
    for (int t = 0; t < R_SWEEP; ++t) {
        acc0[(size_t)t * (size_t)blockDim.x + (size_t)threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    const int chunks = W64 / K_WORDS64;
    if (chunks <= 0) return;

#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier_t bar[stages];
    if (threadIdx.x == 0) {
#pragma unroll
        for (int s = 0; s < stages; ++s) {
            init(&bar[s], 1);
        }
    }
    __syncthreads();

    const bool is_leader = (threadIdx.x == 0);
    uint32_t parity[stages] = {0u, 0u};

    constexpr uint32_t ROW_BYTES = (uint32_t)(K_WORDS64 * sizeof(unsigned long long));
    constexpr uint32_t TX_B_BYTES = (uint32_t)((size_t)R_SWEEP * (size_t)TN * (size_t)SB * (size_t)ROW_BYTES);

    auto wait_stage_leader = [&](int stage) {
        auto handle = cuda::device::barrier_native_handle(bar[stage]);
        while (!ptx::mbarrier_try_wait_parity(ptx::sem_acquire, ptx::scope_cta, handle, parity[stage])) {
        }
        parity[stage] ^= 1u;
    };

    auto prefetch_b_stage = [&](int stage, int chunk, void* __restrict__ B_dst_bytes) {
        if (!B_tensor_map) return;
        if (is_leader) {
            (void)cuda::device::barrier_arrive_tx(bar[stage], 1, TX_B_BYTES);
            const int32_t coords[4] = {(int32_t)(chunk * (int)ROW_BYTES), 0, 0, (int32_t)tile_base};
            auto handle = cuda::device::barrier_native_handle(bar[stage]);
            ptx::cp_async_bulk_tensor(ptx::space_cluster, ptx::space_global, B_dst_bytes, B_tensor_map, coords, handle);
        }
    };

    {
        uint32_t* A_bits = A_bits0;
        constexpr int K_WORDS64_16B = K_WORDS64 / 2;
        for (int idx = threadIdx.x; idx < TM_TOTAL * SA * K_WORDS64_16B; idx += blockDim.x) {
            int t = idx;
            const int w64_pair = t & (K_WORDS64_16B - 1);
            t >>= 1;
            const int m = t & (TM_TOTAL - 1);
            const int i = t >> 5;
            const int q = q0 + m;
            const unsigned long long* a_slice = A + ((size_t)q * (size_t)SA + (size_t)i) * (size_t)W64;
            const int w64_i = w64_pair << 1;
            const int base = ((i * TM_TOTAL + m) * K_STRIDE32) + (w64_i << 1);
            bsi_cp_async_cg_16B(A_bits + base, &a_slice[(size_t)0 * (size_t)K_WORDS64 + (size_t)w64_i]);
        }
        bsi_cp_async_commit_group();
        prefetch_b_stage(0, 0, B_bits0);
        bsi_cp_async_wait_all();
        if (is_leader) wait_stage_leader(0);
        __syncthreads();
    }

    int stage = 0;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        uint32_t* A_bits = (stage == 0) ? A_bits0 : (A_bits0 + A_words);
        uint32_t* B_bits = (stage == 0) ? B_bits0 : (B_bits0 + B_words_sweep);

        const int next_chunk = chunk + 1;
        const int next_stage = stage ^ 1;
        if (next_chunk < chunks) {
            uint32_t* A_bits_next = (next_stage == 0) ? A_bits0 : (A_bits0 + A_words);
            uint32_t* B_bits_next = (next_stage == 0) ? B_bits0 : (B_bits0 + B_words_sweep);

            constexpr int K_WORDS64_16B = K_WORDS64 / 2;
            for (int idx = threadIdx.x; idx < TM_TOTAL * SA * K_WORDS64_16B; idx += blockDim.x) {
                int t = idx;
                const int w64_pair = t & (K_WORDS64_16B - 1);
                t >>= 1;
                const int m = t & (TM_TOTAL - 1);
                const int i = t >> 5;
                const int q = q0 + m;
                const unsigned long long* a_slice = A + ((size_t)q * (size_t)SA + (size_t)i) * (size_t)W64;
                const int w64_i = w64_pair << 1;
                const int base = ((i * TM_TOTAL + m) * K_STRIDE32) + (w64_i << 1);
                bsi_cp_async_cg_16B(
                    A_bits_next + base,
                    &a_slice[(size_t)next_chunk * (size_t)K_WORDS64 + (size_t)w64_i]);
            }
            bsi_cp_async_commit_group();
            prefetch_b_stage(next_stage, next_chunk, B_bits_next);
        }

        float qscale_m0 = 1.0f;
        float qscale_m1 = 1.0f;
        if (threadID == 0) {
            const int q_m0 = q0 + m0;
            const int q_m1 = q0 + m1;
            qscale_m0 = __ldg(&A_chunk_scales[(size_t)q_m0 * (size_t)A_scale_stride + (size_t)chunk]);
            qscale_m1 = __ldg(&A_chunk_scales[(size_t)q_m1 * (size_t)A_scale_stride + (size_t)chunk]);
        }
        qscale_m0 = __shfl_sync(0xffffffff, qscale_m0, lane & ~3);
        qscale_m1 = __shfl_sync(0xffffffff, qscale_m1, lane & ~3);

        // Branchless software swizzle mask setup to eliminate bank conflicts
        const int swap_mask = (groupID & 4);
        const int idx_0 = threadID + swap_mask;
        const int idx_1 = threadID + (swap_mask ^ 4);

#pragma unroll
        for (int t_pair = 0; t_pair < R_SWEEP; t_pair += 2) {
            const int t0 = t_pair;
            const int t1 = t_pair + 1;
            const uint32_t* B0 = B_bits + (size_t)t0 * B_words;
            const uint32_t* B1 = B_bits + (size_t)t1 * B_words;

            uint32_t b0_0[SB];
            uint32_t b1_0[SB];
            uint32_t b0_1[SB];
            uint32_t b1_1[SB];

            const int n = col_base + groupID;
            const int b_slice_stride = TN * K_STRIDE32;
            const uint32_t* b_col_base0 = B0 + n * K_STRIDE32;
            const uint32_t* b_col_base1 = B1 + n * K_STRIDE32;
            
#pragma unroll
            for (int j = 0; j < SB; ++j) {
                const uint32_t* b_col0 = b_col_base0 + j * b_slice_stride;
                const uint32_t* b_col1 = b_col_base1 + j * b_slice_stride;
                
                uint32_t b0_0_raw = b_col0[idx_0];
                uint32_t b1_0_raw = b_col0[idx_1];
                b0_0[j] = swap_mask ? b1_0_raw : b0_0_raw;
                b1_0[j] = swap_mask ? b0_0_raw : b1_0_raw;

                uint32_t b0_1_raw = b_col1[idx_0];
                uint32_t b1_1_raw = b_col1[idx_1];
                b0_1[j] = swap_mask ? b1_1_raw : b0_1_raw;
                b1_1[j] = swap_mask ? b0_1_raw : b1_1_raw;
            }

            int chunk00_0 = 0, chunk01_0 = 0, chunk10_0 = 0, chunk11_0 = 0;
            int chunk00_1 = 0, chunk01_1 = 0, chunk10_1 = 0, chunk11_1 = 0;
            
#pragma unroll
            for (int i = 0; i < SA; ++i) {
                const uint32_t* A_i = A_bits + (size_t)i * (size_t)TM_TOTAL * (size_t)K_STRIDE32;
                
                uint32_t a0_raw = A_i[(size_t)m0 * (size_t)K_STRIDE32 + idx_0];
                uint32_t a2_raw = A_i[(size_t)m0 * (size_t)K_STRIDE32 + idx_1];
                const uint32_t a0 = swap_mask ? a2_raw : a0_raw;
                const uint32_t a2 = swap_mask ? a0_raw : a2_raw;

                uint32_t a1_raw = A_i[(size_t)m1 * (size_t)K_STRIDE32 + idx_0];
                uint32_t a3_raw = A_i[(size_t)m1 * (size_t)K_STRIDE32 + idx_1];
                const uint32_t a1 = swap_mask ? a3_raw : a1_raw;
                const uint32_t a3 = swap_mask ? a1_raw : a3_raw;

#pragma unroll
                for (int j = 0; j < SB; ++j) {
                    const int shift = i + j;
                    const bool neg = ((i == (SA - 1)) ^ (j == (SB - 1)));

                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                    asm volatile(
                        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_0[j]), "r"(b1_0[j]));

                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    asm volatile(
                        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0_1[j]), "r"(b1_1[j]));

                    const int v0 = c0 << shift; const int v1 = c1 << shift; const int v2 = c2 << shift; const int v3 = c3 << shift;
                    const int w0 = d0 << shift; const int w1 = d1 << shift; const int w2 = d2 << shift; const int w3 = d3 << shift;
                    if (neg) {
                        chunk00_0 -= v0; chunk01_0 -= v1; chunk10_0 -= v2; chunk11_0 -= v3;
                        chunk00_1 -= w0; chunk01_1 -= w1; chunk10_1 -= w2; chunk11_1 -= w3;
                    } else {
                        chunk00_0 += v0; chunk01_0 += v1; chunk10_0 += v2; chunk11_0 += v3;
                        chunk00_1 += w0; chunk01_1 += w1; chunk10_1 += w2; chunk11_1 += w3;
                    }
                }
            }

            {
                const float s00 = qscale_m0 * bscale0[t0];
                const float s01 = qscale_m0 * bscale1[t0];
                const float s10 = qscale_m1 * bscale0[t0];
                const float s11 = qscale_m1 * bscale1[t0];
                float4 acc = acc0[(size_t)t0 * (size_t)blockDim.x + (size_t)threadIdx.x];
                acc.x = __fmaf_rn(static_cast<float>(chunk00_0), s00, acc.x);
                acc.y = __fmaf_rn(static_cast<float>(chunk01_0), s01, acc.y);
                acc.z = __fmaf_rn(static_cast<float>(chunk10_0), s10, acc.z);
                acc.w = __fmaf_rn(static_cast<float>(chunk11_0), s11, acc.w);
                acc0[(size_t)t0 * (size_t)blockDim.x + (size_t)threadIdx.x] = acc;
            }
            {
                const float s00 = qscale_m0 * bscale0[t1];
                const float s01 = qscale_m0 * bscale1[t1];
                const float s10 = qscale_m1 * bscale0[t1];
                const float s11 = qscale_m1 * bscale1[t1];
                float4 acc = acc0[(size_t)t1 * (size_t)blockDim.x + (size_t)threadIdx.x];
                acc.x = __fmaf_rn(static_cast<float>(chunk00_1), s00, acc.x);
                acc.y = __fmaf_rn(static_cast<float>(chunk01_1), s01, acc.y);
                acc.z = __fmaf_rn(static_cast<float>(chunk10_1), s10, acc.z);
                acc.w = __fmaf_rn(static_cast<float>(chunk11_1), s11, acc.w);
                acc0[(size_t)t1 * (size_t)blockDim.x + (size_t)threadIdx.x] = acc;
            }
        }

        if (next_chunk < chunks) {
            bsi_cp_async_wait_all();
            if (is_leader) wait_stage_leader(next_stage);
        }
        __syncthreads();
        stage ^= 1;
    }

    const int q_out0 = q0 + m0;
    const int q_out1 = q0 + m1;
#pragma unroll
    for (int t = 0; t < R_SWEEP; ++t) {
        const int r0 = r_base + t * TN;
        const int r_out0 = r0 + col0;
        const int r_out1 = r0 + col1;
        const float4 acc = acc0[(size_t)t * (size_t)blockDim.x + (size_t)threadIdx.x];
        out_global[(size_t)q_out0 * (size_t)R_total + (size_t)r_out0] = acc.x * scale_inv;
        out_global[(size_t)q_out0 * (size_t)R_total + (size_t)r_out1] = acc.y * scale_inv;
        out_global[(size_t)q_out1 * (size_t)R_total + (size_t)r_out0] = acc.z * scale_inv;
        out_global[(size_t)q_out1 * (size_t)R_total + (size_t)r_out1] = acc.w * scale_inv;
    }
#else
    (void)A;
    (void)A_chunk_scales;
    (void)A_scale_stride;
    (void)W64;
    (void)B;
    (void)Bw;
    (void)B_tensor_map;
    (void)R;
    (void)Q;
    (void)scale_inv;
    (void)R_total;
    (void)out_global;
#endif
}

extern "C" __global__ __launch_bounds__(256, 2)
void popcount_weighted_keys_literal_fused_bmma_tc_kernel_tm32_fixed76_chunkscale_rsweep2_tma_tensorB(
    const unsigned long long* __restrict__ A,
    const float* __restrict__ A_chunk_scales,
    int A_scale_stride,
    int W64,
    const unsigned long long* __restrict__ B,
    const float* __restrict__ Bw,
    const void* __restrict__ B_tensor_map,
    int R,
    int Q,
    float scale_inv,
    int R_total,
    float* __restrict__ out_global)
{
    extern __shared__ unsigned char smem_raw[];
    bsi_fixed76_tm32_chunkscale_rsweep_body_tma_tensorB<2>(
        smem_raw, A, A_chunk_scales, A_scale_stride, W64, B, Bw, B_tensor_map, R, Q, scale_inv, R_total, out_global);
}