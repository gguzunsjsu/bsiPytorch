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
#include <pybind11/pybind11.h>

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
//
// We only use these in the SM90 BMMA TC kernels, and we keep a safe fallback for
// non-async architectures / host compilation.
__device__ __forceinline__ void bsi_cp_async_cg_16B(void* dst_shared, const void* src_global) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    const unsigned int dst = __cvta_generic_to_shared(dst_shared);
    // SM90 ptxas (NVHPC 24.11) only accepts 16B for cp.async.
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
    const unsigned long long* __restrict__ A,    // [Q, Sa, W]
    const float* __restrict__ Aw,                // [Q, Sa]
    int Sa,
    int W,
    const unsigned long long* __restrict__ B,    // [R, Sb, W]
    const float* __restrict__ Bw,                // [R, Sb]
    int Sb,
    int R,
    int Q,
    int q_tile,
    int r_tile,
    const long long* __restrict__ key_indices,   // [R]
    const long long* __restrict__ query_indices, // [Q]
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

// NOTE: Legacy non-TC "warp-out" dot kernels were removed to reduce code size.
// The fallback kernel for non-TC execution is `popcount_weighted_keys_literal_fused_multiq_kernel`.
// Tensor-core BMMA path (SM90+): 8 warps per block (32x32 output tile).
template <int SA, int SB, int TM_ROWS>
__device__ __forceinline__ void bsi_bmma_tm32_accum_hot(
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

__device__ __forceinline__ const uint32_t* bsi_sm90_query_packed_row_ptr(
    const uint32_t* __restrict__ A_packed,
    int packed_chunks,
    int Sa,
    int q,
    int chunk,
    int i)
{
    return A_packed +
        (((((size_t)(q >> 5) * (size_t)packed_chunks + (size_t)chunk) * (size_t)Sa + (size_t)i) * 32ull +
          (size_t)(q & 31)) *
         8ull);
}

__device__ __forceinline__ const uint32_t* bsi_sm90_key_packed_row_ptr(
    const uint32_t* __restrict__ B_packed,
    int packed_chunks,
    int Sb,
    int r,
    int chunk,
    int j)
{
    return B_packed +
        (((((size_t)(r >> 5) * (size_t)packed_chunks + (size_t)chunk) * (size_t)Sb + (size_t)j) * 32ull +
          (size_t)(r & 31)) *
         8ull);
}

extern "C" __global__ __launch_bounds__(128)
void popcount_weighted_keys_literal_fused_packed_reference_kernel(
    const uint32_t* __restrict__ A_packed_u32,   // [Q_tiles32, chunks, Sa, 32, 8]
    const float* __restrict__ Aw,                // [Q, Sa]
    const float* __restrict__ A_chunk_scales,    // [Q, chunks] or nullptr
    int A_scale_stride,                          // chunks or 0
    int Sa,
    int packed_chunks,
    const uint32_t* __restrict__ B_packed_u32,   // [R_tiles32, chunks, Sb, 32, 8]
    const float* __restrict__ Bw,                // [R, Sb]
    int Sb,
    int q_begin,
    int q_count,
    int r_begin,
    int r_count,
    const long long* __restrict__ key_indices,   // [r_count] or identity
    const long long* __restrict__ query_indices, // [q_count] or identity
    float scale_inv,
    int R_total,
    float* __restrict__ out_global)
{
#if defined(__CUDA_ARCH__)
    const int q_local = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    const int r_local = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (q_local >= q_count || r_local >= r_count) return;

    const int q = q_begin + q_local;
    const int r = r_begin + r_local;

    float acc = 0.0f;
    for (int chunk = 0; chunk < packed_chunks; ++chunk) {
        const float qscale =
            (A_chunk_scales != nullptr && A_scale_stride > 0)
                ? __ldg(&A_chunk_scales[(size_t)q * (size_t)A_scale_stride + (size_t)chunk])
                : 1.0f;

        for (int i = 0; i < Sa; ++i) {
            const float aw = __ldg(&Aw[(size_t)q * (size_t)Sa + (size_t)i]) * qscale;
            const uint32_t* a_row = bsi_sm90_query_packed_row_ptr(A_packed_u32, packed_chunks, Sa, q, chunk, i);
            for (int j = 0; j < Sb; ++j) {
                const float bw = __ldg(&Bw[(size_t)r * (size_t)Sb + (size_t)j]);
                const uint32_t* b_row = bsi_sm90_key_packed_row_ptr(B_packed_u32, packed_chunks, Sb, r, chunk, j);
                int cnt = 0;
#pragma unroll
                for (int w32 = 0; w32 < 8; ++w32) {
                    cnt += __popc(a_row[w32] & b_row[w32]);
                }
                acc = __fmaf_rn(static_cast<float>(cnt), aw * bw, acc);
            }
        }
    }

    const long long gq = query_indices ? __ldg(query_indices + q_local) : static_cast<long long>(q);
    const long long gr = key_indices ? __ldg(key_indices + r_local) : static_cast<long long>(r);
    out_global[(size_t)gq * (size_t)R_total + (size_t)gr] = acc * scale_inv;
#else
    (void)A_packed_u32;
    (void)Aw;
    (void)A_chunk_scales;
    (void)A_scale_stride;
    (void)Sa;
    (void)packed_chunks;
    (void)B_packed_u32;
    (void)Bw;
    (void)Sb;
    (void)q_begin;
    (void)q_count;
    (void)r_begin;
    (void)r_count;
    (void)key_indices;
    (void)query_indices;
    (void)scale_inv;
    (void)R_total;
    (void)out_global;
#endif
}

extern "C" __global__ __launch_bounds__(128, 2)
void popcount_weighted_keys_literal_fused_bmma_tc_kernel_tile16x32(
    const uint32_t* __restrict__ A_packed_u32,   // [Q_tiles32, chunks, Sa, 32, 8]
    const float* __restrict__ Aw,                // [Q, Sa]
    const float* __restrict__ A_chunk_scales,    // [Q, chunks] or nullptr
    int A_scale_stride,                          // chunks (W64/4) or 0
    int Sa,
    int packed_chunks,
    const uint32_t* __restrict__ B_packed_u32,   // [R_tiles32, chunks, Sb, 32, 8]
    const float* __restrict__ Bw,                // [R, Sb]
    int Sb,
    int R,
    int Q,
    const long long* __restrict__ key_indices,   // [R]
    const long long* __restrict__ query_indices, // [Q]
    float scale_inv,
    int R_total,
    float* __restrict__ out_global,
    int use_cpasync)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    constexpr int TM = 16;
    constexpr int TN = 32;
    constexpr int K_WORDS32 = 8;
    constexpr int K_STRIDE32 = K_WORDS32 + 4;
    constexpr int K_SEGMENTS = 2;

    if (blockDim.x != 128) return;

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;  // 0..3
    const int groupID = lane >> 2;         // 0..7
    const int threadID = lane & 3;         // 0..3
    const int row0 = groupID;
    const int row1 = groupID + 8;
    const int col_base = warp_id * 8;
    const int col0 = col_base + threadID * 2;
    const int col1 = col0 + 1;

    const int q0 = blockIdx.y * TM;
    const int r0 = blockIdx.x * TN;
    if ((q0 + TM) > Q || (r0 + TN) > R) return;

    extern __shared__ unsigned char smem_raw[];
    uintptr_t p = reinterpret_cast<uintptr_t>(smem_raw);
    p = (p + 15u) & ~uintptr_t(15u);

    const int stages = (use_cpasync != 0 && packed_chunks > 1) ? 2 : 1;
    const size_t A_words = (size_t)Sa * (size_t)TM * (size_t)K_STRIDE32;
    const size_t B_words = (size_t)Sb * (size_t)TN * (size_t)K_STRIDE32;
    auto* A_bits0 = reinterpret_cast<uint32_t*>(p);
    p += (size_t)stages * A_words * sizeof(uint32_t);
    auto* B_bits0 = reinterpret_cast<uint32_t*>(p);
    (void)p;

    const bool can_cpasync = (stages == 2);

    if (can_cpasync) {
        uint32_t* A_bits = A_bits0;
        uint32_t* B_bits = B_bits0;
        for (int idx = threadIdx.x; idx < TM * Sa * K_SEGMENTS; idx += blockDim.x) {
            int t = idx;
            const int seg = t & (K_SEGMENTS - 1);
            t >>= 1;
            const int m = t & (TM - 1);
            const int i = t >> 4; // /16
            const int base = ((i * TM + m) * K_STRIDE32) + (seg << 2);
            const uint32_t* a_row = bsi_sm90_query_packed_row_ptr(A_packed_u32, packed_chunks, Sa, q0 + m, 0, i);
            bsi_cp_async_cg_16B(A_bits + base, a_row + (seg << 2));
        }
        for (int idx = threadIdx.x; idx < TM * Sa * (K_STRIDE32 - K_WORDS32); idx += blockDim.x) {
            int t = idx;
            const int pad = t & ((K_STRIDE32 - K_WORDS32) - 1);
            t >>= 2;
            const int m = t & (TM - 1);
            const int i = t >> 4;
            const int base = ((i * TM + m) * K_STRIDE32) + K_WORDS32 + pad;
            A_bits[base] = 0u;
        }
        for (int idx = threadIdx.x; idx < TN * Sb * K_SEGMENTS; idx += blockDim.x) {
            int t = idx;
            const int seg = t & (K_SEGMENTS - 1);
            t >>= 1;
            const int n = t & (TN - 1);
            const int j = t >> 5; // /32
            const int base = ((j * TN + n) * K_STRIDE32) + (seg << 2);
            const uint32_t* b_row = bsi_sm90_key_packed_row_ptr(B_packed_u32, packed_chunks, Sb, r0 + n, 0, j);
            bsi_cp_async_cg_16B(B_bits + base, b_row + (seg << 2));
        }
        for (int idx = threadIdx.x; idx < TN * Sb * (K_STRIDE32 - K_WORDS32); idx += blockDim.x) {
            int t = idx;
            const int pad = t & ((K_STRIDE32 - K_WORDS32) - 1);
            t >>= 2;
            const int n = t & (TN - 1);
            const int j = t >> 5;
            const int base = ((j * TN + n) * K_STRIDE32) + K_WORDS32 + pad;
            B_bits[base] = 0u;
        }
        bsi_cp_async_commit_group();
        bsi_cp_async_wait_all();
        __syncthreads();
    }

    float acc00 = 0.0f;
    float acc01 = 0.0f;
    float acc10 = 0.0f;
    float acc11 = 0.0f;
    int stage = 0;

    for (int chunk = 0; chunk < packed_chunks; ++chunk) {
        uint32_t* A_bits = (stage == 0) ? A_bits0 : (A_bits0 + A_words);
        uint32_t* B_bits = (stage == 0) ? B_bits0 : (B_bits0 + B_words);

        if (!can_cpasync) {
            for (int idx = threadIdx.x; idx < TM * Sa * K_WORDS32; idx += blockDim.x) {
                int t = idx;
                const int w32_i = t & (K_WORDS32 - 1);
                t >>= 3;
                const int m = t & (TM - 1);
                const int i = t >> 4;
                const uint32_t* a_row = bsi_sm90_query_packed_row_ptr(A_packed_u32, packed_chunks, Sa, q0 + m, chunk, i);
                const int base = ((i * TM + m) * K_STRIDE32) + w32_i;
                A_bits[base] = __ldg(a_row + w32_i);
            }
            for (int idx = threadIdx.x; idx < TM * Sa * (K_STRIDE32 - K_WORDS32); idx += blockDim.x) {
                int t = idx;
                const int pad = t & ((K_STRIDE32 - K_WORDS32) - 1);
                t >>= 2;
                const int m = t & (TM - 1);
                const int i = t >> 4;
                const int base = ((i * TM + m) * K_STRIDE32) + K_WORDS32 + pad;
                A_bits[base] = 0u;
            }
            for (int idx = threadIdx.x; idx < TN * Sb * K_WORDS32; idx += blockDim.x) {
                int t = idx;
                const int w32_i = t & (K_WORDS32 - 1);
                t >>= 3;
                const int n = t & (TN - 1);
                const int j = t >> 5;
                const uint32_t* b_row = bsi_sm90_key_packed_row_ptr(B_packed_u32, packed_chunks, Sb, r0 + n, chunk, j);
                const int base = ((j * TN + n) * K_STRIDE32) + w32_i;
                B_bits[base] = __ldg(b_row + w32_i);
            }
            for (int idx = threadIdx.x; idx < TN * Sb * (K_STRIDE32 - K_WORDS32); idx += blockDim.x) {
                int t = idx;
                const int pad = t & ((K_STRIDE32 - K_WORDS32) - 1);
                t >>= 2;
                const int n = t & (TN - 1);
                const int j = t >> 5;
                const int base = ((j * TN + n) * K_STRIDE32) + K_WORDS32 + pad;
                B_bits[base] = 0u;
            }
            __syncthreads();
        } else if (chunk + 1 < packed_chunks) {
            const int next_stage = stage ^ 1;
            uint32_t* A_bits_next = (next_stage == 0) ? A_bits0 : (A_bits0 + A_words);
            uint32_t* B_bits_next = (next_stage == 0) ? B_bits0 : (B_bits0 + B_words);
            const int next_chunk = chunk + 1;
            for (int idx = threadIdx.x; idx < TM * Sa * K_SEGMENTS; idx += blockDim.x) {
                int t = idx;
                const int seg = t & (K_SEGMENTS - 1);
                t >>= 1;
                const int m = t & (TM - 1);
                const int i = t >> 4;
                const int base = ((i * TM + m) * K_STRIDE32) + (seg << 2);
                const uint32_t* a_row =
                    bsi_sm90_query_packed_row_ptr(A_packed_u32, packed_chunks, Sa, q0 + m, next_chunk, i);
                bsi_cp_async_cg_16B(A_bits_next + base, a_row + (seg << 2));
            }
            for (int idx = threadIdx.x; idx < TM * Sa * (K_STRIDE32 - K_WORDS32); idx += blockDim.x) {
                int t = idx;
                const int pad = t & ((K_STRIDE32 - K_WORDS32) - 1);
                t >>= 2;
                const int m = t & (TM - 1);
                const int i = t >> 4;
                const int base = ((i * TM + m) * K_STRIDE32) + K_WORDS32 + pad;
                A_bits_next[base] = 0u;
            }
            for (int idx = threadIdx.x; idx < TN * Sb * K_SEGMENTS; idx += blockDim.x) {
                int t = idx;
                const int seg = t & (K_SEGMENTS - 1);
                t >>= 1;
                const int n = t & (TN - 1);
                const int j = t >> 5;
                const int base = ((j * TN + n) * K_STRIDE32) + (seg << 2);
                const uint32_t* b_row =
                    bsi_sm90_key_packed_row_ptr(B_packed_u32, packed_chunks, Sb, r0 + n, next_chunk, j);
                bsi_cp_async_cg_16B(B_bits_next + base, b_row + (seg << 2));
            }
            for (int idx = threadIdx.x; idx < TN * Sb * (K_STRIDE32 - K_WORDS32); idx += blockDim.x) {
                int t = idx;
                const int pad = t & ((K_STRIDE32 - K_WORDS32) - 1);
                t >>= 2;
                const int n = t & (TN - 1);
                const int j = t >> 5;
                const int base = ((j * TN + n) * K_STRIDE32) + K_WORDS32 + pad;
                B_bits_next[base] = 0u;
            }
            bsi_cp_async_commit_group();
        }

        const bool use_chunk_scale = (A_chunk_scales != nullptr) && (A_scale_stride > 0);
        float qscale_m0 = 1.0f;
        float qscale_m1 = 1.0f;
        if (use_chunk_scale) {
            if (threadID == 0) {
                qscale_m0 = __ldg(&A_chunk_scales[(size_t)(q0 + row0) * (size_t)A_scale_stride + (size_t)chunk]);
                qscale_m1 = __ldg(&A_chunk_scales[(size_t)(q0 + row1) * (size_t)A_scale_stride + (size_t)chunk]);
            }
            qscale_m0 = __shfl_sync(0xffffffff, qscale_m0, lane & ~3);
            qscale_m1 = __shfl_sync(0xffffffff, qscale_m1, lane & ~3);
        }

        const uint32_t* b_col_base = B_bits + (size_t)(col_base + groupID) * (size_t)K_STRIDE32;
        const float* bw_col0 = Bw + (size_t)(r0 + col0) * (size_t)Sb;
        const float* bw_col1 = Bw + (size_t)(r0 + col1) * (size_t)Sb;
        const float* Aw_tile = Aw + (size_t)q0 * (size_t)Sa;
        const int b_slice_stride = TN * K_STRIDE32;

        if (Sa == 7 && Sb == 7) {
            bsi_bmma_tm32_accum_hot<7, 7, 16>(
                A_bits, Aw_tile, b_col_base, bw_col0, bw_col1, threadID, row0, row1,
                use_chunk_scale, qscale_m0, qscale_m1, acc00, acc01, acc10, acc11);
        } else if (Sa == 7 && Sb == 6) {
            bsi_bmma_tm32_accum_hot<7, 6, 16>(
                A_bits, Aw_tile, b_col_base, bw_col0, bw_col1, threadID, row0, row1,
                use_chunk_scale, qscale_m0, qscale_m1, acc00, acc01, acc10, acc11);
        } else if (Sa == 8 && Sb == 7) {
            bsi_bmma_tm32_accum_hot<8, 7, 16>(
                A_bits, Aw_tile, b_col_base, bw_col0, bw_col1, threadID, row0, row1,
                use_chunk_scale, qscale_m0, qscale_m1, acc00, acc01, acc10, acc11);
        } else if (Sa == 6 && Sb == 5) {
            bsi_bmma_tm32_accum_hot<6, 5, 16>(
                A_bits, Aw_tile, b_col_base, bw_col0, bw_col1, threadID, row0, row1,
                use_chunk_scale, qscale_m0, qscale_m1, acc00, acc01, acc10, acc11);
        } else {
            for (int i = 0; i < Sa; ++i) {
                const float aw0 = __ldg(&Aw_tile[(size_t)row0 * (size_t)Sa + (size_t)i]) * qscale_m0;
                const float aw1 = __ldg(&Aw_tile[(size_t)row1 * (size_t)Sa + (size_t)i]) * qscale_m1;
                const uint32_t* A_i = A_bits + (size_t)i * (size_t)TM * (size_t)K_STRIDE32;
                const uint32_t a0 = A_i[(size_t)row0 * (size_t)K_STRIDE32 + (size_t)threadID];
                const uint32_t a1 = A_i[(size_t)row1 * (size_t)K_STRIDE32 + (size_t)threadID];
                const uint32_t a2 = A_i[(size_t)row0 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];
                const uint32_t a3 = A_i[(size_t)row1 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];

                float sum00 = 0.0f;
                float sum01 = 0.0f;
                float sum10 = 0.0f;
                float sum11 = 0.0f;
                for (int j = 0; j < Sb; ++j) {
                    const uint32_t* b_col = b_col_base + (size_t)j * (size_t)b_slice_stride;
                    int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                    asm volatile(
                        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%0, %1, %2, %3};\n"
                        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b_col[threadID]), "r"(b_col[threadID + 4]));

                    const float bw0 = __ldg(&bw_col0[j]);
                    const float bw1 = __ldg(&bw_col1[j]);
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

        if (can_cpasync) {
            bsi_cp_async_wait_all();
            __syncthreads();
            stage ^= 1;
        }
    }

    const int q_out0 = q0 + row0;
    const int q_out1 = q0 + row1;
    const int r_out0 = r0 + col0;
    const int r_out1 = r0 + col1;
    const bool identity_q = (query_indices == nullptr);
    const bool identity_r = (key_indices == nullptr);

    if (identity_q && identity_r) {
        out_global[(size_t)q_out0 * (size_t)R_total + (size_t)r_out0] = acc00 * scale_inv;
        out_global[(size_t)q_out0 * (size_t)R_total + (size_t)r_out1] = acc01 * scale_inv;
        out_global[(size_t)q_out1 * (size_t)R_total + (size_t)r_out0] = acc10 * scale_inv;
        out_global[(size_t)q_out1 * (size_t)R_total + (size_t)r_out1] = acc11 * scale_inv;
    } else {
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
    }
#else
    (void)A_packed_u32;
    (void)Aw;
    (void)A_chunk_scales;
    (void)A_scale_stride;
    (void)Sa;
    (void)packed_chunks;
    (void)B_packed_u32;
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

extern "C" __global__ __launch_bounds__(256)
void popcount_weighted_keys_literal_fused_bmma_tc_kernel_tm32(
    const unsigned long long* __restrict__ A,    // [Q, Sa, W64]
    const uint32_t* __restrict__ A_packed_u32,   // [Q_tiles, chunks, Sa, 32, 8] or nullptr
    const float* __restrict__ Aw,                // [Q, Sa]
    const float* __restrict__ A_chunk_scales,    // [Q, chunks] or nullptr
    int A_scale_stride,                          // chunks (W64/4) or 0
    int Sa,
    int W64,
    const unsigned long long* __restrict__ B,    // [R, Sb, W64]
    const uint32_t* __restrict__ B_packed_u32,   // [R_tiles8, chunks, Sb, 8, 8] or nullptr
    const float* __restrict__ Bw,                // [R, Sb]
    int Sb,
    int R,
    int Q,
    int packed_chunks,
    const long long* __restrict__ key_indices,   // [R]
    const long long* __restrict__ query_indices, // [Q]
    float scale_inv,
    int R_total,
    float* __restrict__ out_global,
    int use_cpasync)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    constexpr int TM_TOTAL = 32;
    constexpr int TM = 16;
    constexpr int TN = 32;
    constexpr int WARPS_PER_QTILE = 4; // 32 cols / 8 cols per warp
    constexpr int QTILES = TM_TOTAL / TM; // 2
    constexpr int SB_MAX = 16;
    constexpr int K_BITS = 256;
    constexpr int K_WORDS64 = K_BITS / 64;   // 4
    constexpr int K_WORDS32 = K_BITS / 32;   // 8
    // Stride=12 is conflict-free for the mma.sync lane->fragment mapping used by this kernel.
    constexpr int K_STRIDE32 = K_WORDS32 + 4; // 12 (padding)

    if (blockDim.x != (WARPS_PER_QTILE * QTILES * 32)) return; // 256
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;          // 0..7
    const int q_tile_id = warp_id >> 2;            // /4 -> 0..1
    const int warp_in_tile = warp_id & (WARPS_PER_QTILE - 1); // 0..3

    const int q0 = blockIdx.y * TM_TOTAL;
    const int r0 = blockIdx.x * TN;
    const bool full_q_tile = (q0 + TM_TOTAL) <= Q;
    const bool full_r_tile = (r0 + TN) <= R;

    extern __shared__ unsigned char smem_raw[];
    uintptr_t p = reinterpret_cast<uintptr_t>(smem_raw);
    p = (p + 15u) & ~uintptr_t(15u);

    // Optional double-buffering for pipelined per-chunk loads.
    const int stages = (use_cpasync != 0) ? 2 : 1;
    const size_t A_words = (size_t)Sa * (size_t)TM_TOTAL * (size_t)K_STRIDE32;
    const size_t B_words = (size_t)Sb * (size_t)TN * (size_t)K_STRIDE32;

    auto* A_bits0 = reinterpret_cast<uint32_t*>(p); // [stages, Sa, TM_TOTAL, K]
    p += (size_t)stages * A_words * sizeof(uint32_t);
    auto* B_bits0 = reinterpret_cast<uint32_t*>(p); // [stages, Sb, TN, K]
    p += (size_t)stages * B_words * sizeof(uint32_t);
    auto* Aw_tile = reinterpret_cast<float*>(p);   // [TM_TOTAL, Sa]
    p += (size_t)TM_TOTAL * (size_t)Sa * sizeof(float);
    auto* Bw_tile = reinterpret_cast<float*>(p);   // [TN, Sb]
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

    const int groupID = lane >> 2;            // 0..7
    const int threadID = lane & 3;            // 0..3
    const int row0 = groupID;                 // 0..7
    const int row1 = groupID + 8;             // 8..15
    const int col_base = warp_in_tile * 8;    // 0..24
    const int col0 = col_base + threadID * 2; // 0..31 even
    const int col1 = col0 + 1;                // odd
    const int m0 = q_tile_id * TM + row0;     // 0..31
    const int m1 = q_tile_id * TM + row1;     // 0..31

    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    // Cache Bw (and B fragments) in small blocks to reduce per-thread register footprint.
    const bool cache_sb = (Sb <= SB_MAX);
    const float* bw_col0 = Bw_tile + (size_t)col0 * (size_t)Sb;
    const float* bw_col1 = Bw_tile + (size_t)col1 * (size_t)Sb;

    const bool use_packed_ab = (A_packed_u32 != nullptr) && (B_packed_u32 != nullptr);
    const int chunks = use_packed_ab ? packed_chunks : (W64 / K_WORDS64);
    const bool can_cpasync = (use_cpasync != 0) && full_q_tile && full_r_tile && (chunks > 1);

    // Prime the pipeline: load chunk 0 into stage 0.
    if (can_cpasync) {
        uint32_t* A_bits = A_bits0;
        uint32_t* B_bits = B_bits0;
        constexpr int WORD32_SEGS = 2; // 2x16B per 256b chunk
        for (int idx = threadIdx.x; idx < TM_TOTAL * Sa * WORD32_SEGS; idx += blockDim.x) {
            int t = idx;
            const int seg = t & (WORD32_SEGS - 1);
            t >>= 1;
            const int m = t & (TM_TOTAL - 1);
            const int i = t >> 5; // /32

            const int base = ((i * TM_TOTAL + m) * K_STRIDE32) + (seg << 2);
            if (use_packed_ab) {
                const int q = q0 + m;
                const uint32_t* a_row = bsi_sm90_query_packed_row_ptr(A_packed_u32, chunks, Sa, q, 0, i);
                bsi_cp_async_cg_16B(A_bits + base, a_row + (seg << 2));
            } else {
                const int q = q0 + m;
                const unsigned long long* a_slice = A + ((size_t)q * (size_t)Sa + (size_t)i) * (size_t)W64;
                const int w64_i = seg << 1;
                bsi_cp_async_cg_16B(A_bits + base, &a_slice[(size_t)w64_i]);
            }
        }
        for (int idx = threadIdx.x; idx < TM_TOTAL * Sa * (K_STRIDE32 - K_WORDS32); idx += blockDim.x) {
            int t = idx;
            const int pad = t & ((K_STRIDE32 - K_WORDS32) - 1);
            t >>= 2;
            const int m = t & (TM_TOTAL - 1);
            const int i = t >> 5;
            const int base = ((i * TM_TOTAL + m) * K_STRIDE32) + K_WORDS32 + pad;
            A_bits[base] = 0u;
        }
        for (int idx = threadIdx.x; idx < TN * Sb * WORD32_SEGS; idx += blockDim.x) {
            int t = idx;
            const int seg = t & (WORD32_SEGS - 1);
            t >>= 1;
            const int n = t & (TN - 1);
            const int j = t >> 5; // /32

            const int base = ((j * TN + n) * K_STRIDE32) + (seg << 2);
            if (use_packed_ab) {
                const int r = r0 + n;
                const uint32_t* b_row = bsi_sm90_key_packed_row_ptr(B_packed_u32, chunks, Sb, r, 0, j);
                bsi_cp_async_cg_16B(B_bits + base, b_row + (seg << 2));
            } else {
                const int r = r0 + n;
                const unsigned long long* b_slice = B + ((size_t)r * (size_t)Sb + (size_t)j) * (size_t)W64;
                const int w64_i = seg << 1;
                bsi_cp_async_cg_16B(B_bits + base, &b_slice[(size_t)w64_i]);
            }
        }
        for (int idx = threadIdx.x; idx < TN * Sb * (K_STRIDE32 - K_WORDS32); idx += blockDim.x) {
            int t = idx;
            const int pad = t & ((K_STRIDE32 - K_WORDS32) - 1);
            t >>= 2;
            const int n = t & (TN - 1);
            const int j = t >> 5;
            const int base = ((j * TN + n) * K_STRIDE32) + K_WORDS32 + pad;
            B_bits[base] = 0u;
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
            // Synchronous global->shared path (handles boundary tiles).
            if (use_packed_ab) {
                for (int idx = threadIdx.x; idx < TM_TOTAL * Sa * K_WORDS32; idx += blockDim.x) {
                    int t = idx;
                    const int w32_i = t & (K_WORDS32 - 1);
                    t >>= 3;
                    const int m = t & (TM_TOTAL - 1);
                    const int i = t >> 5;

                    uint32_t v = 0u;
                    const int q = q0 + m;
                    if (full_q_tile || q < Q) {
                        const uint32_t* a_row = bsi_sm90_query_packed_row_ptr(A_packed_u32, chunks, Sa, q, chunk, i);
                        v = __ldg(a_row + w32_i);
                    }
                    const int base = ((i * TM_TOTAL + m) * K_STRIDE32) + w32_i;
                    A_bits[base] = v;
                }
                for (int idx = threadIdx.x; idx < TM_TOTAL * Sa * (K_STRIDE32 - K_WORDS32); idx += blockDim.x) {
                    int t = idx;
                    const int pad = t & ((K_STRIDE32 - K_WORDS32) - 1);
                    t >>= 2;
                    const int m = t & (TM_TOTAL - 1);
                    const int i = t >> 5;
                    const int base = ((i * TM_TOTAL + m) * K_STRIDE32) + K_WORDS32 + pad;
                    A_bits[base] = 0u;
                }
                for (int idx = threadIdx.x; idx < TN * Sb * K_WORDS32; idx += blockDim.x) {
                    int t = idx;
                    const int w32_i = t & (K_WORDS32 - 1);
                    t >>= 3;
                    const int n = t & (TN - 1);
                    const int j = t >> 5;

                    uint32_t v = 0u;
                    const int r = r0 + n;
                    if (full_r_tile || r < R) {
                        const uint32_t* b_row = bsi_sm90_key_packed_row_ptr(B_packed_u32, chunks, Sb, r, chunk, j);
                        v = __ldg(b_row + w32_i);
                    }
                    const int base = ((j * TN + n) * K_STRIDE32) + w32_i;
                    B_bits[base] = v;
                }
                for (int idx = threadIdx.x; idx < TN * Sb * (K_STRIDE32 - K_WORDS32); idx += blockDim.x) {
                    int t = idx;
                    const int pad = t & ((K_STRIDE32 - K_WORDS32) - 1);
                    t >>= 2;
                    const int n = t & (TN - 1);
                    const int j = t >> 5;
                    const int base = ((j * TN + n) * K_STRIDE32) + K_WORDS32 + pad;
                    B_bits[base] = 0u;
                }
            } else {
                if (full_q_tile) {
                    for (int idx = threadIdx.x; idx < TM_TOTAL * Sa * K_WORDS64; idx += blockDim.x) {
                        int t = idx;
                        const int w64_i = t & (K_WORDS64 - 1);
                        t >>= 2;
                        const int m = t & (TM_TOTAL - 1);
                        const int i = t >> 5; // /32

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
                        const int i = t >> 5; // /32

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
                        const int j = t >> 5; // /32

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
                        const int j = t >> 5; // /32

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
            }
            __syncthreads();
        } else {
            // Prefetch next chunk into the other stage while we compute this one.
            if (chunk + 1 < chunks) {
                const int next_stage = stage ^ 1;
                uint32_t* A_bits_next = (next_stage == 0) ? A_bits0 : (A_bits0 + A_words);
                uint32_t* B_bits_next = (next_stage == 0) ? B_bits0 : (B_bits0 + B_words);

                const int next_chunk = chunk + 1;
                constexpr int WORD32_SEGS = 2;
                for (int idx = threadIdx.x; idx < TM_TOTAL * Sa * WORD32_SEGS; idx += blockDim.x) {
                    int t = idx;
                    const int seg = t & (WORD32_SEGS - 1);
                    t >>= 1;
                    const int m = t & (TM_TOTAL - 1);
                    const int i = t >> 5; // /32

                    const int base = ((i * TM_TOTAL + m) * K_STRIDE32) + (seg << 2);
                    if (use_packed_ab) {
                        const int q = q0 + m;
                        const uint32_t* a_row =
                            bsi_sm90_query_packed_row_ptr(A_packed_u32, chunks, Sa, q, next_chunk, i);
                        bsi_cp_async_cg_16B(A_bits_next + base, a_row + (seg << 2));
                    } else {
                        const int q = q0 + m;
                        const unsigned long long* a_slice = A + ((size_t)q * (size_t)Sa + (size_t)i) * (size_t)W64;
                        const int w64_i = seg << 1;
                        bsi_cp_async_cg_16B(
                            A_bits_next + base,
                            &a_slice[(size_t)next_chunk * (size_t)K_WORDS64 + (size_t)w64_i]);
                    }
                }
                for (int idx = threadIdx.x; idx < TM_TOTAL * Sa * (K_STRIDE32 - K_WORDS32); idx += blockDim.x) {
                    int t = idx;
                    const int pad = t & ((K_STRIDE32 - K_WORDS32) - 1);
                    t >>= 2;
                    const int m = t & (TM_TOTAL - 1);
                    const int i = t >> 5;
                    const int base = ((i * TM_TOTAL + m) * K_STRIDE32) + K_WORDS32 + pad;
                    A_bits_next[base] = 0u;
                }
                for (int idx = threadIdx.x; idx < TN * Sb * WORD32_SEGS; idx += blockDim.x) {
                    int t = idx;
                    const int seg = t & (WORD32_SEGS - 1);
                    t >>= 1;
                    const int n = t & (TN - 1);
                    const int j = t >> 5; // /32

                    const int base = ((j * TN + n) * K_STRIDE32) + (seg << 2);
                    if (use_packed_ab) {
                        const int r = r0 + n;
                        const uint32_t* b_row =
                            bsi_sm90_key_packed_row_ptr(B_packed_u32, chunks, Sb, r, next_chunk, j);
                        bsi_cp_async_cg_16B(B_bits_next + base, b_row + (seg << 2));
                    } else {
                        const int r = r0 + n;
                        const unsigned long long* b_slice = B + ((size_t)r * (size_t)Sb + (size_t)j) * (size_t)W64;
                        const int w64_i = seg << 1;
                        bsi_cp_async_cg_16B(
                            B_bits_next + base,
                            &b_slice[(size_t)next_chunk * (size_t)K_WORDS64 + (size_t)w64_i]);
                    }
                }
                for (int idx = threadIdx.x; idx < TN * Sb * (K_STRIDE32 - K_WORDS32); idx += blockDim.x) {
                    int t = idx;
                    const int pad = t & ((K_STRIDE32 - K_WORDS32) - 1);
                    t >>= 2;
                    const int n = t & (TN - 1);
                    const int j = t >> 5;
                    const int base = ((j * TN + n) * K_STRIDE32) + K_WORDS32 + pad;
                    B_bits_next[base] = 0u;
                }
                bsi_cp_async_commit_group();
            }
        }

        const bool use_chunk_scale = (A_chunk_scales != nullptr) && (A_scale_stride > 0);
        float qscale_m0 = 1.0f;
        float qscale_m1 = 1.0f;
        if (use_chunk_scale) {
            // Each 4-lane group shares the same query rows (m0/m1). Broadcast to reduce redundant loads.
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
            // Sb==7/6 are hot configurations (fixed-bit weights) and tend to regress if the compiler
            // can't specialize away the tail checks. Keep explicit fast paths.
            if (Sa == 7 && Sb == 7) {
                bsi_bmma_tm32_accum_hot<7, 7, 32>(
                    A_bits,
                    Aw_tile,
                    b_col_base,
                    bw_col0,
                    bw_col1,
                    threadID,
                    m0,
                    m1,
                    use_chunk_scale,
                    qscale_m0,
                    qscale_m1,
                    acc00,
                    acc01,
                    acc10,
                    acc11);
            } else if (Sa == 7 && Sb == 6) {
                bsi_bmma_tm32_accum_hot<7, 6, 32>(
                    A_bits,
                    Aw_tile,
                    b_col_base,
                    bw_col0,
                    bw_col1,
                    threadID,
                    m0,
                    m1,
                    use_chunk_scale,
                    qscale_m0,
                    qscale_m1,
                    acc00,
                    acc01,
                    acc10,
                    acc11);
            } else if (Sa == 8 && Sb == 7) {
                bsi_bmma_tm32_accum_hot<8, 7, 32>(
                    A_bits,
                    Aw_tile,
                    b_col_base,
                    bw_col0,
                    bw_col1,
                    threadID,
                    m0,
                    m1,
                    use_chunk_scale,
                    qscale_m0,
                    qscale_m1,
                    acc00,
                    acc01,
                    acc10,
                    acc11);
            } else if (Sa == 6 && Sb == 5) {
                bsi_bmma_tm32_accum_hot<6, 5, 32>(
                    A_bits,
                    Aw_tile,
                    b_col_base,
                    bw_col0,
                    bw_col1,
                    threadID,
                    m0,
                    m1,
                    use_chunk_scale,
                    qscale_m0,
                    qscale_m1,
                    acc00,
                    acc01,
                    acc10,
                    acc11);
            } else if (Sb == 7) {
                // j0 = 0..3
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
                        // Issue BMMA in pairs to increase ILP.
                        {
                            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                            asm volatile(
                                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                "{%0, %1, %2, %3}, "
                                "{%4, %5, %6, %7}, "
                                "{%8, %9}, "
                                "{%0, %1, %2, %3};\n"
                                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                  "r"(b0_cache[0]), "r"(b1_cache[0]));
                            asm volatile(
                                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                "{%0, %1, %2, %3}, "
                                "{%4, %5, %6, %7}, "
                                "{%8, %9}, "
                                "{%0, %1, %2, %3};\n"
                                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                  "r"(b0_cache[1]), "r"(b1_cache[1]));
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
                                "{%0, %1, %2, %3}, "
                                "{%4, %5, %6, %7}, "
                                "{%8, %9}, "
                                "{%0, %1, %2, %3};\n"
                                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                  "r"(b0_cache[2]), "r"(b1_cache[2]));
                            asm volatile(
                                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                "{%0, %1, %2, %3}, "
                                "{%4, %5, %6, %7}, "
                                "{%8, %9}, "
                                "{%0, %1, %2, %3};\n"
                                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                  "r"(b0_cache[3]), "r"(b1_cache[3]));
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

                // j0 = 4..6 (3 slices, no bounds checks)
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
                                "{%0, %1, %2, %3}, "
                                "{%4, %5, %6, %7}, "
                                "{%8, %9}, "
                                "{%0, %1, %2, %3};\n"
                                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                  "r"(b0_cache[0]), "r"(b1_cache[0]));
                            asm volatile(
                                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                "{%0, %1, %2, %3}, "
                                "{%4, %5, %6, %7}, "
                                "{%8, %9}, "
                                "{%0, %1, %2, %3};\n"
                                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                  "r"(b0_cache[1]), "r"(b1_cache[1]));
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
                                "{%0, %1, %2, %3}, "
                                "{%4, %5, %6, %7}, "
                                "{%8, %9}, "
                                "{%0, %1, %2, %3};\n"
                                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                  "r"(b0_cache[2]), "r"(b1_cache[2]));
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
                // Sb==6 hot path: keep a single i-loop (avoid reloading A/Aw for the 4+2 slice split).
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

                    // Issue BMMA in pairs to increase ILP.
                    {
                        int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                        asm volatile(
                            "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                            "{%0, %1, %2, %3}, "
                            "{%4, %5, %6, %7}, "
                            "{%8, %9}, "
                            "{%0, %1, %2, %3};\n"
                            : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                              "r"(b0_cache0[0]), "r"(b1_cache0[0]));
                        asm volatile(
                            "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                            "{%0, %1, %2, %3}, "
                            "{%4, %5, %6, %7}, "
                            "{%8, %9}, "
                            "{%0, %1, %2, %3};\n"
                            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                              "r"(b0_cache0[1]), "r"(b1_cache0[1]));

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
                            "{%0, %1, %2, %3}, "
                            "{%4, %5, %6, %7}, "
                            "{%8, %9}, "
                            "{%0, %1, %2, %3};\n"
                            : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                              "r"(b0_cache0[2]), "r"(b1_cache0[2]));
                        asm volatile(
                            "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                            "{%0, %1, %2, %3}, "
                            "{%4, %5, %6, %7}, "
                            "{%8, %9}, "
                            "{%0, %1, %2, %3};\n"
                            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                              "r"(b0_cache0[3]), "r"(b1_cache0[3]));

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
                            "{%0, %1, %2, %3}, "
                            "{%4, %5, %6, %7}, "
                            "{%8, %9}, "
                            "{%0, %1, %2, %3};\n"
                            : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                              "r"(b0_cache1[0]), "r"(b1_cache1[0]));
                        asm volatile(
                            "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                            "{%0, %1, %2, %3}, "
                            "{%4, %5, %6, %7}, "
                            "{%8, %9}, "
                            "{%0, %1, %2, %3};\n"
                            : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                              "r"(b0_cache1[1]), "r"(b1_cache1[1]));

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
                const int Sb_full = Sb & ~(JBLOCK - 1); // fast path for full JBLOCKs
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

                        // Issue BMMA in pairs to increase ILP (helps hide fixed BMMA latency).
                        {
                            int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                            asm volatile(
                                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                "{%0, %1, %2, %3}, "
                                "{%4, %5, %6, %7}, "
                                "{%8, %9}, "
                                "{%0, %1, %2, %3};\n"
                                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                  "r"(b0_cache[0]), "r"(b1_cache[0]));
                            asm volatile(
                                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                "{%0, %1, %2, %3}, "
                                "{%4, %5, %6, %7}, "
                                "{%8, %9}, "
                                "{%0, %1, %2, %3};\n"
                                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                  "r"(b0_cache[1]), "r"(b1_cache[1]));

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
                                "{%0, %1, %2, %3}, "
                                "{%4, %5, %6, %7}, "
                                "{%8, %9}, "
                                "{%0, %1, %2, %3};\n"
                                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                  "r"(b0_cache[2]), "r"(b1_cache[2]));
                            asm volatile(
                                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                "{%0, %1, %2, %3}, "
                                "{%4, %5, %6, %7}, "
                                "{%8, %9}, "
                                "{%0, %1, %2, %3};\n"
                                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                  "r"(b0_cache[3]), "r"(b1_cache[3]));

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

                // Tail (<JBLOCK): specialize by tail size to avoid per-iteration bounds checks in the inner loop.
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
                                "{%0, %1, %2, %3}, "
                                "{%4, %5, %6, %7}, "
                                "{%8, %9}, "
                                "{%0, %1, %2, %3};\n"
                                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                  "r"(b0_tail[0]), "r"(b1_tail[0]));

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
                                "{%0, %1, %2, %3}, "
                                "{%4, %5, %6, %7}, "
                                "{%8, %9}, "
                                "{%0, %1, %2, %3};\n"
                                : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                  "r"(b0_tail[0]), "r"(b1_tail[0]));
                            asm volatile(
                                "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                "{%0, %1, %2, %3}, "
                                "{%4, %5, %6, %7}, "
                                "{%8, %9}, "
                                "{%0, %1, %2, %3};\n"
                                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                                : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                  "r"(b0_tail[1]), "r"(b1_tail[1]));

                            acc00 = __fmaf_rn(aw0, static_cast<float>(c0) * bw0_tail[0] + static_cast<float>(d0) * bw0_tail[1], acc00);
                            acc01 = __fmaf_rn(aw0, static_cast<float>(c1) * bw1_tail[0] + static_cast<float>(d1) * bw1_tail[1], acc01);
                            acc10 = __fmaf_rn(aw1, static_cast<float>(c2) * bw0_tail[0] + static_cast<float>(d2) * bw0_tail[1], acc10);
                            acc11 = __fmaf_rn(aw1, static_cast<float>(c3) * bw1_tail[0] + static_cast<float>(d3) * bw1_tail[1], acc11);
                        }
                    } else { // tail == 3
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
                                    "{%0, %1, %2, %3}, "
                                    "{%4, %5, %6, %7}, "
                                    "{%8, %9}, "
                                    "{%0, %1, %2, %3};\n"
                                    : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                      "r"(b0_tail[0]), "r"(b1_tail[0]));
                                asm volatile(
                                    "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                                    "{%0, %1, %2, %3}, "
                                    "{%4, %5, %6, %7}, "
                                    "{%8, %9}, "
                                    "{%0, %1, %2, %3};\n"
                                    : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                      "r"(b0_tail[1]), "r"(b1_tail[1]));

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
                                    "{%0, %1, %2, %3}, "
                                    "{%4, %5, %6, %7}, "
                                    "{%8, %9}, "
                                    "{%0, %1, %2, %3};\n"
                                    : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                                      "r"(b0_tail[2]), "r"(b1_tail[2]));

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
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%0, %1, %2, %3};\n"
                        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1));

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

    // Indices are heavily reused within a warp:
    // - 4 lanes share the same query row (threadID varies)
    // - 8 lanes share the same key col (groupID varies)
    // Broadcast to reduce redundant global loads.
    const bool identity_q = (query_indices == nullptr);
    const bool identity_r = (key_indices == nullptr);
    if (full_q_tile && full_r_tile && identity_q && identity_r) {
        // Packed query path + fixed-bit key fast path: avoid index loads/shuffles.
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
    (void)A_packed_u32;
    (void)Aw;
    (void)Sa;
    (void)W64;
    (void)B;
    (void)B_packed_u32;
    (void)Bw;
    (void)Sb;
    (void)R;
    (void)Q;
    (void)packed_chunks;
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
    const unsigned long long* __restrict__ A, // [Q, 7, W64]
    const float* __restrict__ A_chunk_scales, // [Q, chunks]
    int A_scale_stride,
    int W64,
    const unsigned long long* __restrict__ B, // [R, 6, W64]
    const float* __restrict__ Bw,             // [R, 6]
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
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0_cache[j]), "r"(b1_cache[j]));

                int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                asm volatile(
                    "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                      "r"(b0_cache[j + 1]), "r"(b1_cache[j + 1]));

                const int shift0 = i + j;
                const int shift1 = shift0 + 1;
                const bool neg0 = ((i == (SA - 1)) ^ (j == (SB - 1)));
                const bool neg1 = ((i == (SA - 1)) ^ ((j + 1) == (SB - 1)));

                const int v0 = c0 << shift0;
                const int v1 = c1 << shift0;
                const int v2 = c2 << shift0;
                const int v3 = c3 << shift0;
                if (neg0) {
                    chunk00 -= v0;
                    chunk01 -= v1;
                    chunk10 -= v2;
                    chunk11 -= v3;
                } else {
                    chunk00 += v0;
                    chunk01 += v1;
                    chunk10 += v2;
                    chunk11 += v3;
                }

                const int w0 = d0 << shift1;
                const int w1 = d1 << shift1;
                const int w2 = d2 << shift1;
                const int w3 = d3 << shift1;
                if (neg1) {
                    chunk00 -= w0;
                    chunk01 -= w1;
                    chunk10 -= w2;
                    chunk11 -= w3;
                } else {
                    chunk00 += w0;
                    chunk01 += w1;
                    chunk10 += w2;
                    chunk11 += w3;
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

template <bool PACKED_A>
__device__ __forceinline__ const unsigned long long* bsi_fixed76_a_chunk_base_ptr(
    const unsigned long long* __restrict__ A,
    int q0,
    int q_tile,
    int chunks,
    int W64,
    int chunk,
    int i,
    int m)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    constexpr int SA = 7;
    constexpr int TM_TOTAL = 32;
    constexpr int K_WORDS64 = 4;
    if constexpr (PACKED_A) {
        return A + (((((size_t)q_tile * (size_t)chunks + (size_t)chunk) * (size_t)SA + (size_t)i) *
                      (size_t)TM_TOTAL +
                      (size_t)m) *
                     (size_t)K_WORDS64);
    }
    const int q = q0 + m;
    return A + (((size_t)q * (size_t)SA + (size_t)i) * (size_t)W64 +
                (size_t)chunk * (size_t)K_WORDS64);
#else
    (void)q0;
    (void)q_tile;
    (void)chunks;
    (void)W64;
    (void)chunk;
    (void)i;
    (void)m;
    return A;
#endif
}

template <int R_SWEEP, bool PACKED_A>
__device__ __forceinline__ void bsi_fixed76_tm32_chunkscale_rsweep_body(
    unsigned char* __restrict__ smem_raw,
    const unsigned long long* __restrict__ A, // [Q, 7, W64]
    const float* __restrict__ A_chunk_scales, // [Q, chunks]
    int A_scale_stride,
    int W64,
    const unsigned long long* __restrict__ B, // [R, 6, W64]
    const float* __restrict__ Bw,             // [R, 6]
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
    const int q_tile = blockIdx.y;
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
    float4* acc0 = nullptr;
    if constexpr (R_SWEEP != 4) {
        acc0 = reinterpret_cast<float4*>(p);
        p += (size_t)R_SWEEP * (size_t)blockDim.x * sizeof(float4);
    }
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

    float4 acc_reg[R_SWEEP];
#pragma unroll
    for (int t = 0; t < R_SWEEP; ++t) {
        if constexpr (R_SWEEP == 4) {
            acc_reg[t] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        } else {
            acc0[(size_t)t * (size_t)blockDim.x + (size_t)threadIdx.x] =
                make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
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
            const unsigned long long* a_chunk = bsi_fixed76_a_chunk_base_ptr<PACKED_A>(
                A, q0, q_tile, chunks, W64, /*chunk=*/0, i, m);
            const int w64_i = w64_pair << 1;
            const int base = ((i * TM_TOTAL + m) * K_STRIDE32) + (w64_i << 1);
            bsi_cp_async_cg_16B(A_bits + base, &a_chunk[(size_t)w64_i]);
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
                const unsigned long long* a_chunk = bsi_fixed76_a_chunk_base_ptr<PACKED_A>(
                    A, q0, q_tile, chunks, W64, next_chunk, i, m);
                const int w64_i = w64_pair << 1;
                const int base = ((i * TM_TOTAL + m) * K_STRIDE32) + (w64_i << 1);
                bsi_cp_async_cg_16B(
                    A_bits_next + base,
                    &a_chunk[(size_t)w64_i]);
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
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%0, %1, %2, %3};\n"
                        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0_0[j]), "r"(b1_0[j]));

                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    asm volatile(
                        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%0, %1, %2, %3};\n"
                        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0_1[j]), "r"(b1_1[j]));

                    const int v0 = c0 << shift;
                    const int v1 = c1 << shift;
                    const int v2 = c2 << shift;
                    const int v3 = c3 << shift;
                    const int w0 = d0 << shift;
                    const int w1 = d1 << shift;
                    const int w2 = d2 << shift;
                    const int w3 = d3 << shift;
                    if (neg) {
                        chunk00_0 -= v0;
                        chunk01_0 -= v1;
                        chunk10_0 -= v2;
                        chunk11_0 -= v3;
                        chunk00_1 -= w0;
                        chunk01_1 -= w1;
                        chunk10_1 -= w2;
                        chunk11_1 -= w3;
                    } else {
                        chunk00_0 += v0;
                        chunk01_0 += v1;
                        chunk10_0 += v2;
                        chunk11_0 += v3;
                        chunk00_1 += w0;
                        chunk01_1 += w1;
                        chunk10_1 += w2;
                        chunk11_1 += w3;
                    }
                }
            }

            {
                const float s00 = qscale_m0 * bscale0[t0];
                const float s01 = qscale_m0 * bscale1[t0];
                const float s10 = qscale_m1 * bscale0[t0];
                const float s11 = qscale_m1 * bscale1[t0];
                if constexpr (R_SWEEP == 4) {
                    float4& acc = acc_reg[t0];
                    acc.x = __fmaf_rn(static_cast<float>(chunk00_0), s00, acc.x);
                    acc.y = __fmaf_rn(static_cast<float>(chunk01_0), s01, acc.y);
                    acc.z = __fmaf_rn(static_cast<float>(chunk10_0), s10, acc.z);
                    acc.w = __fmaf_rn(static_cast<float>(chunk11_0), s11, acc.w);
                } else {
                    float4 acc = acc0[(size_t)t0 * (size_t)blockDim.x + (size_t)threadIdx.x];
                    acc.x = __fmaf_rn(static_cast<float>(chunk00_0), s00, acc.x);
                    acc.y = __fmaf_rn(static_cast<float>(chunk01_0), s01, acc.y);
                    acc.z = __fmaf_rn(static_cast<float>(chunk10_0), s10, acc.z);
                    acc.w = __fmaf_rn(static_cast<float>(chunk11_0), s11, acc.w);
                    acc0[(size_t)t0 * (size_t)blockDim.x + (size_t)threadIdx.x] = acc;
                }
            }
            {
                const float s00 = qscale_m0 * bscale0[t1];
                const float s01 = qscale_m0 * bscale1[t1];
                const float s10 = qscale_m1 * bscale0[t1];
                const float s11 = qscale_m1 * bscale1[t1];
                if constexpr (R_SWEEP == 4) {
                    float4& acc = acc_reg[t1];
                    acc.x = __fmaf_rn(static_cast<float>(chunk00_1), s00, acc.x);
                    acc.y = __fmaf_rn(static_cast<float>(chunk01_1), s01, acc.y);
                    acc.z = __fmaf_rn(static_cast<float>(chunk10_1), s10, acc.z);
                    acc.w = __fmaf_rn(static_cast<float>(chunk11_1), s11, acc.w);
                } else {
                    float4 acc = acc0[(size_t)t1 * (size_t)blockDim.x + (size_t)threadIdx.x];
                    acc.x = __fmaf_rn(static_cast<float>(chunk00_1), s00, acc.x);
                    acc.y = __fmaf_rn(static_cast<float>(chunk01_1), s01, acc.y);
                    acc.z = __fmaf_rn(static_cast<float>(chunk10_1), s10, acc.z);
                    acc.w = __fmaf_rn(static_cast<float>(chunk11_1), s11, acc.w);
                    acc0[(size_t)t1 * (size_t)blockDim.x + (size_t)threadIdx.x] = acc;
                }
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
        float4 acc;
        if constexpr (R_SWEEP == 4) {
            acc = acc_reg[t];
        } else {
            acc = acc0[(size_t)t * (size_t)blockDim.x + (size_t)threadIdx.x];
        }
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
    bsi_fixed76_tm32_chunkscale_rsweep_body<2, false>(
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
    bsi_fixed76_tm32_chunkscale_rsweep_body<4, false>(
        smem_raw, A, A_chunk_scales, A_scale_stride, W64, B, Bw, R, Q, scale_inv, R_total, out_global);
}

extern "C" __global__ __launch_bounds__(256, 2)
void popcount_weighted_keys_literal_fused_bmma_tc_kernel_tm32_fixed76_chunkscale_rsweep4_packedA(
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
    bsi_fixed76_tm32_chunkscale_rsweep_body<4, true>(
        smem_raw, A, A_chunk_scales, A_scale_stride, W64, B, Bw, R, Q, scale_inv, R_total, out_global);
}

template <int R_SWEEP, bool PACKED_A>
__device__ __forceinline__ void bsi_fixed76_tm32_chunkscale_rsweep_body_tma_tensorB(
    unsigned char* __restrict__ smem_raw,
    const unsigned long long* __restrict__ A, // [Q, 7, W64]
    const float* __restrict__ A_chunk_scales, // [Q, chunks]
    int A_scale_stride,
    int W64,
    const unsigned long long* __restrict__ B, // [R, 6, W64]
    const float* __restrict__ Bw,             // [R, 6]
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
    const int q_tile = blockIdx.y;
    const int r_base = blockIdx.x * (TN * R_SWEEP);
    const int tile_base = blockIdx.x * R_SWEEP;

    uintptr_t p = reinterpret_cast<uintptr_t>(smem_raw);
    // TMA bulk tensor copies are picky about destination alignment. Keep all stage
    // buffers 128B-aligned to avoid misaligned-address faults.
    p = (p + 127u) & ~uintptr_t(127u);

    constexpr int stages = 2;
    constexpr size_t A_words = (size_t)SA * (size_t)TM_TOTAL * (size_t)K_STRIDE32;
    constexpr size_t B_words = (size_t)SB * (size_t)TN * (size_t)K_STRIDE32;
    constexpr size_t B_words_total = (size_t)R_SWEEP * B_words;
    auto* A_bits0 = reinterpret_cast<uint32_t*>(p);
    p += (size_t)stages * A_words * sizeof(uint32_t);
    auto* B_bits0 = reinterpret_cast<uint32_t*>(p);
    p += (size_t)stages * B_words_total * sizeof(uint32_t);
    float4* acc0 = nullptr;
    if constexpr (R_SWEEP != 4) {
        acc0 = reinterpret_cast<float4*>(p);
        p += (size_t)R_SWEEP * (size_t)blockDim.x * sizeof(float4);
    }
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

    float4 acc_reg[R_SWEEP];
#pragma unroll
    for (int t = 0; t < R_SWEEP; ++t) {
        if constexpr (R_SWEEP == 4) {
            acc_reg[t] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        } else {
            acc0[(size_t)t * (size_t)blockDim.x + (size_t)threadIdx.x] =
                make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }

    const int chunks = W64 / K_WORDS64;
    if (chunks <= 0) return;

    // Shared barrier per stage for the bulk-tensor copy of B.
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

    constexpr uint32_t ROW_BYTES = (uint32_t)(K_WORDS64 * sizeof(unsigned long long)); // 32B
    constexpr uint32_t TX_B_BYTES =
        (uint32_t)((size_t)R_SWEEP * (size_t)TN * (size_t)SB * (size_t)ROW_BYTES);

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
            auto handle = cuda::device::barrier_native_handle(bar[stage]);
            // NVHPC's CUDA 12.6 CCCL only provides the mbarrier completion overload for
            // shared::cluster -> global. With cluster size 1, shared::cluster behaves
            // like regular shared memory, so this is safe.
            const int32_t coords[4] = {
                (int32_t)(chunk * (int)ROW_BYTES),
                0,
                0,
                (int32_t)tile_base,
            };
            ptx::cp_async_bulk_tensor(
                ptx::space_cluster, ptx::space_global, B_dst_bytes, B_tensor_map, coords, handle);
        }
    };

    // Prime stage 0 with chunk 0.
    {
        uint32_t* A_bits = A_bits0;
        constexpr int K_WORDS64_16B = K_WORDS64 / 2;
        for (int idx = threadIdx.x; idx < TM_TOTAL * SA * K_WORDS64_16B; idx += blockDim.x) {
            int t = idx;
            const int w64_pair = t & (K_WORDS64_16B - 1);
            t >>= 1;
            const int m = t & (TM_TOTAL - 1);
            const int i = t >> 5;
            const unsigned long long* a_chunk = bsi_fixed76_a_chunk_base_ptr<PACKED_A>(
                A, q0, q_tile, chunks, W64, /*chunk=*/0, i, m);
            const int w64_i = w64_pair << 1;
            const int base = ((i * TM_TOTAL + m) * K_STRIDE32) + (w64_i << 1);
            bsi_cp_async_cg_16B(A_bits + base, &a_chunk[(size_t)w64_i]);
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
        uint32_t* B_bits = (stage == 0) ? B_bits0 : (B_bits0 + B_words_total);

        const int next_chunk = chunk + 1;
        const int next_stage = stage ^ 1;
        if (next_chunk < chunks) {
            uint32_t* A_bits_next = (next_stage == 0) ? A_bits0 : (A_bits0 + A_words);
            uint32_t* B_bits_next = (next_stage == 0) ? B_bits0 : (B_bits0 + B_words_total);

            constexpr int K_WORDS64_16B = K_WORDS64 / 2;
            for (int idx = threadIdx.x; idx < TM_TOTAL * SA * K_WORDS64_16B; idx += blockDim.x) {
                int t = idx;
                const int w64_pair = t & (K_WORDS64_16B - 1);
                t >>= 1;
                const int m = t & (TM_TOTAL - 1);
                const int i = t >> 5;
                const unsigned long long* a_chunk = bsi_fixed76_a_chunk_base_ptr<PACKED_A>(
                    A, q0, q_tile, chunks, W64, next_chunk, i, m);
                const int w64_i = w64_pair << 1;
                const int base = ((i * TM_TOTAL + m) * K_STRIDE32) + (w64_i << 1);
                bsi_cp_async_cg_16B(
                    A_bits_next + base,
                    &a_chunk[(size_t)w64_i]);
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

#pragma unroll
        for (int t_pair = 0; t_pair < R_SWEEP; t_pair += 2) {
            const int t0 = t_pair;
            const int t1 = t0 + 1;
            const uint32_t* B0 = B_bits + (size_t)t0 * B_words;
            const uint32_t* B1 = B_bits + (size_t)t1 * B_words;

            uint32_t b0_0[SB];
            uint32_t b1_0[SB];
            uint32_t b0_1[SB];
            uint32_t b1_1[SB];

            // B shared layout is flattened tile-major: [tile, sb, n, w_bytes].
            const int n = col_base + groupID;
            const int b_slice_stride = TN * K_STRIDE32;
            const uint32_t* b_col_base0 = B0 + n * K_STRIDE32;
            const uint32_t* b_col_base1 = B1 + n * K_STRIDE32;
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
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%0, %1, %2, %3};\n"
                        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0_0[j]), "r"(b1_0[j]));

                    int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                    asm volatile(
                        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%0, %1, %2, %3};\n"
                        : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0_1[j]), "r"(b1_1[j]));

                    const int v0 = c0 << shift;
                    const int v1 = c1 << shift;
                    const int v2 = c2 << shift;
                    const int v3 = c3 << shift;
                    const int w0 = d0 << shift;
                    const int w1 = d1 << shift;
                    const int w2 = d2 << shift;
                    const int w3 = d3 << shift;
                    if (neg) {
                        chunk00_0 -= v0;
                        chunk01_0 -= v1;
                        chunk10_0 -= v2;
                        chunk11_0 -= v3;
                        chunk00_1 -= w0;
                        chunk01_1 -= w1;
                        chunk10_1 -= w2;
                        chunk11_1 -= w3;
                    } else {
                        chunk00_0 += v0;
                        chunk01_0 += v1;
                        chunk10_0 += v2;
                        chunk11_0 += v3;
                        chunk00_1 += w0;
                        chunk01_1 += w1;
                        chunk10_1 += w2;
                        chunk11_1 += w3;
                    }
                }
            }
            {
                const float s00 = qscale_m0 * bscale0[t0];
                const float s01 = qscale_m0 * bscale1[t0];
                const float s10 = qscale_m1 * bscale0[t0];
                const float s11 = qscale_m1 * bscale1[t0];
                if constexpr (R_SWEEP == 4) {
                    float4& acc = acc_reg[t0];
                    acc.x = __fmaf_rn(static_cast<float>(chunk00_0), s00, acc.x);
                    acc.y = __fmaf_rn(static_cast<float>(chunk01_0), s01, acc.y);
                    acc.z = __fmaf_rn(static_cast<float>(chunk10_0), s10, acc.z);
                    acc.w = __fmaf_rn(static_cast<float>(chunk11_0), s11, acc.w);
                } else {
                    float4 acc = acc0[(size_t)t0 * (size_t)blockDim.x + (size_t)threadIdx.x];
                    acc.x = __fmaf_rn(static_cast<float>(chunk00_0), s00, acc.x);
                    acc.y = __fmaf_rn(static_cast<float>(chunk01_0), s01, acc.y);
                    acc.z = __fmaf_rn(static_cast<float>(chunk10_0), s10, acc.z);
                    acc.w = __fmaf_rn(static_cast<float>(chunk11_0), s11, acc.w);
                    acc0[(size_t)t0 * (size_t)blockDim.x + (size_t)threadIdx.x] = acc;
                }
            }
            {
                const float s00 = qscale_m0 * bscale0[t1];
                const float s01 = qscale_m0 * bscale1[t1];
                const float s10 = qscale_m1 * bscale0[t1];
                const float s11 = qscale_m1 * bscale1[t1];
                if constexpr (R_SWEEP == 4) {
                    float4& acc = acc_reg[t1];
                    acc.x = __fmaf_rn(static_cast<float>(chunk00_1), s00, acc.x);
                    acc.y = __fmaf_rn(static_cast<float>(chunk01_1), s01, acc.y);
                    acc.z = __fmaf_rn(static_cast<float>(chunk10_1), s10, acc.z);
                    acc.w = __fmaf_rn(static_cast<float>(chunk11_1), s11, acc.w);
                } else {
                    float4 acc = acc0[(size_t)t1 * (size_t)blockDim.x + (size_t)threadIdx.x];
                    acc.x = __fmaf_rn(static_cast<float>(chunk00_1), s00, acc.x);
                    acc.y = __fmaf_rn(static_cast<float>(chunk01_1), s01, acc.y);
                    acc.z = __fmaf_rn(static_cast<float>(chunk10_1), s10, acc.z);
                    acc.w = __fmaf_rn(static_cast<float>(chunk11_1), s11, acc.w);
                    acc0[(size_t)t1 * (size_t)blockDim.x + (size_t)threadIdx.x] = acc;
                }
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
        float4 acc;
        if constexpr (R_SWEEP == 4) {
            acc = acc_reg[t];
        } else {
            acc = acc0[(size_t)t * (size_t)blockDim.x + (size_t)threadIdx.x];
        }
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
    bsi_fixed76_tm32_chunkscale_rsweep_body_tma_tensorB<2, false>(
        smem_raw, A, A_chunk_scales, A_scale_stride, W64, B, Bw, B_tensor_map, R, Q, scale_inv, R_total, out_global);
}

extern "C" __global__ __launch_bounds__(256, 2)
void popcount_weighted_keys_literal_fused_bmma_tc_kernel_tm32_fixed76_chunkscale_rsweep4_tma_tensorB(
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
    bsi_fixed76_tm32_chunkscale_rsweep_body_tma_tensorB<4, false>(
        smem_raw, A, A_chunk_scales, A_scale_stride, W64, B, Bw, B_tensor_map, R, Q, scale_inv, R_total, out_global);
}

extern "C" __global__ __launch_bounds__(256, 2)
void popcount_weighted_keys_literal_fused_bmma_tc_kernel_tm32_fixed76_chunkscale_rsweep4_packedA_tma_tensorB(
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
    bsi_fixed76_tm32_chunkscale_rsweep_body_tma_tensorB<4, true>(
        smem_raw, A, A_chunk_scales, A_scale_stride, W64, B, Bw, B_tensor_map, R, Q, scale_inv, R_total, out_global);
}


template <int SA, int SB, int TM_ROWS>
__device__ __forceinline__ void bsi_bmma_tm32_accum_hot(
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
    float& acc11)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    static_assert(SA >= 4 && SA <= 10, "SA must be in [4, 10]");
    static_assert(SB >= 4 && SB <= 10, "SB must be in [4, 10]");
    constexpr int TM_TOTAL = TM_ROWS;
    constexpr int TN = 32;
    constexpr int K_STRIDE32 = 12;
    const int b_slice_stride = TN * K_STRIDE32;

    float chunk00 = 0.0f, chunk01 = 0.0f, chunk10 = 0.0f, chunk11 = 0.0f;
#pragma unroll
    for (int i = 0; i < SA; ++i) {
        const uint32_t* A_i = A_bits + (size_t)i * (size_t)TM_TOTAL * (size_t)K_STRIDE32;
        const uint32_t a0 = A_i[(size_t)m0 * (size_t)K_STRIDE32 + (size_t)threadID];
        const uint32_t a1 = A_i[(size_t)m1 * (size_t)K_STRIDE32 + (size_t)threadID];
        const uint32_t a2 = A_i[(size_t)m0 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];
        const uint32_t a3 = A_i[(size_t)m1 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];

        float sum00 = 0.0f, sum01 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;
#pragma unroll
        for (int j = 0; j < SB; j += 2) {
            {
                const uint32_t* b_col = b_col_base + j * b_slice_stride;
                const uint32_t b0 = b_col[threadID];
                const uint32_t b1 = b_col[threadID + 4];
                const float bw0 = __ldg(&bw_col0[j]);
                const float bw1 = __ldg(&bw_col1[j]);
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                asm volatile(
                    "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));

                sum00 = __fmaf_rn(static_cast<float>(c0), bw0, sum00);
                sum01 = __fmaf_rn(static_cast<float>(c1), bw1, sum01);
                sum10 = __fmaf_rn(static_cast<float>(c2), bw0, sum10);
                sum11 = __fmaf_rn(static_cast<float>(c3), bw1, sum11);
            }
            if (j + 1 < SB) {
                const uint32_t* b_col = b_col_base + (j + 1) * b_slice_stride;
                const uint32_t b0 = b_col[threadID];
                const uint32_t b1 = b_col[threadID + 4];
                const float bw0 = __ldg(&bw_col0[j + 1]);
                const float bw1 = __ldg(&bw_col1[j + 1]);
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                asm volatile(
                    "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));

                sum00 = __fmaf_rn(static_cast<float>(c0), bw0, sum00);
                sum01 = __fmaf_rn(static_cast<float>(c1), bw1, sum01);
                sum10 = __fmaf_rn(static_cast<float>(c2), bw0, sum10);
                sum11 = __fmaf_rn(static_cast<float>(c3), bw1, sum11);
            }
        }

        const float aw0 = Aw_tile[(size_t)m0 * (size_t)SA + (size_t)i];
        const float aw1 = Aw_tile[(size_t)m1 * (size_t)SA + (size_t)i];
        chunk00 = __fmaf_rn(aw0, sum00, chunk00);
        chunk01 = __fmaf_rn(aw0, sum01, chunk01);
        chunk10 = __fmaf_rn(aw1, sum10, chunk10);
        chunk11 = __fmaf_rn(aw1, sum11, chunk11);
    }

    const float qmul0 = use_chunk_scale ? qscale_m0 : 1.0f;
    const float qmul1 = use_chunk_scale ? qscale_m1 : 1.0f;
    acc00 = __fmaf_rn(qmul0, chunk00, acc00);
    acc01 = __fmaf_rn(qmul0, chunk01, acc01);
    acc10 = __fmaf_rn(qmul1, chunk10, acc10);
    acc11 = __fmaf_rn(qmul1, chunk11, acc11);
#else
    (void)A_bits;
    (void)Aw_tile;
    (void)b_col_base;
    (void)bw_col0;
    (void)bw_col1;
    (void)threadID;
    (void)m0;
    (void)m1;
    (void)use_chunk_scale;
    (void)qscale_m0;
    (void)qscale_m1;
    (void)acc00;
    (void)acc01;
    (void)acc10;
    (void)acc11;
#endif
}

struct BsiSharedLimits {
    int max_shared_default = 0;
    int max_shared_optin = 0;
    int max_shared = 0;
};

static inline const BsiSharedLimits& bsi_get_shared_limits_cached() {
    static BsiSharedLimits limits = []() {
        BsiSharedLimits out;
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&out.max_shared_default, cudaDevAttrMaxSharedMemoryPerBlock, dev);
        cudaDeviceGetAttribute(&out.max_shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        out.max_shared = (out.max_shared_optin > out.max_shared_default)
            ? out.max_shared_optin
            : out.max_shared_default;
        return out;
    }();
    return limits;
}

#if !defined(__CUDA_ARCH__)
namespace bsi_tma {

// Some CUDA toolchains ship the cuTensorMapEncodeTiled declaration but not the
// PFN_cuTensorMapEncodeTiled typedef. Use our own function pointer type so we
// can still resolve the symbol via cudaGetDriverEntryPointByVersion.
using BsiCuTensorMapEncodeTiledFn = CUresult (*)(
    CUtensorMap*,
    CUtensorMapDataType,
    unsigned int,
    void*,
    const cuuint64_t*,
    const cuuint64_t*,
    const cuuint32_t*,
    const cuuint32_t*,
    CUtensorMapInterleave,
    CUtensorMapSwizzle,
    CUtensorMapL2promotion,
    CUtensorMapFloatOOBfill);

struct TmaTensorMapKey {
    const void* base = nullptr; // global base pointer encoded into the tensor map
    int device = 0;
    int w64 = 0;
    int r_total = 0;
    int r_sweep = 0;
};

struct TmaTensorMapKeyHash {
    size_t operator()(const TmaTensorMapKey& k) const noexcept {
        size_t h = 0;
        auto mix = [&](size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        };
        mix(std::hash<const void*>()(k.base));
        mix(std::hash<int>()(k.device));
        mix(std::hash<int>()(k.w64));
        mix(std::hash<int>()(k.r_total));
        mix(std::hash<int>()(k.r_sweep));
        return h;
    }
};

static inline bool operator==(const TmaTensorMapKey& a, const TmaTensorMapKey& b) {
    return (a.base == b.base) && (a.device == b.device) && (a.w64 == b.w64) && (a.r_total == b.r_total) &&
        (a.r_sweep == b.r_sweep);
}

struct TmaTensorMapValue {
    void* d_tma = nullptr; // device pointer to CUtensorMap
};

static inline BsiCuTensorMapEncodeTiledFn bsi_get_encode_tiled_fn() {
    static BsiCuTensorMapEncodeTiledFn fn = nullptr;
    static int initialized = 0;
    if (!initialized) {
        initialized = 1;
        cudaDriverEntryPointQueryResult driver_status = cudaDriverEntryPointSymbolNotFound;
        void* p = nullptr;
        // Use the compile-time runtime version to avoid ABI mismatches.
        const unsigned int cuda_version = static_cast<unsigned int>(CUDART_VERSION);
        const cudaError_t err = cudaGetDriverEntryPointByVersion(
            "cuTensorMapEncodeTiled",
            &p,
            cuda_version,
            cudaEnableDefault,
            &driver_status);
        if (err == cudaSuccess && driver_status == cudaDriverEntryPointSuccess && p != nullptr) {
            fn = reinterpret_cast<BsiCuTensorMapEncodeTiledFn>(p);
        }
    }
    return fn;
}

static inline void* bsi_get_or_create_b_fixed76_rsweep_tensor_map(
    const unsigned long long* B,
    int W64,
    int R_total,
    int R_sweep,
    cudaStream_t stream)
{
    constexpr int TN = 32;
    constexpr int SB = 6;
    constexpr int BYTES_PER_CHUNK = 32; // K=256b

    if (B == nullptr) return nullptr;
    if (W64 <= 0 || (W64 & 3) != 0) return nullptr;
    if (R_total <= 0 || (R_total & (TN - 1)) != 0) return nullptr;
    if (!(R_sweep == 2 || R_sweep == 4)) return nullptr;

    int dev = 0;
    cudaGetDevice(&dev);

    const TmaTensorMapKey key{
        /*base=*/B,
        /*device=*/dev,
        /*w64=*/W64,
        /*r_total=*/R_total,
        /*r_sweep=*/R_sweep,
    };

    static std::mutex mu;
    static std::unordered_map<TmaTensorMapKey, TmaTensorMapValue, TmaTensorMapKeyHash> cache;

    {
        std::lock_guard<std::mutex> lock(mu);
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second.d_tma;
        }
    }

    BsiCuTensorMapEncodeTiledFn encode = bsi_get_encode_tiled_fn();
    if (!encode) return nullptr;

    // Tensor map dims order is innermost->outermost.
    // We encode B as: [w_bytes, n(32), sb, tile(R_total/32)].
    //
    // This makes the TMA destination shared layout (outermost->innermost):
    //   [tile, sb, n, w_bytes]
    // which matches the baseline cp.async staging layout but without needing
    // padding in the innermost dimension.
    const cuuint64_t w_bytes = static_cast<cuuint64_t>(W64) * 8ull;
    const cuuint64_t tile_count = static_cast<cuuint64_t>(R_total / TN);

    const cuuint64_t global_dim[4] = {w_bytes, (cuuint64_t)TN, (cuuint64_t)SB, tile_count};
    const cuuint64_t global_strides[3] = {
        // stride (bytes) for n (r within tile)
        w_bytes * (cuuint64_t)SB,
        // stride (bytes) for sb
        w_bytes,
        // stride (bytes) for tile
        w_bytes * (cuuint64_t)SB * (cuuint64_t)TN,
    };
    const cuuint32_t box_dim[4] = {(cuuint32_t)BYTES_PER_CHUNK, (cuuint32_t)TN, (cuuint32_t)SB, (cuuint32_t)R_sweep};
    const cuuint32_t elem_strides[4] = {1u, 1u, 1u, 1u};

    alignas(64) CUtensorMap tma;
    const CUresult res = encode(
        &tma,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        /*tensorRank=*/4,
        /*globalAddress=*/(void*)B,
        global_dim,
        global_strides,
        box_dim,
        elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if (res != CUDA_SUCCESS) return nullptr;

    void* d_tma = nullptr;
    if (cudaMalloc(&d_tma, sizeof(CUtensorMap)) != cudaSuccess) return nullptr;
    if (cudaMemcpyAsync(d_tma, &tma, sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        cudaFree(d_tma);
        return nullptr;
    }

    {
        std::lock_guard<std::mutex> lock(mu);
        cache.emplace(key, TmaTensorMapValue{d_tma});
    }
    return d_tma;
}

} // namespace bsi_tma
#else
namespace bsi_tma {
static inline void* bsi_get_or_create_b_fixed76_rsweep_tensor_map(
    const unsigned long long*,
    int,
    int,
    int,
    cudaStream_t)
{
    return nullptr;
}
} // namespace bsi_tma
#endif // !defined(__CUDA_ARCH__)

#if !defined(__CUDA_ARCH__)
namespace {
static int g_last_dot_predicted_active_blocks_per_sm = 0;
static size_t g_last_dot_dynamic_shared_bytes = 0;
static int g_last_dot_registers_per_thread = 0;
static int g_last_dot_used_sm90_packed = 0;
static int g_last_dot_used_cpasync = 0;
static int g_last_dot_used_tma = 0;
static int g_last_dot_packed_chunks = 0;
static const char* g_last_dot_kernel_kind = "none";
static const char* g_last_dot_tile_shape = "";
static const char* g_last_dot_pack_layout = "sm90_b1_u32_tile32_v2";
}

extern "C" pybind11::dict bsi_get_last_dot_launch_stats_cuda() {
    pybind11::dict out;
    out["predicted_active_blocks_per_sm"] = pybind11::int_(g_last_dot_predicted_active_blocks_per_sm);
    out["dynamic_shared_bytes"] = pybind11::int_(static_cast<long long>(g_last_dot_dynamic_shared_bytes));
    out["registers_per_thread"] = pybind11::int_(g_last_dot_registers_per_thread);
    out["used_sm90_packed"] = pybind11::bool_(g_last_dot_used_sm90_packed != 0);
    out["used_cpasync"] = pybind11::bool_(g_last_dot_used_cpasync != 0);
    out["used_tma"] = pybind11::bool_(g_last_dot_used_tma != 0);
    out["packed_chunks"] = pybind11::int_(g_last_dot_packed_chunks);
    out["kernel_kind"] = pybind11::str(g_last_dot_kernel_kind);
    out["tile_shape"] = pybind11::str(g_last_dot_tile_shape);
    out["pack_layout"] = pybind11::str(g_last_dot_pack_layout);
    return out;
}
#endif


extern "C" void launch_popcount_weighted_keys_literal_fused_multiq(
    const unsigned long long* A,
    const unsigned int* A_packed_u32,
    const float* Aw,
    const float* A_chunk_scales,
    int A_scale_stride,
    int Sa,
    int W,
    int packed_chunks,
    const unsigned long long* B,
    const unsigned int* B_packed_u32,
    const float* Bw,
    int Sb,
    int R,
    int Q,
    int q_tile,
    int r_tile,
    const long long* indices_r,
    const long long* indices_q,
    float scale_inv,
    int R_total,
    float* out_global,
    cudaStream_t stream,
    int reference_only)
{
    const bool use_sm90_packed = (A_packed_u32 != nullptr) && (B_packed_u32 != nullptr) && (packed_chunks > 0);
#if !defined(__CUDA_ARCH__)
    g_last_dot_predicted_active_blocks_per_sm = 0;
    g_last_dot_dynamic_shared_bytes = 0;
    g_last_dot_registers_per_thread = 0;
    g_last_dot_used_sm90_packed = 0;
    g_last_dot_used_cpasync = 0;
    g_last_dot_used_tma = 0;
    g_last_dot_packed_chunks = packed_chunks;
    g_last_dot_kernel_kind = "none";
    g_last_dot_tile_shape = "";
#endif
    const bool use_tc = use_sm90_packed || (A_chunk_scales != nullptr && A_scale_stride > 0);
    static int cached_tc_ok = -1;
    if (cached_tc_ok < 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        cached_tc_ok = (major >= 9) ? 1 : 0;
    }

    auto launch_packed_reference = [&](int q_begin, int q_count, int r_begin, int r_count) {
        if (q_count <= 0 || r_count <= 0) return;
        dim3 block_ref(16, 8, 1);
        dim3 grid_ref((r_count + block_ref.x - 1) / block_ref.x, (q_count + block_ref.y - 1) / block_ref.y, 1);
        popcount_weighted_keys_literal_fused_packed_reference_kernel<<<grid_ref, block_ref, 0, stream>>>(
            A_packed_u32,
            Aw,
            A_chunk_scales,
            A_scale_stride,
            Sa,
            packed_chunks,
            B_packed_u32,
            Bw,
            Sb,
            q_begin,
            q_count,
            r_begin,
            r_count,
            indices_r ? (indices_r + r_begin) : nullptr,
            indices_q ? (indices_q + q_begin) : nullptr,
            scale_inv,
            R_total,
            out_global);
    };

    if (use_tc && cached_tc_ok && use_sm90_packed && Sa > 0 && Sb > 0) {
        const int Q_full = (Q / 16) * 16;
        const int R_full = (R / 32) * 32;
        const auto& smem_limits = bsi_get_shared_limits_cached();
        const int max_shared_default = smem_limits.max_shared_default;
        const int max_shared = smem_limits.max_shared;

        auto try_tile16x32 = [&]() -> bool {
            constexpr int TM = 16;
            constexpr int TN = 32;
            constexpr int K_WORDS32 = 8;
            constexpr int K_STRIDE32 = K_WORDS32 + 4;
            dim3 block_tc(128, 1, 1);
            const int use_cpasync_requested = 1;
            int use_cpasync = use_cpasync_requested;
            const int stages = use_cpasync ? 2 : 1;
            size_t shared_bytes =
                16u +
                (size_t)stages * (size_t)Sa * (size_t)TM * (size_t)K_STRIDE32 * sizeof(uint32_t) +
                (size_t)stages * (size_t)Sb * (size_t)TN * (size_t)K_STRIDE32 * sizeof(uint32_t);
            if (shared_bytes > (size_t)max_shared && use_cpasync) {
                use_cpasync = 0;
                shared_bytes =
                    16u +
                    (size_t)Sa * (size_t)TM * (size_t)K_STRIDE32 * sizeof(uint32_t) +
                    (size_t)Sb * (size_t)TN * (size_t)K_STRIDE32 * sizeof(uint32_t);
            }
            if (shared_bytes > (size_t)max_shared) return false;

            static size_t configured_shared_tile16x32 = 0;
            if (shared_bytes > (size_t)max_shared_default && shared_bytes > configured_shared_tile16x32) {
                cudaFuncSetAttribute(
                    popcount_weighted_keys_literal_fused_bmma_tc_kernel_tile16x32,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    (int)shared_bytes);
                cudaFuncSetAttribute(
                    popcount_weighted_keys_literal_fused_bmma_tc_kernel_tile16x32,
                    cudaFuncAttributePreferredSharedMemoryCarveout,
                    100);
                configured_shared_tile16x32 = shared_bytes;
            }

#if !defined(__CUDA_ARCH__)
            cudaFuncAttributes attrs{};
            cudaFuncGetAttributes(&attrs, popcount_weighted_keys_literal_fused_bmma_tc_kernel_tile16x32);
            int predicted_blocks = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &predicted_blocks,
                popcount_weighted_keys_literal_fused_bmma_tc_kernel_tile16x32,
                block_tc.x,
                shared_bytes);
            g_last_dot_predicted_active_blocks_per_sm = predicted_blocks;
            g_last_dot_dynamic_shared_bytes = shared_bytes;
            g_last_dot_registers_per_thread = attrs.numRegs;
            g_last_dot_used_sm90_packed = 1;
            g_last_dot_used_cpasync = use_cpasync;
            g_last_dot_used_tma = 0;
            g_last_dot_tile_shape = "16x32x256";
            if (reference_only) {
                g_last_dot_kernel_kind = "reference_only";
            } else if (Sa == 7 && Sb == 6) {
                g_last_dot_kernel_kind = "specialized_7_6";
            } else if (Sa == 8 && Sb == 7) {
                g_last_dot_kernel_kind = "specialized_8_7";
            } else if (Sa == 6 && Sb == 5) {
                g_last_dot_kernel_kind = "specialized_6_5";
            } else if (Sa == 7 && Sb == 7) {
                g_last_dot_kernel_kind = "specialized_7_7";
            } else {
                g_last_dot_kernel_kind = "generic";
            }
#endif
            if (reference_only) {
#if !defined(__CUDA_ARCH__)
                g_last_dot_predicted_active_blocks_per_sm = 0;
                g_last_dot_dynamic_shared_bytes = 0;
                g_last_dot_registers_per_thread = 0;
                g_last_dot_used_cpasync = 0;
                g_last_dot_kernel_kind = "reference_only";
                g_last_dot_tile_shape = "scalar";
#endif
                launch_packed_reference(0, Q, 0, R);
                return true;
            }
            if (Q_full > 0 && R_full > 0) {
                dim3 grid_tc(R_full / TN, Q_full / TM, 1);
                popcount_weighted_keys_literal_fused_bmma_tc_kernel_tile16x32<<<grid_tc, block_tc, shared_bytes, stream>>>(
                    A_packed_u32,
                    Aw,
                    A_chunk_scales,
                    A_scale_stride,
                    Sa,
                    packed_chunks,
                    B_packed_u32,
                    Bw,
                    Sb,
                    R_full,
                    Q_full,
                    indices_r,
                    indices_q,
                    scale_inv,
                    R_total,
                    out_global,
                    use_cpasync);
            }
            if (R_full < R) {
                launch_packed_reference(0, Q, R_full, R - R_full);
            }
            if (Q_full < Q) {
                launch_packed_reference(Q_full, Q - Q_full, 0, R_full);
            }
            return true;
        };
        if (try_tile16x32()) return;
    }

    if (use_sm90_packed) {
#if !defined(__CUDA_ARCH__)
        g_last_dot_kernel_kind = "reference_fallback";
        g_last_dot_tile_shape = "scalar";
        g_last_dot_used_sm90_packed = 1;
        g_last_dot_used_cpasync = 0;
        g_last_dot_used_tma = 0;
#endif
        launch_packed_reference(0, Q, 0, R);
        return;
    }

    const int tile_q = (q_tile > 0) ? q_tile : 1;
    const int tile_r = (r_tile > 0) ? r_tile : 1;

    // Simple non-TC fallback.
    dim3 block(256, 1, 1);
    dim3 grid((R + tile_r - 1) / tile_r, (Q + tile_q - 1) / tile_q);
    size_t shared_bytes =
        ((size_t)Sa * (size_t)W + (size_t)Sb * (size_t)W) * sizeof(unsigned long long) +
        (size_t)Sa * (size_t)Sb * (sizeof(float) + 2 * sizeof(int)) +
        ((size_t)Sa + (size_t)Sb) * sizeof(float);
#if !defined(__CUDA_ARCH__)
    g_last_dot_kernel_kind = "generic_scalar";
    g_last_dot_tile_shape = "legacy";
#endif
    popcount_weighted_keys_literal_fused_multiq_kernel<<<grid, block, shared_bytes, stream>>>(
        A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r, indices_r, indices_q, scale_inv, R_total, out_global);
}
