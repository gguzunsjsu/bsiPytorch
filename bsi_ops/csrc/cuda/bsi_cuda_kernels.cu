#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>

__inline__ __device__ unsigned long long warp_reduce_sum_ull(unsigned long long v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__inline__ __device__ double warp_reduce_sum_double(double v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__inline__ __device__ float warp_reduce_sum_float(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// EWAH decompress: interpret buffer of RLWs and literal words into W literal words.
// Assumes u64 words and runninglengthbits = 32, literalbits = 31.
extern "C" __global__
void ewah_decompress_kernel(
    const unsigned long long* __restrict__ in,
    int in_len,
    int W,
    unsigned long long* __restrict__ out)
{
    if (threadIdx.x != 0) return; // simple sequential decode per slice
    const unsigned long long RUNLEN_MASK = (1ULL << 32) - 1ULL; // 32 bits for runlen
    int idx = 0;
    int out_idx = 0;
    while (idx < in_len && out_idx < W) {
        unsigned long long rlw = in[idx++];
        bool running_bit = (rlw & 1ULL) != 0ULL;
        unsigned int run_len = (unsigned int)((rlw >> 1) & RUNLEN_MASK);
        unsigned int lit_words = (unsigned int)(rlw >> (1 + 32));
        // run
        unsigned long long run_val = running_bit ? ~0ULL : 0ULL;
        for (unsigned int k=0; k<run_len && out_idx < W; ++k) {
            out[out_idx++] = run_val;
        }
        // literal words
        for (unsigned int k=0; k<lit_words && out_idx < W && idx < in_len; ++k) {
            out[out_idx++] = in[idx++];
        }
    }
    // pad zeros
    while (out_idx < W) out[out_idx++] = 0ULL;
}

extern "C" __global__
void pack_bits_all_kernel(
    const int64_t* __restrict__ values,
    int64_t n,
    int slices,
    int words_per_slice,
    unsigned long long value_mask,
    unsigned long long* __restrict__ out)
{
    int total = slices * words_per_slice;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int slice = idx / words_per_slice;
    int word_idx = idx % words_per_slice;
    int64_t base_row = static_cast<int64_t>(word_idx) * 64;

    unsigned long long word = 0ULL;
    for (int bit = 0; bit < 64; ++bit) {
        int64_t row = base_row + bit;
        if (row >= n) break;
        unsigned long long v = static_cast<unsigned long long>(values[row]) & value_mask;
        unsigned long long bit_val = (v >> slice) & 1ULL;
        word |= (bit_val << bit);
    }
    out[idx] = word;
}

// Warp-ballot optimized pack: one block (64 threads) packs one 64-row word for a given slice
// Multi-warp ballot packer: each warp packs one 64-row word with two ballots
extern "C" __global__
void pack_bits_all_ballot_multi_kernel(
    const long long* __restrict__ values,
    long long n,
    int slices,
    int words_per_slice,
    unsigned long long value_mask,
    unsigned long long* __restrict__ out)
{
    int slice = blockIdx.y;
    if (slice >= slices) return;

    int warps_per_block = blockDim.x >> 5; // blockDim.x / 32
    int warp = threadIdx.x >> 5;           // 0..warps_per_block-1
    int lane = threadIdx.x & 31;           // 0..31
    int words_group = blockIdx.x;
    int word_idx = words_group * warps_per_block + warp;
    if (word_idx >= words_per_slice) return;

    long long row0 = (long long)word_idx * 64LL + (long long)lane;
    long long row1 = row0 + 32LL;

    bool b0 = false, b1 = false;
    if (row0 < n) {
        unsigned long long v0 = (static_cast<unsigned long long>(values[row0]) & value_mask);
        b0 = ((v0 >> slice) & 1ULL) != 0ULL;
    }
    if (row1 < n) {
        unsigned long long v1 = (static_cast<unsigned long long>(values[row1]) & value_mask);
        b1 = ((v1 >> slice) & 1ULL) != 0ULL;
    }
    unsigned lo = __ballot_sync(0xffffffff, b0);
    unsigned hi = __ballot_sync(0xffffffff, b1);
    if (lane == 0) {
        unsigned long long word = (unsigned long long)lo | ((unsigned long long)hi << 32);
        out[(size_t)slice * words_per_slice + word_idx] = word;
    }
}

// Multi-query fused version: process Q queries and multiple keys per block; tiles both axes to shrink grid
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

    for (int tq = 0; tq < tile_q; ++tq) {
        int q = q_start + tq;
        if (q >= Q) break;

        long long global_q = __ldg(&query_indices[q]);
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

            long long global_r = __ldg(&key_indices[r]);
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

// Warp-per-output variant (no coefficient cache, lower shared memory).
extern "C" __global__
void popcount_weighted_keys_literal_fused_multiq_kernel_warp_out_nocoeff(
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

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

    unsigned long long* A_sh = reinterpret_cast<unsigned long long*>(shmem);
    unsigned long long* B_sh = A_sh + (size_t)Sa * (size_t)W;
    float* Aw_sh = reinterpret_cast<float*>(B_sh + (size_t)tile_r * (size_t)Sb * (size_t)W);
    float* Bw_sh = Aw_sh + Sa;

    for (int tr = 0; tr < tile_r; ++tr) {
        int r = r_start + tr;
        if (r >= R) continue;
        const unsigned long long* B_base = B + ((size_t)r * Sb * W);
        const float* Bw_base = Bw + ((size_t)r * Sb);
        for (int idx = threadIdx.x; idx < Sb * W; idx += blockDim.x) {
            B_sh[(size_t)tr * (size_t)Sb * (size_t)W + (size_t)idx] = __ldg(&B_base[idx]);
        }
        for (int idx = threadIdx.x; idx < Sb; idx += blockDim.x) {
            Bw_sh[(size_t)tr * (size_t)Sb + (size_t)idx] = __ldg(&Bw_base[idx]);
        }
    }
    __syncthreads();

    for (int tq = 0; tq < tile_q; ++tq) {
        int q = q_start + tq;
        if (q >= Q) break;

        long long global_q = __ldg(&query_indices[q]);
        const unsigned long long* A_base = A + ((size_t)q * Sa * W);
        const float* Aw_base = Aw + ((size_t)q * Sa);

        for (int idx = threadIdx.x; idx < Sa * W; idx += blockDim.x) {
            A_sh[idx] = __ldg(&A_base[idx]);
        }
        for (int idx = threadIdx.x; idx < Sa; idx += blockDim.x) {
            Aw_sh[idx] = __ldg(&Aw_base[idx]);
        }
        __syncthreads();

        for (int out_idx = warp_id; out_idx < tile_r; out_idx += num_warps) {
            int r = r_start + out_idx;
            if (r >= R) continue;
            long long global_r = __ldg(&key_indices[r]);

            float local = 0.0f;
            if (W <= 32) {
                if (lane < W) {
                    const unsigned long long* b_row =
                        B_sh + (size_t)out_idx * (size_t)Sb * (size_t)W + (size_t)lane;
                    const float* bw_row = Bw_sh + (size_t)out_idx * (size_t)Sb;
                    bool use_bw_cache = Sb <= 16;
                    float bw_cache[16];
                    if (use_bw_cache) {
#pragma unroll
                        for (int j = 0; j < 16; ++j) {
                            if (j < Sb) bw_cache[j] = bw_row[j];
                        }
                    }
                    const unsigned long long* a_ptr = A_sh + (size_t)lane;
                    const float* aw_ptr = Aw_sh;
                    for (int i = 0; i < Sa; ++i) {
                        float aw = *aw_ptr++;
                        unsigned long long a_val = *a_ptr;
                        a_ptr += W;
                        const unsigned long long* b_ptr = b_row;
                        if (use_bw_cache) {
                            for (int j = 0; j < Sb; ++j) {
                                unsigned long long b_val = *b_ptr;
                                int cnt = __popcll(a_val & b_val);
                                local += (float)cnt * aw * bw_cache[j];
                                b_ptr += W;
                            }
                        } else {
                            const float* bw_ptr = bw_row;
                            for (int j = 0; j < Sb; ++j) {
                                unsigned long long b_val = *b_ptr;
                                int cnt = __popcll(a_val & b_val);
                                local += (float)cnt * aw * (*bw_ptr);
                                b_ptr += W;
                                ++bw_ptr;
                            }
                        }
                    }
                }
            } else {
                const unsigned long long* b_row = B_sh + (size_t)out_idx * (size_t)Sb * (size_t)W;
                const float* bw_row = Bw_sh + (size_t)out_idx * (size_t)Sb;
                bool use_bw_cache = Sb <= 16;
                float bw_cache[16];
                if (use_bw_cache) {
#pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        if (j < Sb) bw_cache[j] = bw_row[j];
                    }
                }
                for (int i = 0; i < Sa; ++i) {
                    float aw = Aw_sh[i];
                    const unsigned long long* a_row = A_sh + (size_t)i * (size_t)W;
                    for (int w = lane; w < W; w += 32) {
                        unsigned long long a_val = a_row[(size_t)w];
                        const unsigned long long* b_ptr = b_row + (size_t)w;
                        if (use_bw_cache) {
                            for (int j = 0; j < Sb; ++j) {
                                unsigned long long b_val = *b_ptr;
                                int cnt = __popcll(a_val & b_val);
                                local += (float)cnt * aw * bw_cache[j];
                                b_ptr += W;
                            }
                        } else {
                            const float* bw_ptr = bw_row;
                            for (int j = 0; j < Sb; ++j) {
                                unsigned long long b_val = *b_ptr;
                                int cnt = __popcll(a_val & b_val);
                                local += (float)cnt * aw * (*bw_ptr);
                                b_ptr += W;
                                ++bw_ptr;
                            }
                        }
                    }
                }
            }

            local = warp_reduce_sum_float(local);
            if (lane == 0) {
                out_global[((size_t)global_q * (size_t)R_total) + (size_t)global_r] = local * scale_inv;
            }
        }
        __syncthreads();
    }
}

// W==32 fast path for small Sb (<=16).
template <int SB>
__global__ void popcount_weighted_keys_literal_fused_multiq_kernel_warp_out_w32_sb(
    const unsigned long long* __restrict__ A,    // [Q, Sa, 32]
    const float* __restrict__ Aw,                // [Q, Sa]
    int Sa,
    int W,
    const unsigned long long* __restrict__ B,    // [R, SB, 32]
    const float* __restrict__ Bw,                // [R, SB]
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
    constexpr int Wc = 32;
    (void)W;
    (void)Sb;
    static_assert(SB >= 1 && SB <= 16, "SB out of range");

    extern __shared__ unsigned char shmem[];
    int r_block = blockIdx.x;
    int q_block = blockIdx.y;
    const int tile_q = (q_tile > 0) ? q_tile : 1;
    const int tile_r = (r_tile > 0) ? r_tile : 1;
    const int q_start = q_block * tile_q;
    const int r_start = r_block * tile_r;
    if (q_start >= Q) return;

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

    unsigned long long* A_sh = reinterpret_cast<unsigned long long*>(shmem);
    unsigned long long* B_sh = A_sh + (size_t)Sa * (size_t)Wc;
    float* Aw_sh = reinterpret_cast<float*>(B_sh + (size_t)tile_r * (size_t)SB * (size_t)Wc);
    float* Bw_sh = Aw_sh + Sa;

    for (int tr = 0; tr < tile_r; ++tr) {
        int r = r_start + tr;
        if (r >= R) continue;
        const unsigned long long* B_base = B + ((size_t)r * (size_t)SB * (size_t)Wc);
        const float* Bw_base = Bw + ((size_t)r * (size_t)SB);
        for (int idx = threadIdx.x; idx < SB * Wc; idx += blockDim.x) {
            B_sh[(size_t)tr * (size_t)SB * (size_t)Wc + (size_t)idx] = __ldg(&B_base[idx]);
        }
        for (int idx = threadIdx.x; idx < SB; idx += blockDim.x) {
            Bw_sh[(size_t)tr * (size_t)SB + (size_t)idx] = __ldg(&Bw_base[idx]);
        }
    }
    __syncthreads();

    for (int tq = 0; tq < tile_q; ++tq) {
        int q = q_start + tq;
        if (q >= Q) break;

        long long global_q = __ldg(&query_indices[q]);
        const unsigned long long* A_base = A + ((size_t)q * (size_t)Sa * (size_t)Wc);
        const float* Aw_base = Aw + ((size_t)q * (size_t)Sa);

        for (int idx = threadIdx.x; idx < Sa * Wc; idx += blockDim.x) {
            A_sh[idx] = __ldg(&A_base[idx]);
        }
        for (int idx = threadIdx.x; idx < Sa; idx += blockDim.x) {
            Aw_sh[idx] = __ldg(&Aw_base[idx]);
        }
        __syncthreads();

        for (int out_idx = warp_id; out_idx < tile_r; out_idx += num_warps) {
            int r = r_start + out_idx;
            if (r >= R) continue;
            long long global_r = __ldg(&key_indices[r]);

            float local = 0.0f;
            const unsigned long long* b_row =
                B_sh + (size_t)out_idx * (size_t)SB * (size_t)Wc + (size_t)lane;
            const float* bw_row = Bw_sh + (size_t)out_idx * (size_t)SB;
            const unsigned int full_mask = 0xffffffffu;

            float bw_cache[SB];
#pragma unroll
            for (int j = 0; j < SB; ++j) {
                bw_cache[j] = bw_row[j];
            }
            unsigned long long b_cache[SB];
            const unsigned long long* b_ptr_init = b_row;
#pragma unroll
            for (int j = 0; j < SB; ++j) {
                b_cache[j] = *b_ptr_init;
                b_ptr_init += Wc;
            }

            const unsigned long long* a_ptr = A_sh + (size_t)lane;
            if (Sa <= 16) {
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    float aw = 0.0f;
                    if (i < Sa) {
                        if (lane == 0) aw = Aw_sh[i];
                        aw = __shfl_sync(full_mask, aw, 0);
                        unsigned long long a_val = *a_ptr;
#pragma unroll
                        for (int j = 0; j < SB; ++j) {
                            unsigned long long b_val = b_cache[j];
                            int cnt = __popcll(a_val & b_val);
                            local += (float)cnt * aw * bw_cache[j];
                        }
                    }
                    a_ptr += Wc;
                }
            } else {
                for (int i = 0; i < Sa; ++i) {
                    float aw = 0.0f;
                    if (lane == 0) aw = Aw_sh[i];
                    aw = __shfl_sync(full_mask, aw, 0);
                    unsigned long long a_val = *a_ptr;
                    a_ptr += Wc;
#pragma unroll
                    for (int j = 0; j < SB; ++j) {
                        unsigned long long b_val = b_cache[j];
                        int cnt = __popcll(a_val & b_val);
                        local += (float)cnt * aw * bw_cache[j];
                    }
                }
            }

            local = warp_reduce_sum_float(local);
            if (lane == 0) {
                out_global[((size_t)global_q * (size_t)R_total) + (size_t)global_r] = local * scale_inv;
            }
        }
        __syncthreads();
    }
}

// W==32 fast path: two outputs per warp (out0/out1).
template <int SB>
__global__ void popcount_weighted_keys_literal_fused_multiq_kernel_warp_out_w32_sb2(
    const unsigned long long* __restrict__ A,    // [Q, Sa, 32]
    const float* __restrict__ Aw,                // [Q, Sa]
    int Sa,
    int W,
    const unsigned long long* __restrict__ B,    // [R, SB, 32]
    const float* __restrict__ Bw,                // [R, SB]
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
    constexpr int Wc = 32;
    (void)W;
    (void)Sb;
    static_assert(SB >= 1 && SB <= 16, "SB out of range");

    extern __shared__ unsigned char shmem[];
    int r_block = blockIdx.x;
    int q_block = blockIdx.y;
    const int tile_q = (q_tile > 0) ? q_tile : 1;
    const int tile_r = (r_tile > 0) ? r_tile : 1;
    const int q_start = q_block * tile_q;
    const int r_start = r_block * tile_r;
    if (q_start >= Q) return;

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int num_warps = (blockDim.x + 31) >> 5;

    unsigned long long* A_sh = reinterpret_cast<unsigned long long*>(shmem);
    unsigned long long* B_sh = A_sh + (size_t)Sa * (size_t)Wc;
    float* Aw_sh = reinterpret_cast<float*>(B_sh + (size_t)tile_r * (size_t)SB * (size_t)Wc);
    float* Bw_sh = Aw_sh + Sa;

    for (int tr = 0; tr < tile_r; ++tr) {
        int r = r_start + tr;
        if (r >= R) continue;
        const unsigned long long* B_base = B + ((size_t)r * (size_t)SB * (size_t)Wc);
        const float* Bw_base = Bw + ((size_t)r * (size_t)SB);
        for (int idx = threadIdx.x; idx < SB * Wc; idx += blockDim.x) {
            B_sh[(size_t)tr * (size_t)SB * (size_t)Wc + (size_t)idx] = __ldg(&B_base[idx]);
        }
        for (int idx = threadIdx.x; idx < SB; idx += blockDim.x) {
            Bw_sh[(size_t)tr * (size_t)SB + (size_t)idx] = __ldg(&Bw_base[idx]);
        }
    }
    __syncthreads();

    for (int tq = 0; tq < tile_q; ++tq) {
        int q = q_start + tq;
        if (q >= Q) break;

        long long global_q = __ldg(&query_indices[q]);
        const unsigned long long* A_base = A + ((size_t)q * (size_t)Sa * (size_t)Wc);
        const float* Aw_base = Aw + ((size_t)q * (size_t)Sa);

        for (int idx = threadIdx.x; idx < Sa * Wc; idx += blockDim.x) {
            A_sh[idx] = __ldg(&A_base[idx]);
        }
        for (int idx = threadIdx.x; idx < Sa; idx += blockDim.x) {
            Aw_sh[idx] = __ldg(&Aw_base[idx]);
        }
        __syncthreads();

        int out0 = warp_id * 2;
        int out1 = out0 + 1;
        bool valid0 = out0 < tile_r;
        bool valid1 = out1 < tile_r;
        int r0 = r_start + out0;
        int r1 = r_start + out1;
        if (!valid0 && !valid1) {
            __syncthreads();
            continue;
        }

        long long global_r0 = valid0 ? __ldg(&key_indices[r0]) : 0;
        long long global_r1 = valid1 ? __ldg(&key_indices[r1]) : 0;
        const unsigned int full_mask = 0xffffffffu;

        float bw0[SB];
        float bw1[SB];
        unsigned long long b0[SB];
        unsigned long long b1[SB];

        if (valid0) {
            const float* bw_row0 = Bw_sh + (size_t)out0 * (size_t)SB;
            const unsigned long long* b_row0 =
                B_sh + (size_t)out0 * (size_t)SB * (size_t)Wc + (size_t)lane;
#pragma unroll
            for (int j = 0; j < SB; ++j) {
                bw0[j] = bw_row0[j];
                b0[j] = *b_row0;
                b_row0 += Wc;
            }
        }
        if (valid1) {
            const float* bw_row1 = Bw_sh + (size_t)out1 * (size_t)SB;
            const unsigned long long* b_row1 =
                B_sh + (size_t)out1 * (size_t)SB * (size_t)Wc + (size_t)lane;
#pragma unroll
            for (int j = 0; j < SB; ++j) {
                bw1[j] = bw_row1[j];
                b1[j] = *b_row1;
                b_row1 += Wc;
            }
        }

        float local0 = 0.0f;
        float local1 = 0.0f;
        const unsigned long long* a_ptr = A_sh + (size_t)lane;
        if (Sa <= 16) {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                float aw = 0.0f;
                if (i < Sa) {
                    if (lane == 0) aw = Aw_sh[i];
                    aw = __shfl_sync(full_mask, aw, 0);
                    unsigned long long a_val = *a_ptr;
#pragma unroll
                    for (int j = 0; j < SB; ++j) {
                        if (valid0) {
                            int cnt0 = __popcll(a_val & b0[j]);
                            local0 += (float)cnt0 * aw * bw0[j];
                        }
                        if (valid1) {
                            int cnt1 = __popcll(a_val & b1[j]);
                            local1 += (float)cnt1 * aw * bw1[j];
                        }
                    }
                }
                a_ptr += Wc;
            }
        } else {
            for (int i = 0; i < Sa; ++i) {
                float aw = 0.0f;
                if (lane == 0) aw = Aw_sh[i];
                aw = __shfl_sync(full_mask, aw, 0);
                unsigned long long a_val = *a_ptr;
                a_ptr += Wc;
#pragma unroll
                for (int j = 0; j < SB; ++j) {
                    if (valid0) {
                        int cnt0 = __popcll(a_val & b0[j]);
                        local0 += (float)cnt0 * aw * bw0[j];
                    }
                    if (valid1) {
                        int cnt1 = __popcll(a_val & b1[j]);
                        local1 += (float)cnt1 * aw * bw1[j];
                    }
                }
            }
        }

        local0 = warp_reduce_sum_float(local0);
        local1 = warp_reduce_sum_float(local1);
        if (lane == 0) {
            if (valid0) out_global[((size_t)global_q * (size_t)R_total) + (size_t)global_r0] = local0 * scale_inv;
            if (valid1) out_global[((size_t)global_q * (size_t)R_total) + (size_t)global_r1] = local1 * scale_inv;
        }
        __syncthreads();
    }
}

template <int SB>
static inline void launch_w32_sb_kernel(
    const unsigned long long* A,
    const float* Aw,
    int Sa,
    int W,
    const unsigned long long* B,
    const float* Bw,
    int Sb,
    int R,
    int Q,
    int tile_q,
    int tile_r,
    const long long* indices_r,
    const long long* indices_q,
    float scale_inv,
    int R_total,
    float* out_global,
    size_t shared_bytes,
    dim3 grid_warp,
    dim3 block,
    cudaStream_t stream,
    int max_shared_default)
{
    if (shared_bytes > (size_t)max_shared_default) {
        cudaFuncSetAttribute(
            popcount_weighted_keys_literal_fused_multiq_kernel_warp_out_w32_sb<SB>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)shared_bytes);
    }
    popcount_weighted_keys_literal_fused_multiq_kernel_warp_out_w32_sb<SB><<<grid_warp, block, shared_bytes, stream>>>(
        A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r, indices_r, indices_q, scale_inv, R_total, out_global);
}

template <int SB>
static inline void launch_w32_sb2_kernel(
    const unsigned long long* A,
    const float* Aw,
    int Sa,
    int W,
    const unsigned long long* B,
    const float* Bw,
    int Sb,
    int R,
    int Q,
    int tile_q,
    int tile_r,
    const long long* indices_r,
    const long long* indices_q,
    float scale_inv,
    int R_total,
    float* out_global,
    size_t shared_bytes,
    dim3 grid_warp,
    dim3 block,
    cudaStream_t stream,
    int max_shared_default)
{
    if (shared_bytes > (size_t)max_shared_default) {
        cudaFuncSetAttribute(
            popcount_weighted_keys_literal_fused_multiq_kernel_warp_out_w32_sb2<SB>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)shared_bytes);
    }
    popcount_weighted_keys_literal_fused_multiq_kernel_warp_out_w32_sb2<SB><<<grid_warp, block, shared_bytes, stream>>>(
        A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r, indices_r, indices_q, scale_inv, R_total, out_global);
}

extern "C" void launch_popcount_weighted_keys_literal_fused_multiq(
    const unsigned long long* A,
    const float* Aw,
    int Sa,
    int W,
    const unsigned long long* B,
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
    cudaStream_t stream)
{
    static int cached_block = 0;
    if (cached_block == 0) {
        int v = 256;
        if (const char* s = getenv("BSI_CK_BLOCK")) {
            int t = atoi(s);
            if (t > 0) v = t;
        }
        if (v < 64) v = 64;
        if (v > 1024) v = 1024;
        cached_block = (v / 32) * 32;
        if (cached_block == 0) cached_block = 32;
    }
    int tile_q = (q_tile > 0) ? q_tile : 1;
    int tile_r = (r_tile > 0) ? r_tile : 1;
    dim3 block(cached_block);
    bool use_warp_out = false;
    if (const char* s = getenv("BSI_WARP_OUT")) {
        int v = atoi(s);
        use_warp_out = (v != 0);
    }
    bool use_warp_out2 = false;
    if (const char* s = getenv("BSI_WARP_OUT2")) {
        int v = atoi(s);
        use_warp_out2 = (v != 0);
    }
    bool launch_base = !use_warp_out;
    if (use_warp_out) {
        int dev = 0;
        cudaGetDevice(&dev);
        int max_shared_default = 0;
        int max_shared_optin = 0;
        cudaDeviceGetAttribute(&max_shared_default, cudaDevAttrMaxSharedMemoryPerBlock, dev);
        cudaDeviceGetAttribute(&max_shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        int max_shared = (max_shared_optin > max_shared_default) ? max_shared_optin : max_shared_default;
        int tile_r_eff = tile_r;
        size_t shared_bytes = 0;
        while (tile_r_eff > 0) {
            shared_bytes =
                ((size_t)Sa * (size_t)W + (size_t)tile_r_eff * (size_t)Sb * (size_t)W) * sizeof(unsigned long long) +
                ((size_t)Sa + (size_t)tile_r_eff * (size_t)Sb) * sizeof(float);
            if (shared_bytes <= (size_t)max_shared) break;
            tile_r_eff = (tile_r_eff + 1) / 2;
            if (tile_r_eff == tile_r) {
                tile_r_eff = tile_r - 1;
            }
        }
        if (tile_r_eff <= 0 || shared_bytes > (size_t)max_shared) {
            launch_base = true;
        } else {
            dim3 grid_warp((R + tile_r_eff - 1) / tile_r_eff, (Q + tile_q - 1) / tile_q);
            if (W == 32 && Sb >= 1 && Sb <= 16) {
                switch (Sb) {
                    case 1:
                        if (use_warp_out2) {
                            launch_w32_sb2_kernel<1>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        } else {
                            launch_w32_sb_kernel<1>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                    scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        }
                        break;
                    case 2:
                        if (use_warp_out2) {
                            launch_w32_sb2_kernel<2>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        } else {
                            launch_w32_sb_kernel<2>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                    scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        }
                        break;
                    case 3:
                        if (use_warp_out2) {
                            launch_w32_sb2_kernel<3>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        } else {
                            launch_w32_sb_kernel<3>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                    scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        }
                        break;
                    case 4:
                        if (use_warp_out2) {
                            launch_w32_sb2_kernel<4>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        } else {
                            launch_w32_sb_kernel<4>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                    scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        }
                        break;
                    case 5:
                        if (use_warp_out2) {
                            launch_w32_sb2_kernel<5>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        } else {
                            launch_w32_sb_kernel<5>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                    scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        }
                        break;
                    case 6:
                        if (use_warp_out2) {
                            launch_w32_sb2_kernel<6>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        } else {
                            launch_w32_sb_kernel<6>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                    scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        }
                        break;
                    case 7:
                        if (use_warp_out2) {
                            launch_w32_sb2_kernel<7>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        } else {
                            launch_w32_sb_kernel<7>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                    scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        }
                        break;
                    case 8:
                        if (use_warp_out2) {
                            launch_w32_sb2_kernel<8>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        } else {
                            launch_w32_sb_kernel<8>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                    scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        }
                        break;
                    case 9:
                        if (use_warp_out2) {
                            launch_w32_sb2_kernel<9>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        } else {
                            launch_w32_sb_kernel<9>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                    scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        }
                        break;
                    case 10:
                        if (use_warp_out2) {
                            launch_w32_sb2_kernel<10>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                      scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        } else {
                            launch_w32_sb_kernel<10>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        }
                        break;
                    case 11:
                        if (use_warp_out2) {
                            launch_w32_sb2_kernel<11>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                      scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        } else {
                            launch_w32_sb_kernel<11>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        }
                        break;
                    case 12:
                        if (use_warp_out2) {
                            launch_w32_sb2_kernel<12>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                      scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        } else {
                            launch_w32_sb_kernel<12>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        }
                        break;
                    case 13:
                        if (use_warp_out2) {
                            launch_w32_sb2_kernel<13>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                      scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        } else {
                            launch_w32_sb_kernel<13>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        }
                        break;
                    case 14:
                        if (use_warp_out2) {
                            launch_w32_sb2_kernel<14>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                      scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        } else {
                            launch_w32_sb_kernel<14>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        }
                        break;
                    case 15:
                        if (use_warp_out2) {
                            launch_w32_sb2_kernel<15>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                      scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        } else {
                            launch_w32_sb_kernel<15>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        }
                        break;
                    case 16:
                        if (use_warp_out2) {
                            launch_w32_sb2_kernel<16>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                      scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        } else {
                            launch_w32_sb_kernel<16>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        }
                        break;
                    default:
                        if (use_warp_out2) {
                            launch_w32_sb2_kernel<16>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                      scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        } else {
                            launch_w32_sb_kernel<16>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        }
                        break;
                }
            } else {
                if (shared_bytes > (size_t)max_shared_default) {
                    cudaFuncSetAttribute(
                        popcount_weighted_keys_literal_fused_multiq_kernel_warp_out_nocoeff,
                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                        (int)shared_bytes);
                }
                popcount_weighted_keys_literal_fused_multiq_kernel_warp_out_nocoeff<<<grid_warp, block, shared_bytes, stream>>>(
                    A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q, scale_inv, R_total, out_global);
            }
        }
    }
    if (launch_base) {
        dim3 grid((R + tile_r - 1) / tile_r, (Q + tile_q - 1) / tile_q);
        size_t shared_bytes =
            ((size_t)Sa * (size_t)W + (size_t)Sb * (size_t)W) * sizeof(unsigned long long) +
            (size_t)Sa * (size_t)Sb * (sizeof(float) + 2 * sizeof(int)) +
            ((size_t)Sa + (size_t)Sb) * sizeof(float);
        popcount_weighted_keys_literal_fused_multiq_kernel<<<grid, block, shared_bytes, stream>>>(
            A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r, indices_r, indices_q, scale_inv, R_total, out_global);
    }
}

extern "C" void launch_ewah_decompress(
    const unsigned long long* in,
    int in_len,
    int W,
    unsigned long long* out,
    cudaStream_t stream) {
    ewah_decompress_kernel<<<1,1,0,stream>>>(in, in_len, W, out);
}

extern "C" void launch_pack_bits_all(
    const int64_t* values,
    int64_t n,
    int slices,
    int words_per_slice,
    unsigned long long value_mask,
    unsigned long long* out,
    cudaStream_t stream)
{
    int total = slices * words_per_slice;
    if (total <= 0) {
        return;
    }
    int threads = 128;
    int blocks = (total + threads - 1) / threads;
    pack_bits_all_kernel<<<blocks, threads, 0, stream>>>(
        values,
        n,
        slices,
        words_per_slice,
        value_mask,
        out);
}

extern "C" void launch_pack_bits_all_ballot(
    const long long* values,
    long long n,
    int slices,
    int words_per_slice,
    unsigned long long value_mask,
    unsigned long long* out,
    cudaStream_t stream)
{
    if (slices <= 0 || words_per_slice <= 0) return;
    // Use 8 warps (256 threads) per block by default; grid.x covers groups of 8 words per slice
    const int warps_per_block = 8;
    dim3 block(warps_per_block * 32);
    dim3 grid((words_per_slice + warps_per_block - 1) / warps_per_block, slices);
    pack_bits_all_ballot_multi_kernel<<<grid, block, 0, stream>>>(
        values, n, slices, words_per_slice, value_mask, out);
}

// ---------------- Hybrid (EWAH) helper kernels -----------------

// Compute popcount sum per slice across W words
extern "C" __global__
void slice_popcount_sum_kernel(
    const unsigned long long* __restrict__ words, // [S*W]
    int S,
    int W,
    unsigned long long* __restrict__ out_counts)  // [S]
{
    int s = blockIdx.x;
    if (s >= S) return;
    unsigned long long local = 0ULL;
    for (int w = threadIdx.x; w < W; w += blockDim.x) {
        local += __popcll(words[(size_t)s * W + w]);
    }
    __shared__ unsigned long long smem[32];
    unsigned long long warp = warp_reduce_sum_ull(local);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = warp;
    __syncthreads();
    unsigned long long block_sum = 0ULL;
    int num_warps = blockDim.x >> 5;
    if (threadIdx.x < num_warps) block_sum = smem[threadIdx.x];
    block_sum = warp_reduce_sum_ull(block_sum);
    if (threadIdx.x == 0) out_counts[s] = block_sum;
}

extern "C" void launch_slice_popcount_sum(
    const unsigned long long* words,
    int S,
    int W,
    unsigned long long* out_counts,
    cudaStream_t stream)
{
    dim3 grid(S);
    dim3 block(256);
    slice_popcount_sum_kernel<<<grid, block, 0, stream>>>(words, S, W, out_counts);
}

// Compute compress flags (1=compress, 0=literal) from popcount density
extern "C" __global__
void compress_flags_from_density_kernel(
    const unsigned long long* __restrict__ counts, // [S]
    int S,
    int W,
    double threshold,
    int* __restrict__ out_flags) // [S]
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= S) return;
    double total_bits = (double)W * 64.0;
    double p = (total_bits > 0.0) ? ((double)counts[s] / total_bits) : 0.0;
    if (p < 0.0) p = 0.0; if (p > 1.0) p = 1.0;
    // M = 1 - (1-p)^(128) - p^(128)
    double a = pow(1.0 - p, 128.0);
    double b = pow(p, 128.0);
    double M = 1.0 - a - b;
    out_flags[s] = (M <= threshold) ? 1 : 0;
}

extern "C" void launch_compress_flags_from_density(
    const unsigned long long* counts,
    int S,
    int W,
    double threshold,
    int* out_flags,
    cudaStream_t stream)
{
    int threads = 128;
    int blocks = (S + threads - 1) / threads;
    compress_flags_from_density_kernel<<<blocks, threads, 0, stream>>>(counts, S, W, threshold, out_flags);
}

// Size pass: compute number of u64 words to emit per slice for EWAH encoding
extern "C" __global__
void ewah_size_kernel(
    const unsigned long long* __restrict__ words, // [S*W]
    int S,
    int W,
    const int* __restrict__ flags,                // [S] 1=compress, 0=literal-only
    unsigned long long* __restrict__ sizes)       // [S]
{
    int s = blockIdx.x;
    if (s >= S) return;
    if (threadIdx.x != 0) return; // single thread per slice does sequential scan
    const unsigned long long* base = words + (size_t)s * W;
    unsigned long long out_size = 0ULL;
    int idx = 0;
    int compress = flags[s];
    while (idx < W) {
        int run_len = 0;
        int run_bit = 0;
        if (compress && idx < W) {
            unsigned long long v = base[idx];
            bool cz = (v == 0ULL);
            bool co = (v == ~0ULL);
            if (cz || co) {
                run_bit = co ? 1 : 0;
                unsigned long long fill = co ? ~0ULL : 0ULL;
                int k = idx;
                while (k < W && base[k] == fill) { ++k; }
                run_len = k - idx;
                idx = k;
            }
        }
        int lit_count = 0;
        int lit_start = idx;
        while (idx < W) {
            unsigned long long v = base[idx];
            bool cz = (v == 0ULL), co = (v == ~0ULL);
            if (compress && (cz || co)) break;
            ++lit_count;
            ++idx;
        }
        // one RLW + lit_count literal words
        out_size += 1ULL + (unsigned long long)lit_count;
    }
    sizes[s] = out_size;
}

extern "C" void launch_ewah_size(
    const unsigned long long* words,
    int S,
    int W,
    const int* flags,
    unsigned long long* sizes,
    cudaStream_t stream)
{
    dim3 grid(S);
    dim3 block(1);
    ewah_size_kernel<<<grid, block, 0, stream>>>(words, S, W, flags, sizes);
}

// Emit pass: write RLWs and literal payload into output with given per-slice offsets
extern "C" __global__
void ewah_emit_kernel(
    const unsigned long long* __restrict__ words, // [S*W]
    int S,
    int W,
    const int* __restrict__ flags,                // [S]
    const unsigned long long* __restrict__ off,   // [S]
    unsigned long long* __restrict__ out,         // [total]
    int* __restrict__ out_len)                    // [S]
{
    int s = blockIdx.x;
    if (s >= S) return;
    if (threadIdx.x != 0) return;
    const unsigned long long* base = words + (size_t)s * W;
    unsigned long long o = off[s];
    int idx = 0;
    int compress = flags[s];
    const unsigned long long RUNLEN_MASK = (1ULL << 32) - 1ULL;
    while (idx < W) {
        int run_len = 0;
        int run_bit = 0;
        if (compress && idx < W) {
            unsigned long long v = base[idx];
            bool cz = (v == 0ULL);
            bool co = (v == ~0ULL);
            if (cz || co) {
                run_bit = co ? 1 : 0;
                unsigned long long fill = co ? ~0ULL : 0ULL;
                int k = idx;
                while (k < W && base[k] == fill) { ++k; }
                run_len = k - idx;
                idx = k;
            }
        }
        int lit_count = 0;
        int lit_start = idx;
        while (idx < W) {
            unsigned long long v = base[idx];
            bool cz = (v == 0ULL), co = (v == ~0ULL);
            if (compress && (cz || co)) break;
            ++lit_count;
            ++idx;
        }
        // Encode RLW: bit0=run_bit; bits[1..32]=run_len; bits[33..63]=lit_count
        unsigned long long rlw = (unsigned long long)run_bit
                               | ((unsigned long long)(run_len & RUNLEN_MASK) << 1)
                               | ((unsigned long long)lit_count << (1 + 32));
        out[o++] = rlw;
        // Literals
        for (int k = 0; k < lit_count; ++k) {
            out[o++] = base[lit_start + k];
        }
    }
    out_len[s] = (int)(o - off[s]);
}

extern "C" void launch_ewah_emit(
    const unsigned long long* words,
    int S,
    int W,
    const int* flags,
    const unsigned long long* off,
    unsigned long long* out,
    int* out_len,
    cudaStream_t stream)
{
    dim3 grid(S);
    dim3 block(1);
    ewah_emit_kernel<<<grid, block, 0, stream>>>(words, S, W, flags, off, out, out_len);
}
