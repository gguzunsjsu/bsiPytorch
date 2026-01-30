#pragma once

// Multi-query fused version: process Q queries and multiple keys per block; tiles both axes to shrink grid
#include <stdint.h>
#ifdef __CUDACC__
#include <mma.h>
#endif

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

    int q_end = q_start + tile_q;
    if (q_end > Q) q_end = Q;
    for (int q = q_start; q < q_end; ++q) {

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
                    unsigned long long b_cache[16];
                    if (use_bw_cache) {
#pragma unroll
                        for (int j = 0; j < 16; ++j) {
                            if (j < Sb) {
                                bw_cache[j] = bw_row[j];
                                b_cache[j] = b_row[(size_t)j * (size_t)W];
                            }
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
#pragma unroll
                            for (int j = 0; j < 16; ++j) {
                                if (j < Sb) {
                                    int cnt = __popcll(a_val & b_cache[j]);
                                    local += (float)cnt * aw * bw_cache[j];
                                }
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
                    float coeff_cache[16];
                    if (use_bw_cache) {
#pragma unroll
                        for (int j = 0; j < 16; ++j) {
                            if (j < Sb) coeff_cache[j] = aw * bw_cache[j];
                        }
                    }
                    for (int w = lane; w < W; w += 32) {
                        unsigned long long a_val = a_row[(size_t)w];
                        const unsigned long long* b_ptr = b_row + (size_t)w;
                        if (use_bw_cache) {
#pragma unroll
                            for (int j = 0; j < 16; ++j) {
                                if (j < Sb) {
                                    unsigned long long b_val = *b_ptr;
                                    int cnt = __popcll(a_val & b_val);
                                    local += (float)cnt * coeff_cache[j];
                                    b_ptr += W;
                                }
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
        if (q + 1 < q_end) {
            __syncthreads();
        }
    }
}

// W==32 fast path (template-specialized for Sb <= 32).
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
    static_assert(SB >= 1 && SB <= 32, "SB out of range");

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

    unsigned long long* A_sh0 = reinterpret_cast<unsigned long long*>(shmem);
    unsigned long long* A_sh1 = A_sh0 + (size_t)Sa * (size_t)Wc;
    unsigned long long* B_sh = A_sh1 + (size_t)Sa * (size_t)Wc;
    float* Aw_sh0 = reinterpret_cast<float*>(B_sh + (size_t)tile_r * (size_t)SB * (size_t)Wc);
    float* Aw_sh1 = Aw_sh0 + Sa;
    float* Bw_sh = Aw_sh1 + Sa;

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

    int q_end = q_start + tile_q;
    if (q_end > Q) q_end = Q;
    if (q_start >= q_end) return;

    int buf0 = q_start & 1;
    const unsigned long long* A_base0 = A + ((size_t)q_start * (size_t)Sa * (size_t)Wc);
    if (buf0 == 0) {
        cp_async_copy_ull(A_sh0, A_base0, Sa * Wc);
    } else {
        cp_async_copy_ull(A_sh1, A_base0, Sa * Wc);
    }
    cp_async_commit();
    cp_async_wait();
    if (buf0 == 0) {
        cp_async_tail_ull(A_sh0, A_base0, Sa * Wc);
    } else {
        cp_async_tail_ull(A_sh1, A_base0, Sa * Wc);
    }
    __syncthreads();

    for (int q = q_start; q < q_end; ++q) {
        int buf = q & 1;
        unsigned long long* A_sh = buf ? A_sh1 : A_sh0;
        float* Aw_sh = buf ? Aw_sh1 : Aw_sh0;

        long long global_q = __ldg(&query_indices[q]);
        const float* Aw_base = Aw + ((size_t)q * (size_t)Sa);
        int q_next = q + 1;
        unsigned long long* A_sh_next = nullptr;
        const unsigned long long* A_base_next = nullptr;
        if (q_next < q_end) {
            A_base_next = A + ((size_t)q_next * (size_t)Sa * (size_t)Wc);
            A_sh_next = buf ? A_sh0 : A_sh1;
            cp_async_copy_ull(A_sh_next, A_base_next, Sa * Wc);
            cp_async_commit();
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

            // Cache Bw + B words in registers (B does not depend on i). This avoids
            // rereading shared memory inside the Sa loop, which is a big win on H100.
            float bw_cache[SB];
#pragma unroll
            for (int j = 0; j < SB; ++j) bw_cache[j] = bw_row[j];
            unsigned long long b_cache[SB];
            const unsigned long long* b_ptr_init = b_row;
#pragma unroll
            for (int j = 0; j < SB; ++j) {
                b_cache[j] = *b_ptr_init;
                b_ptr_init += Wc;
            }

            if (Sa <= 16) {
                const unsigned long long* a_ptr = A_sh + (size_t)lane;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    if (i < Sa) {
                        const float aw = Aw_sh[i];
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
                const unsigned long long* a_ptr = A_sh + (size_t)lane;
                for (int i = 0; i < Sa; ++i) {
                    const float aw = Aw_sh[i];
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
        if (q_next < q_end) {
            cp_async_wait();
            cp_async_tail_ull(A_sh_next, A_base_next, Sa * Wc);
            __syncthreads();
        }
    }
}

template <int SB>
__global__ void popcount_weighted_keys_literal_fused_multiq_kernel_warp_out_w128_sb(
    const unsigned long long* __restrict__ A,    // [Q, Sa, 128]
    const float* __restrict__ Aw,                // [Q, Sa]
    int Sa,
    int W,
    const unsigned long long* __restrict__ B,    // [R, SB, 128]
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
    constexpr int Wc = 128;
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

    // NOTE: Prefer smaller shared memory footprint here. W=128 kernels are often
    // occupancy-limited due to large B tiles; dropping A double-buffering
    // increases resident blocks/SM on H100 without changing math.
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

    int q_end = q_start + tile_q;
    if (q_end > Q) q_end = Q;
    if (q_start >= q_end) return;

    for (int q = q_start; q < q_end; ++q) {
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

            if constexpr (SB <= 8) {
                float bw_cache[SB];
#pragma unroll
                for (int j = 0; j < SB; ++j) {
                    bw_cache[j] = bw_row[j];
                }
                unsigned long long b_cache0[SB];
                unsigned long long b_cache1[SB];
                unsigned long long b_cache2[SB];
                unsigned long long b_cache3[SB];
                const unsigned long long* b_ptr_init = b_row;
#pragma unroll
                for (int j = 0; j < SB; ++j) {
                    b_cache0[j] = b_ptr_init[0];
                    b_cache1[j] = b_ptr_init[32];
                    b_cache2[j] = b_ptr_init[64];
                    b_cache3[j] = b_ptr_init[96];
                    b_ptr_init += Wc;
                }

                if (Sa <= 16) {
                    const unsigned long long* a_ptr = A_sh + (size_t)lane;
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        if (i < Sa) {
                            const float aw = Aw_sh[i];
                            unsigned long long a0 = a_ptr[0];
                            unsigned long long a1 = a_ptr[32];
                            unsigned long long a2 = a_ptr[64];
                            unsigned long long a3 = a_ptr[96];
#pragma unroll
                            for (int j = 0; j < SB; ++j) {
                                int cnt = __popcll(a0 & b_cache0[j]);
                                cnt += __popcll(a1 & b_cache1[j]);
                                cnt += __popcll(a2 & b_cache2[j]);
                                cnt += __popcll(a3 & b_cache3[j]);
                                local += (float)cnt * aw * bw_cache[j];
                            }
                        }
                        a_ptr += Wc;
                    }
                } else {
                    const unsigned long long* a_ptr = A_sh + (size_t)lane;
                    for (int i = 0; i < Sa; ++i) {
                        const float aw = Aw_sh[i];
                        unsigned long long a0 = a_ptr[0];
                        unsigned long long a1 = a_ptr[32];
                        unsigned long long a2 = a_ptr[64];
                        unsigned long long a3 = a_ptr[96];
                        a_ptr += Wc;
#pragma unroll
                        for (int j = 0; j < SB; ++j) {
                            int cnt = __popcll(a0 & b_cache0[j]);
                            cnt += __popcll(a1 & b_cache1[j]);
                            cnt += __popcll(a2 & b_cache2[j]);
                            cnt += __popcll(a3 & b_cache3[j]);
                            local += (float)cnt * aw * bw_cache[j];
                        }
                    }
                }
            } else {
                // SB > 8: Avoid large per-thread B caches (register pressure/spills).
                // Load B directly from shared memory inside the inner loop.
                if (Sa <= 16) {
                    const unsigned long long* a_ptr = A_sh + (size_t)lane;
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        if (i < Sa) {
                            const float aw = Aw_sh[i];
                            const unsigned long long a0 = a_ptr[0];
                            const unsigned long long a1 = a_ptr[32];
                            const unsigned long long a2 = a_ptr[64];
                            const unsigned long long a3 = a_ptr[96];
#pragma unroll
                            for (int j = 0; j < SB; ++j) {
                                const unsigned long long* b_ptr = b_row + (size_t)j * (size_t)Wc;
                                int cnt = __popcll(a0 & b_ptr[0]);
                                cnt += __popcll(a1 & b_ptr[32]);
                                cnt += __popcll(a2 & b_ptr[64]);
                                cnt += __popcll(a3 & b_ptr[96]);
                                local += (float)cnt * aw * bw_row[j];
                            }
                        }
                        a_ptr += Wc;
                    }
                } else {
                    const unsigned long long* a_ptr = A_sh + (size_t)lane;
                    for (int i = 0; i < Sa; ++i) {
                        const float aw = Aw_sh[i];
                        const unsigned long long a0 = a_ptr[0];
                        const unsigned long long a1 = a_ptr[32];
                        const unsigned long long a2 = a_ptr[64];
                        const unsigned long long a3 = a_ptr[96];
                        a_ptr += Wc;
#pragma unroll
                        for (int j = 0; j < SB; ++j) {
                            const unsigned long long* b_ptr = b_row + (size_t)j * (size_t)Wc;
                            int cnt = __popcll(a0 & b_ptr[0]);
                            cnt += __popcll(a1 & b_ptr[32]);
                            cnt += __popcll(a2 & b_ptr[64]);
                            cnt += __popcll(a3 & b_ptr[96]);
                            local += (float)cnt * aw * bw_row[j];
                        }
                    }
                }
            }

            local = warp_reduce_sum_float(local);
            if (lane == 0) {
                out_global[((size_t)global_q * (size_t)R_total) + (size_t)global_r] = local * scale_inv;
            }
        }
        // Make sure all warps are done with shared memory before the next query loads.
        if (q + 1 < q_end) __syncthreads();
    }
}

// Tensor-core BMMA path for H100/A100+: compute a 16x8 output tile per block using 1-bit MMA.
//
// This is the only plausible route to a large dot_ms drop: the existing popcount kernels are
// already ALU-bound (SM busy ~90-95% in NCU) so cache/tiling tweaks don't move the needle.
//
// The kernel computes:
//   out[q, r] += sum_{i,j} Aw[q,i] * Bw[r,j] * popcount( A[q,i] & B[r,j] )
// by processing the bit dimension in 256-bit chunks using BMMA m16n8k256.
extern "C" __global__
void popcount_weighted_keys_literal_fused_bmma_tc_kernel(
    const unsigned long long* __restrict__ A,    // [Q, Sa, W64]
    const float* __restrict__ Aw,                // [Q, Sa]
    int Sa,
    int W64,
    const unsigned long long* __restrict__ B,    // [R, Sb, W64]
    const float* __restrict__ Bw,                // [R, Sb]
    int Sb,
    int R,
    int Q,
    const long long* __restrict__ key_indices,   // [R]
    const long long* __restrict__ query_indices, // [Q]
    float scale_inv,
    int R_total,
    float* __restrict__ out_global)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    using namespace nvcuda;
    using namespace nvcuda::wmma;

    constexpr int TM = 16;
    constexpr int TN = 8;
    constexpr int K_BITS = 256;
    constexpr int K_WORDS64 = K_BITS / 64;   // 4
    constexpr int K_WORDS32 = K_BITS / 32;   // 8

    // This kernel is defined as 1 warp per block.
    if (blockDim.x != 32) return;
    const int lane = threadIdx.x & 31;

    const int q0 = blockIdx.y * TM;
    const int r0 = blockIdx.x * TN;

    // --- Shared memory layout (dynamic) ---
    extern __shared__ unsigned char smem_raw[];
    uintptr_t p = reinterpret_cast<uintptr_t>(smem_raw);
    // Keep WMMA loads happy; 16B alignment is generally sufficient for bit-packed tiles.
    p = (p + 15u) & ~uintptr_t(15u);

    auto* A_bits = reinterpret_cast<uint32_t*>(p); // [Sa, TM, 8]
    p += (size_t)Sa * (size_t)TM * (size_t)K_WORDS32 * sizeof(uint32_t);
    auto* B_bits = reinterpret_cast<uint32_t*>(p); // [Sb, TN, 8] (columns stored contiguously)
    p += (size_t)Sb * (size_t)TN * (size_t)K_WORDS32 * sizeof(uint32_t);
    auto* Aw_tile = reinterpret_cast<float*>(p);   // [TM, Sa]
    p += (size_t)TM * (size_t)Sa * sizeof(float);
    auto* Bw_tile = reinterpret_cast<float*>(p);   // [TN, Sb]
    p += (size_t)TN * (size_t)Sb * sizeof(float);
    auto* C_tile = reinterpret_cast<int*>(p);      // [TM, TN]
    p += (size_t)TM * (size_t)TN * sizeof(int);
    auto* q_ids = reinterpret_cast<long long*>(p); // [TM]
    p += (size_t)TM * sizeof(long long);
    auto* r_ids = reinterpret_cast<long long*>(p); // [TN]
    (void)p;

    // Load q/r indices once for the tile (used for output scatter).
    if (lane < TM) {
        const int q = q0 + lane;
        q_ids[lane] = (q < Q) ? __ldg(&query_indices[q]) : -1;
    }
    if (lane < TN) {
        const int r = r0 + lane;
        r_ids[lane] = (r < R) ? __ldg(&key_indices[r]) : -1;
    }

    // Load all slice weights for the tile.
    for (int idx = lane; idx < TM * Sa; idx += 32) {
        const int m = idx / Sa;
        const int i = idx - m * Sa;
        const int q = q0 + m;
        Aw_tile[(size_t)m * (size_t)Sa + (size_t)i] =
            (q < Q) ? __ldg(&Aw[(size_t)q * (size_t)Sa + (size_t)i]) : 0.0f;
    }
    for (int idx = lane; idx < TN * Sb; idx += 32) {
        const int n = idx / Sb;
        const int j = idx - n * Sb;
        const int r = r0 + n;
        Bw_tile[(size_t)n * (size_t)Sb + (size_t)j] =
            (r < R) ? __ldg(&Bw[(size_t)r * (size_t)Sb + (size_t)j]) : 0.0f;
    }
    __syncwarp();

    // Each thread accumulates 4 outputs in row-major order.
    const int out0 = lane * 4 + 0;
    const int out1 = lane * 4 + 1;
    const int out2 = lane * 4 + 2;
    const int out3 = lane * 4 + 3;
    const int r0_0 = out0 >> 3, c0_0 = out0 & 7;
    const int r0_1 = out1 >> 3, c0_1 = out1 & 7;
    const int r0_2 = out2 >> 3, c0_2 = out2 & 7;
    const int r0_3 = out3 >> 3, c0_3 = out3 & 7;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

    // Number of 256-bit chunks in the bit dimension.
    const int chunks = W64 / K_WORDS64;

    // Main loop: iterate over chunks, load all slice bits for the tile into shared, then BMMA.
    for (int chunk = 0; chunk < chunks; ++chunk) {
        // Load A bits for all slices and all 16 queries for this chunk into shared.
        // A_bits layout: ((i*TM + m)*8 + w32)
        for (int idx = lane; idx < TM * Sa * K_WORDS32; idx += 32) {
            int t = idx;
            const int w32 = t & (K_WORDS32 - 1);
            t >>= 3; // /8
            const int m = t & (TM - 1);
            const int i = t >> 4; // /16

            uint32_t v = 0;
            const int q = q0 + m;
            if (q < Q) {
                const unsigned long long* a_slice = A + ((size_t)q * (size_t)Sa + (size_t)i) * (size_t)W64;
                const unsigned long long w64 = __ldg(&a_slice[(size_t)chunk * (size_t)K_WORDS64 + (size_t)(w32 >> 1)]);
                v = (w32 & 1) ? static_cast<uint32_t>(w64 >> 32) : static_cast<uint32_t>(w64 & 0xffffffffu);
            }
            A_bits[idx] = v;
        }

        // Load B bits for all slices and all 8 keys for this chunk into shared.
        // B_bits layout: ((j*TN + n)*8 + w32)
        for (int idx = lane; idx < TN * Sb * K_WORDS32; idx += 32) {
            int t = idx;
            const int w32 = t & (K_WORDS32 - 1);
            t >>= 3;
            const int n = t & (TN - 1);
            const int j = t >> 3; // /8

            uint32_t v = 0;
            const int r = r0 + n;
            if (r < R) {
                const unsigned long long* b_slice = B + ((size_t)r * (size_t)Sb + (size_t)j) * (size_t)W64;
                const unsigned long long w64 = __ldg(&b_slice[(size_t)chunk * (size_t)K_WORDS64 + (size_t)(w32 >> 1)]);
                v = (w32 & 1) ? static_cast<uint32_t>(w64 >> 32) : static_cast<uint32_t>(w64 & 0xffffffffu);
            }
            B_bits[idx] = v;
        }
        __syncwarp();

        // For each slice pair (i,j), compute 16x8 popcounts for this 256-bit chunk and
        // accumulate into the final float output tile with per-row/per-col weights.
        for (int i = 0; i < Sa; ++i) {
            // Warp-uniform skip: if the whole tile has Aw==0 for slice i, skip.
            bool row_nz = false;
            if (lane < TM) {
                row_nz = (Aw_tile[(size_t)lane * (size_t)Sa + (size_t)i] != 0.0f);
            }
            if (__ballot_sync(0xffffffffu, row_nz) == 0u) continue;

            const float aw0 = Aw_tile[(size_t)r0_0 * (size_t)Sa + (size_t)i];
            const float aw1 = Aw_tile[(size_t)r0_1 * (size_t)Sa + (size_t)i];
            const float aw2 = Aw_tile[(size_t)r0_2 * (size_t)Sa + (size_t)i];
            const float aw3 = Aw_tile[(size_t)r0_3 * (size_t)Sa + (size_t)i];

            const uint32_t* A_ptr = A_bits + (size_t)i * (size_t)TM * (size_t)K_WORDS32;

            for (int j = 0; j < Sb; ++j) {
                // Warp-uniform skip: if the whole tile has Bw==0 for slice j, skip.
                bool col_nz = false;
                if (lane < TN) {
                    col_nz = (Bw_tile[(size_t)lane * (size_t)Sb + (size_t)j] != 0.0f);
                }
                if (__ballot_sync(0xffffffffu, col_nz) == 0u) continue;

                const float bw0 = Bw_tile[(size_t)c0_0 * (size_t)Sb + (size_t)j];
                const float bw1 = Bw_tile[(size_t)c0_1 * (size_t)Sb + (size_t)j];
                const float bw2 = Bw_tile[(size_t)c0_2 * (size_t)Sb + (size_t)j];
                const float bw3 = Bw_tile[(size_t)c0_3 * (size_t)Sb + (size_t)j];

                const uint32_t* B_ptr = B_bits + (size_t)j * (size_t)TN * (size_t)K_WORDS32;

                using FragA = fragment<matrix_a, TM, TN, K_BITS, experimental::precision::b1, row_major>;
                using FragB = fragment<matrix_b, TM, TN, K_BITS, experimental::precision::b1, col_major>;
                using FragC = fragment<accumulator, TM, TN, K_BITS, int>;
                static_assert(FragA::num_elements == 4, "BMMA m16n8k256 expects 4 A regs/thread");
                static_assert(FragB::num_elements == 2, "BMMA m16n8k256 expects 2 B regs/thread");
                static_assert(FragC::num_elements == 4, "BMMA m16n8k256 expects 4 C regs/thread");

                FragA a_frag;
                FragB b_frag;

                // Leading dimension is in 32-bit words for bit-packed operands (K_BITS/32 == 8).
                load_matrix_sync(a_frag, reinterpret_cast<const unsigned int*>(A_ptr), K_WORDS32);
                load_matrix_sync(b_frag, reinterpret_cast<const unsigned int*>(B_ptr), K_WORDS32);

                // BMMA: exact op we want, on tensor cores.
                int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
                asm volatile(
                    "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                    : "r"(a_frag.x[0]),
                      "r"(a_frag.x[1]),
                      "r"(a_frag.x[2]),
                      "r"(a_frag.x[3]),
                      "r"(b_frag.x[0]),
                      "r"(b_frag.x[1]));

                FragC c_frag;
                c_frag.x[0] = c0;
                c_frag.x[1] = c1;
                c_frag.x[2] = c2;
                c_frag.x[3] = c3;

                store_matrix_sync(C_tile, c_frag, TN, mem_row_major);
                __syncwarp();

                // Update 4 outputs owned by this thread (row-major order).
                const float k0 = aw0 * bw0;
                const float k1 = aw1 * bw1;
                const float k2 = aw2 * bw2;
                const float k3 = aw3 * bw3;

                acc0 += static_cast<float>(C_tile[out0]) * k0;
                acc1 += static_cast<float>(C_tile[out1]) * k1;
                acc2 += static_cast<float>(C_tile[out2]) * k2;
                acc3 += static_cast<float>(C_tile[out3]) * k3;
            }
        }
        __syncwarp();
    }

    // Scatter store the 16x8 tile back to the global output.
    // Each thread stores its 4 outputs (if in-bounds).
    const int q_out0 = q0 + r0_0;
    const int q_out1 = q0 + r0_1;
    const int q_out2 = q0 + r0_2;
    const int q_out3 = q0 + r0_3;
    const int r_out0 = r0 + c0_0;
    const int r_out1 = r0 + c0_1;
    const int r_out2 = r0 + c0_2;
    const int r_out3 = r0 + c0_3;

    if (q_out0 < Q && r_out0 < R) {
        const long long gq = q_ids[r0_0];
        const long long gr = r_ids[c0_0];
        if (gq >= 0 && gr >= 0) out_global[(size_t)gq * (size_t)R_total + (size_t)gr] = acc0 * scale_inv;
    }
    if (q_out1 < Q && r_out1 < R) {
        const long long gq = q_ids[r0_1];
        const long long gr = r_ids[c0_1];
        if (gq >= 0 && gr >= 0) out_global[(size_t)gq * (size_t)R_total + (size_t)gr] = acc1 * scale_inv;
    }
    if (q_out2 < Q && r_out2 < R) {
        const long long gq = q_ids[r0_2];
        const long long gr = r_ids[c0_2];
        if (gq >= 0 && gr >= 0) out_global[(size_t)gq * (size_t)R_total + (size_t)gr] = acc2 * scale_inv;
    }
    if (q_out3 < Q && r_out3 < R) {
        const long long gq = q_ids[r0_3];
        const long long gr = r_ids[c0_3];
        if (gq >= 0 && gr >= 0) out_global[(size_t)gq * (size_t)R_total + (size_t)gr] = acc3 * scale_inv;
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
#endif
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
static inline void launch_w128_sb_kernel(
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
            popcount_weighted_keys_literal_fused_multiq_kernel_warp_out_w128_sb<SB>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)shared_bytes);
    }
    popcount_weighted_keys_literal_fused_multiq_kernel_warp_out_w128_sb<SB><<<grid_warp, block, shared_bytes, stream>>>(
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
    // Optional tensor-core (BMMA) path. On H100 this is the only realistic way to
    // get a large dot_ms win; the scalar popcount kernels are already ALU-bound.
    static int cached_tc = -1;
    if (cached_tc < 0) {
        int v = 0;
        if (const char* s = getenv("BSI_TC_DOT")) v = (atoi(s) != 0) ? 1 : 0;
        cached_tc = v;
    }
    if (cached_tc) {
        // BMMA is available on SM80+. Keep a runtime guard so we can still build
        // fatbins / run on older GPUs using the existing kernels.
        static int cached_tc_ok = -1;
        if (cached_tc_ok < 0) {
            int dev = 0;
            cudaGetDevice(&dev);
            int major = 0;
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
            cached_tc_ok = (major >= 8) ? 1 : 0;
        }
        if (cached_tc_ok && Sa > 0 && Sb > 0 && (W % 4 == 0) && W >= 4) {
            constexpr int TM = 16;
            constexpr int TN = 8;
            constexpr int K_WORDS32 = 8; // 256 bits / 32

            dim3 block_tc(32, 1, 1);
            dim3 grid_tc((R + TN - 1) / TN, (Q + TM - 1) / TM, 1);

            // +16 bytes for internal alignment padding in the kernel.
            size_t shared_bytes =
                16u +
                (size_t)Sa * (size_t)TM * (size_t)K_WORDS32 * sizeof(uint32_t) +
                (size_t)Sb * (size_t)TN * (size_t)K_WORDS32 * sizeof(uint32_t) +
                (size_t)TM * (size_t)Sa * sizeof(float) +
                (size_t)TN * (size_t)Sb * sizeof(float) +
                (size_t)TM * (size_t)TN * sizeof(int) +
                (size_t)TM * sizeof(long long) +
                (size_t)TN * sizeof(long long);

            popcount_weighted_keys_literal_fused_bmma_tc_kernel<<<grid_tc, block_tc, shared_bytes, stream>>>(
                A,
                Aw,
                Sa,
                W,
                B,
                Bw,
                Sb,
                R,
                Q,
                indices_r,
                indices_q,
                scale_inv,
                R_total,
                out_global);
            return;
        }
    }

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
    // Optional auto-tiling: can help some shapes by increasing data reuse, but it can also
    // reduce occupancy if shared memory gets too large. Keep it opt-in.
    bool auto_tiles = false;
    if (const char* s = getenv("BSI_AUTO_TILES")) {
        auto_tiles = (atoi(s) != 0);
    }
    if (auto_tiles) {
        const int num_warps = cached_block >> 5;
        int target_r = num_warps * 4; // allow each warp to handle multiple outputs
        if (W == 128) {
            if (target_r > 8) target_r = 8;
        } else if (W == 32) {
            if (target_r > 32) target_r = 32;
        } else {
            if (target_r > 16) target_r = 16;
        }
        if (target_r > tile_r) tile_r = target_r;

        int target_q = (W == 128) ? 16 : 32;
        if (target_q > tile_q) tile_q = target_q;
    }
    dim3 block(cached_block);
    bool use_warp_out = true; // default to warp-out path for better perf/SMEM usage
    if (const char* s = getenv("BSI_WARP_OUT")) {
        int v = atoi(s);
        use_warp_out = (v != 0);
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
        bool use_w32 = (W == 32 && Sb >= 1 && Sb <= 32);
        bool use_w128 = (W == 128 && Sb >= 1 && Sb <= 16);
        // W==32 kernel double-buffers A (+Aw) to overlap copy/compute; W==128 kernel does not.
        size_t a_word_factor = use_w32 ? 2u : 1u;
        size_t a_float_factor = use_w32 ? 2u : 1u;
        int tile_r_eff = tile_r;
        size_t shared_bytes = 0;
        while (tile_r_eff > 0) {
            shared_bytes =
                ((size_t)Sa * (size_t)W * a_word_factor +
                 (size_t)tile_r_eff * (size_t)Sb * (size_t)W) * sizeof(unsigned long long) +
                ((size_t)Sa * a_float_factor + (size_t)tile_r_eff * (size_t)Sb) * sizeof(float);
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
            if (use_w32) {
                switch (Sb) {
#define LAUNCH_W32_SB_CASE(N) \
                    case N: \
                        launch_w32_sb_kernel<N>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q, \
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default); \
                        break;
                    LAUNCH_W32_SB_CASE(1)
                    LAUNCH_W32_SB_CASE(2)
                    LAUNCH_W32_SB_CASE(3)
                    LAUNCH_W32_SB_CASE(4)
                    LAUNCH_W32_SB_CASE(5)
                    LAUNCH_W32_SB_CASE(6)
                    LAUNCH_W32_SB_CASE(7)
                    LAUNCH_W32_SB_CASE(8)
                    LAUNCH_W32_SB_CASE(9)
                    LAUNCH_W32_SB_CASE(10)
                    LAUNCH_W32_SB_CASE(11)
                    LAUNCH_W32_SB_CASE(12)
                    LAUNCH_W32_SB_CASE(13)
                    LAUNCH_W32_SB_CASE(14)
                    LAUNCH_W32_SB_CASE(15)
                    LAUNCH_W32_SB_CASE(16)
                    LAUNCH_W32_SB_CASE(17)
                    LAUNCH_W32_SB_CASE(18)
                    LAUNCH_W32_SB_CASE(19)
                    LAUNCH_W32_SB_CASE(20)
                    LAUNCH_W32_SB_CASE(21)
                    LAUNCH_W32_SB_CASE(22)
                    LAUNCH_W32_SB_CASE(23)
                    LAUNCH_W32_SB_CASE(24)
                    LAUNCH_W32_SB_CASE(25)
                    LAUNCH_W32_SB_CASE(26)
                    LAUNCH_W32_SB_CASE(27)
                    LAUNCH_W32_SB_CASE(28)
                    LAUNCH_W32_SB_CASE(29)
                    LAUNCH_W32_SB_CASE(30)
                    LAUNCH_W32_SB_CASE(31)
                    LAUNCH_W32_SB_CASE(32)
#undef LAUNCH_W32_SB_CASE
                    default:
                        launch_w32_sb_kernel<32>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        break;
                }
            } else if (use_w128) {
                switch (Sb) {
#define LAUNCH_W128_SB_CASE(N) \
                    case N: \
                        launch_w128_sb_kernel<N>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q, \
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default); \
                        break;
                    LAUNCH_W128_SB_CASE(1)
                    LAUNCH_W128_SB_CASE(2)
                    LAUNCH_W128_SB_CASE(3)
                    LAUNCH_W128_SB_CASE(4)
                    LAUNCH_W128_SB_CASE(5)
                    LAUNCH_W128_SB_CASE(6)
                    LAUNCH_W128_SB_CASE(7)
                    LAUNCH_W128_SB_CASE(8)
                    LAUNCH_W128_SB_CASE(9)
                    LAUNCH_W128_SB_CASE(10)
                    LAUNCH_W128_SB_CASE(11)
                    LAUNCH_W128_SB_CASE(12)
                    LAUNCH_W128_SB_CASE(13)
                    LAUNCH_W128_SB_CASE(14)
                    LAUNCH_W128_SB_CASE(15)
                    LAUNCH_W128_SB_CASE(16)
#undef LAUNCH_W128_SB_CASE
                    default:
                        launch_w128_sb_kernel<16>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
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
