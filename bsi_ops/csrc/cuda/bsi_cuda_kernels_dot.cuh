#pragma once

// Multi-query fused version: process Q queries and multiple keys per block; tiles both axes to shrink grid
#include <stdint.h>

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

// Tensor-core BMMA path (SM90+): computes a TMxTN output tile using
// mma.sync.aligned.m16n8k256.*.b1.b1.s32.and.popc over 256-bit chunks.
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
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    constexpr int TM = 16;
    // Use 4 warps per block to cover a 16x32 output tile. This increases reuse of
    // the A tile across more output columns and reduces redundant loads of A across blocks.
    constexpr int TN = 32;
    constexpr int SB_MAX = 16;
    constexpr int K_BITS = 256;
    constexpr int K_WORDS64 = K_BITS / 64;   // 4
    constexpr int K_WORDS32 = K_BITS / 32;   // 8
    // Pad K by 1x32-bit word in shared to reduce 2-way bank conflicts in BMMA fragment loads.
    constexpr int K_STRIDE32 = K_WORDS32 + 1; // 9

    // This kernel is defined as 4 warps per block.
    if (blockDim.x != 128) return;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5; // 0..3

    const int q0 = blockIdx.y * TM;
    const int r0 = blockIdx.x * TN;

    // --- Shared memory layout (dynamic) ---
    extern __shared__ unsigned char smem_raw[];
    uintptr_t p = reinterpret_cast<uintptr_t>(smem_raw);
    // 16B alignment keeps PTX loads/stores happy and avoids bank conflicts on most layouts.
    p = (p + 15u) & ~uintptr_t(15u);

    auto* A_bits = reinterpret_cast<uint32_t*>(p); // [Sa, TM, 8]
    p += (size_t)Sa * (size_t)TM * (size_t)K_STRIDE32 * sizeof(uint32_t);
    auto* B_bits = reinterpret_cast<uint32_t*>(p); // [Sb, TN, 8] (columns stored contiguously)
    p += (size_t)Sb * (size_t)TN * (size_t)K_STRIDE32 * sizeof(uint32_t);
    auto* Aw_tile = reinterpret_cast<float*>(p);   // [TM, Sa]
    p += (size_t)TM * (size_t)Sa * sizeof(float);
    auto* Bw_tile = reinterpret_cast<float*>(p);   // [TN, Sb]
    p += (size_t)TN * (size_t)Sb * sizeof(float);
    (void)p;

    // Load all slice weights for the tile.
    for (int idx = threadIdx.x; idx < TM * Sa; idx += blockDim.x) {
        const int m = idx / Sa;
        const int i = idx - m * Sa;
        const int q = q0 + m;
        Aw_tile[(size_t)m * (size_t)Sa + (size_t)i] =
            (q < Q) ? __ldg(&Aw[(size_t)q * (size_t)Sa + (size_t)i]) : 0.0f;
    }
    for (int idx = threadIdx.x; idx < TN * Sb; idx += blockDim.x) {
        const int n = idx / Sb;
        const int j = idx - n * Sb;
        const int r = r0 + n;
        Bw_tile[(size_t)n * (size_t)Sb + (size_t)j] =
            (r < R) ? __ldg(&Bw[(size_t)r * (size_t)Sb + (size_t)j]) : 0.0f;
    }
    __syncthreads();

    // PTX fragment mapping for mma.sync.aligned.m16n8k256.*.b1.b1.s32.and.popc:
    // groupID = laneid >> 2, threadID = laneid & 3.
    // Each lane owns 4 accumulators at:
    //   (row=groupID,   col=threadID*2 + 0/1) and
    //   (row=groupID+8, col=threadID*2 + 0/1).
    const int groupID = lane >> 2;           // 0..7
    const int threadID = lane & 3;           // 0..3
    const int row0 = groupID;                // 0..7
    const int row1 = groupID + 8;            // 8..15
    const int col_base = warp_id * 8;        // 0,8,16,24
    const int col0 = col_base + threadID * 2; // 0..31 even
    const int col1 = col0 + 1;               // 1,3,5,7
    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    // Cache Bw for the 2 columns this lane owns (reused across all chunks and Sa slices).
    // This cuts shared-memory traffic in the hot inner loop.
    float bw0_cache[SB_MAX];
    float bw1_cache[SB_MAX];
    const bool cache_sb = (Sb <= SB_MAX);
    if (cache_sb) {
        const float* bw_col0 = Bw_tile + (size_t)col0 * (size_t)Sb;
        const float* bw_col1 = Bw_tile + (size_t)col1 * (size_t)Sb;
#pragma unroll
        for (int j = 0; j < SB_MAX; ++j) {
            if (j < Sb) {
                bw0_cache[j] = bw_col0[j];
                bw1_cache[j] = bw_col1[j];
            } else {
                bw0_cache[j] = 0.0f;
                bw1_cache[j] = 0.0f;
            }
        }
    }

    // Number of 256-bit chunks in the bit dimension.
    const int chunks = W64 / K_WORDS64;

    // Main loop: iterate over chunks, load all slice bits for the tile into shared, then BMMA.
    for (int chunk = 0; chunk < chunks; ++chunk) {
        // Load A bits for all slices and all 16 queries for this chunk into shared.
        // A_bits layout: ((i*TM + m)*8 + w32)
        for (int idx = threadIdx.x; idx < TM * Sa * K_WORDS32; idx += blockDim.x) {
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
            A_bits[((i * TM + m) * K_STRIDE32) + w32] = v;
        }

        // Load B bits for all slices and all TN keys for this chunk into shared.
        // B_bits layout: ((j*TN + n)*8 + w32)
        for (int idx = threadIdx.x; idx < TN * Sb * K_WORDS32; idx += blockDim.x) {
            int t = idx;
            const int w32 = t & (K_WORDS32 - 1);
            t >>= 3;
            const int n = t & (TN - 1);      // /TN
            const int j = t >> 5;            // /32

            uint32_t v = 0;
            const int r = r0 + n;
            if (r < R) {
                const unsigned long long* b_slice = B + ((size_t)r * (size_t)Sb + (size_t)j) * (size_t)W64;
                const unsigned long long w64 = __ldg(&b_slice[(size_t)chunk * (size_t)K_WORDS64 + (size_t)(w32 >> 1)]);
                v = (w32 & 1) ? static_cast<uint32_t>(w64 >> 32) : static_cast<uint32_t>(w64 & 0xffffffffu);
            }
            B_bits[((j * TN + n) * K_STRIDE32) + w32] = v;
        }
        __syncthreads();

        // For each slice pair (i,j), compute 16x8 popcounts for this 256-bit chunk and
        // accumulate into the final float output tile with per-row/per-col weights.
        if (cache_sb) {
            // Cache the 2 B fragment regs for each j (reused across all Sa slices).
            uint32_t b0_cache[SB_MAX];
            uint32_t b1_cache[SB_MAX];
            const size_t b_slice_stride = (size_t)TN * (size_t)K_STRIDE32;
            const uint32_t* b_col_base = B_bits + (size_t)(col_base + groupID) * (size_t)K_STRIDE32;
#pragma unroll
            for (int j = 0; j < SB_MAX; ++j) {
                if (j < Sb) {
                    const uint32_t* b_col = b_col_base + (size_t)j * b_slice_stride;
                    b0_cache[j] = b_col[(size_t)threadID];
                    b1_cache[j] = b_col[(size_t)(threadID + 4)];
                } else {
                    b0_cache[j] = 0u;
                    b1_cache[j] = 0u;
                }
            }

            for (int i = 0; i < Sa; ++i) {
                const float aw0 = Aw_tile[(size_t)row0 * (size_t)Sa + (size_t)i];
                const float aw1 = Aw_tile[(size_t)row1 * (size_t)Sa + (size_t)i];

                const uint32_t* A_i = A_bits + (size_t)i * (size_t)TM * (size_t)K_STRIDE32;
                const uint32_t a0 = A_i[(size_t)row0 * (size_t)K_STRIDE32 + (size_t)threadID];
                // NOTE: For SM90 BMMA the second 32b A reg for row1 uses the same 0..31 column window as row0.
                const uint32_t a1 = A_i[(size_t)row1 * (size_t)K_STRIDE32 + (size_t)threadID];
                const uint32_t a2 = A_i[(size_t)row0 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];
                const uint32_t a3 = A_i[(size_t)row1 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];

                float sum00 = 0.0f, sum01 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;
                // IMPORTANT: mma.sync is warp-synchronous. Do not per-lane branch around it.
#pragma unroll
                for (int j = 0; j < SB_MAX; ++j) {
                    if (j < Sb) {
                        const float bw0 = bw0_cache[j];
                        const float bw1 = bw1_cache[j];
                        const uint32_t b0 = b0_cache[j];
                        const uint32_t b1 = b1_cache[j];

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
                }
                acc00 = __fmaf_rn(aw0, sum00, acc00);
                acc01 = __fmaf_rn(aw0, sum01, acc01);
                acc10 = __fmaf_rn(aw1, sum10, acc10);
                acc11 = __fmaf_rn(aw1, sum11, acc11);
            }
        } else {
            for (int i = 0; i < Sa; ++i) {
                const float aw0 = Aw_tile[(size_t)row0 * (size_t)Sa + (size_t)i];
                const float aw1 = Aw_tile[(size_t)row1 * (size_t)Sa + (size_t)i];

                const uint32_t* A_i = A_bits + (size_t)i * (size_t)TM * (size_t)K_STRIDE32;
                const uint32_t a0 = A_i[(size_t)row0 * (size_t)K_STRIDE32 + (size_t)threadID];
                const uint32_t a1 = A_i[(size_t)row1 * (size_t)K_STRIDE32 + (size_t)threadID];
                const uint32_t a2 = A_i[(size_t)row0 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];
                const uint32_t a3 = A_i[(size_t)row1 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];

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
        __syncthreads();
    }

    // Each lane stores its 4 outputs (if in-bounds). Each warp covers an independent 16x8 tile.
    const int q_out0 = q0 + row0;
    const int q_out1 = q0 + row1;
    const int r_out0 = r0 + col0;
    const int r_out1 = r0 + col1;

    const long long gq0 = (q_out0 < Q) ? __ldg(&query_indices[q_out0]) : 0;
    const long long gq1 = (q_out1 < Q) ? __ldg(&query_indices[q_out1]) : 0;
    const long long gr0 = (r_out0 < R) ? __ldg(&key_indices[r_out0]) : 0;
    const long long gr1 = (r_out1 < R) ? __ldg(&key_indices[r_out1]) : 0;

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

// BMMA TC variant: 8 warps per block (16x64 output tile). This reduces redundant
// A-tile reloads across the R dimension (each A tile is reused across twice as many
// output columns vs TN=32).
extern "C" __global__
void popcount_weighted_keys_literal_fused_bmma_tc_kernel_tn64(
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
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    constexpr int TM = 16;
    constexpr int TN = 64;
    constexpr int SB_MAX = 16;
    constexpr int K_BITS = 256;
    constexpr int K_WORDS64 = K_BITS / 64;   // 4
    constexpr int K_WORDS32 = K_BITS / 32;   // 8
    // Pad K by 1x32-bit word in shared to reduce 2-way bank conflicts in BMMA fragment loads.
    constexpr int K_STRIDE32 = K_WORDS32 + 1; // 9

    if (blockDim.x != 256) return;
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5; // 0..7

    const int q0 = blockIdx.y * TM;
    const int r0 = blockIdx.x * TN;

    extern __shared__ unsigned char smem_raw[];
    uintptr_t p = reinterpret_cast<uintptr_t>(smem_raw);
    p = (p + 15u) & ~uintptr_t(15u);

    auto* A_bits = reinterpret_cast<uint32_t*>(p); // [Sa, TM, 8]
    p += (size_t)Sa * (size_t)TM * (size_t)K_STRIDE32 * sizeof(uint32_t);
    auto* B_bits = reinterpret_cast<uint32_t*>(p); // [Sb, TN, 8]
    p += (size_t)Sb * (size_t)TN * (size_t)K_STRIDE32 * sizeof(uint32_t);
    auto* Aw_tile = reinterpret_cast<float*>(p);   // [TM, Sa]
    p += (size_t)TM * (size_t)Sa * sizeof(float);
    auto* Bw_tile = reinterpret_cast<float*>(p);   // [TN, Sb]
    p += (size_t)TN * (size_t)Sb * sizeof(float);
    (void)p;

    for (int idx = threadIdx.x; idx < TM * Sa; idx += blockDim.x) {
        const int m = idx / Sa;
        const int i = idx - m * Sa;
        const int q = q0 + m;
        Aw_tile[(size_t)m * (size_t)Sa + (size_t)i] =
            (q < Q) ? __ldg(&Aw[(size_t)q * (size_t)Sa + (size_t)i]) : 0.0f;
    }
    for (int idx = threadIdx.x; idx < TN * Sb; idx += blockDim.x) {
        const int n = idx / Sb;
        const int j = idx - n * Sb;
        const int r = r0 + n;
        Bw_tile[(size_t)n * (size_t)Sb + (size_t)j] =
            (r < R) ? __ldg(&Bw[(size_t)r * (size_t)Sb + (size_t)j]) : 0.0f;
    }
    __syncthreads();

    const int groupID = lane >> 2;            // 0..7
    const int threadID = lane & 3;            // 0..3
    const int row0 = groupID;                 // 0..7
    const int row1 = groupID + 8;             // 8..15
    const int col_base = warp_id * 8;         // 0..56
    const int col0 = col_base + threadID * 2; // 0..63 even
    const int col1 = col0 + 1;                // odd
    float acc00 = 0.0f, acc01 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;

    float bw0_cache[SB_MAX];
    float bw1_cache[SB_MAX];
    const bool cache_sb = (Sb <= SB_MAX);
    if (cache_sb) {
        const float* bw_col0 = Bw_tile + (size_t)col0 * (size_t)Sb;
        const float* bw_col1 = Bw_tile + (size_t)col1 * (size_t)Sb;
#pragma unroll
        for (int j = 0; j < SB_MAX; ++j) {
            if (j < Sb) {
                bw0_cache[j] = bw_col0[j];
                bw1_cache[j] = bw_col1[j];
            } else {
                bw0_cache[j] = 0.0f;
                bw1_cache[j] = 0.0f;
            }
        }
    }

    const int chunks = W64 / K_WORDS64;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        for (int idx = threadIdx.x; idx < TM * Sa * K_WORDS32; idx += blockDim.x) {
            int t = idx;
            const int w32 = t & (K_WORDS32 - 1);
            t >>= 3;
            const int m = t & (TM - 1);
            const int i = t >> 4;

            uint32_t v = 0;
            const int q = q0 + m;
            if (q < Q) {
                const unsigned long long* a_slice = A + ((size_t)q * (size_t)Sa + (size_t)i) * (size_t)W64;
                const unsigned long long w64 = __ldg(&a_slice[(size_t)chunk * (size_t)K_WORDS64 + (size_t)(w32 >> 1)]);
                v = (w32 & 1) ? static_cast<uint32_t>(w64 >> 32) : static_cast<uint32_t>(w64 & 0xffffffffu);
            }
            A_bits[((i * TM + m) * K_STRIDE32) + w32] = v;
        }

        for (int idx = threadIdx.x; idx < TN * Sb * K_WORDS32; idx += blockDim.x) {
            int t = idx;
            const int w32 = t & (K_WORDS32 - 1);
            t >>= 3;
            const int n = t & (TN - 1);      // /TN
            const int j = t >> 6;            // /64

            uint32_t v = 0;
            const int r = r0 + n;
            if (r < R) {
                const unsigned long long* b_slice = B + ((size_t)r * (size_t)Sb + (size_t)j) * (size_t)W64;
                const unsigned long long w64 = __ldg(&b_slice[(size_t)chunk * (size_t)K_WORDS64 + (size_t)(w32 >> 1)]);
                v = (w32 & 1) ? static_cast<uint32_t>(w64 >> 32) : static_cast<uint32_t>(w64 & 0xffffffffu);
            }
            B_bits[((j * TN + n) * K_STRIDE32) + w32] = v;
        }
        __syncthreads();

        if (cache_sb) {
            uint32_t b0_cache[SB_MAX];
            uint32_t b1_cache[SB_MAX];
            const size_t b_slice_stride = (size_t)TN * (size_t)K_STRIDE32;
            const uint32_t* b_col_base = B_bits + (size_t)(col_base + groupID) * (size_t)K_STRIDE32;
#pragma unroll
            for (int j = 0; j < SB_MAX; ++j) {
                if (j < Sb) {
                    const uint32_t* b_col = b_col_base + (size_t)j * b_slice_stride;
                    b0_cache[j] = b_col[(size_t)threadID];
                    b1_cache[j] = b_col[(size_t)(threadID + 4)];
                } else {
                    b0_cache[j] = 0u;
                    b1_cache[j] = 0u;
                }
            }

            for (int i = 0; i < Sa; ++i) {
                const float aw0 = Aw_tile[(size_t)row0 * (size_t)Sa + (size_t)i];
                const float aw1 = Aw_tile[(size_t)row1 * (size_t)Sa + (size_t)i];

                const uint32_t* A_i = A_bits + (size_t)i * (size_t)TM * (size_t)K_STRIDE32;
                const uint32_t a0 = A_i[(size_t)row0 * (size_t)K_STRIDE32 + (size_t)threadID];
                const uint32_t a1 = A_i[(size_t)row1 * (size_t)K_STRIDE32 + (size_t)threadID];
                const uint32_t a2 = A_i[(size_t)row0 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];
                const uint32_t a3 = A_i[(size_t)row1 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];

                float sum00 = 0.0f, sum01 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;
#pragma unroll
                for (int j = 0; j < SB_MAX; ++j) {
                    if (j < Sb) {
                        const float bw0 = bw0_cache[j];
                        const float bw1 = bw1_cache[j];
                        const uint32_t b0 = b0_cache[j];
                        const uint32_t b1 = b1_cache[j];

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
                }
                acc00 = __fmaf_rn(aw0, sum00, acc00);
                acc01 = __fmaf_rn(aw0, sum01, acc01);
                acc10 = __fmaf_rn(aw1, sum10, acc10);
                acc11 = __fmaf_rn(aw1, sum11, acc11);
            }
        } else {
            for (int i = 0; i < Sa; ++i) {
                const float aw0 = Aw_tile[(size_t)row0 * (size_t)Sa + (size_t)i];
                const float aw1 = Aw_tile[(size_t)row1 * (size_t)Sa + (size_t)i];

                const uint32_t* A_i = A_bits + (size_t)i * (size_t)TM * (size_t)K_STRIDE32;
                const uint32_t a0 = A_i[(size_t)row0 * (size_t)K_STRIDE32 + (size_t)threadID];
                const uint32_t a1 = A_i[(size_t)row1 * (size_t)K_STRIDE32 + (size_t)threadID];
                const uint32_t a2 = A_i[(size_t)row0 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];
                const uint32_t a3 = A_i[(size_t)row1 * (size_t)K_STRIDE32 + (size_t)(threadID + 4)];

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
        __syncthreads();
    }

    const int q_out0 = q0 + row0;
    const int q_out1 = q0 + row1;
    const int r_out0 = r0 + col0;
    const int r_out1 = r0 + col1;

    const long long gq0 = (q_out0 < Q) ? __ldg(&query_indices[q_out0]) : 0;
    const long long gq1 = (q_out1 < Q) ? __ldg(&query_indices[q_out1]) : 0;
    const long long gr0 = (r_out0 < R) ? __ldg(&key_indices[r_out0]) : 0;
    const long long gr1 = (r_out1 < R) ? __ldg(&key_indices[r_out1]) : 0;

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
    // Optional SM90+ tensor-core (BMMA) path (guarded by BSI_TC_DOT).
    int use_tc = 0;
    if (const char* s = getenv("BSI_TC_DOT")) use_tc = (atoi(s) != 0) ? 1 : 0;
    if (use_tc) {
        // BMMA (1-bit MMA with AND+POPC) is available on H100+ (SM90+). Keep a runtime
        // guard so we can still build fatbins / run on older GPUs using the existing kernels.
        static int cached_tc_ok = -1;
        if (cached_tc_ok < 0) {
            int dev = 0;
            cudaGetDevice(&dev);
            int major = 0;
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
            cached_tc_ok = (major >= 9) ? 1 : 0;
        }
        if (cached_tc_ok && Sa > 0 && Sb > 0 && (W % 4 == 0) && W >= 4) {
            constexpr int TM = 16;
            constexpr int K_WORDS32 = 8;           // 256 bits / 32
            constexpr int K_STRIDE32 = K_WORDS32 + 1; // padding to reduce bank conflicts

            int tc_tn = 32;
            if (const char* s = getenv("BSI_TC_TN")) {
                int t = atoi(s);
                if (t == 64) tc_tn = 64;
            }

            int dev = 0;
            cudaGetDevice(&dev);
            int max_shared_default = 0;
            int max_shared_optin = 0;
            cudaDeviceGetAttribute(&max_shared_default, cudaDevAttrMaxSharedMemoryPerBlock, dev);
            cudaDeviceGetAttribute(&max_shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
            int max_shared = (max_shared_optin > max_shared_default) ? max_shared_optin : max_shared_default;

            if (tc_tn == 64) {
                constexpr int TN = 64;
                dim3 block_tc(256, 1, 1);
                dim3 grid_tc((R + TN - 1) / TN, (Q + TM - 1) / TM, 1);

                size_t shared_bytes =
                    16u +
                    (size_t)Sa * (size_t)TM * (size_t)K_STRIDE32 * sizeof(uint32_t) +
                    (size_t)Sb * (size_t)TN * (size_t)K_STRIDE32 * sizeof(uint32_t) +
                    (size_t)TM * (size_t)Sa * sizeof(float) +
                    (size_t)TN * (size_t)Sb * sizeof(float);

                if (shared_bytes <= (size_t)max_shared) {
                    if (shared_bytes > (size_t)max_shared_default) {
                        cudaFuncSetAttribute(
                            popcount_weighted_keys_literal_fused_bmma_tc_kernel_tn64,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            (int)shared_bytes);
                    }
                    popcount_weighted_keys_literal_fused_bmma_tc_kernel_tn64<<<grid_tc, block_tc, shared_bytes, stream>>>(
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
            } else {
                constexpr int TN = 32;
                dim3 block_tc(128, 1, 1);
                dim3 grid_tc((R + TN - 1) / TN, (Q + TM - 1) / TM, 1);

                size_t shared_bytes =
                    16u +
                    (size_t)Sa * (size_t)TM * (size_t)K_STRIDE32 * sizeof(uint32_t) +
                    (size_t)Sb * (size_t)TN * (size_t)K_STRIDE32 * sizeof(uint32_t) +
                    (size_t)TM * (size_t)Sa * sizeof(float) +
                    (size_t)TN * (size_t)Sb * sizeof(float);

                if (shared_bytes <= (size_t)max_shared) {
                    if (shared_bytes > (size_t)max_shared_default) {
                        cudaFuncSetAttribute(
                            popcount_weighted_keys_literal_fused_bmma_tc_kernel,
                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                            (int)shared_bytes);
                    }
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
