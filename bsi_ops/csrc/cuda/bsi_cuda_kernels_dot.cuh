#pragma once

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
        if (q + 1 < q_end) {
            __syncthreads();
        }
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
    const float* Aw_base0 = Aw + ((size_t)q_start * (size_t)Sa);
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

            if (Sa <= 16) {
                const unsigned long long* a_ptr = A_sh + (size_t)lane;
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
                const unsigned long long* a_ptr = A_sh + (size_t)lane;
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
        if (q_next < q_end) {
            cp_async_wait();
            cp_async_tail_ull(A_sh_next, A_base_next, Sa * Wc);
            __syncthreads();
        }
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
    bool launch_base = !use_warp_out;
    if (use_warp_out) {
        int dev = 0;
        cudaGetDevice(&dev);
        int max_shared_default = 0;
        int max_shared_optin = 0;
        cudaDeviceGetAttribute(&max_shared_default, cudaDevAttrMaxSharedMemoryPerBlock, dev);
        cudaDeviceGetAttribute(&max_shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        int max_shared = (max_shared_optin > max_shared_default) ? max_shared_optin : max_shared_default;
        bool use_w32 = (W == 32 && Sb >= 1 && Sb <= 16);
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
            if (W == 32 && Sb >= 1 && Sb <= 16) {
                switch (Sb) {
                    case 1:
                        launch_w32_sb_kernel<1>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                    scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        break;
                    case 2:
                        launch_w32_sb_kernel<2>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                    scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        break;
                    case 3:
                        launch_w32_sb_kernel<3>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                    scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        break;
                    case 4:
                        launch_w32_sb_kernel<4>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                    scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        break;
                    case 5:
                        launch_w32_sb_kernel<5>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                    scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        break;
                    case 6:
                        launch_w32_sb_kernel<6>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                    scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        break;
                    case 7:
                        launch_w32_sb_kernel<7>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                    scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        break;
                    case 8:
                        launch_w32_sb_kernel<8>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                    scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        break;
                    case 9:
                        launch_w32_sb_kernel<9>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                    scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        break;
                    case 10:
                        launch_w32_sb_kernel<10>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        break;
                    case 11:
                        launch_w32_sb_kernel<11>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        break;
                    case 12:
                        launch_w32_sb_kernel<12>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        break;
                    case 13:
                        launch_w32_sb_kernel<13>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        break;
                    case 14:
                        launch_w32_sb_kernel<14>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        break;
                    case 15:
                        launch_w32_sb_kernel<15>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        break;
                    case 16:
                        launch_w32_sb_kernel<16>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
                                                     scale_inv, R_total, out_global, shared_bytes, grid_warp, block, stream, max_shared_default);
                        break;
                    default:
                        launch_w32_sb_kernel<16>(A, Aw, Sa, W, B, Bw, Sb, R, Q, tile_q, tile_r_eff, indices_r, indices_q,
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
