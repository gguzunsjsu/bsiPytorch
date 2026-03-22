#pragma once

#include "bsi_word_config.h"

// ---------------------------------------------------------------------------
// EWAH decompress: interpret buffer of RLWs and literal words into W words.
// RLW layout uses kBsiRlwRunBits / kBsiRlwLitShift from bsi_word_config.h.
// ---------------------------------------------------------------------------
extern "C" __global__
void ewah_decompress_kernel(
    const bsi_word_t* __restrict__ in,
    int in_len,
    int W,
    bsi_word_t* __restrict__ out)
{
    if (threadIdx.x != 0) return; // simple sequential decode per slice

    int idx = 0;
    int out_idx = 0;
    while (idx < in_len && out_idx < W) {
        bsi_word_t rlw = in[idx++];
        bool running_bit = bsi_rlw_run_bit(rlw);
        unsigned int run_len = bsi_rlw_run_len(rlw);
        unsigned int lit_words = bsi_rlw_lit_count(rlw);

        // Emit run words
#if BSI_WORD_BITS <= 64
        bsi_word_t run_val = running_bit ? ~static_cast<bsi_word_t>(0) : static_cast<bsi_word_t>(0);
#else
        bsi_word_t run_val = running_bit ? bsi_word_ones() : bsi_word_zero();
#endif
        for (unsigned int k = 0; k < run_len && out_idx < W; ++k) {
            out[out_idx++] = run_val;
        }
        // Emit literal words
        for (unsigned int k = 0; k < lit_words && out_idx < W && idx < in_len; ++k) {
            out[out_idx++] = in[idx++];
        }
    }
    // Pad zeros
#if BSI_WORD_BITS <= 64
    while (out_idx < W) out[out_idx++] = static_cast<bsi_word_t>(0);
#else
    while (out_idx < W) out[out_idx++] = bsi_word_zero();
#endif
}

extern "C" void launch_ewah_decompress(
    const bsi_word_t* in,
    int in_len,
    int W,
    bsi_word_t* out,
    cudaStream_t stream) {
    ewah_decompress_kernel<<<1,1,0,stream>>>(in, in_len, W, out);
}

// ---------------- Hybrid (EWAH) helper kernels -----------------

// Compute popcount sum per slice across W words
extern "C" __global__
void slice_popcount_sum_kernel(
    const bsi_word_t* __restrict__ words, // [S*W]
    int S,
    int W,
    unsigned long long* __restrict__ out_counts)  // [S]
{
    int s = blockIdx.x;
    if (s >= S) return;
    unsigned long long local = 0ULL;
    for (int w = threadIdx.x; w < W; w += blockDim.x) {
        local += bsi_popc(words[(size_t)s * W + w]);
    }
    __shared__ unsigned long long smem[32];
    unsigned long long warp = warp_reduce_sum_ull(local);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = warp;
    __syncthreads();
    unsigned long long block_sum = 0ULL;
    int num_warps = blockDim.x >> 5;
    if (threadIdx.x < (unsigned)num_warps) block_sum = smem[threadIdx.x];
    block_sum = warp_reduce_sum_ull(block_sum);
    if (threadIdx.x == 0) out_counts[s] = block_sum;
}

extern "C" void launch_slice_popcount_sum(
    const bsi_word_t* words,
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
    double total_bits = (double)W * (double)kBsiWordBits;
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

// ---------------------------------------------------------------------------
// Zero/ones helpers for comparisons in EWAH size/emit kernels
// ---------------------------------------------------------------------------
#if BSI_WORD_BITS <= 64
__device__ __forceinline__ bool bsi_word_is_zero(bsi_word_t v) { return v == static_cast<bsi_word_t>(0); }
__device__ __forceinline__ bool bsi_word_is_ones(bsi_word_t v) { return v == ~static_cast<bsi_word_t>(0); }
#else
__device__ __forceinline__ bool bsi_word_is_zero(const bsi_word_t& v) {
    for (int i = 0; i < kBsiWordParts; ++i) if (v.parts[i] != 0ULL) return false;
    return true;
}
__device__ __forceinline__ bool bsi_word_is_ones(const bsi_word_t& v) {
    for (int i = 0; i < kBsiWordParts; ++i) if (v.parts[i] != ~0ULL) return false;
    return true;
}
#endif

// Size pass: compute number of words to emit per slice for EWAH encoding
extern "C" __global__
void ewah_size_kernel(
    const bsi_word_t* __restrict__ words, // [S*W]
    int S,
    int W,
    const int* __restrict__ flags,                // [S] 1=compress, 0=literal-only
    unsigned long long* __restrict__ sizes)       // [S]
{
    int s = blockIdx.x;
    if (s >= S) return;
    if (threadIdx.x != 0) return;
    const bsi_word_t* base = words + (size_t)s * W;
    unsigned long long out_size = 0ULL;
    int idx = 0;
    int compress = flags[s];
    while (idx < W) {
        int run_len = 0;
        if (compress && idx < W) {
            bsi_word_t v = base[idx];
            bool cz = bsi_word_is_zero(v);
            bool co = bsi_word_is_ones(v);
            if (cz || co) {
                int k = idx;
                while (k < W) {
                    if (co && !bsi_word_is_ones(base[k])) break;
                    if (cz && !bsi_word_is_zero(base[k])) break;
                    ++k;
                }
                run_len = k - idx;
                idx = k;
            }
        }
        int lit_count = 0;
        while (idx < W) {
            bsi_word_t v = base[idx];
            bool cz = bsi_word_is_zero(v), co = bsi_word_is_ones(v);
            if (compress && (cz || co)) break;
            ++lit_count;
            ++idx;
        }
        out_size += 1ULL + (unsigned long long)lit_count;
    }
    sizes[s] = out_size;
}

extern "C" void launch_ewah_size(
    const bsi_word_t* words,
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
    const bsi_word_t* __restrict__ words, // [S*W]
    int S,
    int W,
    const int* __restrict__ flags,                // [S]
    const unsigned long long* __restrict__ off,   // [S]
    bsi_word_t* __restrict__ out,                 // [total]
    int* __restrict__ out_len)                    // [S]
{
    int s = blockIdx.x;
    if (s >= S) return;
    if (threadIdx.x != 0) return;
    const bsi_word_t* base = words + (size_t)s * W;
    unsigned long long o = off[s];
    int idx = 0;
    int compress = flags[s];
    while (idx < W) {
        int run_len = 0;
        int run_bit = 0;
        if (compress && idx < W) {
            bsi_word_t v = base[idx];
            bool cz = bsi_word_is_zero(v);
            bool co = bsi_word_is_ones(v);
            if (cz || co) {
                run_bit = co ? 1 : 0;
                int k = idx;
                while (k < W) {
                    if (co && !bsi_word_is_ones(base[k])) break;
                    if (cz && !bsi_word_is_zero(base[k])) break;
                    ++k;
                }
                run_len = k - idx;
                idx = k;
            }
        }
        int lit_count = 0;
        int lit_start = idx;
        while (idx < W) {
            bsi_word_t v = base[idx];
            bool cz = bsi_word_is_zero(v), co = bsi_word_is_ones(v);
            if (compress && (cz || co)) break;
            ++lit_count;
            ++idx;
        }
        // Encode RLW using word-size-aware encoding
        bsi_word_t rlw = bsi_rlw_encode(run_bit != 0, (unsigned int)run_len, (unsigned int)lit_count);
        out[o++] = rlw;
        for (int k = 0; k < lit_count; ++k) {
            out[o++] = base[lit_start + k];
        }
    }
    out_len[s] = (int)(o - off[s]);
}

extern "C" void launch_ewah_emit(
    const bsi_word_t* words,
    int S,
    int W,
    const int* flags,
    const unsigned long long* off,
    bsi_word_t* out,
    int* out_len,
    cudaStream_t stream)
{
    dim3 grid(S);
    dim3 block(1);
    ewah_emit_kernel<<<grid, block, 0, stream>>>(words, S, W, flags, off, out, out_len);
}
