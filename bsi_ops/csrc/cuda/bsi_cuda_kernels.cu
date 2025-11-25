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

// Multi-query fused version: process Q queries in a single launch (2D grid)
extern "C" __global__
void popcount_weighted_keys_literal_fused_multiq_kernel(
    const unsigned long long* __restrict__ A,    // [Q, Sa, W]
    const double* __restrict__ Aw,               // [Q, Sa]
    int Sa,
    int W,
    const unsigned long long* __restrict__ B,    // [R, Sb, W]
    const double* __restrict__ Bw,               // [R, Sb]
    int Sb,
    int R,
    int Q,
    const long long* __restrict__ key_indices,   // [R]
    const long long* __restrict__ query_indices, // [Q]
    double scale_inv,
    int R_total,
    double* __restrict__ out_global)
{
    int r = blockIdx.x;
    int q = blockIdx.y;
    if (r >= R || q >= Q) return;

    long long global_r = __ldg(&key_indices[r]);
    long long global_q = __ldg(&query_indices[q]);

    const unsigned long long* A_base = A + ((size_t)q * Sa * W);
    const double* Aw_base = Aw + (size_t)q * Sa;
    const unsigned long long* B_base = B + ((size_t)r * Sb * W);
    const double* Bw_base = Bw + (size_t)r * Sb;

    double local = 0.0;
    long long total = (long long)Sa * (long long)Sb * (long long)W;
    for (long long idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int i = static_cast<int>(idx / (Sb * (long long)W));
        int rem = static_cast<int>(idx % (Sb * (long long)W));
        int j = rem / W;
        int w = rem % W;

        const unsigned long long* ai = A_base + (size_t)i * W;
        const unsigned long long* bj = B_base + (size_t)j * W;

        unsigned long long a_val = __ldg(&ai[w]);
        unsigned long long b_val = __ldg(&bj[w]);
        int cnt = __popcll(a_val & b_val);

        double aw = __ldg(&Aw_base[i]);
        double bw = __ldg(&Bw_base[j]);

        local += (double)cnt * aw * bw;
    }

    // Warp-level reduction (no divergence within warp)
    local = warp_reduce_sum_double(local);

    // Block-level reduction using shared memory
    __shared__ double warp_buf[32];
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    if (lane == 0) {
        warp_buf[warp_id] = local;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) >> 5;
        double val = (lane < num_warps) ? warp_buf[lane] : 0.0;
        val = warp_reduce_sum_double(val);

        if (lane == 0) {
            // write directly into the correct row/column
            out_global[(size_t)global_q * (size_t)R_total + (size_t)global_r] = val * scale_inv;
        }
    }
}

extern "C" void launch_popcount_weighted_keys_literal_fused_multiq(
    const unsigned long long* A,
    const double* Aw,
    int Sa,
    int W,
    const unsigned long long* B,
    const double* Bw,
    int Sb,
    int R,
    int Q,
    const long long* indices_r,
    const long long* indices_q,
    double scale_inv,
    int R_total,
    double* out_global,
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
    dim3 grid(R, Q);
    dim3 block(cached_block);
    popcount_weighted_keys_literal_fused_multiq_kernel<<<grid, block, 0, stream>>>(
        A, Aw, Sa, W, B, Bw, Sb, R, Q, indices_r, indices_q, scale_inv, R_total, out_global);
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
