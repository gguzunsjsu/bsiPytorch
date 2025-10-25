// Minimal CUDA kernels for BSI operations (verbatim words popcount)
#include <cuda_runtime.h>
#include <stdint.h>

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

extern "C" __global__
void popcount_pairwise_kernel(
    const unsigned long long* __restrict__ A, // [Sa * W]
    const unsigned long long* __restrict__ B, // [Sb * W]
    int Sa, int Sb, int W,
    unsigned long long* __restrict__ out // [Sb * Sa] (row-major j,i)
) {
    int i = blockIdx.x;   // slice index in A
    int j = blockIdx.y;   // slice index in B
    if (i >= Sa || j >= Sb) return;

    const unsigned long long* a = A + (size_t)i * W;
    const unsigned long long* b = B + (size_t)j * W;

    unsigned long long partial = 0;
    for (int w = threadIdx.x; w < W; w += blockDim.x) {
        partial += __popcll(a[w] & b[w]);
    }

    // Reduce within block
    __shared__ unsigned long long smem[32];
    unsigned long long warp = warp_reduce_sum_ull(partial);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = warp;
    __syncthreads();

    unsigned long long block_sum = 0;
    int num_warps = blockDim.x >> 5;
    if (threadIdx.x < num_warps) block_sum = smem[threadIdx.x];
    block_sum = warp_reduce_sum_ull(block_sum);

    if (threadIdx.x == 0) {
        out[(size_t)j * Sa + i] = block_sum;
    }
}
// Simple scatter: out[idx[i]] = src[i]
extern "C" __global__
void scatter_set_double_kernel(
    const long long* __restrict__ idx,
    const double* __restrict__ src,
    int n,
    double* __restrict__ out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        long long j = idx[i];
        out[j] = src[i];
    }
}

extern "C" void launch_scatter_set_double(
    const long long* idx,
    const double* src,
    int n,
    double* out,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    scatter_set_double_kernel<<<blocks, threads, 0, stream>>>(idx, src, n, out);
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

// Host-callable launchers (called from C++ host code)
extern "C" void launch_popcount_pairwise(
    const unsigned long long* A,
    const unsigned long long* B,
    int Sa, int Sb, int W,
    unsigned long long* out,
    cudaStream_t stream) {
    dim3 grid(Sa, Sb);
    dim3 block(256);
    popcount_pairwise_kernel<<<grid, block, 0, stream>>>(A, B, Sa, Sb, W, out);
}

extern "C" __global__
void weighted_popcount_counts_kernel(
    const unsigned long long* __restrict__ A,
    const double* __restrict__ Aw,
    int Sa,
    int W,
    double* __restrict__ out)
{
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    if (w >= W) return;
    double total = 0.0;
    for (int i = 0; i < Sa; ++i) {
        const unsigned long long* ai = A + (size_t)i * W;
        total += static_cast<double>(__popcll(ai[w])) * Aw[i];
    }
    out[w] = total;
}

extern "C" void launch_weighted_popcount_counts(
    const unsigned long long* A,
    const double* Aw,
    int Sa,
    int W,
    double* out_counts,
    cudaStream_t stream)
{
    if (W <= 0) return;
    dim3 block(128);
    dim3 grid((W + block.x - 1) / block.x);
    weighted_popcount_counts_kernel<<<grid, block, 0, stream>>>(A, Aw, Sa, W, out_counts);
}

extern "C" __global__
void popcount_weighted_keys_hybrid_kernel(
    const unsigned long long* __restrict__ A,
    const double* __restrict__ Aw,
    const double* __restrict__ prefix,
    int Sa,
    int W,
    const unsigned long long* __restrict__ B_words,
    const unsigned long long* __restrict__ comp_words,
    const long long* __restrict__ comp_off_abs,
    const int* __restrict__ comp_len,
    const uint8_t* __restrict__ flags,
    const double* __restrict__ Bw,
    int Sb,
    int R,
    double scale_inv,
    double* __restrict__ out)
{
    int r = blockIdx.x;
    if (r >= R) return;

    extern __shared__ double shmem[];
    double* shAw = shmem;
    for (int t = threadIdx.x; t < Sa; t += blockDim.x) {
        shAw[t] = Aw[t];
    }
    __syncthreads();

    __shared__ double warp_buf_a[32];
    __shared__ double warp_buf_b[32];
    __shared__ double warp_buf_final[32];

    double local = 0.0;
    for (int j = 0; j < Sb; ++j) {
        double bw = Bw[(size_t)r * Sb + j];
        if (bw == 0.0) continue;
        uint8_t flag = flags[(size_t)r * Sb + j];
        if (!flag) {
            const unsigned long long* b = B_words + ((size_t)r * Sb + j) * W;
            for (int w = 0; w < W; ++w) {
                unsigned long long bword = b[w];
                double part = 0.0;
                for (int i = threadIdx.x; i < Sa; i += blockDim.x) {
                    const unsigned long long* ai = A + (size_t)i * W;
                    int cnt = __popcll(ai[w] & bword);
                    part += (double)cnt * shAw[i];
                }
                part = warp_reduce_sum_double(part);
                if ((threadIdx.x & 31) == 0) warp_buf_a[threadIdx.x >> 5] = part;
                __syncthreads();
                double sum = 0.0;
                if (threadIdx.x < 32) {
                    int nw = blockDim.x >> 5;
                    sum = (threadIdx.x < nw) ? warp_buf_a[threadIdx.x] : 0.0;
                    sum = warp_reduce_sum_double(sum);
                }
                if (threadIdx.x == 0) local += sum * bw;
                __syncthreads();
            }
        } else {
            if (!comp_words || !comp_off_abs || !comp_len) continue;
            long long off = comp_off_abs[(size_t)r * Sb + j];
            int len = comp_len[(size_t)r * Sb + j];
            const unsigned long long* cw = comp_words + off;
            int pos = 0;
            for (int idx = 0; idx < len;) {
                unsigned long long rlw = cw[idx++];
                int run_bit = (int)(rlw & 1ULL);
                int run_len = (int)((rlw >> 1) & ((1u << 32) - 1));
                int lit_cnt = (int)(rlw >> (1 + 32));
                if (run_bit) {
                    if (threadIdx.x == 0) {
                        int endp = min(pos + run_len, W);
                        local += (prefix[endp] - prefix[pos]) * bw;
                    }
                }
                pos += run_len;
                for (int k = 0; k < lit_cnt && idx < len && pos < W; ++k) {
                    unsigned long long bword = cw[idx++];
                    double part = 0.0;
                    for (int i = threadIdx.x; i < Sa; i += blockDim.x) {
                        const unsigned long long* ai = A + (size_t)i * W;
                        int cnt = __popcll(ai[pos] & bword);
                        part += (double)cnt * shAw[i];
                    }
                    part = warp_reduce_sum_double(part);
                    if ((threadIdx.x & 31) == 0) warp_buf_b[threadIdx.x >> 5] = part;
                    __syncthreads();
                    double sum = 0.0;
                    if (threadIdx.x < 32) {
                        int nw = blockDim.x >> 5;
                        sum = (threadIdx.x < nw) ? warp_buf_b[threadIdx.x] : 0.0;
                        sum = warp_reduce_sum_double(sum);
                    }
                    if (threadIdx.x == 0) local += sum * bw;
                    ++pos;
                    __syncthreads();
                }
            }
        }
    }

    local = warp_reduce_sum_double(local);
    if ((threadIdx.x & 31) == 0) warp_buf_final[threadIdx.x >> 5] = local;
    __syncthreads();
    double total = 0.0;
    if (threadIdx.x < 32) {
        int nw = blockDim.x >> 5;
        total = (threadIdx.x < nw) ? warp_buf_final[threadIdx.x] : 0.0;
        total = warp_reduce_sum_double(total);
    }
    if (threadIdx.x == 0) out[r] = total * scale_inv;
}

extern "C" void launch_popcount_weighted_keys_hybrid(
    const unsigned long long* A,
    const double* Aw,
    const double* prefix,
    int Sa,
    int W,
    const unsigned long long* B_words,
    const unsigned long long* comp_words,
    const long long* comp_off_abs,
    const int* comp_len,
    const uint8_t* flags,
    const double* Bw,
    int Sb,
    int R,
    double scale_inv,
    double* out,
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
    size_t shmem = static_cast<size_t>(Sa) * sizeof(double);
    dim3 grid(R);
    dim3 block(cached_block);
    popcount_weighted_keys_hybrid_kernel<<<grid, block, shmem, stream>>>(
        A, Aw, prefix, Sa, W, B_words, comp_words, comp_off_abs, comp_len, flags, Bw, Sb, R, scale_inv, out);
}

extern "C" __global__
void popcount_weighted_keys_hybrid_tiled_kernel(
    const unsigned long long* __restrict__ A,
    const double* __restrict__ Aw,
    const double* __restrict__ prefix,
    int Sa,
    int W,
    const unsigned long long* __restrict__ B_words,
    const unsigned long long* __restrict__ comp_words,
    const long long* __restrict__ comp_off_abs,
    const int* __restrict__ comp_len,
    const uint8_t* __restrict__ flags,
    const double* __restrict__ Bw,
    int Sb,
    int R,
    int jtile,
    double scale_inv,
    double* __restrict__ out)
{
    int r = blockIdx.x;
    if (r >= R) return;
    int tile = blockIdx.y;
    int j_begin = tile * jtile;
    int j_end = min(j_begin + jtile, Sb);
    if (j_begin >= Sb) return;

    extern __shared__ double shmem[];
    double* shAw = shmem;
    for (int t = threadIdx.x; t < Sa; t += blockDim.x) shAw[t] = Aw[t];
    __syncthreads();

    __shared__ double warp_buf_a[32];
    __shared__ double warp_buf_b[32];
    __shared__ double warp_buf_final[32];

    double local = 0.0;
    for (int j = j_begin; j < j_end; ++j) {
        double bw = Bw[(size_t)r * Sb + j];
        if (bw == 0.0) continue;
        uint8_t flag = flags[(size_t)r * Sb + j];
        if (!flag) {
            const unsigned long long* b = B_words + ((size_t)r * Sb + j) * W;
            for (int w = 0; w < W; ++w) {
                unsigned long long bword = b[w];
                double part = 0.0;
                for (int i = threadIdx.x; i < Sa; i += blockDim.x) {
                    const unsigned long long* ai = A + (size_t)i * W;
                    int cnt = __popcll(ai[w] & bword);
                    part += (double)cnt * shAw[i];
                }
                part = warp_reduce_sum_double(part);
                if ((threadIdx.x & 31) == 0) warp_buf_a[threadIdx.x >> 5] = part;
                __syncthreads();
                double sum = 0.0;
                if (threadIdx.x < 32) {
                    int nw = blockDim.x >> 5;
                    sum = (threadIdx.x < nw) ? warp_buf_a[threadIdx.x] : 0.0;
                    sum = warp_reduce_sum_double(sum);
                }
                if (threadIdx.x == 0) local += sum * bw;
                __syncthreads();
            }
        } else {
            if (!comp_words || !comp_off_abs || !comp_len) continue;
            long long off = comp_off_abs[(size_t)r * Sb + j];
            int len = comp_len[(size_t)r * Sb + j];
            const unsigned long long* cw = comp_words + off;
            int pos = 0;
            for (int idx = 0; idx < len;) {
                unsigned long long rlw = cw[idx++];
                int run_bit = (int)(rlw & 1ULL);
                int run_len = (int)((rlw >> 1) & ((1u << 32) - 1));
                int lit_cnt = (int)(rlw >> (1 + 32));
                if (run_bit) {
                    if (threadIdx.x == 0) {
                        int endp = min(pos + run_len, W);
                        local += (prefix[endp] - prefix[pos]) * bw;
                    }
                }
                pos += run_len;
                for (int k = 0; k < lit_cnt && idx < len && pos < W; ++k) {
                    unsigned long long bword = cw[idx++];
                    double part = 0.0;
                    for (int i = threadIdx.x; i < Sa; i += blockDim.x) {
                        const unsigned long long* ai = A + (size_t)i * W;
                        int cnt = __popcll(ai[pos] & bword);
                        part += (double)cnt * shAw[i];
                    }
                    part = warp_reduce_sum_double(part);
                    if ((threadIdx.x & 31) == 0) warp_buf_b[threadIdx.x >> 5] = part;
                    __syncthreads();
                    double sum = 0.0;
                    if (threadIdx.x < 32) {
                        int nw = blockDim.x >> 5;
                        sum = (threadIdx.x < nw) ? warp_buf_b[threadIdx.x] : 0.0;
                        sum = warp_reduce_sum_double(sum);
                    }
                    if (threadIdx.x == 0) local += sum * bw;
                    ++pos;
                    __syncthreads();
                }
            }
        }
    }

    local = warp_reduce_sum_double(local);
    if ((threadIdx.x & 31) == 0) warp_buf_final[threadIdx.x >> 5] = local;
    __syncthreads();
    double total = 0.0;
    if (threadIdx.x < 32) {
        int nw = blockDim.x >> 5;
        total = (threadIdx.x < nw) ? warp_buf_final[threadIdx.x] : 0.0;
        total = warp_reduce_sum_double(total);
    }
    if (threadIdx.x == 0) atomicAdd(&out[r], total * scale_inv);
}

extern "C" void launch_popcount_weighted_keys_hybrid_tiled(
    const unsigned long long* A,
    const double* Aw,
    const double* prefix,
    int Sa,
    int W,
    const unsigned long long* B_words,
    const unsigned long long* comp_words,
    const long long* comp_off_abs,
    const int* comp_len,
    const uint8_t* flags,
    const double* Bw,
    int Sb,
    int R,
    int jtile,
    double scale_inv,
    double* out,
    cudaStream_t stream)
{
    if (jtile <= 0) {
        launch_popcount_weighted_keys_hybrid(A, Aw, prefix, Sa, W, B_words, comp_words, comp_off_abs, comp_len, flags, Bw, Sb, R, scale_inv, out, stream);
        return;
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
    int tiles = (Sb + jtile - 1) / jtile;
    size_t shmem = static_cast<size_t>(Sa) * sizeof(double);
    dim3 grid(R, tiles);
    dim3 block(cached_block);
    popcount_weighted_keys_hybrid_tiled_kernel<<<grid, block, shmem, stream>>>(
        A, Aw, prefix, Sa, W, B_words, comp_words, comp_off_abs, comp_len, flags, Bw, Sb, R, jtile, scale_inv, out);
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
