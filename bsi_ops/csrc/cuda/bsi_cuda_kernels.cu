// Minimal CUDA kernels for BSI operations (verbatim words popcount)
#include <cuda_runtime.h>
#include <stdint.h>

__inline__ __device__ unsigned long long warp_reduce_sum_ull(unsigned long long v) {
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

extern "C" __global__
void popcount_weighted_kernel(
    const unsigned long long* __restrict__ A, // [Sa * W]
    const double* __restrict__ Aw,            // [Sa]
    int Sa, int W,
    const unsigned long long* __restrict__ B, // [Sb * W]
    const double* __restrict__ Bw,            // [Sb]
    int Sb,
    double* __restrict__ out)
{
    int i = blockIdx.x;
    int j = blockIdx.y;
    if (i >= Sa || j >= Sb) return;

    const unsigned long long* a = A + (size_t)i * W;
    const unsigned long long* b = B + (size_t)j * W;

    unsigned long long partial = 0;
    for (int w = threadIdx.x; w < W; w += blockDim.x) {
        partial += __popcll(a[w] & b[w]);
    }

    __shared__ unsigned long long smem[32];
    unsigned long long warp = warp_reduce_sum_ull(partial);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = warp;
    __syncthreads();

    unsigned long long block_sum = 0;
    int num_warps = blockDim.x >> 5;
    if (threadIdx.x < num_warps) block_sum = smem[threadIdx.x];
    block_sum = warp_reduce_sum_ull(block_sum);

    if (threadIdx.x == 0) {
        double contrib = static_cast<double>(block_sum) * Aw[i] * Bw[j];
        atomicAdd(out, contrib);
    }
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

extern "C" void launch_popcount_weighted(
    const unsigned long long* A,
    const double* Aw,
    int Sa, int W,
    const unsigned long long* B,
    const double* Bw,
    int Sb,
    double* out,
    cudaStream_t stream) {
    dim3 grid(Sa, Sb);
    dim3 block(256);
    popcount_weighted_kernel<<<grid, block, 0, stream>>>(A, Aw, Sa, W, B, Bw, Sb, out);
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

// -----------------------
// GPU EWAH compression (per-slice sequential emitter into 2*W staging)

__device__ inline bool is_clean_word(unsigned long long w, bool& bit) {
    if (w == 0ULL) { bit = false; return true; }
    if (w == ~0ULL) { bit = true; return true; }
    return false;
}

extern "C" __global__
void ewah_compress_slices_kernel(
    const unsigned long long* __restrict__ words, // [S * W]
    int S,
    int W,
    unsigned long long* __restrict__ out_tmp,     // [S * (2*W)]
    int tmp_stride,                               // = 2*W
    int* __restrict__ out_lengths,                // [S]
    unsigned long long* __restrict__ out_stats)   // [S*2]: run_words, lit_words
{
    int s = blockIdx.x;
    if (s >= S) return;
    if (threadIdx.x != 0) return; // sequential per-slice emitter

    const unsigned long long* in = words + (size_t)s * W;
    unsigned long long* out = out_tmp + (size_t)s * tmp_stride;

    // EWAH fields for u64: runninglengthbits=32, literalbits=31, bit0 = running_bit
    const unsigned long long RUNLEN_MAX = (1ULL << 32) - 1ULL;
    const unsigned long long LITCNT_MAX = (1ULL << 31) - 1ULL;

    int out_idx = 0;
    unsigned long long run_words = 0;
    unsigned long long lit_words = 0;

    unsigned long long rlw = 0ULL;
    unsigned long long cur_run_bit = 0ULL; // 0 or 1
    unsigned long long run_len = 0ULL;
    unsigned long long litcnt = 0ULL;
    bool have_rlw = false;

    auto flush_rlw = [&]() {
        // write RLW header if we opened one
        if (!have_rlw) return;
        // pack: bit0=cur_run_bit, bits[1..32]=run_len, bits[33..63]=litcnt
        unsigned long long hdr = 0ULL;
        if (cur_run_bit) hdr |= 1ULL;
        hdr |= (run_len & RUNLEN_MAX) << 1;
        hdr |= (litcnt & LITCNT_MAX) << (1 + 32);
        out[out_idx++] = hdr;
        run_words += run_len;
        have_rlw = false;
        run_len = 0ULL; litcnt = 0ULL; cur_run_bit = 0ULL;
    };

    auto start_rlw = [&](unsigned long long bit) {
        rlw = 0ULL; cur_run_bit = bit; run_len = 0ULL; litcnt = 0ULL; have_rlw = true;
    };

    int i = 0;
    while (i < W) {
        unsigned long long w = in[i];
        bool clean; bool bit = false;
        clean = is_clean_word(w, bit);
        if (clean) {
            // If we have no RLW open or run_bit changed or litcnt already started, flush and start a new RLW
            if (!have_rlw || litcnt > 0 || bit != (cur_run_bit != 0ULL)) {
                flush_rlw();
                start_rlw(bit ? 1ULL : 0ULL);
            }
            // Accumulate run, splitting if exceeds max
            while (i < W) {
                unsigned long long w2 = in[i]; bool bit2=false; bool c2 = is_clean_word(w2, bit2);
                if (!c2 || bit2 != (cur_run_bit != 0ULL)) break;
                if (run_len == RUNLEN_MAX) {
                    // RLW full, flush it
                    flush_rlw();
                    start_rlw(bit ? 1ULL : 0ULL);
                }
                ++run_len; ++i;
            }
            continue; // loop continues; do not advance i here (done in while)
        }

        // literal word
        if (!have_rlw) start_rlw(0ULL); // run_len=0 allowed
        // Append literal, splitting if litcnt full
        if (litcnt == LITCNT_MAX) {
            flush_rlw();
            start_rlw(0ULL);
        }
        ++litcnt;
        out[out_idx++] = w; // literal word follows its RLW
        ++lit_words;
        ++i;
        // after literals, next iteration will handle clean/literal transitions
    }

    // done, flush any pending RLW
    flush_rlw();

    out_lengths[s] = out_idx;
    if (out_stats) {
        out_stats[(size_t)s * 2 + 0] = run_words;
        out_stats[(size_t)s * 2 + 1] = lit_words;
    }
}

extern "C" void launch_ewah_compress(
    const unsigned long long* words,
    int S, int W,
    unsigned long long* out_tmp,
    int tmp_stride,
    int* out_lengths,
    unsigned long long* out_stats,
    cudaStream_t stream)
{
    dim3 grid(S);
    dim3 block(1);
    ewah_compress_slices_kernel<<<grid, block, 0, stream>>>(
        words, S, W, out_tmp, tmp_stride, out_lengths, out_stats);
}

// Compact copy from per-slice staging to flat buffer according to offsets/lengths
extern "C" __global__
void compact_copy_kernel(
    const unsigned long long* __restrict__ tmp,
    int tmp_stride,
    const int* __restrict__ lengths,
    const int* __restrict__ offsets,
    int S,
    unsigned long long* __restrict__ out)
{
    int s = blockIdx.x;
    if (s >= S) return;
    int len = lengths[s];
    int off = offsets[s];
    const unsigned long long* src = tmp + (size_t)s * tmp_stride;
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        out[(size_t)off + i] = src[i];
    }
}

extern "C" void launch_compact_copy(
    const unsigned long long* tmp,
    int tmp_stride,
    const int* lengths,
    const int* offsets,
    int S,
    unsigned long long* out,
    cudaStream_t stream)
{
    dim3 grid(S);
    dim3 block(128);
    compact_copy_kernel<<<grid, block, 0, stream>>>(tmp, tmp_stride, lengths, offsets, S, out);
}

// Decompress many slices using RLW input (one block per slice, sequential decode)
extern "C" __global__
void ewah_decompress_slices_kernel(
    const unsigned long long* __restrict__ in,
    const int* __restrict__ offsets,
    const int* __restrict__ lengths,
    int S,
    int W,
    unsigned long long* __restrict__ out)
{
    int s = blockIdx.x;
    if (s >= S) return;
    if (threadIdx.x != 0) return;
    int off = offsets[s];
    int len = lengths[s];
    const unsigned long long* ptr = in + (size_t)off;
    unsigned long long* dst = out + (size_t)s * W;
    // reuse one-slice decoder
    const unsigned long long RUNLEN_MASK = (1ULL << 32) - 1ULL; // 32 bits for runlen
    int idx = 0;
    int out_idx = 0;
    while (idx < len && out_idx < W) {
        unsigned long long rlw = ptr[idx++];
        bool running_bit = (rlw & 1ULL) != 0ULL;
        unsigned int run_len = (unsigned int)((rlw >> 1) & RUNLEN_MASK);
        unsigned int lit_words = (unsigned int)(rlw >> (1 + 32));
        unsigned long long run_val = running_bit ? ~0ULL : 0ULL;
        for (unsigned int k = 0; k < run_len && out_idx < W; ++k) dst[out_idx++] = run_val;
        for (unsigned int k = 0; k < lit_words && out_idx < W && idx < len; ++k) dst[out_idx++] = ptr[idx++];
    }
    while (out_idx < W) dst[out_idx++] = 0ULL;
}

extern "C" void launch_ewah_decompress_slices(
    const unsigned long long* in,
    const int* offsets,
    const int* lengths,
    int S,
    int W,
    unsigned long long* out,
    cudaStream_t stream)
{
    dim3 grid(S);
    dim3 block(1);
    ewah_decompress_slices_kernel<<<grid, block, 0, stream>>>(in, offsets, lengths, S, W, out);
}
