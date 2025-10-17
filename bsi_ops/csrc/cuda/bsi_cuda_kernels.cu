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

// Batched: compute R keys in a single launch
// A: [Sa, W], Aw: [Sa]
// B: [R, Sb, W], Bw: [R, Sb]
// out: [R]
extern "C" __global__
void popcount_weighted_batch_kernel(
    const unsigned long long* __restrict__ A,
    const double* __restrict__ Aw,
    int Sa, int W,
    const unsigned long long* __restrict__ B,
    const double* __restrict__ Bw,
    int Sb,
    int R,
    double* __restrict__ out)
{
    int i = blockIdx.x; // slice in A [0..Sa)
    int j = blockIdx.y; // slice in B [0..Sb)
    int r = blockIdx.z; // key index [0..R)
    if (i >= Sa || j >= Sb || r >= R) return;

    const unsigned long long* a = A + (size_t)i * W;
    const unsigned long long* b = B + ((size_t)r * Sb + j) * W;

    unsigned long long partial = 0ULL;
    for (int w = threadIdx.x; w < W; w += blockDim.x) {
        partial += __popcll(a[w] & b[w]);
    }

    __shared__ unsigned long long smem[32];
    unsigned long long warp = warp_reduce_sum_ull(partial);
    if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = warp;
    __syncthreads();

    unsigned long long block_sum = 0ULL;
    int num_warps = blockDim.x >> 5;
    if (threadIdx.x < num_warps) block_sum = smem[threadIdx.x];
    block_sum = warp_reduce_sum_ull(block_sum);

    if (threadIdx.x == 0) {
        double contrib = static_cast<double>(block_sum) * Aw[i] * Bw[(size_t)r * Sb + j];
        atomicAdd(&out[r], contrib);
    }
}

extern "C" void launch_popcount_weighted_batch(
    const unsigned long long* A,
    const double* Aw,
    int Sa, int W,
    const unsigned long long* B,
    const double* Bw,
    int Sb,
    int R,
    double* out,
    cudaStream_t stream)
{
    dim3 grid(Sa, Sb, R);
    dim3 block(256);
    popcount_weighted_batch_kernel<<<grid, block, 0, stream>>>(A, Aw, Sa, W, B, Bw, Sb, R, out);
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
