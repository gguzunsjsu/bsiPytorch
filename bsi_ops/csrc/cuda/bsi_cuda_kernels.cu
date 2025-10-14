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
