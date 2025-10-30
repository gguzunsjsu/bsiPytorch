// Optimized CUDA kernels for BSI operations - Phase 1 optimizations
// Major improvements:
// 1. Vectorized loads (128-bit) for better memory bandwidth
// 2. No division/modulo in hot loops
// 3. Coalesced memory access patterns
// 4. Block-level reductions without atomicAdd serialization
// 5. Batched query building

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>

// ============================================================================
// Utility Functions
// ============================================================================

__inline__ __device__ unsigned long long warp_reduce_sum_ull(unsigned long long v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__inline__ __device__ double warp_reduce_sum_double(double v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__inline__ __device__ double block_reduce_sum_double(double val) {
    __shared__ double smem[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;

    val = warp_reduce_sum_double(val);

    if (lane == 0) smem[wid] = val;
    __syncthreads();

    int num_warps = blockDim.x >> 5;
    val = (threadIdx.x < num_warps) ? smem[lane] : 0.0;

    if (wid == 0) val = warp_reduce_sum_double(val);

    return val;
}

// ============================================================================
// Optimized Dot Product Kernel - Main Workhorse
// ============================================================================

// Optimized kernel: processes one result per block
// Key optimizations:
// - Vectorized 128-bit loads (uint4)
// - Query slices loaded to shared memory once
// - No divisions in inner loop
// - Coalesced memory access
// - Single block reduction, no atomicAdd
extern "C" __global__
void popcount_weighted_optimized_kernel(
    const unsigned long long* __restrict__ A_words,  // Query: [Sa, W]
    const double* __restrict__ A_weights,            // [Sa]
    int Sa,
    int W,
    const unsigned long long* __restrict__ B_words,  // Keys: [R, Sb, W]
    const double* __restrict__ B_weights,            // [R, Sb]
    int Sb,
    int R,
    double scale_inv,
    double* __restrict__ out)                        // [R]
{
    int r = blockIdx.x;
    if (r >= R) return;

    // Shared memory layout: query weights + temp storage for popcount reduction
    extern __shared__ char smem_bytes[];
    double* Aw_shared = reinterpret_cast<double*>(smem_bytes);
    unsigned long long* smem_pop = reinterpret_cast<unsigned long long*>(
        smem_bytes + Sa * sizeof(double));

    for (int i = threadIdx.x; i < Sa; i += blockDim.x) {
        Aw_shared[i] = A_weights[i];
    }
    __syncthreads();

    // Each thread accumulates its portion
    double thread_sum = 0.0;

    // Process this key (r) against all query slices
    const unsigned long long* B_base = B_words + (size_t)r * Sb * W;
    const double* Bw_row = B_weights + (size_t)r * Sb;

    // Iterate over key slices
    for (int j = 0; j < Sb; ++j) {
        double wb = Bw_row[j];
        if (wb == 0.0) continue;

        const unsigned long long* B_slice = B_base + (size_t)j * W;

        // Iterate over query slices
        for (int i = 0; i < Sa; ++i) {
            double wa = Aw_shared[i];
            if (wa == 0.0) continue;

            const unsigned long long* A_slice = A_words + (size_t)i * W;

            // Popcount across W words with thread-level parallelism
            unsigned long long local_pop = 0ULL;

            // Vectorized path: process 4 words at once (128-bit loads)
            int W_vec4 = W / 4;
            const uint4* A_vec4 = reinterpret_cast<const uint4*>(A_slice);
            const uint4* B_vec4 = reinterpret_cast<const uint4*>(B_slice);

            for (int w4 = threadIdx.x; w4 < W_vec4; w4 += blockDim.x) {
                uint4 a4 = A_vec4[w4];
                uint4 b4 = B_vec4[w4];

                local_pop += __popcll(a4.x & b4.x);
                local_pop += __popcll(a4.y & b4.y);
                local_pop += __popcll(a4.z & b4.z);
                local_pop += __popcll(a4.w & b4.w);
            }

            // Handle remaining words (W % 4)
            int W_remainder_start = W_vec4 * 4;
            for (int w = W_remainder_start + threadIdx.x; w < W; w += blockDim.x) {
                local_pop += __popcll(A_slice[w] & B_slice[w]);
            }

            // Reduce popcount across block (all threads participate)
            local_pop = warp_reduce_sum_ull(local_pop);

            int lane = threadIdx.x & 31;
            int wid = threadIdx.x >> 5;

            if (lane == 0) smem_pop[wid] = local_pop;
            __syncthreads();

            // Final warp reduction
            int num_warps = (blockDim.x + 31) >> 5;
            local_pop = (threadIdx.x < num_warps) ? smem_pop[lane] : 0ULL;
            if (wid == 0) local_pop = warp_reduce_sum_ull(local_pop);

            // Only thread 0 has the final popcount, accumulate weighted sum
            if (threadIdx.x == 0) {
                thread_sum += (double)local_pop * wa * wb;
            }
        }
    }

    // Only thread 0 has accumulated the full sum
    if (threadIdx.x == 0) {
        out[r] = thread_sum * scale_inv;
    }
}

// ============================================================================
// Batched Query Builder - Build Multiple Queries in One Kernel
// ============================================================================

// Builds Q queries in parallel on GPU
// Input: [Q, d] float tensor
// Output: [Q, slices, words_per_slice] packed bitplanes
extern "C" __global__
void pack_queries_batched_kernel(
    const long long* __restrict__ values,  // [Q, n] quantized values
    int Q,                                   // Number of queries
    long long n,                            // Vector dimension
    int slices,
    int words_per_slice,
    unsigned long long value_mask,
    unsigned long long* __restrict__ out)  // [Q, slices, words_per_slice]
{
    // Each block handles one query
    int q = blockIdx.x;
    if (q >= Q) return;

    const long long* query_vals = values + (long long)q * n;
    unsigned long long* query_out = out + (size_t)q * slices * words_per_slice;

    // Each warp handles one word for all slices (ballot-based packing)
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;

    for (int word_idx = warp_id; word_idx < words_per_slice; word_idx += warps_per_block) {
        long long row0 = (long long)word_idx * 64LL + (long long)lane;
        long long row1 = row0 + 32LL;

        // Each warp packs all slices for this word
        for (int slice = 0; slice < slices; ++slice) {
            bool b0 = false, b1 = false;

            if (row0 < n) {
                unsigned long long v0 = (unsigned long long)(query_vals[row0]) & value_mask;
                b0 = ((v0 >> slice) & 1ULL) != 0ULL;
            }
            if (row1 < n) {
                unsigned long long v1 = (unsigned long long)(query_vals[row1]) & value_mask;
                b1 = ((v1 >> slice) & 1ULL) != 0ULL;
            }

            unsigned lo = __ballot_sync(0xffffffff, b0);
            unsigned hi = __ballot_sync(0xffffffff, b1);

            if (lane == 0) {
                unsigned long long word = (unsigned long long)lo | ((unsigned long long)hi << 32);
                query_out[(size_t)slice * words_per_slice + word_idx] = word;
            }
        }
    }
}

// ============================================================================
// Multi-Query Optimized Kernel
// ============================================================================

// Process multiple queries against keys in one kernel launch
// More efficient than launching per-query
extern "C" __global__
void popcount_multiquery_optimized_kernel(
    const unsigned long long* __restrict__ A_words,  // Queries: [Q, Sa, W]
    const double* __restrict__ A_weights,            // [Q, Sa]
    const int* __restrict__ Sa_array,                // [Q] slices per query
    int Q,                                             // Number of queries
    int W,
    const unsigned long long* __restrict__ B_words,  // Keys: [R, Sb, W]
    const double* __restrict__ B_weights,            // [R, Sb]
    int Sb,
    int R,
    double scale_inv,
    double* __restrict__ out)                        // [Q, R]
{
    // Grid: (Q * R) blocks, each block computes one (q, r) dot product
    int global_id = blockIdx.x;
    int q = global_id / R;
    int r = global_id % R;

    if (q >= Q || r >= R) return;

    int Sa = Sa_array[q];

    // Load query weights to shared memory
    extern __shared__ double smem[];
    double* Aw_shared = smem;

    const double* Aw_query = A_weights + (size_t)q * 64;  // Max 64 slices
    for (int i = threadIdx.x; i < Sa; i += blockDim.x) {
        Aw_shared[i] = Aw_query[i];
    }
    __syncthreads();

    double thread_sum = 0.0;

    const unsigned long long* A_base = A_words + (size_t)q * 64 * W;  // Max 64 slices
    const unsigned long long* B_base = B_words + (size_t)r * Sb * W;
    const double* Bw_row = B_weights + (size_t)r * Sb;

    for (int j = 0; j < Sb; ++j) {
        double wb = Bw_row[j];
        if (wb == 0.0) continue;

        const unsigned long long* B_slice = B_base + (size_t)j * W;

        for (int i = 0; i < Sa; ++i) {
            double wa = Aw_shared[i];
            if (wa == 0.0) continue;

            const unsigned long long* A_slice = A_base + (size_t)i * W;

            unsigned long long local_pop = 0ULL;

            int W_vec4 = W / 4;
            const uint4* A_vec4 = reinterpret_cast<const uint4*>(A_slice);
            const uint4* B_vec4 = reinterpret_cast<const uint4*>(B_slice);

            for (int w4 = threadIdx.x; w4 < W_vec4; w4 += blockDim.x) {
                uint4 a4 = A_vec4[w4];
                uint4 b4 = B_vec4[w4];

                local_pop += __popcll(a4.x & b4.x);
                local_pop += __popcll(a4.y & b4.y);
                local_pop += __popcll(a4.z & b4.z);
                local_pop += __popcll(a4.w & b4.w);
            }

            int W_remainder_start = W_vec4 * 4;
            for (int w = W_remainder_start + threadIdx.x; w < W; w += blockDim.x) {
                local_pop += __popcll(A_slice[w] & B_slice[w]);
            }

            local_pop = warp_reduce_sum_ull(local_pop);

            if ((threadIdx.x & 31) == 0) {
                thread_sum += (double)local_pop * wa * wb;
            }
        }
    }

    double block_sum = block_reduce_sum_double(thread_sum);

    if (threadIdx.x == 0) {
        out[(size_t)q * R + r] = block_sum * scale_inv;
    }
}

// ============================================================================
// Host Launcher Functions
// ============================================================================

extern "C" void launch_popcount_weighted_optimized(
    const unsigned long long* A_words,
    const double* A_weights,
    int Sa,
    int W,
    const unsigned long long* B_words,
    const double* B_weights,
    int Sb,
    int R,
    double scale_inv,
    double* out,
    cudaStream_t stream)
{
    if (R <= 0) return;

    // One block per result
    dim3 grid(R);
    // Use 256 threads per block (8 warps)
    dim3 block(256);

    // Shared memory: query weights + popcount reduction temp (32 warps max)
    size_t smem_size = Sa * sizeof(double) + 32 * sizeof(unsigned long long);

    popcount_weighted_optimized_kernel<<<grid, block, smem_size, stream>>>(
        A_words, A_weights, Sa, W, B_words, B_weights, Sb, R, scale_inv, out);
}

extern "C" void launch_pack_queries_batched(
    const long long* values,
    int Q,
    long long n,
    int slices,
    int words_per_slice,
    unsigned long long value_mask,
    unsigned long long* out,
    cudaStream_t stream)
{
    if (Q <= 0) return;

    // One block per query
    dim3 grid(Q);
    // 256 threads = 8 warps
    dim3 block(256);

    pack_queries_batched_kernel<<<grid, block, 0, stream>>>(
        values, Q, n, slices, words_per_slice, value_mask, out);
}

extern "C" void launch_popcount_multiquery_optimized(
    const unsigned long long* A_words,
    const double* A_weights,
    const int* Sa_array,
    int Q,
    int W,
    const unsigned long long* B_words,
    const double* B_weights,
    int Sb,
    int R,
    double scale_inv,
    double* out,
    cudaStream_t stream)
{
    if (Q <= 0 || R <= 0) return;

    // One block per (query, key) pair
    dim3 grid(Q * R);
    dim3 block(256);

    size_t smem_size = 64 * sizeof(double);  // Max 64 slices

    popcount_multiquery_optimized_kernel<<<grid, block, smem_size, stream>>>(
        A_words, A_weights, Sa_array, Q, W, B_words, B_weights, Sb, R, scale_inv, out);
}

// ============================================================================
// Note: scatter, ewah_decompress, and pack_bits_all_ballot are defined in
// bsi_cuda_kernels.cu (old file). We don't redefine them here to avoid
// linker errors. Both .cu files are compiled together.
// ============================================================================
