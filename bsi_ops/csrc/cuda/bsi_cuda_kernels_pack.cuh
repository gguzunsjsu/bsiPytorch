#pragma once

#include <stdint.h>

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

extern "C" __global__
void pack_bits_all_ballot_multi_kernel_oneshot(
    const long long* __restrict__ values,
    long long n,
    int slices,
    int words_per_slice,
    unsigned long long value_mask,
    unsigned long long* __restrict__ out)
{
    const int warps_per_block = blockDim.x >> 5;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int words_group = blockIdx.x;
    const int word_idx = words_group * warps_per_block + warp;
    if (word_idx >= words_per_slice) return;

    const long long row0 = (long long)word_idx * 64LL + (long long)lane;
    const long long row1 = row0 + 32LL;

    unsigned long long v0 = 0ULL;
    unsigned long long v1 = 0ULL;
    if (row0 < n) {
        v0 = (static_cast<unsigned long long>(values[row0]) & value_mask);
    }
    if (row1 < n) {
        v1 = (static_cast<unsigned long long>(values[row1]) & value_mask);
    }

    for (int slice = 0; slice < slices; ++slice) {
        const bool b0 = ((v0 >> slice) & 1ULL) != 0ULL;
        const bool b1 = ((v1 >> slice) & 1ULL) != 0ULL;
        const unsigned lo = __ballot_sync(0xffffffff, b0);
        const unsigned hi = __ballot_sync(0xffffffff, b1);
        if (lane == 0) {
            const unsigned long long word = (unsigned long long)lo | ((unsigned long long)hi << 32);
            out[(size_t)slice * (size_t)words_per_slice + (size_t)word_idx] = word;
        }
    }
}

extern "C" __global__
void pack_bits_all_ballot_multi_kernel_batch(
    const long long* __restrict__ values,
    int Q,
    long long n,
    int slices,
    int words_per_slice,
    unsigned long long value_mask,
    unsigned long long* __restrict__ out)
{
    int q = blockIdx.z;
    if (q >= Q) return;
    int slice = blockIdx.y;
    if (slice >= slices) return;

    int warps_per_block = blockDim.x >> 5; // blockDim.x / 32
    int warp = threadIdx.x >> 5;           // 0..warps_per_block-1
    int lane = threadIdx.x & 31;           // 0..31
    int words_group = blockIdx.x;
    int word_idx = words_group * warps_per_block + warp;
    if (word_idx >= words_per_slice) return;

    const long long* vals = values + (long long)q * n;
    unsigned long long* out_q = out + ((size_t)q * (size_t)slices * (size_t)words_per_slice);

    long long row0 = (long long)word_idx * 64LL + (long long)lane;
    long long row1 = row0 + 32LL;

    bool b0 = false, b1 = false;
    if (row0 < n) {
        unsigned long long v0 = (static_cast<unsigned long long>(vals[row0]) & value_mask);
        b0 = ((v0 >> slice) & 1ULL) != 0ULL;
    }
    if (row1 < n) {
        unsigned long long v1 = (static_cast<unsigned long long>(vals[row1]) & value_mask);
        b1 = ((v1 >> slice) & 1ULL) != 0ULL;
    }
    unsigned lo = __ballot_sync(0xffffffff, b0);
    unsigned hi = __ballot_sync(0xffffffff, b1);
    if (lane == 0) {
        unsigned long long word = (unsigned long long)lo | ((unsigned long long)hi << 32);
        out_q[(size_t)slice * (size_t)words_per_slice + (size_t)word_idx] = word;
    }
}

extern "C" __global__
void pack_bits_all_ballot_multi_kernel_batch_oneshot(
    const long long* __restrict__ values,
    int Q,
    long long n,
    int slices,
    int words_per_slice,
    unsigned long long value_mask,
    unsigned long long* __restrict__ out)
{
    const int q = blockIdx.z;
    if (q >= Q) return;

    const int warps_per_block = blockDim.x >> 5;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int words_group = blockIdx.x;
    const int word_idx = words_group * warps_per_block + warp;
    if (word_idx >= words_per_slice) return;

    const long long* vals = values + (long long)q * n;
    unsigned long long* out_q = out + ((size_t)q * (size_t)slices * (size_t)words_per_slice);

    const long long row0 = (long long)word_idx * 64LL + (long long)lane;
    const long long row1 = row0 + 32LL;

    unsigned long long v0 = 0ULL;
    unsigned long long v1 = 0ULL;
    if (row0 < n) {
        v0 = (static_cast<unsigned long long>(vals[row0]) & value_mask);
    }
    if (row1 < n) {
        v1 = (static_cast<unsigned long long>(vals[row1]) & value_mask);
    }

    for (int slice = 0; slice < slices; ++slice) {
        const bool b0 = ((v0 >> slice) & 1ULL) != 0ULL;
        const bool b1 = ((v1 >> slice) & 1ULL) != 0ULL;
        const unsigned lo = __ballot_sync(0xffffffff, b0);
        const unsigned hi = __ballot_sync(0xffffffff, b1);
        if (lane == 0) {
            const unsigned long long word = (unsigned long long)lo | ((unsigned long long)hi << 32);
            out_q[(size_t)slice * (size_t)words_per_slice + (size_t)word_idx] = word;
        }
    }
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
    // Use 8 warps (256 threads) per block by default.
    const int warps_per_block = 8;
    dim3 block(warps_per_block * 32);
    dim3 grid((words_per_slice + warps_per_block - 1) / warps_per_block);
    int use_oneshot = 1;
    if (const char* s = getenv("BSI_PACK_ONESHOT")) {
        use_oneshot = (atoi(s) != 0) ? 1 : 0;
    }
    if (use_oneshot) {
        pack_bits_all_ballot_multi_kernel_oneshot<<<grid, block, 0, stream>>>(
            values, n, slices, words_per_slice, value_mask, out);
    } else {
        dim3 grid_legacy((words_per_slice + warps_per_block - 1) / warps_per_block, slices);
        pack_bits_all_ballot_multi_kernel<<<grid_legacy, block, 0, stream>>>(
            values, n, slices, words_per_slice, value_mask, out);
    }
}

extern "C" void launch_pack_bits_all_ballot_batch(
    const long long* values,
    int Q,
    long long n,
    int slices,
    int words_per_slice,
    unsigned long long value_mask,
    unsigned long long* out,
    cudaStream_t stream)
{
    if (Q <= 0 || slices <= 0 || words_per_slice <= 0) return;
    const int warps_per_block = 8;
    dim3 block(warps_per_block * 32);
    dim3 grid((words_per_slice + warps_per_block - 1) / warps_per_block, 1, Q);
    int use_oneshot = 1;
    if (const char* s = getenv("BSI_PACK_ONESHOT")) {
        use_oneshot = (atoi(s) != 0) ? 1 : 0;
    }
    if (use_oneshot) {
        pack_bits_all_ballot_multi_kernel_batch_oneshot<<<grid, block, 0, stream>>>(
            values, Q, n, slices, words_per_slice, value_mask, out);
    } else {
        dim3 grid_legacy((words_per_slice + warps_per_block - 1) / warps_per_block, slices, Q);
        pack_bits_all_ballot_multi_kernel_batch<<<grid_legacy, block, 0, stream>>>(
            values, Q, n, slices, words_per_slice, value_mask, out);
    }
}

extern "C" __global__
void bsi_words_to_tc_layout_kernel(
    const unsigned long long* __restrict__ in_words, // [rows, S, W64]
    int rows,
    int S,
    int W64,
    uint32_t* __restrict__ out_words_tc) // [rows, S, chunks, 12]
{
    const int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int total = rows * S * W64;
    if (idx >= total) return;
    const int w64_i = idx % W64;
    int t = idx / W64;
    const int s = t % S;
    const int row = t / S;
    const int chunks = W64 >> 2;
    const int chunk = w64_i >> 2;
    const int within = w64_i & 3;

    const unsigned long long w = in_words[idx];
    const uint32_t lo = static_cast<uint32_t>(w);
    const uint32_t hi = static_cast<uint32_t>(w >> 32);
    const int base = (((row * S + s) * chunks + chunk) * 12) + (within << 1);
    out_words_tc[base] = lo;
    out_words_tc[base + 1] = hi;
}

extern "C" void launch_bsi_words_to_tc_layout(
    const unsigned long long* in_words,
    int rows,
    int S,
    int W64,
    uint32_t* out_words_tc,
    cudaStream_t stream)
{
    if (rows <= 0 || S <= 0 || W64 <= 0) return;
    if ((W64 & 3) != 0) return;
    const int total = rows * S * W64;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    bsi_words_to_tc_layout_kernel<<<blocks, threads, 0, stream>>>(
        in_words, rows, S, W64, out_words_tc);
}
