#pragma once

#include <stdint.h>
#include "bsi_word_config.h"

// ---------------------------------------------------------------------------
// Helper: assemble one bsi_word_t from ballot results.
//
// For kBsiWordBits >= 32, each word needs kBsiWordBits/32 ballot calls.
// For kBsiWordBits < 32, one ballot produces 32/kBsiWordBits words; the
// caller selects which sub-word via `sub_idx`.
//
// `values`: quantized int64 array, `n`: total rows
// `value_mask`: mask for valid slices, `slice`: current slice
// `base_row`: first row for this word (= word_idx * kBsiWordBits)
// `lane`: lane within warp (0..31)
// ---------------------------------------------------------------------------

#if BSI_WORD_BITS >= 64

// --- Large words (64, 128, 256): multiple ballots per word ---

__device__ __forceinline__
bsi_word_t bsi_pack_word(const long long* __restrict__ values,
                         long long n,
                         unsigned long long value_mask,
                         int slice,
                         long long base_row,
                         int lane)
{
  #if BSI_WORD_BITS == 64
    // Two ballots -> lo | (hi << 32)
    long long r0 = base_row + lane;
    long long r1 = r0 + 32LL;
    bool b0 = false, b1 = false;
    if (r0 < n) {
        unsigned long long v = static_cast<unsigned long long>(values[r0]) & value_mask;
        b0 = ((v >> slice) & 1ULL) != 0ULL;
    }
    if (r1 < n) {
        unsigned long long v = static_cast<unsigned long long>(values[r1]) & value_mask;
        b1 = ((v >> slice) & 1ULL) != 0ULL;
    }
    unsigned lo = __ballot_sync(0xffffffff, b0);
    unsigned hi = __ballot_sync(0xffffffff, b1);
    return static_cast<bsi_word_t>(lo) | (static_cast<bsi_word_t>(hi) << 32);

  #else // 128 or 256: kBsiBallotsPerWord ballots, combine into uint64 parts
    bsi_word_t word;
    #pragma unroll
    for (int p = 0; p < kBsiWordParts; ++p) {
        long long r0 = base_row + (long long)(p * 64) + lane;
        long long r1 = r0 + 32LL;
        bool b0 = false, b1 = false;
        if (r0 < n) {
            unsigned long long v = static_cast<unsigned long long>(values[r0]) & value_mask;
            b0 = ((v >> slice) & 1ULL) != 0ULL;
        }
        if (r1 < n) {
            unsigned long long v = static_cast<unsigned long long>(values[r1]) & value_mask;
            b1 = ((v >> slice) & 1ULL) != 0ULL;
        }
        unsigned lo = __ballot_sync(0xffffffff, b0);
        unsigned hi = __ballot_sync(0xffffffff, b1);
        word.parts[p] = static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
    }
    return word;
  #endif
}

#elif BSI_WORD_BITS == 32

// --- 32-bit: one ballot = one word ---

__device__ __forceinline__
bsi_word_t bsi_pack_word(const long long* __restrict__ values,
                         long long n,
                         unsigned long long value_mask,
                         int slice,
                         long long base_row,
                         int lane)
{
    long long r = base_row + lane;
    bool b = false;
    if (r < n) {
        unsigned long long v = static_cast<unsigned long long>(values[r]) & value_mask;
        b = ((v >> slice) & 1ULL) != 0ULL;
    }
    return __ballot_sync(0xffffffff, b);
}

#else // BSI_WORD_BITS < 32 (8 or 16)

// --- Small words (8, 16): one ballot -> multiple words ---
// Returns a single 32-bit ballot result. The caller extracts sub-words.

__device__ __forceinline__
unsigned int bsi_pack_ballot32(const long long* __restrict__ values,
                               long long n,
                               unsigned long long value_mask,
                               int slice,
                               long long base_row,
                               int lane)
{
    long long r = base_row + lane;
    bool b = false;
    if (r < n) {
        unsigned long long v = static_cast<unsigned long long>(values[r]) & value_mask;
        b = ((v >> slice) & 1ULL) != 0ULL;
    }
    return __ballot_sync(0xffffffff, b);
}

// Extract sub-word from a 32-bit ballot result.
__device__ __forceinline__
bsi_word_t bsi_extract_subword(unsigned int ballot, int sub_idx) {
  #if BSI_WORD_BITS == 8
    return static_cast<bsi_word_t>((ballot >> (sub_idx * 8)) & 0xFFu);
  #elif BSI_WORD_BITS == 16
    return static_cast<bsi_word_t>((ballot >> (sub_idx * 16)) & 0xFFFFu);
  #endif
}

#endif // BSI_WORD_BITS

// ===========================================================================
// Kernel: per-slice grid (slice = blockIdx.y)
// ===========================================================================
extern "C" __global__
void pack_bits_all_ballot_multi_kernel(
    const long long* __restrict__ values,
    long long n,
    int slices,
    int words_per_slice,
    unsigned long long value_mask,
    bsi_word_t* __restrict__ out)
{
    int slice = blockIdx.y;
    if (slice >= slices) return;

    int warps_per_block = blockDim.x >> 5;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

#if BSI_WORD_BITS >= 32
    int word_idx = blockIdx.x * warps_per_block + warp;
    if (word_idx >= words_per_slice) return;

    long long base_row = (long long)word_idx * (long long)kBsiWordBits;
    bsi_word_t word = bsi_pack_word(values, n, value_mask, slice, base_row, lane);
    if (lane == 0) {
        out[(size_t)slice * words_per_slice + word_idx] = word;
    }
#else
    // Small words: each warp processes one group of 32 rows -> kBsiWordsPerBallot words
    int group_idx = blockIdx.x * warps_per_block + warp;
    int first_word = group_idx * kBsiWordsPerBallot;
    if (first_word >= words_per_slice) return;

    long long base_row = (long long)group_idx * 32LL;
    unsigned int ballot = bsi_pack_ballot32(values, n, value_mask, slice, base_row, lane);
    if (lane == 0) {
        #pragma unroll
        for (int s = 0; s < kBsiWordsPerBallot; ++s) {
            int wi = first_word + s;
            if (wi < words_per_slice) {
                out[(size_t)slice * words_per_slice + wi] = bsi_extract_subword(ballot, s);
            }
        }
    }
#endif
}

// ===========================================================================
// Kernel: oneshot (loop over slices within kernel)
// ===========================================================================
extern "C" __global__
void pack_bits_all_ballot_multi_kernel_oneshot(
    const long long* __restrict__ values,
    long long n,
    int slices,
    int words_per_slice,
    unsigned long long value_mask,
    bsi_word_t* __restrict__ out)
{
    const int warps_per_block = blockDim.x >> 5;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

#if BSI_WORD_BITS >= 32
    const int word_idx = blockIdx.x * warps_per_block + warp;
    if (word_idx >= words_per_slice) return;

    const long long base_row = (long long)word_idx * (long long)kBsiWordBits;

    // Pre-load values for all slices (reuse across slice loop)
    // For large words we still iterate ballots inside bsi_pack_word per slice.
    for (int slice = 0; slice < slices; ++slice) {
        bsi_word_t word = bsi_pack_word(values, n, value_mask, slice, base_row, lane);
        if (lane == 0) {
            out[(size_t)slice * (size_t)words_per_slice + (size_t)word_idx] = word;
        }
    }
#else
    const int group_idx = blockIdx.x * warps_per_block + warp;
    const int first_word = group_idx * kBsiWordsPerBallot;
    if (first_word >= words_per_slice) return;

    const long long base_row = (long long)group_idx * 32LL;

    for (int slice = 0; slice < slices; ++slice) {
        unsigned int ballot = bsi_pack_ballot32(values, n, value_mask, slice, base_row, lane);
        if (lane == 0) {
            #pragma unroll
            for (int s = 0; s < kBsiWordsPerBallot; ++s) {
                int wi = first_word + s;
                if (wi < words_per_slice) {
                    out[(size_t)slice * words_per_slice + wi] = bsi_extract_subword(ballot, s);
                }
            }
        }
    }
#endif
}

// ===========================================================================
// Kernel: batch (Q queries, slice = blockIdx.y, q = blockIdx.z)
// ===========================================================================
extern "C" __global__
void pack_bits_all_ballot_multi_kernel_batch(
    const long long* __restrict__ values,
    int Q,
    long long n,
    int slices,
    int words_per_slice,
    unsigned long long value_mask,
    bsi_word_t* __restrict__ out)
{
    int q = blockIdx.z;
    if (q >= Q) return;
    int slice = blockIdx.y;
    if (slice >= slices) return;

    int warps_per_block = blockDim.x >> 5;
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    const long long* vals = values + (long long)q * n;
    bsi_word_t* out_q = out + ((size_t)q * (size_t)slices * (size_t)words_per_slice);

#if BSI_WORD_BITS >= 32
    int word_idx = blockIdx.x * warps_per_block + warp;
    if (word_idx >= words_per_slice) return;

    long long base_row = (long long)word_idx * (long long)kBsiWordBits;
    bsi_word_t word = bsi_pack_word(vals, n, value_mask, slice, base_row, lane);
    if (lane == 0) {
        out_q[(size_t)slice * (size_t)words_per_slice + (size_t)word_idx] = word;
    }
#else
    int group_idx = blockIdx.x * warps_per_block + warp;
    int first_word = group_idx * kBsiWordsPerBallot;
    if (first_word >= words_per_slice) return;

    long long base_row = (long long)group_idx * 32LL;
    unsigned int ballot = bsi_pack_ballot32(vals, n, value_mask, slice, base_row, lane);
    if (lane == 0) {
        #pragma unroll
        for (int s = 0; s < kBsiWordsPerBallot; ++s) {
            int wi = first_word + s;
            if (wi < words_per_slice) {
                out_q[(size_t)slice * words_per_slice + wi] = bsi_extract_subword(ballot, s);
            }
        }
    }
#endif
}

// ===========================================================================
// Kernel: batch oneshot
// ===========================================================================
extern "C" __global__
void pack_bits_all_ballot_multi_kernel_batch_oneshot(
    const long long* __restrict__ values,
    int Q,
    long long n,
    int slices,
    int words_per_slice,
    unsigned long long value_mask,
    bsi_word_t* __restrict__ out)
{
    const int q = blockIdx.z;
    if (q >= Q) return;

    const int warps_per_block = blockDim.x >> 5;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    const long long* vals = values + (long long)q * n;
    bsi_word_t* out_q = out + ((size_t)q * (size_t)slices * (size_t)words_per_slice);

#if BSI_WORD_BITS >= 32
    const int word_idx = blockIdx.x * warps_per_block + warp;
    if (word_idx >= words_per_slice) return;

    const long long base_row = (long long)word_idx * (long long)kBsiWordBits;

    for (int slice = 0; slice < slices; ++slice) {
        bsi_word_t word = bsi_pack_word(vals, n, value_mask, slice, base_row, lane);
        if (lane == 0) {
            out_q[(size_t)slice * (size_t)words_per_slice + (size_t)word_idx] = word;
        }
    }
#else
    const int group_idx = blockIdx.x * warps_per_block + warp;
    const int first_word = group_idx * kBsiWordsPerBallot;
    if (first_word >= words_per_slice) return;

    const long long base_row = (long long)group_idx * 32LL;

    for (int slice = 0; slice < slices; ++slice) {
        unsigned int ballot = bsi_pack_ballot32(vals, n, value_mask, slice, base_row, lane);
        if (lane == 0) {
            #pragma unroll
            for (int s = 0; s < kBsiWordsPerBallot; ++s) {
                int wi = first_word + s;
                if (wi < words_per_slice) {
                    out_q[(size_t)slice * words_per_slice + wi] = bsi_extract_subword(ballot, s);
                }
            }
        }
    }
#endif
}

// ===========================================================================
// Launchers
// ===========================================================================

extern "C" void launch_pack_bits_all_ballot(
    const long long* values,
    long long n,
    int slices,
    int words_per_slice,
    unsigned long long value_mask,
    bsi_word_t* out,
    cudaStream_t stream)
{
    if (slices <= 0 || words_per_slice <= 0) return;
    const int warps_per_block = 8;
    dim3 block(warps_per_block * 32);

#if BSI_WORD_BITS >= 32
    dim3 grid((words_per_slice + warps_per_block - 1) / warps_per_block);
#else
    // Each warp processes one group of 32 rows -> kBsiWordsPerBallot words
    int groups = (words_per_slice + kBsiWordsPerBallot - 1) / kBsiWordsPerBallot;
    dim3 grid((groups + warps_per_block - 1) / warps_per_block);
#endif

    int use_oneshot = 1;
    if (const char* s = getenv("BSI_PACK_ONESHOT")) {
        use_oneshot = (atoi(s) != 0) ? 1 : 0;
    }
    if (use_oneshot) {
        pack_bits_all_ballot_multi_kernel_oneshot<<<grid, block, 0, stream>>>(
            values, n, slices, words_per_slice, value_mask, out);
    } else {
        dim3 grid_legacy(grid.x, slices);
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
    bsi_word_t* out,
    cudaStream_t stream)
{
    if (Q <= 0 || slices <= 0 || words_per_slice <= 0) return;
    const int warps_per_block = 8;
    dim3 block(warps_per_block * 32);

#if BSI_WORD_BITS >= 32
    dim3 grid((words_per_slice + warps_per_block - 1) / warps_per_block, 1, Q);
#else
    int groups = (words_per_slice + kBsiWordsPerBallot - 1) / kBsiWordsPerBallot;
    dim3 grid((groups + warps_per_block - 1) / warps_per_block, 1, Q);
#endif

    int use_oneshot = 1;
    if (const char* s = getenv("BSI_PACK_ONESHOT")) {
        use_oneshot = (atoi(s) != 0) ? 1 : 0;
    }
    if (use_oneshot) {
        pack_bits_all_ballot_multi_kernel_batch_oneshot<<<grid, block, 0, stream>>>(
            values, Q, n, slices, words_per_slice, value_mask, out);
    } else {
        dim3 grid_legacy(grid.x, slices, Q);
        pack_bits_all_ballot_multi_kernel_batch<<<grid_legacy, block, 0, stream>>>(
            values, Q, n, slices, words_per_slice, value_mask, out);
    }
}
