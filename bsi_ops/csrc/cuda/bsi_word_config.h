#pragma once

#include <cstdint>

#ifndef BSI_WORD_BITS
#define BSI_WORD_BITS 64
#endif

static_assert(BSI_WORD_BITS == 8 || BSI_WORD_BITS == 16 ||
              BSI_WORD_BITS == 32 || BSI_WORD_BITS == 64 ||
              BSI_WORD_BITS == 128 || BSI_WORD_BITS == 256,
              "BSI_WORD_BITS must be 8, 16, 32, 64, 128, or 256");

constexpr int kBsiWordBits  = BSI_WORD_BITS;
constexpr int kBsiWordBytes = BSI_WORD_BITS / 8;

// ---------------------------------------------------------------------------
// Native word type (8-64) vs composite (128-256)
// ---------------------------------------------------------------------------
#if BSI_WORD_BITS <= 64

  #if BSI_WORD_BITS == 8
    using bsi_word_t  = uint8_t;
    using bsi_sword_t = int8_t;
  #elif BSI_WORD_BITS == 16
    using bsi_word_t  = uint16_t;
    using bsi_sword_t = int16_t;
  #elif BSI_WORD_BITS == 32
    using bsi_word_t  = uint32_t;
    using bsi_sword_t = int32_t;
  #elif BSI_WORD_BITS == 64
    using bsi_word_t  = uint64_t;
    using bsi_sword_t = int64_t;
  #endif

  constexpr int kBsiWordParts = 1;

#else // BSI_WORD_BITS > 64 (composite)

  constexpr int kBsiWordParts = BSI_WORD_BITS / 64;

  struct alignas(kBsiWordParts * 8) bsi_word_t {
      uint64_t parts[kBsiWordParts];
  };

  // Signed counterpart not meaningful for composite; use the struct for both.
  using bsi_sword_t = bsi_word_t;

#endif

// ---------------------------------------------------------------------------
// Torch dtype for word tensors
// ---------------------------------------------------------------------------
// For composite types (128/256), we store as kInt64 tensors with the last
// dimension multiplied by kBsiWordParts.
#if BSI_WORD_BITS == 8
  // Note: torch::kUInt8 exists but may not support all ops; kInt8 is safer.
  #define BSI_TORCH_WORD_DTYPE torch::kInt8
#elif BSI_WORD_BITS == 16
  #define BSI_TORCH_WORD_DTYPE torch::kInt16
#elif BSI_WORD_BITS == 32
  #define BSI_TORCH_WORD_DTYPE torch::kInt32
#else
  // 64, 128, 256 all stored as int64 tensors
  #define BSI_TORCH_WORD_DTYPE torch::kInt64
#endif

// ---------------------------------------------------------------------------
// Helper: how many bsi_word_t to cover d bits
// ---------------------------------------------------------------------------
inline constexpr int bsi_words_per_slice(int64_t d) {
    return (d > 0) ? static_cast<int>((d + kBsiWordBits - 1) / kBsiWordBits) : 1;
}

// Number of torch-dtype elements per logical word in tensors.
// For composite types (128/256), each logical word occupies kBsiWordParts
// int64 elements in the tensor.
inline constexpr int bsi_torch_elems_per_word() {
    return (kBsiWordBits <= 64) ? 1 : kBsiWordParts;
}

// Number of torch-dtype elements for the word dimension of a tensor
// that stores words_per_slice logical words.
inline constexpr int bsi_torch_word_dim(int words_per_slice) {
    return words_per_slice * bsi_torch_elems_per_word();
}

// ---------------------------------------------------------------------------
// EWAH run-length-word (RLW) encoding constants
// Matches hybridbitmap runninglengthword.h: runninglengthbits = sizeof(uword)*4
// ---------------------------------------------------------------------------
constexpr int kBsiRlwRunBits = kBsiWordBytes * 4;  // half the word bits
constexpr int kBsiRlwLitBits = kBsiWordBits - 1 - kBsiRlwRunBits;
// Shift amount for literal count field: 1 (run_bit) + runninglengthbits
constexpr int kBsiRlwLitShift = 1 + kBsiRlwRunBits;

// ---------------------------------------------------------------------------
// Device-side helpers (CUDA only)
// ---------------------------------------------------------------------------
#if defined(__CUDACC__)

// Popcount for bsi_word_t
__device__ __forceinline__ int bsi_popc(bsi_word_t v) {
  #if BSI_WORD_BITS <= 32
    return __popc(static_cast<unsigned int>(v));
  #elif BSI_WORD_BITS == 64
    return __popcll(v);
  #else
    // Composite: sum popcounts of all uint64 parts
    int count = 0;
    #pragma unroll
    for (int i = 0; i < kBsiWordParts; ++i)
        count += __popcll(v.parts[i]);
    return count;
  #endif
}

// Bitwise AND
#if BSI_WORD_BITS > 64
__device__ __forceinline__ bsi_word_t operator&(const bsi_word_t& a,
                                                 const bsi_word_t& b) {
    bsi_word_t r;
    #pragma unroll
    for (int i = 0; i < kBsiWordParts; ++i)
        r.parts[i] = a.parts[i] & b.parts[i];
    return r;
}

__device__ __forceinline__ bsi_word_t operator|(const bsi_word_t& a,
                                                 const bsi_word_t& b) {
    bsi_word_t r;
    #pragma unroll
    for (int i = 0; i < kBsiWordParts; ++i)
        r.parts[i] = a.parts[i] | b.parts[i];
    return r;
}

// Zero constant
__device__ __forceinline__ bsi_word_t bsi_word_zero() {
    bsi_word_t r;
    #pragma unroll
    for (int i = 0; i < kBsiWordParts; ++i) r.parts[i] = 0ULL;
    return r;
}

// All-ones constant
__device__ __forceinline__ bsi_word_t bsi_word_ones() {
    bsi_word_t r;
    #pragma unroll
    for (int i = 0; i < kBsiWordParts; ++i) r.parts[i] = ~0ULL;
    return r;
}
#endif // BSI_WORD_BITS > 64

// ---------------------------------------------------------------------------
// EWAH RLW device helpers
// ---------------------------------------------------------------------------

// Build an RLW from components. For native types uses bit ops directly.
// For composite types, encodes into the first uint64 part (matches hybridbitmap).
__device__ __forceinline__ bsi_word_t bsi_rlw_encode(bool run_bit,
                                                      unsigned int run_len,
                                                      unsigned int lit_count) {
  #if BSI_WORD_BITS <= 64
    bsi_word_t rlw = static_cast<bsi_word_t>(run_bit ? 1 : 0);
    rlw |= static_cast<bsi_word_t>(run_len) << 1;
    rlw |= static_cast<bsi_word_t>(lit_count) << kBsiRlwLitShift;
    return rlw;
  #else
    // For composite, encode into parts[0], zero the rest.
    bsi_word_t rlw = bsi_word_zero();
    rlw.parts[0] = static_cast<uint64_t>(run_bit ? 1 : 0)
                  | (static_cast<uint64_t>(run_len) << 1)
                  | (static_cast<uint64_t>(lit_count) << kBsiRlwLitShift);
    return rlw;
  #endif
}

// Decode RLW fields
__device__ __forceinline__ bool bsi_rlw_run_bit(bsi_word_t rlw) {
  #if BSI_WORD_BITS <= 64
    return (rlw & static_cast<bsi_word_t>(1)) != 0;
  #else
    return (rlw.parts[0] & 1ULL) != 0ULL;
  #endif
}

__device__ __forceinline__ unsigned int bsi_rlw_run_len(bsi_word_t rlw) {
  #if BSI_WORD_BITS <= 64
    constexpr bsi_word_t mask = (static_cast<bsi_word_t>(1) << kBsiRlwRunBits) - 1;
    return static_cast<unsigned int>((rlw >> 1) & mask);
  #else
    constexpr uint64_t mask = (1ULL << kBsiRlwRunBits) - 1ULL;
    return static_cast<unsigned int>((rlw.parts[0] >> 1) & mask);
  #endif
}

__device__ __forceinline__ unsigned int bsi_rlw_lit_count(bsi_word_t rlw) {
  #if BSI_WORD_BITS <= 64
    return static_cast<unsigned int>(rlw >> kBsiRlwLitShift);
  #else
    return static_cast<unsigned int>(rlw.parts[0] >> kBsiRlwLitShift);
  #endif
}

// ---------------------------------------------------------------------------
// Pack kernel helpers: how __ballot_sync maps to bsi_word_t
// ---------------------------------------------------------------------------

// Number of 32-bit ballot results needed to fill one bsi_word_t.
// For words < 32 bits, one ballot produces multiple words, so this is 0.
constexpr int kBsiBallotsPerWord = (kBsiWordBits >= 32) ? (kBsiWordBits / 32) : 0;

// Number of words produced from a single 32-bit ballot result.
// For words >= 32 bits, this is 0 (use kBsiBallotsPerWord instead).
constexpr int kBsiWordsPerBallot = (kBsiWordBits < 32) ? (32 / kBsiWordBits) : 0;

#endif // __CUDACC__

// ---------------------------------------------------------------------------
// Host-side EWAH RLW encoding (non-CUDA)
// ---------------------------------------------------------------------------
#if !defined(__CUDACC__)

// Host version of RLW encode for use in bsi_vector_utils.h
template <typename U>
inline U bsi_host_rlw_encode(bool run_bit, U run_len, U lit_count) {
    constexpr int run_bits = sizeof(U) * 4;
    constexpr int lit_shift = 1 + run_bits;
    U rlw = static_cast<U>(run_bit ? 1 : 0);
    rlw |= static_cast<U>(run_len) << 1;
    rlw |= static_cast<U>(lit_count) << lit_shift;
    return rlw;
}

#endif // !__CUDACC__
