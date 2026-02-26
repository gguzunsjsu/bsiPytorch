#pragma once

#include <torch/extension.h>

#include <cstdint>
#include <tuple>

#include "bsi_vector_utils.h"

struct BsiVectorCudaData {
    int64_t rows = 0;
    int slices = 0;
    int words_per_slice = 0;
    int offset = 0;
    int decimals = 0;
    bool twos_complement = false;
    // Additional per-vector scaling factor used by fixed-bit quantization modes.
    // Default is 1.0 (no scaling).
    float scale = 1.0f;
    torch::Tensor words;      // [slices, words_per_slice] (verbatim; optional)
    torch::Tensor metadata;   // placeholder for future hybrid metadata
    // Hybrid (EWAH) compressed representation (always populated in hybrid builder)
    torch::Tensor comp_words; // [u64_total]
    torch::Tensor comp_off;   // [slices] int64 offsets into comp_words
    torch::Tensor comp_len;   // [slices] int32 lengths (u64 count) per slice

    void log(const char* tag = nullptr) const;
};

struct BsiQueryBatchCudaData {
    int64_t rows = 0;
    int slices = 0;
    int words_per_slice = 0;
    int offset = 0;
    torch::Tensor words;         // [Q, slices, words_per_slice]
    torch::Tensor slice_weights; // [Q, slices]
    // Optional per-query, per-256-element-chunk power-of-two scales used by
    // fixed-bit "block floating" modes. Shape: [Q, chunks] where
    // chunks = words_per_slice / 4 (i.e., 256 bits per chunk).
    // When undefined, slice_weights already include any fixed-bit scaling.
    torch::Tensor chunk_scales;  // [Q, chunks] or undefined
};

bool bsi_cuda_should_log();

BsiVectorCudaData build_bsi_vector_from_float_tensor(const torch::Tensor& values,
                                                     int decimal_places,
                                                     const torch::Device& device,
                                                     bool verbose = false);

// Hybrid (EWAH) builder on GPU: builds compressed representation directly on device
BsiVectorCudaData build_bsi_vector_from_float_tensor_hybrid(const torch::Tensor& values,
                                                            int decimal_places,
                                                            double compress_threshold,
                                                            const torch::Device& device,
                                                            bool verbose = false);

BsiQueryBatchCudaData build_bsi_queries_cuda_batch_data(const torch::Tensor& values,
                                                        int decimal_places,
                                                        const torch::Device& device,
                                                        bool verbose = false,
                                                        bool for_keys = false);

// Exposed for tests/debug: quantise floats to int64 with CPU parity (half-away-from-zero).
torch::Tensor bsi_cuda_quantize_to_int64(const torch::Tensor& values,
                                         int decimal_places,
                                         const torch::Device& device);

// Debug helper: return (scaled_fp, rounded_fp, staged_int) heads (first k elements) on CPU
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
bsi_cuda_quantize_debug(const torch::Tensor& values,
                        int decimal_places,
                        const torch::Device& device,
                        int64_t k = 8);

BsiVectorCudaData create_bsi_vector_cuda_from_cpu(const BsiVector<uint64_t>& src,
                                                  const torch::Device& device,
                                                  bool verbose = false);

// Last-call query build profiling (set by build_bsi_queries_cuda_batch_data when BSI_PROFILE=1).
// Returns (quantize_ns, pack_ns, total_ns). Values are 0 when profiling is disabled.
std::tuple<uint64_t, uint64_t, uint64_t> bsi_cuda_get_last_query_build_profile();
void bsi_cuda_reset_last_query_build_profile();
