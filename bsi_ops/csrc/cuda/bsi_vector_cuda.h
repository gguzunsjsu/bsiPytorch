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
    torch::Tensor words;           // [slices, words_per_slice] verbatim bitplanes (kept for compatibility)
    // Hybrid (EWAH) compressed representation
    torch::Tensor cwords;          // [total_compressed_words] flat buffer: RLWs + literals per slice
    torch::Tensor comp_offsets;    // [slices] int32 offsets into cwords
    torch::Tensor comp_lengths;    // [slices] int32 lengths (#u64) per slice
    torch::Tensor comp_stats;      // [slices, 2] int32: {run_words, literal_words} (optional)
    torch::Tensor metadata;        // reserved for future use

    void log(const char* tag = nullptr) const;
};

bool bsi_cuda_should_log();

BsiVectorCudaData build_bsi_vector_from_float_tensor(const torch::Tensor& values,
                                                     int decimal_places,
                                                     const torch::Device& device,
                                                     bool verbose = false);

// Build EWAH-compressed representation from verbatim words in-place on device.
void bsi_cuda_build_compressed_view(BsiVectorCudaData& data);

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
