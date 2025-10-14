#pragma once

#include <torch/extension.h>

#include <cstdint>

#include "bsi_vector_utils.h"

struct BsiVectorCudaData {
    int64_t rows = 0;
    int slices = 0;
    int words_per_slice = 0;
    int offset = 0;
    int decimals = 0;
    bool twos_complement = false;
    torch::Tensor words;      // [slices, words_per_slice]
    torch::Tensor metadata;   // placeholder for future hybrid metadata

    void log(const char* tag = nullptr) const;
};

bool bsi_cuda_should_log();

BsiVectorCudaData create_bsi_vector_cuda_from_cpu(const BsiVector<uint64_t>& src,
                                                  const torch::Device& device,
                                                  bool verbose = false);
