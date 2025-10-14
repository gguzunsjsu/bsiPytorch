#include "bsi_vector_cuda.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include <c10/cuda/CUDAStream.h>
#include <ATen/ops/bitwise_right_shift.h>
#include <ATen/ops/floor.h>
#include <ATen/ops/where.h>

namespace {
inline torch::Tensor make_words_tensor(const std::vector<uint64_t>& words,
                                       int slices,
                                       int words_per_slice,
                                       const torch::Device& device) {
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(device);
    if (words.empty()) {
        return torch::zeros({slices, words_per_slice}, options);
    }
    return torch::from_blob(const_cast<uint64_t*>(words.data()),
                            {slices, words_per_slice},
                            torch::TensorOptions().dtype(torch::kInt64))
        .clone()
        .to(device, /*non_blocking=*/true);
}
} // namespace

extern "C" void launch_pack_bits_all(const int64_t* values,
                                     int64_t n,
                                     int slices,
                                     int words_per_slice,
                                     unsigned long long value_mask,
                                     unsigned long long* out,
                                     cudaStream_t stream);

bool bsi_cuda_should_log() {
    static bool cached = []() {
        const char* flag = std::getenv("BSI_DEBUG");
        return flag != nullptr;
    }();
    return cached;
}

void BsiVectorCudaData::log(const char* tag) const {
    if (!bsi_cuda_should_log()) {
        return;
    }
    std::ostream& os = std::cout;
    if (tag) {
        os << "[BSI_CUDA] " << tag << ": ";
    } else {
        os << "[BSI_CUDA]";
    }
    os << "rows=" << rows
       << " slices=" << slices
       << " words_per_slice=" << words_per_slice
       << " offset=" << offset
       << " decimals=" << decimals
       << " twos_complement=" << (twos_complement ? 1 : 0)
       << std::endl;
}

BsiVectorCudaData build_bsi_vector_from_float_tensor(const torch::Tensor& input,
                                                     int decimal_places,
                                                     const torch::Device& device,
                                                     bool verbose) {
    TORCH_CHECK(input.dim() == 1, "build_bsi_vector_from_float_tensor expects 1D tensor");
    auto values = input.to(device, torch::kFloat64, /*non_blocking=*/true).contiguous();

    const double scale = std::pow(10.0, static_cast<double>(decimal_places));
    auto scaled_fp = values * scale;
    auto rounded = torch::where(
        scaled_fp.ge(0),
        torch::floor(scaled_fp + 0.5),
        -torch::floor(-scaled_fp + 0.5));
    auto scaled = rounded.to(torch::kInt64);
    const int64_t rows = scaled.size(0);

    bool any_non_zero = (rows > 0) && scaled.ne(0).any().item<bool>();
    bool has_negative = (rows > 0) && scaled.lt(0).any().item<bool>();

    long long max_abs = 0;
    if (any_non_zero) {
        max_abs = scaled.abs().max().item<int64_t>();
    }

    int magnitude_bits = any_non_zero
        ? std::max(1, static_cast<int>(std::bit_width(static_cast<unsigned long long>(max_abs))))
        : 1;
    int total_slices = std::min(64, magnitude_bits + 2);

    if (!any_non_zero) {
        total_slices = 1;
        has_negative = false;
    }

    int offset = 0;
    if (any_non_zero) {
        auto abs_vals = scaled.abs();
        for (int candidate = 0; candidate < total_slices - 1; ++candidate) {
            if (abs_vals.bitwise_and(1).ne(0).any().item<bool>()) {
                break;
            }
            abs_vals = at::bitwise_right_shift(abs_vals, 1);
            ++offset;
        }
        // Avoid dropping all slices; keep at least the sign slice.
        if (offset >= total_slices) {
            offset = total_slices - 1;
        }
    }

    int stored_slices = std::max(1, total_slices - offset);
    torch::Tensor shifted = scaled;
    if (offset > 0 && rows > 0) {
        shifted = at::bitwise_right_shift(scaled, offset);
    }
    shifted = shifted.contiguous();

    const int words_per_slice = rows > 0 ? static_cast<int>((rows + 63) / 64) : 1;
    auto words = torch::zeros({stored_slices, words_per_slice},
                              torch::TensorOptions().dtype(torch::kInt64).device(device));

    if (rows > 0) {
        unsigned long long value_mask = (stored_slices >= 64)
            ? ~0ULL
            : ((1ULL << stored_slices) - 1ULL);
        auto stream = at::cuda::getCurrentCUDAStream();
        launch_pack_bits_all(
            shifted.data_ptr<int64_t>(),
            rows,
            stored_slices,
            words_per_slice,
            value_mask,
            reinterpret_cast<unsigned long long*>(words.data_ptr<int64_t>()),
            stream.stream());
    }

    BsiVectorCudaData data;
    data.rows = rows;
    data.slices = stored_slices;
    data.words_per_slice = words_per_slice;
    data.offset = offset;
    data.decimals = decimal_places;
    data.twos_complement = has_negative;
    data.words = words;
    data.metadata = torch::empty({stored_slices, 0},
                                 torch::TensorOptions().dtype(torch::kInt32).device(device));

    if (verbose || bsi_cuda_should_log()) {
        data.log("build_bsi_vector_from_float_tensor");
    }
    return data;
}

BsiVectorCudaData create_bsi_vector_cuda_from_cpu(const BsiVector<uint64_t>& src,
                                                  const torch::Device& device,
                                                  bool verbose) {
    std::vector<uint64_t> words;
    int slices = 0;
    int words_per_slice = 0;
    bsi_flatten_words_gpu_helper<uint64_t>(src, words, slices, words_per_slice);

    BsiVectorCudaData data;
    data.rows = src.getNumberOfRows();
    data.slices = slices;
    data.words_per_slice = words_per_slice;
    data.offset = src.offset;
    data.decimals = src.decimals;
    data.twos_complement = src.twosComplement;
    data.words = make_words_tensor(words, slices, words_per_slice, device);
    data.metadata = torch::empty({slices, 0}, torch::TensorOptions().dtype(torch::kInt32).device(device));

    if (verbose || bsi_cuda_should_log()) {
        data.log("create_bsi_vector_cuda_from_cpu");
    }
    return data;
}
