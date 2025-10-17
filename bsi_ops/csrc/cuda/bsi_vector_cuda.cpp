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
template <typename T>
inline T* tensor_data_ptr(torch::Tensor& t) {
    return t.data_ptr<T>();
}

template <typename T>
inline const T* tensor_data_ptr(const torch::Tensor& t) {
    auto& nc = const_cast<torch::Tensor&>(t);
    return const_cast<const T*>(nc.data_ptr<T>());
}

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

// no additional externs

bool bsi_cuda_should_log() {
    static bool cached = []() {
        const char* flag = std::getenv("BSI_DEBUG");
        return flag != nullptr;
    }();
    return cached;
}

static void maybe_log_scaled(const torch::Tensor& scaled);

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
    auto scaled = bsi_cuda_quantize_to_int64(input, decimal_places, device);
    maybe_log_scaled(scaled);
    const int64_t rows = scaled.size(0);

    bool any_non_zero = (rows > 0) && scaled.ne(0).any().item<bool>();
    bool has_negative = (rows > 0) && scaled.lt(0).any().item<bool>();
    bool all_zero = (rows > 0) && !any_non_zero;

    long long max_abs = 0;
    if (any_non_zero) {
        max_abs = scaled.abs().max().item<int64_t>();
    }

    int magnitude_bits = any_non_zero
        ? std::max(1, static_cast<int>(std::bit_width(static_cast<unsigned long long>(max_abs))))
        : 1;
    int total_slices = std::min(64, magnitude_bits + 2);

    if (!any_non_zero) {
        // Match CPU decimals builder: slices = bit_width(0) + 2 = 2
        total_slices = 2;
        has_negative = false;
    }

    // For parity with CPU decimal builder, do not trim low zero bitplanes.
    int offset = 0;
    int stored_slices = std::max(1, total_slices);
    torch::Tensor shifted = scaled.contiguous();

    const int words_per_slice = rows > 0 ? static_cast<int>((rows + 63) / 64) : 1;
    auto words = torch::zeros({stored_slices, words_per_slice},
                              torch::TensorOptions().dtype(torch::kInt64).device(device));

    if (rows > 0) {
        unsigned long long value_mask = (stored_slices >= 64)
            ? ~0ULL
            : ((1ULL << stored_slices) - 1ULL);
        auto stream = at::cuda::getCurrentCUDAStream();
        auto* shifted_ptr = tensor_data_ptr<int64_t>(shifted);
        launch_pack_bits_all(
            shifted_ptr,
            rows,
            stored_slices,
            words_per_slice,
            value_mask,
            reinterpret_cast<unsigned long long*>(tensor_data_ptr<int64_t>(words)),
            stream.stream());
    }

    BsiVectorCudaData data;
    data.rows = rows;
    data.slices = stored_slices;
    data.words_per_slice = words_per_slice;
    data.offset = offset;
    data.decimals = decimal_places;
    // Match CPU decimals builder: two's complement only when negatives present
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

// no compressed view in Phase-2

torch::Tensor bsi_cuda_quantize_to_int64(const torch::Tensor& input,
                                         int decimal_places,
                                         const torch::Device& device) {
    auto values = input.to(device, torch::kFloat64, /*non_blocking=*/true).contiguous();
    const double scale = std::pow(10.0, static_cast<double>(decimal_places));
    auto x = values * scale;
    // std::round (half away from zero): floor(x+0.5) if x>=0; -floor(-x+0.5) if x<0
    auto rounded = torch::where(
        x.ge(0),
        torch::floor(x + 0.5),
        -torch::floor(-x + 0.5)
    );
    return rounded.to(torch::kInt64).contiguous();
}

static void maybe_log_scaled(const torch::Tensor& scaled) {
    if (!bsi_cuda_should_log()) return;
    auto n = std::min<int64_t>(scaled.numel(), 8);
    auto cpu = scaled.to(torch::kCPU);
    auto acc = cpu.accessor<int64_t,1>();
    std::cout << "[BSI_CUDA] scaled_ints:";
    for (int64_t i=0;i<n;++i) std::cout << ' ' << (long long)acc[i];
    std::cout << std::endl;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
bsi_cuda_quantize_debug(const torch::Tensor& input,
                        int decimal_places,
                        const torch::Device& device,
                        int64_t k) {
    auto values = input.to(device, torch::kFloat64, /*non_blocking=*/true).contiguous();
    const double scale = std::pow(10.0, static_cast<double>(decimal_places));
    auto x = values * scale;
    auto rounded = torch::where(
        x.ge(0),
        torch::floor(x + 0.5),
        -torch::floor(-x + 0.5)
    );
    auto ints = rounded.to(torch::kInt64);
    auto take = std::min<int64_t>(x.numel(), k);
    auto x_head = x.narrow(0, 0, take).to(torch::kCPU).contiguous();
    auto r_head = rounded.narrow(0, 0, take).to(torch::kCPU).contiguous();
    auto i_head = ints.narrow(0, 0, take).to(torch::kCPU).contiguous();
    return std::make_tuple(x_head, r_head, i_head);
}
