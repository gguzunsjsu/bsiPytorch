#include "bsi_vector_cuda.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>

#include <cuda_runtime.h>

#include <c10/cuda/CUDAStream.h>
#include <ATen/ops/bitwise_left_shift.h>
#include <ATen/ops/bitwise_right_shift.h>
#include <ATen/ops/clamp.h>
#include <ATen/ops/floor.h>
#include <ATen/ops/log2.h>
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

inline torch::Tensor make_slice_weights_cuda_local(int S,
                                                   int offset,
                                                   bool twos,
                                                   const torch::Device& device) {
    std::vector<float> host(S);
    for (int i = 0; i < S; ++i) {
        int shift = offset + i;
        long double w = (shift >= 0) ? std::ldexp(1.0L, shift) : 0.0L;
        if (twos && i == S - 1) {
            w = -w;
        }
        host[i] = static_cast<float>(w);
    }
    return torch::from_blob(
               host.data(),
               {S},
               torch::TensorOptions().dtype(torch::kFloat32))
        .clone()
        .to(device, /*non_blocking=*/true);
}

inline torch::Tensor make_slice_weights_cuda_cached(int S,
                                                    int offset,
                                                    bool twos,
                                                    const torch::Device& device) {
    // Cache immutable [S] base weights on-device; query build reuses them every forward.
    static std::unordered_map<uint64_t, torch::Tensor> cache;
    const int dev_idx = device.has_index() ? device.index() : 0;
    const uint64_t key =
        (static_cast<uint64_t>(static_cast<uint32_t>(dev_idx)) << 32) ^
        (static_cast<uint64_t>(static_cast<uint16_t>(S)) << 16) ^
        static_cast<uint64_t>(static_cast<uint16_t>(offset)) ^
        (twos ? (1ull << 63) : 0ull);
    auto it = cache.find(key);
    if (it != cache.end() && it->second.defined()) {
        return it->second;
    }
    auto t = make_slice_weights_cuda_local(S, offset, twos, device).contiguous();
    cache.emplace(key, t);
    return t;
}
} // namespace

extern "C" void launch_pack_bits_all_ballot(
    const long long* values,
    long long n,
    int slices,
    int words_per_slice,
    unsigned long long value_mask,
    unsigned long long* out,
    cudaStream_t stream);

extern "C" void launch_pack_bits_all_ballot_batch(
    const long long* values,
    int Q,
    long long n,
    int slices,
    int words_per_slice,
    unsigned long long value_mask,
    unsigned long long* out,
    cudaStream_t stream);

extern "C" void launch_quantize_round_to_int64_batch(
    const void* input,
    int input_dtype,
    int64_t Q,
    int64_t d,
    double dec_scale,
    long long* out_ints,
    cudaStream_t stream);

extern "C" void launch_compute_row_shift_scale(
    const long long* ints,
    int64_t Q,
    int64_t d,
    int fixed_bits,
    float clip_k,
    int* shifts,
    float* scales,
    cudaStream_t stream);

extern "C" void launch_compute_chunk_shift_scale(
    const long long* ints,
    int64_t Q,
    int64_t d,
    int chunks,
    int fixed_bits,
    float clip_k,
    int* shifts,
    float* scales,
    cudaStream_t stream);

extern "C" void launch_apply_row_shift_clamp(
    long long* ints,
    int64_t Q,
    int64_t d,
    const int* shifts,
    int fixed_bits,
    cudaStream_t stream);

extern "C" void launch_apply_chunk_shift_clamp(
    long long* ints,
    int64_t Q,
    int64_t d,
    int chunks,
    const int* shifts,
    int fixed_bits,
    cudaStream_t stream);

extern "C" void launch_compute_row_shift_scale_from_input(
    const void* input,
    int input_dtype,
    int64_t Q,
    int64_t d,
    double dec_scale,
    int fixed_bits,
    float clip_k,
    int* shifts,
    float* scales,
    cudaStream_t stream);

extern "C" void launch_compute_chunk_shift_scale_from_input(
    const void* input,
    int input_dtype,
    int64_t Q,
    int64_t d,
    int chunks,
    double dec_scale,
    int fixed_bits,
    float clip_k,
    int* shifts,
    float* scales,
    cudaStream_t stream);

extern "C" void launch_quantize_shift_pack_row_batch(
    const void* input,
    int input_dtype,
    int64_t Q,
    int64_t d,
    int slices,
    int words_per_slice,
    unsigned long long value_mask,
    double dec_scale,
    int fixed_bits,
    const int* shifts,
    unsigned long long* out,
    cudaStream_t stream);

extern "C" void launch_quantize_shift_pack_chunk_batch(
    const void* input,
    int input_dtype,
    int64_t Q,
    int64_t d,
    int chunks,
    int slices,
    int words_per_slice,
    unsigned long long value_mask,
    double dec_scale,
    int fixed_bits,
    const int* shifts,
    unsigned long long* out,
    cudaStream_t stream);

extern "C" void launch_slice_popcount_sum(
    const unsigned long long* words,
    int S,
    int W,
    unsigned long long* out_counts,
    cudaStream_t stream);

extern "C" void launch_compress_flags_from_density(
    const unsigned long long* counts,
    int S,
    int W,
    double threshold,
    int* out_flags,
    cudaStream_t stream);

extern "C" void launch_ewah_size(
    const unsigned long long* words,
    int S,
    int W,
    const int* flags,
    unsigned long long* sizes,
    cudaStream_t stream);

extern "C" void launch_ewah_emit(
    const unsigned long long* words,
    int S,
    int W,
    const int* flags,
    const unsigned long long* off,
    unsigned long long* out,
    int* out_len,
    cudaStream_t stream);

// no additional externs

bool bsi_cuda_should_log() {
    static bool cached = []() {
        const char* flag = std::getenv("BSI_DEBUG");
        return flag != nullptr;
    }();
    return cached;
}

static int bsi_cuda_fixed_bits() {
    static int cached = []() {
        const char* s = std::getenv("BSI_FIXED_BITS");
        if (s == nullptr) return 0;
        int v = std::atoi(s);
        if (v <= 0) return 0;
        if (v < 2) v = 2;
        if (v > 63) v = 63;
        return v;
    }();
    return cached;
}

static int bsi_cuda_fixed_bits_override(const char* name) {
    const char* s = std::getenv(name);
    if (s == nullptr) return -1;
    int v = std::atoi(s);
    if (v <= 0) return 0;
    if (v < 2) v = 2;
    if (v > 63) v = 63;
    return v;
}

static int bsi_cuda_fixed_bits_queries() {
    static int cached = []() {
        int v = bsi_cuda_fixed_bits_override("BSI_FIXED_BITS_QUERIES");
        if (v >= 0) return v;
        return bsi_cuda_fixed_bits();
    }();
    return cached;
}

static int bsi_cuda_fixed_bits_keys() {
    static int cached = []() {
        int v = bsi_cuda_fixed_bits_override("BSI_FIXED_BITS_KEYS");
        if (v >= 0) return v;
        return bsi_cuda_fixed_bits();
    }();
    return cached;
}

float bsi_cuda_fixed_clip_k() {
    static float cached = []() {
        // Clip extreme values when computing the per-row shift in fixed-bit mode.
        // 0 disables clipping and uses true absmax (no saturation).
        float v = 0.0f;
        if (const char* s = std::getenv("BSI_FIXED_CLIP_K")) {
            v = std::strtof(s, nullptr);
        }
        if (!(v > 0.0f)) v = 0.0f;
        return v;
    }();
    return cached;
}

static bool bsi_cuda_fixed_chunk_scale_queries() {
    static int cached = -1;
    if (cached >= 0) return cached != 0;
    int v = 0;
    if (const char* s = std::getenv("BSI_FIXED_CHUNK_SCALE")) {
        v = std::atoi(s);
    }
    cached = (v != 0) ? 1 : 0;
    return cached != 0;
}

static bool bsi_cuda_profile_enabled_local() {
    static int cached = -1;
    if (cached >= 0) return cached != 0;
    int v = 0;
    if (const char* s = std::getenv("BSI_PROFILE")) {
        v = (std::atoi(s) != 0) ? 1 : 0;
    }
    cached = v;
    return cached != 0;
}

static bool bsi_cuda_custom_quant_enabled() {
    static int cached = -1;
    if (cached >= 0) return cached != 0;
    int v = 1; // default on
    if (const char* s = std::getenv("BSI_CUSTOM_QUANT")) {
        v = (std::atoi(s) != 0) ? 1 : 0;
    }
    cached = v;
    return cached != 0;
}

static bool bsi_cuda_fused_qpack_enabled() {
    static int cached = -1;
    if (cached >= 0) return cached != 0;
    int v = 1; // default on
    if (const char* s = std::getenv("BSI_FUSED_QPACK")) {
        v = (std::atoi(s) != 0) ? 1 : 0;
    }
    cached = v;
    return cached != 0;
}

static int bsi_cuda_input_dtype_code(const torch::Tensor& values) {
    int input_dtype = 0;
    if (values.scalar_type() == torch::kFloat16) input_dtype = 1;
    else if (values.scalar_type() == torch::kBFloat16) input_dtype = 2;
    return input_dtype;
}

static uint64_t g_last_query_build_quantize_ns = 0;
static uint64_t g_last_query_build_pack_ns = 0;
static uint64_t g_last_query_build_total_ns = 0;

std::tuple<uint64_t, uint64_t, uint64_t> bsi_cuda_get_last_query_build_profile() {
    return std::make_tuple(g_last_query_build_quantize_ns,
                           g_last_query_build_pack_ns,
                           g_last_query_build_total_ns);
}

void bsi_cuda_reset_last_query_build_profile() {
    g_last_query_build_quantize_ns = 0;
    g_last_query_build_pack_ns = 0;
    g_last_query_build_total_ns = 0;
}

static inline torch::Tensor round_half_away_from_zero(const torch::Tensor& x) {
    // std::round semantics: half away from zero.
    return torch::where(
        x.ge(0),
        torch::floor(x + 0.5),
        -torch::floor(-x + 0.5));
}

struct QuantizedIntsAndScale {
    torch::Tensor ints;   // int64, same shape as input (quantized values to pack)
    torch::Tensor scale;  // float32, shape [Q] or [Q, chunks] for 2D, scalar for 1D
};

static inline torch::Tensor round_shift_right_signed(const torch::Tensor& ints, const torch::Tensor& shift) {
    // Round-to-nearest (half away from zero) for signed int64 division by 2^shift.
    // shift must be >= 0 and broadcastable to ints.
    auto abs_ints = ints.abs();
    auto round_add = torch::where(
        shift.gt(0),
        torch::bitwise_left_shift(torch::ones_like(shift), shift - 1),
        torch::zeros_like(shift));
    auto abs_rounded = torch::bitwise_right_shift(abs_ints + round_add, shift);
    return torch::where(ints.lt(0), -abs_rounded, abs_rounded);
}

static bool bsi_cuda_build_fixed_query_words_fused(const torch::Tensor& input,
                                                   int decimal_places,
                                                   int fixed_bits,
                                                   bool chunk_scale,
                                                   const torch::Device& device,
                                                   bool profile_cuda,
                                                   torch::Tensor& words_out,
                                                   torch::Tensor& scale_out,
                                                   float& quantize_ms,
                                                   float& pack_ms) {
    if (fixed_bits <= 0 || !input.is_cuda()) {
        return false;
    }
    auto values = input.to(device, input.scalar_type(), /*non_blocking=*/true).contiguous();
    if (!(values.scalar_type() == torch::kFloat32 ||
          values.scalar_type() == torch::kFloat16 ||
          values.scalar_type() == torch::kBFloat16)) {
        values = values.to(torch::kFloat32);
    }
    TORCH_CHECK(values.dim() == 2, "Fused fixed-bit query build expects 2D input");
    const int64_t Q = values.size(0);
    const int64_t d = values.size(1);
    const int slices = std::max(1, fixed_bits);
    const int words_per_slice = (d > 0) ? static_cast<int>((d + 63) / 64) : 1;
    auto stream = at::cuda::getCurrentCUDAStream();

    words_out = torch::zeros({Q, slices, words_per_slice},
                             torch::TensorOptions().dtype(torch::kInt64).device(device));

    if (Q <= 0 || d <= 0) {
        if (chunk_scale) {
            const int chunks = static_cast<int>((d + 255) / 256);
            scale_out = torch::ones({Q, chunks}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        } else {
            scale_out = torch::ones({Q}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        }
        quantize_ms = 0.0f;
        pack_ms = 0.0f;
        return true;
    }

    const int input_dtype = bsi_cuda_input_dtype_code(values);
    const double dec_scale = std::pow(10.0, static_cast<double>(decimal_places));
    const float clip_k = bsi_cuda_fixed_clip_k();
    const unsigned long long value_mask =
        (slices >= 64) ? ~0ULL : ((1ULL << slices) - 1ULL);

    cudaEvent_t quantize_start_evt = nullptr;
    cudaEvent_t quantize_end_evt = nullptr;
    cudaEvent_t pack_start_evt = nullptr;
    cudaEvent_t pack_end_evt = nullptr;
    if (profile_cuda) {
        cudaEventCreate(&quantize_start_evt);
        cudaEventCreate(&quantize_end_evt);
        cudaEventCreate(&pack_start_evt);
        cudaEventCreate(&pack_end_evt);
    }

    if (chunk_scale) {
        const int chunks = static_cast<int>((d + 255) / 256);
        auto shifts = torch::empty({Q, chunks}, torch::TensorOptions().dtype(torch::kInt32).device(device));
        auto scales = torch::empty({Q, chunks}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

        if (profile_cuda) cudaEventRecord(quantize_start_evt, stream.stream());
        launch_compute_chunk_shift_scale_from_input(
            values.data_ptr(),
            input_dtype,
            Q,
            d,
            chunks,
            dec_scale,
            fixed_bits,
            clip_k,
            shifts.data_ptr<int>(),
            tensor_data_ptr<float>(scales),
            stream.stream());
        if (profile_cuda) {
            cudaEventRecord(quantize_end_evt, stream.stream());
            cudaEventSynchronize(quantize_end_evt);
            cudaEventElapsedTime(&quantize_ms, quantize_start_evt, quantize_end_evt);
        }

        if (profile_cuda) cudaEventRecord(pack_start_evt, stream.stream());
        launch_quantize_shift_pack_chunk_batch(
            values.data_ptr(),
            input_dtype,
            Q,
            d,
            chunks,
            slices,
            words_per_slice,
            value_mask,
            dec_scale,
            fixed_bits,
            shifts.data_ptr<int>(),
            reinterpret_cast<unsigned long long*>(tensor_data_ptr<int64_t>(words_out)),
            stream.stream());
        if (profile_cuda) {
            cudaEventRecord(pack_end_evt, stream.stream());
            cudaEventSynchronize(pack_end_evt);
            cudaEventElapsedTime(&pack_ms, pack_start_evt, pack_end_evt);
        }

        scale_out = scales.contiguous();
    } else {
        auto shifts = torch::empty({Q}, torch::TensorOptions().dtype(torch::kInt32).device(device));
        auto scales = torch::empty({Q}, torch::TensorOptions().dtype(torch::kFloat32).device(device));

        if (profile_cuda) cudaEventRecord(quantize_start_evt, stream.stream());
        launch_compute_row_shift_scale_from_input(
            values.data_ptr(),
            input_dtype,
            Q,
            d,
            dec_scale,
            fixed_bits,
            clip_k,
            shifts.data_ptr<int>(),
            tensor_data_ptr<float>(scales),
            stream.stream());
        if (profile_cuda) {
            cudaEventRecord(quantize_end_evt, stream.stream());
            cudaEventSynchronize(quantize_end_evt);
            cudaEventElapsedTime(&quantize_ms, quantize_start_evt, quantize_end_evt);
        }

        if (profile_cuda) cudaEventRecord(pack_start_evt, stream.stream());
        launch_quantize_shift_pack_row_batch(
            values.data_ptr(),
            input_dtype,
            Q,
            d,
            slices,
            words_per_slice,
            value_mask,
            dec_scale,
            fixed_bits,
            shifts.data_ptr<int>(),
            reinterpret_cast<unsigned long long*>(tensor_data_ptr<int64_t>(words_out)),
            stream.stream());
        if (profile_cuda) {
            cudaEventRecord(pack_end_evt, stream.stream());
            cudaEventSynchronize(pack_end_evt);
            cudaEventElapsedTime(&pack_ms, pack_start_evt, pack_end_evt);
        }

        scale_out = scales.contiguous();
    }

    if (profile_cuda) {
        cudaEventDestroy(quantize_start_evt);
        cudaEventDestroy(quantize_end_evt);
        cudaEventDestroy(pack_start_evt);
        cudaEventDestroy(pack_end_evt);
    }
    return true;
}

static QuantizedIntsAndScale bsi_cuda_quantize_to_int64_and_scale(const torch::Tensor& input,
                                                                  int decimal_places,
                                                                  const torch::Device& device,
                                                                  bool per_row_scale,
                                                                  int fixed_bits,
                                                                  bool chunk_scale) {
    if (fixed_bits <= 0) {
        // CPU-parity path (used when keys come from CPU builder).
        auto values = input.to(device, torch::kFloat64, /*non_blocking=*/true).contiguous();
        const double dec_scale = std::pow(10.0, static_cast<double>(decimal_places));
        auto x = values * dec_scale;
        auto rounded = round_half_away_from_zero(x);
        QuantizedIntsAndScale out;
        out.ints = rounded.to(torch::kInt64).contiguous();
        out.scale = per_row_scale
            ? torch::ones({out.ints.size(0)}, torch::TensorOptions().dtype(torch::kFloat32).device(device))
            : torch::ones({}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        return out;
    }

    // Fast custom CUDA path for fixed-bit batched quantization.
    // This avoids large float64 tensor-op graphs (log2/pow/where) and computes shift/scale with custom kernels.
    if (per_row_scale && input.is_cuda() && bsi_cuda_custom_quant_enabled()) {
        auto values = input.to(device, input.scalar_type(), /*non_blocking=*/true).contiguous();
        if (!(values.scalar_type() == torch::kFloat32 ||
              values.scalar_type() == torch::kFloat16 ||
              values.scalar_type() == torch::kBFloat16)) {
            values = values.to(torch::kFloat32);
        }
        TORCH_CHECK(values.dim() == 2, "Custom fixed-bit quantization expects 2D input");
        const int64_t Q = values.size(0);
        const int64_t d = values.size(1);
        auto ints = torch::empty({Q, d}, torch::TensorOptions().dtype(torch::kInt64).device(device));
        const double dec_scale = std::pow(10.0, static_cast<double>(decimal_places));
        int input_dtype = 0;
        if (values.scalar_type() == torch::kFloat16) input_dtype = 1;
        else if (values.scalar_type() == torch::kBFloat16) input_dtype = 2;

        auto stream = at::cuda::getCurrentCUDAStream();
        launch_quantize_round_to_int64_batch(
            values.data_ptr(),
            input_dtype,
            Q,
            d,
            dec_scale,
            reinterpret_cast<long long*>(tensor_data_ptr<int64_t>(ints)),
            stream.stream());

        const float clip_k = bsi_cuda_fixed_clip_k();
        QuantizedIntsAndScale out;
        if (chunk_scale) {
            const int chunks = static_cast<int>((d + 255) / 256);
            auto shifts = torch::empty({Q, chunks}, torch::TensorOptions().dtype(torch::kInt32).device(device));
            auto scales = torch::empty({Q, chunks}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
            launch_compute_chunk_shift_scale(
                reinterpret_cast<const long long*>(tensor_data_ptr<int64_t>(ints)),
                Q,
                d,
                chunks,
                fixed_bits,
                clip_k,
                shifts.data_ptr<int>(),
                tensor_data_ptr<float>(scales),
                stream.stream());
            launch_apply_chunk_shift_clamp(
                reinterpret_cast<long long*>(tensor_data_ptr<int64_t>(ints)),
                Q,
                d,
                chunks,
                shifts.data_ptr<int>(),
                fixed_bits,
                stream.stream());
            out.ints = ints.contiguous();
            out.scale = scales.contiguous();
        } else {
            auto shifts = torch::empty({Q}, torch::TensorOptions().dtype(torch::kInt32).device(device));
            auto scales = torch::empty({Q}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
            launch_compute_row_shift_scale(
                reinterpret_cast<const long long*>(tensor_data_ptr<int64_t>(ints)),
                Q,
                d,
                fixed_bits,
                clip_k,
                shifts.data_ptr<int>(),
                tensor_data_ptr<float>(scales),
                stream.stream());
            launch_apply_row_shift_clamp(
                reinterpret_cast<long long*>(tensor_data_ptr<int64_t>(ints)),
                Q,
                d,
                shifts.data_ptr<int>(),
                fixed_bits,
                stream.stream());
            out.ints = ints.contiguous();
            out.scale = scales.contiguous();
        }
        return out;
    }

    // Fixed-bit path: "block floating" downshift to fit fixed_bits, then compensate via a power-of-two scale.
    // This keeps slice weights exact (powers of two) and matches the original fixed-bit implementation behavior.
    auto values = input.to(device, torch::kFloat64, /*non_blocking=*/true).contiguous();
    const double dec_scale = std::pow(10.0, static_cast<double>(decimal_places));
    auto x = values * dec_scale;
    auto rounded = round_half_away_from_zero(x);
    auto ints = rounded.to(torch::kInt64).contiguous();

    torch::Tensor max_abs;
    torch::Tensor mean_abs;
    torch::Tensor ints_for_shift = ints;
    int64_t chunk_elems = 0;
    int64_t chunks = 0;
    if (chunk_scale && per_row_scale) {
        // Per-256-element chunk scaling (block floating): compute shift per row per chunk.
        // This reduces the effect of activation outliers (one chunk doesn't force a shift for the whole row).
        const int64_t Q = ints.size(0);
        const int64_t d = ints.size(1);
        chunk_elems = 256;
        chunks = (d + chunk_elems - 1) / chunk_elems;
        const int64_t d_pad = chunks * chunk_elems;
        if (d_pad != d) {
            auto pad = torch::zeros({Q, d_pad}, torch::TensorOptions().dtype(torch::kInt64).device(device));
            pad.narrow(1, 0, d).copy_(ints);
            ints_for_shift = pad;
        }
        auto view_abs = ints_for_shift.abs().view({Q, chunks, chunk_elems}); // int64
        max_abs = std::get<0>(view_abs.max(2));                              // [Q, chunks] int64
        mean_abs = view_abs.to(torch::kFloat64).mean(2);                     // [Q, chunks] float64
    } else if (per_row_scale) {
        max_abs = std::get<0>(ints.abs().max(1));                            // [Q] int64
        mean_abs = ints.abs().to(torch::kFloat64).mean(1);                   // [Q] float64
    } else {
        max_abs = ints.abs().max();                                          // scalar int64
        mean_abs = ints.abs().to(torch::kFloat64).mean();                    // scalar float64
    }

    auto effective_max_f = max_abs.to(torch::kFloat64);
    const float clip_k = bsi_cuda_fixed_clip_k();
    if (clip_k > 0.0f) {
        // Outlier handling: cap absmax by (clip_k * mean_abs) when selecting shift.
        // This can reduce over-shifting due to a few extreme values, at the cost of saturating those outliers.
        auto clip_max = mean_abs * static_cast<double>(clip_k);
        effective_max_f = torch::where(clip_max.lt(effective_max_f), clip_max, effective_max_f);
    }

    // Compute shift so that bit_width(effective_max) <= fixed_bits-1 (excluding sign bit).
    auto eff_safe = torch::where(effective_max_f.gt(0.0), effective_max_f, torch::ones_like(effective_max_f));
    auto bits_f = torch::floor(torch::log2(eff_safe)) + 1.0;
    auto shift_f = bits_f - static_cast<double>(fixed_bits - 1);
    auto shift = torch::where(shift_f.gt(0.0), shift_f, torch::zeros_like(shift_f))
                     .to(torch::kInt64)
                     .contiguous();

    // Apply rounded right shift (signed) and clamp to intN.
    torch::Tensor ints_shifted;
    if (chunk_scale && per_row_scale) {
        const int64_t Q = ints_for_shift.size(0);
        TORCH_CHECK(chunks > 0 && chunk_elems == 256, "Invalid chunk_scale state");
        auto ints_view = ints_for_shift.view({Q, chunks, chunk_elems});
        auto shift_view = shift.unsqueeze(2);
        auto shifted_view = round_shift_right_signed(ints_view, shift_view);
        const int64_t d = ints.size(1);
        ints_shifted = shifted_view.view({Q, ints_for_shift.size(1)}).narrow(1, 0, d);
    } else {
        torch::Tensor shift_b = shift;
        if (per_row_scale) {
            shift_b = shift.unsqueeze(1);
        }
        ints_shifted = round_shift_right_signed(ints, shift_b);
    }

    const int64_t qmax_i = (int64_t(1) << (fixed_bits - 1)) - 1;
    const int64_t qmin_i = -(int64_t(1) << (fixed_bits - 1));
    ints_shifted = torch::clamp(ints_shifted, qmin_i, qmax_i).contiguous();

    // Compensate the shift by scaling slice weights by 2^shift (exact in fp32 for shift <= 62).
    // (Clamp shift for integer bitshift safety; larger shifts would overflow int64 anyway.)
    auto shift_safe = torch::clamp(shift, 0, 62);
    auto scale_i64 = torch::bitwise_left_shift(torch::ones_like(shift_safe), shift_safe);
    auto scale = scale_i64.to(torch::kFloat32).contiguous();

    QuantizedIntsAndScale out;
    out.ints = ints_shifted;
    out.scale = scale;
    return out;
}

static inline int choose_total_slices(bool any_non_zero, long long max_abs, int fixed_bits) {
    if (fixed_bits > 0) {
        return fixed_bits;
    }
    const int magnitude_bits = any_non_zero
        ? std::max(1, static_cast<int>(std::bit_width(static_cast<unsigned long long>(max_abs))))
        : 1;
    return any_non_zero ? std::min(64, magnitude_bits + 2) : 2;
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
       << " scale=" << scale
       << std::endl;
}

BsiVectorCudaData build_bsi_vector_from_float_tensor(const torch::Tensor& input,
                                                     int decimal_places,
                                                     const torch::Device& device,
                                                     bool verbose) {
    TORCH_CHECK(input.dim() == 1, "build_bsi_vector_from_float_tensor expects 1D tensor");
    const int fixed_bits = bsi_cuda_fixed_bits_queries();
    auto q = bsi_cuda_quantize_to_int64_and_scale(
        input, decimal_places, device, /*per_row_scale=*/false, fixed_bits, /*chunk_scale=*/false);
    auto scaled = q.ints;
    maybe_log_scaled(scaled);
    const int64_t rows = scaled.size(0);

    bool any_non_zero = (rows > 0) && scaled.ne(0).any().item<bool>();
    bool has_negative = (rows > 0) && scaled.lt(0).any().item<bool>();
    // all_zero is unused; keep any_non_zero for early exits only.

    long long max_abs = 0;
    if (any_non_zero) {
        max_abs = scaled.abs().max().item<int64_t>();
    }

    int total_slices = choose_total_slices(any_non_zero, max_abs, fixed_bits);

    if (!any_non_zero) {
        total_slices = choose_total_slices(false, 0, fixed_bits);
        has_negative = false;
    }

    // For parity with CPU decimal builder, do not trim low zero bitplanes.
    // In fixed-bit mode, slice count is fixed and we carry a floating-point
    // scale (data.scale) to reconstruct decimal-scaled values.
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
        // Use warp-ballot optimized packer for faster bitplane build
        launch_pack_bits_all_ballot(
            reinterpret_cast<const long long*>(shifted_ptr),
            static_cast<long long>(rows),
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
    data.scale = (fixed_bits > 0) ? q.scale.item<float>() : 1.0f;
    data.words = words;
    data.metadata = torch::empty({stored_slices, 0},
                                 torch::TensorOptions().dtype(torch::kInt32).device(device));

    if (verbose || bsi_cuda_should_log()) {
        data.log("build_bsi_vector_from_float_tensor");
    }
    return data;
}

BsiVectorCudaData build_bsi_vector_from_float_tensor_hybrid(const torch::Tensor& input,
                                                            int decimal_places,
                                                            double compress_threshold,
                                                            const torch::Device& device,
                                                            bool verbose) {
    TORCH_CHECK(input.dim() == 1, "build_bsi_vector_from_float_tensor_hybrid expects 1D tensor");
    const int fixed_bits = bsi_cuda_fixed_bits_queries();
    auto q = bsi_cuda_quantize_to_int64_and_scale(
        input, decimal_places, device, /*per_row_scale=*/false, fixed_bits, /*chunk_scale=*/false);
    auto scaled = q.ints;
    const int64_t rows = scaled.size(0);

    bool any_non_zero = (rows > 0) && scaled.ne(0).any().item<bool>();
    bool has_negative = (rows > 0) && scaled.lt(0).any().item<bool>();

    long long max_abs = 0;
    if (any_non_zero) {
        max_abs = scaled.abs().max().item<int64_t>();
    }
    int total_slices = choose_total_slices(any_non_zero, max_abs, fixed_bits);
    if (!any_non_zero) {
        total_slices = choose_total_slices(false, 0, fixed_bits);
        has_negative = false;
    }

    int offset = 0;
    int stored_slices = std::max(1, total_slices);
    const int W = rows > 0 ? static_cast<int>((rows + 63) / 64) : 1;

    // Build temporary verbatim words on device quickly; we release them after compression
    auto words = torch::zeros({stored_slices, W}, torch::dtype(torch::kInt64).device(device));
    if (rows > 0) {
        unsigned long long value_mask = (stored_slices >= 64) ? ~0ULL : ((1ULL << stored_slices) - 1ULL);
        auto stream = at::cuda::getCurrentCUDAStream();
        auto* shifted_ptr = tensor_data_ptr<int64_t>(scaled);
        launch_pack_bits_all_ballot(
            reinterpret_cast<const long long*>(shifted_ptr),
            static_cast<long long>(rows),
            stored_slices,
            W,
            value_mask,
            reinterpret_cast<unsigned long long*>(tensor_data_ptr<int64_t>(words)),
            stream.stream());
    }


    // Per-slice popcounts and compress flags on device
    auto slice_counts = torch::empty({stored_slices}, torch::dtype(torch::kInt64).device(device));
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_slice_popcount_sum(
        reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(words)),
        stored_slices,
        W,
        reinterpret_cast<unsigned long long*>(tensor_data_ptr<int64_t>(slice_counts)),
        stream.stream());

    auto flags = torch::empty({stored_slices}, torch::dtype(torch::kInt32).device(device));
    launch_compress_flags_from_density(
        reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(slice_counts)),
        stored_slices,
        W,
        compress_threshold,
        flags.data_ptr<int>(),
        stream.stream());

    // Size pass per slice
    auto sizes = torch::empty({stored_slices}, torch::dtype(torch::kInt64).device(device));
    launch_ewah_size(
        reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(words)),
        stored_slices,
        W,
        flags.data_ptr<int>(),
        reinterpret_cast<unsigned long long*>(tensor_data_ptr<int64_t>(sizes)),
        stream.stream());

    // Exclusive scan to offsets and total size
    auto inclusive = sizes.cumsum(0);
    int64_t total_u64 = inclusive.index({stored_slices - 1}).item<int64_t>();
    auto comp_off = torch::empty({stored_slices}, torch::dtype(torch::kInt64).device(device));
    if (stored_slices > 0) {
        comp_off.index_put_({0}, 0);
        if (stored_slices > 1) {
            auto shifted = inclusive.narrow(0, 0, stored_slices - 1).contiguous();
            comp_off.narrow(0, 1, stored_slices - 1).copy_(shifted);
        }
    }
    auto comp_words = torch::empty({total_u64}, torch::dtype(torch::kInt64).device(device));
    auto comp_len = torch::empty({stored_slices}, torch::dtype(torch::kInt32).device(device));

    // Emit pass
    launch_ewah_emit(
        reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(words)),
        stored_slices,
        W,
        flags.data_ptr<int>(),
        reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(comp_off)),
        reinterpret_cast<unsigned long long*>(tensor_data_ptr<int64_t>(comp_words)),
        comp_len.data_ptr<int>(),
        stream.stream());

    BsiVectorCudaData data;
    data.rows = rows;
    data.slices = stored_slices;
    data.words_per_slice = W;
    data.offset = offset;
    data.decimals = decimal_places;
    data.twos_complement = has_negative;
    data.scale = (fixed_bits > 0) ? q.scale.item<float>() : 1.0f;
    data.words = torch::empty({0}, torch::dtype(torch::kInt64).device(device)); // not stored
    data.metadata = torch::empty({stored_slices, 0}, torch::dtype(torch::kInt32).device(device));
    data.comp_words = comp_words;
    data.comp_off = comp_off;
    data.comp_len = comp_len;
    if (verbose || bsi_cuda_should_log()) data.log("build_bsi_vector_from_float_tensor_hybrid");
    return data;
}

BsiQueryBatchCudaData build_bsi_queries_cuda_batch_data(const torch::Tensor& input,
                                                        int decimal_places,
                                                        const torch::Device& device,
                                                        bool verbose,
                                                        bool for_keys) {
    TORCH_CHECK(input.dim() == 2, "build_bsi_queries_cuda_batch_data expects 2D tensor [Q, d]");
    const int fixed_bits = for_keys ? bsi_cuda_fixed_bits_keys() : bsi_cuda_fixed_bits_queries();
    // Optional per-256-element chunk scaling for queries (activations) in fixed-bit mode.
    // Keys/weights use per-row scaling by default (for reproducibility + lower overhead).
    bool chunk_scale = false;
    if (!for_keys && fixed_bits > 0 && bsi_cuda_fixed_chunk_scale_queries()) {
        // Only enable chunk scaling when the bitplane word count is compatible with BMMA chunks (256 bits).
        const int64_t d = input.size(1);
        const int words_per_slice = (d > 0) ? static_cast<int>((d + 63) / 64) : 1;
        if ((words_per_slice % 4) == 0) {
            int dev = 0;
            cudaGetDevice(&dev);
            int major = 0;
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
            if (major >= 9) {
                chunk_scale = true;
            }
        }
    }
    const bool profile = bsi_cuda_profile_enabled_local();
    const bool profile_cuda = profile && input.is_cuda();
    float quantize_ms = 0.0f;
    float pack_ms = 0.0f;
    const int64_t Q_input = input.size(0);
    const int64_t d_input = input.size(1);

    // Fused fixed-bit query builder: compute shifts/scales directly from input and
    // quantize+pack directly into bitplanes (no intermediate int64 staging tensor).
    if (!for_keys &&
        fixed_bits > 0 &&
        input.is_cuda() &&
        bsi_cuda_custom_quant_enabled() &&
        bsi_cuda_fused_qpack_enabled()) {
        const int offset = 0;
        const int slices = std::max(1, fixed_bits);
        const int words_per_slice = (d_input > 0) ? static_cast<int>((d_input + 63) / 64) : 1;

        torch::Tensor words;
        torch::Tensor scale;
        const bool fused_ok = bsi_cuda_build_fixed_query_words_fused(
            input,
            decimal_places,
            fixed_bits,
            chunk_scale,
            device,
            profile_cuda,
            words,
            scale,
            quantize_ms,
            pack_ms);

        if (fused_ok) {
            auto weights_twos = make_slice_weights_cuda_cached(slices, offset, true, device);
            auto slice_weights = weights_twos.unsqueeze(0).expand({Q_input, slices});
            BsiQueryBatchCudaData out;
            out.rows = Q_input;
            out.slices = slices;
            out.words_per_slice = words_per_slice;
            out.offset = offset;
            out.words = words.contiguous();
            if (chunk_scale) {
                TORCH_CHECK(scale.defined() && scale.dim() == 2, "Expected [Q, chunks] chunk scales");
                out.slice_weights = slice_weights.contiguous();
                out.chunk_scales = scale.contiguous();
            } else {
                TORCH_CHECK(scale.defined() && scale.dim() == 1, "Expected [Q] row scales");
                out.slice_weights = (slice_weights * scale.unsqueeze(1)).contiguous();
            }

            if (profile_cuda) {
                g_last_query_build_quantize_ns = static_cast<uint64_t>(quantize_ms * 1.0e6);
                g_last_query_build_pack_ns = static_cast<uint64_t>(pack_ms * 1.0e6);
                g_last_query_build_total_ns = g_last_query_build_quantize_ns + g_last_query_build_pack_ns;
            } else {
                g_last_query_build_quantize_ns = 0;
                g_last_query_build_pack_ns = 0;
                g_last_query_build_total_ns = 0;
            }

            if (verbose || bsi_cuda_should_log()) {
                std::cout << "[BSI_CUDA] build_bsi_queries_cuda_batch_data (fused): "
                          << "Q=" << Q_input
                          << " slices=" << slices
                          << " words_per_slice=" << words_per_slice
                          << std::endl;
            }
            return out;
        }
    }

    cudaEvent_t quantize_start_evt = nullptr;
    cudaEvent_t quantize_end_evt = nullptr;
    if (profile_cuda) {
        cudaEventCreate(&quantize_start_evt);
        cudaEventCreate(&quantize_end_evt);
        auto stream = at::cuda::getCurrentCUDAStream();
        cudaEventRecord(quantize_start_evt, stream.stream());
    }
    auto q = bsi_cuda_quantize_to_int64_and_scale(
        input, decimal_places, device, /*per_row_scale=*/true, fixed_bits, chunk_scale);
    if (profile_cuda) {
        auto stream = at::cuda::getCurrentCUDAStream();
        cudaEventRecord(quantize_end_evt, stream.stream());
        cudaEventSynchronize(quantize_end_evt);
        cudaEventElapsedTime(&quantize_ms, quantize_start_evt, quantize_end_evt);
        cudaEventDestroy(quantize_start_evt);
        cudaEventDestroy(quantize_end_evt);
    }
    auto scaled = q.ints;
    const int64_t Q = scaled.size(0);
    const int64_t d = scaled.size(1);

    int total_slices = 2;
    if (fixed_bits > 0) {
        // Fixed-bit mode uses a constant slice count and avoids device->host syncs.
        total_slices = fixed_bits;
    } else {
        bool any_non_zero = (Q > 0) && scaled.ne(0).any().item<bool>();
        long long max_abs = 0;
        if (any_non_zero) {
            max_abs = scaled.abs().max().item<int64_t>();
        }
        total_slices = choose_total_slices(any_non_zero, max_abs, fixed_bits);
        if (!any_non_zero) {
            total_slices = choose_total_slices(false, 0, fixed_bits);
        }
    }

    int offset = 0;
    int slices = std::max(1, total_slices);
    const int words_per_slice = (d > 0) ? static_cast<int>((d + 63) / 64) : 1;

    auto words = torch::zeros({Q, slices, words_per_slice},
                              torch::TensorOptions().dtype(torch::kInt64).device(device));

    if (Q > 0 && d > 0) {
        cudaEvent_t pack_start_evt = nullptr;
        cudaEvent_t pack_end_evt = nullptr;
        if (profile_cuda) {
            cudaEventCreate(&pack_start_evt);
            cudaEventCreate(&pack_end_evt);
            auto stream = at::cuda::getCurrentCUDAStream();
            cudaEventRecord(pack_start_evt, stream.stream());
        }
        unsigned long long value_mask = (slices >= 64)
            ? ~0ULL
            : ((1ULL << slices) - 1ULL);
        auto stream = at::cuda::getCurrentCUDAStream();
        auto* scaled_ptr = tensor_data_ptr<int64_t>(scaled);
        launch_pack_bits_all_ballot_batch(
            reinterpret_cast<const long long*>(scaled_ptr),
            static_cast<int>(Q),
            static_cast<long long>(d),
            slices,
            words_per_slice,
            value_mask,
            reinterpret_cast<unsigned long long*>(tensor_data_ptr<int64_t>(words)),
            stream.stream());
        if (profile_cuda) {
            auto stream2 = at::cuda::getCurrentCUDAStream();
            cudaEventRecord(pack_end_evt, stream2.stream());
            cudaEventSynchronize(pack_end_evt);
            cudaEventElapsedTime(&pack_ms, pack_start_evt, pack_end_evt);
            cudaEventDestroy(pack_start_evt);
            cudaEventDestroy(pack_end_evt);
        }
    }

    torch::Tensor slice_weights;
    if (fixed_bits > 0) {
        // Fixed-bit path always uses signed representation; base twos-complement
        // slice weights are reused across all rows.
        auto weights_twos = make_slice_weights_cuda_cached(slices, offset, true, device);
        slice_weights = weights_twos.unsqueeze(0).expand({Q, slices});
        if (chunk_scale) {
            // Fixed-bit chunk-scale path: per-chunk power-of-two scales are applied inside the dot kernel.
            // Keep slice_weights as the exact base powers-of-two (with sign handling).
            TORCH_CHECK(q.scale.defined() && q.scale.dim() == 2, "Expected [Q, chunks] chunk scales");
        } else {
            // Fixed-bit path: per-row power-of-two scale (2^shift) computed during quantization.
            TORCH_CHECK(q.scale.defined() && q.scale.dim() == 1, "Expected [Q] row scales");
            slice_weights = slice_weights * q.scale.unsqueeze(1);
        }
    } else {
        auto neg_flags = (Q > 0)
            ? scaled.lt(0).any(1)
            : torch::zeros({Q}, torch::TensorOptions().dtype(torch::kBool).device(device));
        auto weights_unsigned = make_slice_weights_cuda_cached(slices, offset, false, device);
        auto weights_twos = make_slice_weights_cuda_cached(slices, offset, true, device);
        slice_weights = torch::where(neg_flags.unsqueeze(1), weights_twos, weights_unsigned);
    }

    BsiQueryBatchCudaData out;
    out.rows = Q;
    out.slices = slices;
    out.words_per_slice = words_per_slice;
    out.offset = offset;
    out.words = words;
    out.slice_weights = slice_weights.contiguous();
    if (fixed_bits > 0 && chunk_scale) {
        out.chunk_scales = q.scale.contiguous();
    }

    if (profile_cuda) {
        g_last_query_build_quantize_ns = static_cast<uint64_t>(quantize_ms * 1.0e6);
        g_last_query_build_pack_ns = static_cast<uint64_t>(pack_ms * 1.0e6);
        g_last_query_build_total_ns = g_last_query_build_quantize_ns + g_last_query_build_pack_ns;
    } else {
        g_last_query_build_quantize_ns = 0;
        g_last_query_build_pack_ns = 0;
        g_last_query_build_total_ns = 0;
    }

    if (verbose || bsi_cuda_should_log()) {
        std::cout << "[BSI_CUDA] build_bsi_queries_cuda_batch_data: "
                  << "Q=" << Q
                  << " slices=" << slices
                  << " words_per_slice=" << words_per_slice
                  << std::endl;
    }
    return out;
}
BsiVectorCudaData create_bsi_vector_cuda_from_cpu(const BsiVector<uint64_t>& src,
                                                  const torch::Device& device,
                                                  bool verbose) {
    std::vector<uint64_t> words;
    int slices = 0;
    int words_per_slice = 0;
    bsi_flatten_words_gpu_helper<uint64_t>(src, words, slices, words_per_slice);
    int offset = src.offset;

    BsiVectorCudaData data;
    data.rows = src.getNumberOfRows();
    data.slices = slices;
    data.words_per_slice = words_per_slice;
    data.offset = offset;
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
    // Keep this helper consistent with the actual CUDA builders:
    // when fixed-bit mode is enabled we return the quantized/clamped ints that are packed into bitplanes.
    auto q = bsi_cuda_quantize_to_int64_and_scale(
        input,
        decimal_places,
        device,
        /*per_row_scale=*/(input.dim() == 2),
        /*fixed_bits=*/bsi_cuda_fixed_bits_queries(),
        /*chunk_scale=*/false);
    return q.ints;
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
