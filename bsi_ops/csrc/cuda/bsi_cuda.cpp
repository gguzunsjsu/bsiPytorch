#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <cstdint>
#include <unordered_map>
#include <c10/cuda/CUDAFunctions.h>
#include <ATen/Parallel.h>
#include <cstdlib>
#include <string>

// bring in BSI core (CPU) to build and access slices
#include "../../../bsiCPP/bsi/BsiVector.hpp"
#include "../../../bsiCPP/bsi/BsiSigned.hpp"
#include "../../../bsiCPP/bsi/BsiUnsigned.hpp"
#include "../../../bsiCPP/bsi/hybridBitmap/hybridbitmap.h"
#include "../../../bsiCPP/bsi/hybridBitmap/runninglengthword.h"

#include "bsi_vector_cuda.h"

using u64 = uint64_t;

// kernel launcher decls (implemented in .cu)
extern "C" void launch_popcount_pairwise(
    const unsigned long long* A,
    const unsigned long long* B,
    int Sa, int Sb, int W,
    unsigned long long* out,
    cudaStream_t stream);

extern "C" void launch_weighted_reduction(
    const unsigned long long* counts,
    const double* Aw,
    const double* Bw,
    int Sa, int Sb, int R,
    const long long* indices,
    double scale_inv,
    double* out_global,
    cudaStream_t stream);

extern "C" void launch_ewah_decompress(
    const unsigned long long* in,
    int in_len,
    int W,
    unsigned long long* out,
    cudaStream_t stream);

// no multi-slice EWAH decompress in Phase-2

extern "C" void launch_scatter_set_double(
    const long long* idx,
    const double* src,
    int n,
    double* out,
    cudaStream_t stream);

extern "C" void launch_popcount_weighted_keys_literal(
    const unsigned long long* A,
    const double* Aw,
    int Sa,
    int W,
    const unsigned long long* B,
    const double* Bw,
    int Sb,
    int R,
    double scale_inv,
    double* out,
    cudaStream_t stream);
extern "C" void launch_popcount_weighted_keys_literal_tiled(
    const unsigned long long* A,
    const double* Aw,
    int Sa,
    int W,
    const unsigned long long* B,
    const double* Bw,
    int Sb,
    int R,
    int tile_size,
    double scale_inv,
    double* out,
    cudaStream_t stream);
extern "C" void launch_popcount_weighted_keys_literal_fused(
    const unsigned long long* A,
    const double* Aw,
    int Sa,
    int W,
    const unsigned long long* B,
    const double* Bw,
    int Sb,
    int R,
    const long long* indices,
    double scale_inv,
    double* out_global,
    cudaStream_t stream);

static bool bsi_cuda_use_tiled() {
    static int cached = -1;
    if (cached >= 0) return cached != 0;
    const char* v = std::getenv("BSI_TILED");
    if (!v) { cached = 1; return true; }
    std::string s(v);
    for (auto& c : s) c = (char)std::tolower(c);
    cached = (s == "1" || s == "true" || s == "yes") ? 1 : 0;
    return cached != 0;
}

static inline uint64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

template <typename T>
inline T* tensor_data_ptr(torch::Tensor& t) {
    return t.data_ptr<T>();
}

template <typename T>
inline const T* tensor_data_ptr(const torch::Tensor& t) {
    auto& nc = const_cast<torch::Tensor&>(t);
    return const_cast<const T*>(nc.data_ptr<T>());
}

// Flatten one HybridBitmap (compressed or verbatim) to verbatim word buffer of length W words
struct PrebuiltBSIKeys {
    std::vector<BsiVector<u64>*> keys;
    int decimalPlaces = 0;
    int64_t d = 0;
    int64_t num_keys = 0;
    float threshold = 0.2f;
};

static PrebuiltBSIKeys* capsule_to_keys(const pybind11::capsule& cap) {
    return reinterpret_cast<PrebuiltBSIKeys*>(cap.get_pointer());
}

static double accumulate_weighted_dot(const BsiVector<u64>* a, const BsiVector<u64>* b,
                                      const at::Tensor& counts_cpu) {
    auto ca = counts_cpu.accessor<int64_t, 2>(); // [Sb, Sa]
    int Sa = a->numSlices;
    int Sb = b->numSlices;

    auto weight_for = [](const BsiVector<u64>* vec, int idx) -> long double {
        int shift = vec->offset + idx;
        long double w = (shift >= 0) ? std::ldexp(1.0L, shift) : 0.0L;
        if (vec->twosComplement && idx == vec->numSlices - 1) w = -w;
        return w;
    };

    long double acc = 0.0L;
    for (int j = 0; j < Sb; ++j) {
        long double wb = weight_for(b, j);
        if (wb == 0) continue;
        for (int i = 0; i < Sa; ++i) {
            long double wa = weight_for(a, i);
            if (wa == 0) continue;
            long double cnt = static_cast<long double>(ca[j][i]);
            if (cnt == 0) continue;
            acc += wa * wb * cnt;
        }
    }
    int totalDecimals = a->decimals + b->decimals;
    long double scale = (totalDecimals > 0) ? std::pow(10.0L, totalDecimals) : 1.0L;
    return static_cast<double>(acc / scale);
}

// --- GPU prebuilt keys (device-packed words) ---
struct KeyMeta {
    int S = 0;
    int offset = 0;
    bool twos = true;
    int decimals = 0;
};

struct PrebuiltBSIKeysCUDA {
    std::vector<at::Tensor> dev_words; // each [S_k, W], int64, cuda
    std::vector<at::Tensor> slice_weights; // each [S_k], float64, cuda
    std::vector<KeyMeta> metas;
    std::vector<BsiVectorCudaData> device_views;
    // Grouped, contiguous views by Sb to avoid per-call stacking
    std::unordered_map<int, at::Tensor> grouped_words;   // Sb -> [R_sb, Sb, W]
    std::unordered_map<int, at::Tensor> grouped_weights; // Sb -> [R_sb, Sb]
    std::unordered_map<int, std::vector<int64_t>> grouped_indices; // Sb -> original key indices
    std::unordered_map<int, at::Tensor> grouped_indices_dev; // Sb -> [R_sb] int64 cuda
    // Compressed grouped layout (EWAH):
    // For each Sb group: a big 1D comp_words buffer, plus absolute per-slice offsets and lengths per key
    std::unordered_map<int, at::Tensor> grouped_comp_words; // Sb -> [total_u64]
    std::unordered_map<int, at::Tensor> grouped_comp_off;   // Sb -> [R_sb, Sb] (int64, absolute offsets)
    std::unordered_map<int, at::Tensor> grouped_comp_len;   // Sb -> [R_sb, Sb] (int32)
    int W = 0;
    int64_t d = 0;
    int64_t num_keys = 0;
    int decimals = 0;
    float threshold = 0.2f;
};

static PrebuiltBSIKeysCUDA* capsule_to_keys_cuda(const pybind11::capsule& cap) {
    return reinterpret_cast<PrebuiltBSIKeysCUDA*>(cap.get_pointer());
}

struct PrebuiltBSIQueryCUDA {
    BsiVector<u64>* vec = nullptr;
    BsiVectorCudaData device_view;
    at::Tensor dev_words;      // alias of device_view.words
    at::Tensor slice_weights;  // [S]
    int S = 0;
    int W = 0;
    size_t mem_bytes = 0;
};

static PrebuiltBSIQueryCUDA* capsule_to_query_cuda(const pybind11::capsule& cap) {
    return reinterpret_cast<PrebuiltBSIQueryCUDA*>(cap.get_pointer());
}

static inline long double weight_for_meta(int offset, int idx, bool twos, int S) {
    int shift = offset + idx;
    long double w = (shift >= 0) ? std::ldexp(1.0L, shift) : 0.0L;
    if (twos && idx == S - 1) w = -w;
    return w;
}

static inline at::Tensor make_slice_weights_cuda(int S, int offset, bool twos) {
    std::vector<double> host(S);
    for (int i = 0; i < S; ++i) {
        long double w = weight_for_meta(offset, i, twos, S);
        host[i] = static_cast<double>(w);
    }
    return torch::from_blob(
               host.data(),
               {S},
               torch::TensorOptions().dtype(torch::kFloat64))
        .clone()
        .to(torch::kCUDA);
}

static double accumulate_weighted_dot_meta(
    int Sa, int offset_a, bool twos_a, int decimals_a,
    const at::Tensor& counts_cpu,
    const KeyMeta& kb)
{
    auto ca = counts_cpu.accessor<int64_t, 2>(); // [Sb, Sa]
    long double acc = 0.0L;
    for (int j = 0; j < kb.S; ++j) {
        long double wb = weight_for_meta(kb.offset, j, kb.twos, kb.S);
        if (wb == 0) continue;
        for (int i = 0; i < Sa; ++i) {
            long double wa = weight_for_meta(offset_a, i, twos_a, Sa);
            if (wa == 0) continue;
            long double cnt = static_cast<long double>(ca[j][i]);
            if (cnt == 0) continue;
            acc += wa * wb * cnt;
        }
    }
    int totalDecimals = decimals_a + kb.decimals;
    long double scale = (totalDecimals > 0) ? std::pow(10.0L, totalDecimals) : 1.0L;
    return static_cast<double>(acc / scale);
}

static pybind11::tuple dot_product_decimal_cuda(torch::Tensor q, torch::Tensor k, int decimalPlaces) {
    TORCH_CHECK(q.dim() == 1 && k.dim() == 1 && q.size(0) == k.size(0), "q,k shapes");
    auto q_cpu = q.detach().to(torch::kFloat32).cpu().contiguous();
    auto k_cpu = k.detach().to(torch::kFloat32).cpu().contiguous();

    // Build BSI on CPU, allow compression per default threshold (0.2)
    std::vector<double> qv(q_cpu.size(0)), kv(k_cpu.size(0));
    auto qa = q_cpu.accessor<float,1>();
    auto ka = k_cpu.accessor<float,1>();
    for (int64_t i = 0; i < qa.size(0); ++i) { qv[i] = qa[i]; kv[i] = ka[i]; }

    BsiSigned<u64> builder;
    BsiVector<u64>* bsi_q = builder.buildBsiVector(qv, decimalPlaces, 0.2f);
    BsiVector<u64>* bsi_k = builder.buildBsiVector(kv, decimalPlaces, 0.2f);
    bsi_q->setPartitionID(0); bsi_q->setFirstSliceFlag(true); bsi_q->setLastSliceFlag(true);
    bsi_k->setPartitionID(0); bsi_k->setFirstSliceFlag(true); bsi_k->setLastSliceFlag(true);

    // Flatten to words [S,W]
    std::vector<u64> A_host, B_host;
    int Sa=0, Wa=0, Sb=0, Wb=0;
    bsi_flatten_words_gpu_helper(*bsi_q, A_host, Sa, Wa);
    bsi_flatten_words_gpu_helper(*bsi_k, B_host, Sb, Wb);
    TORCH_CHECK(Wa == Wb, "word count mismatch");

    // Device buffers
    auto A_dev = torch::from_blob(A_host.data(), {Sa, Wa}, torch::dtype(torch::kInt64)).clone().to(torch::kCUDA);
    auto B_dev = torch::from_blob(B_host.data(), {Sb, Wa}, torch::dtype(torch::kInt64)).clone().to(torch::kCUDA);
    auto counts_dev = torch::empty({Sb, Sa}, torch::device(torch::kCUDA).dtype(torch::kInt64));

    dim3 grid(Sa, Sb);
    dim3 block(256);

    // Kernel timing
    cudaEvent_t start, end;
    cudaEventCreate(&start); cudaEventCreate(&end);
    cudaEventRecord(start);
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_popcount_pairwise(
        reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(A_dev)),
        reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(B_dev)),
        Sa, Sb, Wa,
        reinterpret_cast<unsigned long long*>(tensor_data_ptr<int64_t>(counts_dev)),
        stream.stream());
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start, end);
    cudaEventDestroy(start); cudaEventDestroy(end);

    auto counts_cpu = counts_dev.to(torch::kCPU).contiguous();
    double result = accumulate_weighted_dot(bsi_q, bsi_k, counts_cpu);

    size_t mem_q = bsi_q->getSizeInMemory();
    size_t mem_k = bsi_k->getSizeInMemory();
    delete bsi_q; delete bsi_k;
    uint64_t dot_ns = static_cast<uint64_t>(kernel_ms * 1.0e6);
    return pybind11::make_tuple(result, dot_ns /*total ns*/, (uint64_t)mem_q, (uint64_t)mem_k);
}

static pybind11::tuple build_bsi_query_cuda(torch::Tensor q, int decimalPlaces, float compress_threshold = 0.2f) {
    TORCH_CHECK(q.dim() == 1, "q must be 1D [d]");
    (void)compress_threshold;

    auto* holder = new PrebuiltBSIQueryCUDA();
    holder->vec = nullptr;

    auto device = torch::Device(torch::kCUDA, c10::cuda::current_device());
    bool verbose = bsi_cuda_should_log();
    holder->device_view = build_bsi_vector_from_float_tensor(q.detach(), decimalPlaces, device, verbose);
    holder->S = holder->device_view.slices;
    holder->W = holder->device_view.words_per_slice;
    holder->dev_words = holder->device_view.words;
    holder->slice_weights = make_slice_weights_cuda(holder->S,
                                                    holder->device_view.offset,
                                                    holder->device_view.twos_complement);
    holder->mem_bytes = static_cast<size_t>(holder->dev_words.numel() * holder->dev_words.element_size());

    pybind11::capsule cap(holder, "PrebuiltBSIQueryCUDA",
        [](PyObject* capsule) {
            auto* p = reinterpret_cast<PrebuiltBSIQueryCUDA*>(PyCapsule_GetPointer(capsule, "PrebuiltBSIQueryCUDA"));
            if (p) {
                delete p->vec;
                delete p;
            }
        }
    );

    return pybind11::make_tuple(cap, static_cast<uint64_t>(holder->mem_bytes), holder->S, holder->W);
}

static pybind11::tuple batch_dot_product_prebuilt_cuda(torch::Tensor q, pybind11::capsule keyset_cap,
                                                       float query_threshold = 0.2f) {
    TORCH_CHECK(q.dim() == 1, "q must be 1D [d]");
    auto* keys = capsule_to_keys(keyset_cap);
    TORCH_CHECK(keys != nullptr, "Invalid BSI keys capsule");

    const int64_t d = keys->d;
    TORCH_CHECK(q.size(0) == d, "q.size(0) must equal keys' dimension");
    const int decimalPlaces = keys->decimalPlaces;
    float threshold_val = query_threshold >= 0.0f ? query_threshold : keys->threshold;

    // Build query BSI on CPU
    auto q_cpu = q.detach().to(torch::kFloat32).cpu().contiguous();
    auto qa = q_cpu.accessor<float,1>();
    std::vector<double> qv; qv.reserve(d);
    for (int64_t i = 0; i < d; ++i) qv.push_back(static_cast<double>(qa[i]));

    uint64_t t_build0 = now_ns();
    BsiSigned<u64> b;
    BsiVector<u64>* bsi_q = b.buildBsiVector(qv, decimalPlaces, threshold_val);
    bsi_q->setPartitionID(0); bsi_q->setFirstSliceFlag(true); bsi_q->setLastSliceFlag(true);
    uint64_t build_ns = now_ns() - t_build0;

    // Flatten query
    std::vector<u64> A_host; int Sa=0, Wa=0;
    bsi_flatten_words_gpu_helper(*bsi_q, A_host, Sa, Wa);
    auto A_dev = torch::from_blob(A_host.data(), {Sa, Wa}, torch::dtype(torch::kInt64)).clone().to(torch::kCUDA);

    // Prepare output scores (CPU double)
    auto out = torch::empty({keys->num_keys}, torch::TensorOptions().dtype(torch::kFloat64));
    auto out_a = out.accessor<double,1>();

    float total_kernel_ms = 0.0f;
    for (int64_t r = 0; r < keys->num_keys; ++r) {
        BsiVector<u64>* bsi_k = keys->keys[r];
        TORCH_CHECK(bsi_k != nullptr, "Null BSI key at ", r);

        // Flatten key
        std::vector<u64> B_host; int Sb=0, Wb=0;
        bsi_flatten_words_gpu_helper(*bsi_k, B_host, Sb, Wb);
        TORCH_CHECK(Wb == Wa, "word count mismatch");
        auto B_dev = torch::from_blob(B_host.data(), {Sb, Wa}, torch::dtype(torch::kInt64)).clone().to(torch::kCUDA);
        auto counts_dev = torch::empty({Sb, Sa}, torch::device(torch::kCUDA).dtype(torch::kInt64));

        dim3 grid(Sa, Sb);
        dim3 block(256);
        cudaEvent_t start, end; cudaEventCreate(&start); cudaEventCreate(&end);
        cudaEventRecord(start);
        auto stream1 = at::cuda::getCurrentCUDAStream();
        launch_popcount_pairwise(
            reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(A_dev)),
            reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(B_dev)),
            Sa, Sb, Wa,
            reinterpret_cast<unsigned long long*>(tensor_data_ptr<int64_t>(counts_dev)),
            stream1.stream());
        cudaEventRecord(end); cudaEventSynchronize(end);
        float kernel_ms = 0.0f; cudaEventElapsedTime(&kernel_ms, start, end);
        cudaEventDestroy(start); cudaEventDestroy(end);
        total_kernel_ms += kernel_ms;

        auto counts_cpu = counts_dev.to(torch::kCPU).contiguous();
        double score = accumulate_weighted_dot(bsi_q, bsi_k, counts_cpu);
        out_a[r] = score;
    }

    delete bsi_q;
    uint64_t dot_ns = (uint64_t)(total_kernel_ms * 1.0e6);
    uint64_t total_ns = build_ns + dot_ns;
    size_t mem_q = 0; // not meaningful here beyond bsi_q->getSizeInMemory, but q deleted
    return pybind11::make_tuple(out, total_ns, build_ns, dot_ns, mem_q);
}

static pybind11::tuple batch_dot_product_prebuilt_cuda_caps(pybind11::capsule query_cap,
                                                            pybind11::capsule keyset_cuda_cap) {
    auto* query = capsule_to_query_cuda(query_cap);
    TORCH_CHECK(query != nullptr, "Invalid BSI query capsule");
    auto* keys = capsule_to_keys_cuda(keyset_cuda_cap);
    TORCH_CHECK(keys != nullptr, "Invalid CUDA keys capsule");

    TORCH_CHECK(query->W == keys->W, "Word count mismatch between query and keys");

    auto out_dev = torch::zeros({keys->num_keys}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
    const auto* query_words = reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(query->dev_words));
    const auto* query_weights = tensor_data_ptr<double>(query->slice_weights);

    // Precompute single scale factor when decimals are fixed across keys (they are, per-layer)
    const int totalDecimals = query->device_view.decimals + keys->decimals;
    const double scale_inv = (totalDecimals > 0) ? (1.0 / std::pow(10.0, totalDecimals)) : 1.0;

    auto stream = at::cuda::getCurrentCUDAStream();
    cudaEvent_t start_evt, end_evt;
    cudaEventCreate(&start_evt);
    cudaEventCreate(&end_evt);
    cudaEventRecord(start_evt, stream.stream());

    for (const auto& kv : keys->grouped_indices) {
        int Sb = kv.first;
        const auto& idxs = kv.second;
        const int R = static_cast<int>(idxs.size());
        auto out_slice = torch::zeros({R}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));

        const auto& words = keys->grouped_words.at(Sb);
        const auto& Bw_stacked = keys->grouped_weights.at(Sb);
        const auto* B_words = reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(words));
        const auto* Bw_ptr = tensor_data_ptr<double>(Bw_stacked);

        bool use_tiled = bsi_cuda_use_tiled();
        int tile_size = 64;
        if (const char* s = std::getenv("BSI_TILE_W")) {
            int t = std::atoi(s);
            if (t > 0) tile_size = t;
        }
        if (!use_tiled || query->W <= tile_size) {
            launch_popcount_weighted_keys_literal(
                query_words,
                query_weights,
                query->S,
                query->W,
                B_words,
                Bw_ptr,
                Sb,
                R,
                scale_inv,
                tensor_data_ptr<double>(out_slice),
                stream.stream());
        } else {
            tile_size = std::min(tile_size, query->W);
            launch_popcount_weighted_keys_literal_tiled(
                query_words,
                query_weights,
                query->S,
                query->W,
                B_words,
                Bw_ptr,
                Sb,
                R,
                tile_size,
                scale_inv,
                tensor_data_ptr<double>(out_slice),
                stream.stream());
        }

        const auto& idx_dev = keys->grouped_indices_dev.at(Sb);
        auto stream3 = at::cuda::getCurrentCUDAStream();
        launch_scatter_set_double(
            reinterpret_cast<const long long*>(tensor_data_ptr<int64_t>(idx_dev)),
            tensor_data_ptr<double>(out_slice),
            R,
            tensor_data_ptr<double>(out_dev),
            stream3.stream());
    }

    cudaEventRecord(end_evt, stream.stream());
    cudaEventSynchronize(end_evt);
    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start_evt, end_evt);
    cudaEventDestroy(start_evt);
    cudaEventDestroy(end_evt);

    // Return GPU scores directly
    uint64_t dot_ns = static_cast<uint64_t>(kernel_ms * 1.0e6);
    uint64_t total_ns = dot_ns;
    return pybind11::make_tuple(out_dev, total_ns, static_cast<uint64_t>(0), dot_ns, static_cast<uint64_t>(query->mem_bytes));
}

static std::string cuda_builder_version() {
    static const std::string version = std::string("CUDA_BUILDER_PHASE2_ROUNDING_V2 ") + __DATE__ + " " + __TIME__;
    return version;
}

static pybind11::tuple build_bsi_query_cuda_hybrid(torch::Tensor q, int decimalPlaces, double compress_threshold) {
    TORCH_CHECK(q.dim() == 1, "q must be 1D [d]");
    auto* holder = new PrebuiltBSIQueryCUDA();
    holder->vec = nullptr;
    auto device = torch::Device(torch::kCUDA, c10::cuda::current_device());
    bool verbose = bsi_cuda_should_log();
    holder->device_view = build_bsi_vector_from_float_tensor_hybrid(q.detach(), decimalPlaces, compress_threshold, device, verbose);
    holder->S = holder->device_view.slices;
    holder->W = holder->device_view.words_per_slice;
    holder->dev_words = torch::zeros({holder->S, holder->W}, torch::dtype(torch::kInt64).device(device));
    holder->slice_weights = make_slice_weights_cuda(holder->S,
                                                    holder->device_view.offset,
                                                    holder->device_view.twos_complement);
    if (holder->device_view.comp_words.defined() && holder->device_view.comp_words.numel() > 0) {
        auto stream = at::cuda::getCurrentCUDAStream();
        const int64_t* off_ptr = holder->device_view.comp_off.data_ptr<int64_t>();
        const int* len_ptr = holder->device_view.comp_len.data_ptr<int>();
        for (int s = 0; s < holder->S; ++s) {
            auto off = off_ptr[s];
            auto len = len_ptr[s];
            const unsigned long long* in_ptr = reinterpret_cast<const unsigned long long*>(
                holder->device_view.comp_words.data_ptr<int64_t>() + off);
            unsigned long long* out_ptr = reinterpret_cast<unsigned long long*>(
                holder->dev_words.data_ptr<int64_t>() + static_cast<int64_t>(s) * holder->W);
            launch_ewah_decompress(in_ptr, len, holder->W, out_ptr, stream.stream());
        }
    } else {
        auto fallback = build_bsi_vector_from_float_tensor(q.detach(), decimalPlaces, device, verbose);
        TORCH_CHECK(fallback.words.size(0) == holder->S && fallback.words.size(1) == holder->W,
                    "Hybrid fallback mismatch in query builder");
        holder->dev_words.copy_(fallback.words);
    }
    holder->mem_bytes = static_cast<size_t>(
        holder->device_view.comp_words.numel() * holder->device_view.comp_words.element_size() +
        holder->dev_words.numel() * holder->dev_words.element_size());
    pybind11::capsule cap(holder, "PrebuiltBSIQueryCUDA",
        [](PyObject* capsule) {
            auto* p = reinterpret_cast<PrebuiltBSIQueryCUDA*>(PyCapsule_GetPointer(capsule, "PrebuiltBSIQueryCUDA"));
            if (p) { delete p->vec; delete p; }
        }
    );
    return pybind11::make_tuple(cap, static_cast<uint64_t>(holder->mem_bytes), holder->S, holder->W);
}

static torch::Tensor debug_bsi_query_hybrid_decompress(pybind11::capsule query_cap) {
    auto* query = capsule_to_query_cuda(query_cap);
    TORCH_CHECK(query != nullptr, "Invalid BSI CUDA query capsule");
    const auto& dv = query->device_view;
    TORCH_CHECK(dv.comp_words.defined() && dv.comp_off.defined() && dv.comp_len.defined(), "No compressed view present");
    int S = dv.slices;
    int W = dv.words_per_slice;
    auto out = torch::zeros({S, W}, torch::dtype(torch::kInt64).device(torch::kCUDA));
    auto stream = at::cuda::getCurrentCUDAStream();
    for (int s = 0; s < S; ++s) {
        auto off = dv.comp_off.index({s}).item<int64_t>();
        auto len = dv.comp_len.index({s}).item<int>();
        auto in_ptr = reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(dv.comp_words) + off);
        auto out_ptr = reinterpret_cast<unsigned long long*>(tensor_data_ptr<int64_t>(out) + (long long)s * W);
        launch_ewah_decompress(in_ptr, len, W, out_ptr, stream.stream());
    }
    auto cpu = out.to(torch::kCPU).contiguous();
    return cpu;
}

static pybind11::tuple batch_dot_product_two_stage_cuda_caps(pybind11::list query_caps_list, pybind11::capsule keyset_cuda_cap) {
    auto* keys = capsule_to_keys_cuda(keyset_cuda_cap);
    TORCH_CHECK(keys != nullptr, "Invalid CUDA keys capsule");
    const int64_t R = keys->num_keys;
    const int64_t Q = static_cast<int64_t>(pybind11::len(query_caps_list));
    TORCH_CHECK(R > 0 && Q > 0, "Empty keys or queries");

    auto out_all = torch::zeros({Q, R}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));

    cudaEvent_t start_evt, end_evt;
    cudaEventCreate(&start_evt);
    cudaEventCreate(&end_evt);
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaEventRecord(start_evt, stream.stream());

    for (int64_t qi = 0; qi < Q; ++qi) {
        auto cap_obj = query_caps_list[qi];
        TORCH_CHECK(pybind11::isinstance<pybind11::capsule>(cap_obj), "Each item must be a capsule");
        auto cap = pybind11::cast<pybind11::capsule>(cap_obj);
        auto* query = capsule_to_query_cuda(cap);
        TORCH_CHECK(query != nullptr, "Invalid BSI query capsule at index ", qi);
        TORCH_CHECK(query->W == keys->W, "Word count mismatch between query and keys");

        const auto* A = reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(query->dev_words));
        const auto* Aw = tensor_data_ptr<double>(query->slice_weights);
        auto out_row = out_all[qi];

        int totalDecimals = query->device_view.decimals + keys->decimals;
        const double scale_inv = (totalDecimals > 0) ? (1.0 / std::pow(10.0, totalDecimals)) : 1.0;

        for (const auto& kv2 : keys->grouped_indices) {
            int Sb = kv2.first;
            const auto& idxs = kv2.second;
            const int Rg = static_cast<int>(idxs.size());

            const auto& words = keys->grouped_words.at(Sb);
            const auto& Bw_stacked = keys->grouped_weights.at(Sb);
            const auto* B_words = reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(words));
            const auto* Bw_ptr = tensor_data_ptr<double>(Bw_stacked);
            const auto& idx_dev = keys->grouped_indices_dev.at(Sb);

            // Allocate counts buffer: [Rg, Sb, Sa]
            auto counts = torch::empty({Rg, Sb, query->S},
                torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
            auto* counts_ptr = reinterpret_cast<unsigned long long*>(tensor_data_ptr<int64_t>(counts));

            // Stage 1: Popcount for all (r, j, i) combinations
            for (int r = 0; r < Rg; ++r) {
                const unsigned long long* B_r = B_words + (size_t)r * Sb * query->W;
                unsigned long long* counts_r = counts_ptr + (size_t)r * Sb * query->S;

                launch_popcount_pairwise(
                    A,
                    B_r,
                    query->S,
                    Sb,
                    query->W,
                    counts_r,
                    stream.stream());
            }

            // Stage 2: Weighted reduction
            launch_weighted_reduction(
                counts_ptr,
                Aw,
                Bw_ptr,
                query->S,
                Sb,
                Rg,
                reinterpret_cast<const long long*>(tensor_data_ptr<int64_t>(idx_dev)),
                scale_inv,
                tensor_data_ptr<double>(out_row),
                stream.stream());
        }
    }

    cudaEventRecord(end_evt, stream.stream());
    cudaEventSynchronize(end_evt);
    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start_evt, end_evt);
    cudaEventDestroy(start_evt);
    cudaEventDestroy(end_evt);

    uint64_t dot_ns = static_cast<uint64_t>(kernel_ms * 1.0e6);
    return pybind11::make_tuple(out_all, dot_ns);
}

static pybind11::tuple batch_dot_product_multiquery_cuda_caps(pybind11::list query_caps_list, pybind11::capsule keyset_cuda_cap) {
    auto* keys = capsule_to_keys_cuda(keyset_cuda_cap);
    TORCH_CHECK(keys != nullptr, "Invalid CUDA keys capsule");
    const int64_t R = keys->num_keys;
    const int64_t Q = static_cast<int64_t>(pybind11::len(query_caps_list));
    TORCH_CHECK(R > 0 && Q > 0, "Empty keys or queries");

    auto out_all = torch::zeros({Q, R}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));

    cudaEvent_t start_evt, end_evt;
    cudaEventCreate(&start_evt);
    cudaEventCreate(&end_evt);
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaEventRecord(start_evt, stream.stream());

    for (int64_t qi = 0; qi < Q; ++qi) {
        auto cap_obj = query_caps_list[qi];
        TORCH_CHECK(pybind11::isinstance<pybind11::capsule>(cap_obj), "Each item must be a capsule");
        auto cap = pybind11::cast<pybind11::capsule>(cap_obj);
        auto* query = capsule_to_query_cuda(cap);
        TORCH_CHECK(query != nullptr, "Invalid BSI query capsule at index ", qi);
        TORCH_CHECK(query->W == keys->W, "Word count mismatch between query and keys");

        const auto* A = reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(query->dev_words));
        const auto* Aw = tensor_data_ptr<double>(query->slice_weights);
        auto out_row = out_all[qi];

        int totalDecimals = query->device_view.decimals + keys->decimals;
        const double scale_inv = (totalDecimals > 0) ? (1.0 / std::pow(10.0, totalDecimals)) : 1.0;

        for (const auto& kv2 : keys->grouped_indices) {
            int Sb = kv2.first;
            const auto& idxs = kv2.second;
            const int Rg = static_cast<int>(idxs.size());

            const auto& words = keys->grouped_words.at(Sb);
            const auto& Bw_stacked = keys->grouped_weights.at(Sb);
            const auto* B_words = reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(words));
            const auto* Bw_ptr = tensor_data_ptr<double>(Bw_stacked);
            const auto& idx_dev = keys->grouped_indices_dev.at(Sb);

            launch_popcount_weighted_keys_literal_fused(
                A,
                Aw,
                query->S,
                query->W,
                B_words,
                Bw_ptr,
                Sb,
                Rg,
                reinterpret_cast<const long long*>(tensor_data_ptr<int64_t>(idx_dev)),
                scale_inv,
                tensor_data_ptr<double>(out_row),
                stream.stream());
        }
    }

    cudaEventRecord(end_evt, stream.stream());
    cudaEventSynchronize(end_evt);
    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start_evt, end_evt);
    cudaEventDestroy(start_evt);
    cudaEventDestroy(end_evt);

    uint64_t dot_ns = static_cast<uint64_t>(kernel_ms * 1.0e6);
    return pybind11::make_tuple(out_all, dot_ns);
}

static pybind11::tuple build_bsi_keys_cuda(torch::Tensor K, int decimalPlaces, float compress_threshold) {
    TORCH_CHECK(K.dim() == 2, "K must be 2D [num_keys, d]");
    auto Kc = K.detach().to(torch::kCPU, /*non_blocking=*/false).contiguous();
    const float* Kd = static_cast<const float*>(Kc.data_ptr());
    int64_t num_keys = Kc.size(0);
    int64_t d = Kc.size(1);
    PrebuiltBSIKeysCUDA* holder = new PrebuiltBSIKeysCUDA();
    holder->num_keys = num_keys;
    holder->d = d;
    holder->decimals = decimalPlaces;
    holder->threshold = compress_threshold;

    // Build one to get W
    {
        std::vector<double> tmp_row; tmp_row.reserve(d);
        for (int64_t c=0;c<d;++c) tmp_row.push_back(static_cast<double>(Kd[c]));
        BsiSigned<u64> b; BsiVector<u64>* t = b.buildBsiVector(tmp_row, decimalPlaces, compress_threshold);
        int S0, W0; std::vector<u64> tmp_words; bsi_flatten_words_gpu_helper(*t, tmp_words, S0, W0);
        holder->W = W0; delete t;
    }

    holder->dev_words.clear(); holder->dev_words.reserve(num_keys);
    holder->slice_weights.clear(); holder->slice_weights.reserve(num_keys);
    holder->metas.clear(); holder->metas.reserve(num_keys);
    holder->device_views.clear(); holder->device_views.reserve(num_keys);
    std::vector<std::vector<u64>> comp_words_per_key(num_keys);
    std::vector<std::vector<int>> comp_len_per_key(num_keys);
    size_t total_mem = 0;
    for (int64_t r = 0; r < num_keys; ++r) {
            std::vector<double> kv; kv.reserve(d);
            const float* rowp = Kd + r * d;
            for (int64_t c=0;c<d;++c) kv.push_back(static_cast<double>(rowp[c]));
            BsiSigned<u64> b;
            BsiVector<u64>* bsi_k = b.buildBsiVector(kv, decimalPlaces, compress_threshold);
            bsi_k->setPartitionID(0); bsi_k->setFirstSliceFlag(true); bsi_k->setLastSliceFlag(true);
            auto device = torch::Device(torch::kCUDA, c10::cuda::current_device());
            auto dev_view = create_bsi_vector_cuda_from_cpu(*bsi_k, device, bsi_cuda_should_log());
            int Sb = dev_view.slices;
            int Wb = dev_view.words_per_slice;
            TORCH_CHECK(Wb == holder->W, "word count mismatch while building CUDA keys");
            holder->device_views.push_back(dev_view);
            holder->dev_words.push_back(dev_view.words);
            holder->slice_weights.push_back(make_slice_weights_cuda(Sb, bsi_k->offset, bsi_k->twosComplement));
            KeyMeta meta; meta.S = Sb; meta.offset = bsi_k->offset; meta.twos = bsi_k->twosComplement; meta.decimals = bsi_k->decimals;
            holder->metas.push_back(meta);
            total_mem += bsi_k->getSizeInMemory();
            auto& comp_vec = comp_words_per_key[r]; comp_vec.clear();
            auto& len_vec = comp_len_per_key[r]; len_vec.resize(Sb);
            for (int s = 0; s < Sb; ++s) {
                std::vector<u64> tmp;
                hb_to_ewah_stream_helper<u64>(bsi_k->bsi[s], bsi_k->getNumberOfRows(), tmp);
                len_vec[s] = (int)tmp.size();
                comp_vec.insert(comp_vec.end(), tmp.begin(), tmp.end());
            }
            delete bsi_k;
    }
    {
        std::unordered_map<int, std::vector<int64_t>> groups;
        for (int64_t r=0; r<num_keys; ++r) groups[ holder->metas[r].S ].push_back(r);
        for (auto& kv : groups) {
            int Sb = kv.first; const auto& idxs = kv.second; int R = static_cast<int>(idxs.size());
            auto gw = torch::empty({R, Sb, holder->W}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
            auto gwt = torch::empty({R, Sb}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
            for (int i=0; i<R; ++i) {
                int64_t r = idxs[i];
                gw[i].copy_( holder->dev_words[r] );
                gwt[i].copy_( holder->slice_weights[r] );
            }
            holder->grouped_words[Sb] = gw.contiguous();
            holder->grouped_weights[Sb] = gwt.contiguous();
            holder->grouped_indices[Sb] = idxs;
            auto idx_cpu = torch::from_blob(
                const_cast<int64_t*>(idxs.data()),
                {R},
                torch::TensorOptions().dtype(torch::kInt64)).clone();
            holder->grouped_indices_dev[Sb] = idx_cpu.to(torch::kCUDA).contiguous();
        }
    }
    for (const auto& kv : holder->grouped_indices) {
        int Sb = kv.first; const auto& idxs = kv.second; int R = static_cast<int>(idxs.size());
        long long total_u64 = 0;
        std::vector<long long> h_off; h_off.resize((size_t)R * Sb);
        std::vector<int> h_len; h_len.resize((size_t)R * Sb);
        for (int i = 0; i < R; ++i) {
            int64_t r = idxs[i];
            const auto& lens = comp_len_per_key[r];
            long long base = total_u64;
            for (int s = 0; s < Sb; ++s) {
                int L = lens[s];
                h_off[(size_t)i * Sb + s] = base;
                h_len[(size_t)i * Sb + s] = L;
                base += L;
            }
            total_u64 = base;
        }
        std::vector<u64> h_comp; h_comp.reserve((size_t)total_u64);
        for (int i = 0; i < R; ++i) {
            int64_t r = idxs[i];
            const auto& comp = comp_words_per_key[r];
            h_comp.insert(h_comp.end(), comp.begin(), comp.end());
        }
        auto d_comp = torch::from_blob(h_comp.data(), {total_u64}, torch::TensorOptions().dtype(torch::kInt64)).clone().to(torch::kCUDA);
        auto d_off = torch::from_blob(h_off.data(), {R, Sb}, torch::TensorOptions().dtype(torch::kInt64)).clone().to(torch::kCUDA);
        auto d_len = torch::from_blob(h_len.data(), {R, Sb}, torch::TensorOptions().dtype(torch::kInt32)).clone().to(torch::kCUDA);
        holder->grouped_comp_words[Sb] = d_comp.contiguous();
        holder->grouped_comp_off[Sb] = d_off.contiguous();
        holder->grouped_comp_len[Sb] = d_len.contiguous();
    }
    pybind11::capsule cap(holder, "PrebuiltBSIKeysCUDA",
        [](PyObject* capsule){
            auto* p = reinterpret_cast<PrebuiltBSIKeysCUDA*>(PyCapsule_GetPointer(capsule, "PrebuiltBSIKeysCUDA"));
            if (p) delete p;
        }
    );
    return pybind11::make_tuple(cap, (uint64_t)total_mem, num_keys, d, holder->W);
}

static pybind11::tuple batch_dot_product_prebuilt_cuda_keys(torch::Tensor q, pybind11::capsule keyset_cuda_cap, float query_threshold) {
    TORCH_CHECK(q.dim() == 1, "q must be 1D [d]");
    auto* keys = capsule_to_keys_cuda(keyset_cuda_cap);
    TORCH_CHECK(keys != nullptr, "Invalid CUDA keys capsule");
    TORCH_CHECK(q.size(0) == keys->d, "q.size(0) must match keys d");
    int64_t d = keys->d;
    int decimals = keys->decimals;
    float thr = (query_threshold >= 0.0f) ? query_threshold : keys->threshold;

    auto q_cpu2 = q.detach().to(torch::kFloat32).cpu().contiguous();
    auto qa = q_cpu2.accessor<float,1>();
    std::vector<double> qv; qv.reserve(d);
    for (int64_t i=0;i<d;++i) qv.push_back(static_cast<double>(qa[i]));
    uint64_t t0 = now_ns();
    BsiSigned<u64> b;
    BsiVector<u64>* bsi_q = b.buildBsiVector(qv, decimals, thr);
    bsi_q->setPartitionID(0); bsi_q->setFirstSliceFlag(true); bsi_q->setLastSliceFlag(true);
    uint64_t build_ns = now_ns() - t0;
    int Sa, Wa; std::vector<u64> A_host; bsi_flatten_words_gpu_helper(*bsi_q, A_host, Sa, Wa);
    TORCH_CHECK(Wa == keys->W, "word count mismatch (q vs keys)");
    auto A_dev = torch::from_blob(A_host.data(), {Sa, Wa}, torch::dtype(torch::kInt64)).clone().to(torch::kCUDA);

    auto out = torch::zeros({keys->num_keys}, torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat64));
    auto query_weights = make_slice_weights_cuda(Sa, bsi_q->offset, bsi_q->twosComplement);
    float total_kernel_ms = 0.0f;

    for (int64_t r=0;r<keys->num_keys;++r) {
        const auto& B_dev = keys->dev_words[r];
        const auto& km = keys->metas[r];
        const auto& key_weights = keys->slice_weights[r];
        auto counts_dev = torch::empty({km.S, Sa}, torch::device(torch::kCUDA).dtype(torch::kInt64));
        dim3 grid(Sa, km.S); dim3 block(256);
        cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
        cudaEventRecord(s);
        auto stream2 = at::cuda::getCurrentCUDAStream();
    launch_popcount_pairwise(
        reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(A_dev)),
        reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(B_dev)),
        Sa, km.S, Wa,
        reinterpret_cast<unsigned long long*>(tensor_data_ptr<int64_t>(counts_dev)),
        stream2.stream());
        cudaEventRecord(e); cudaEventSynchronize(e);
        float ms=0.0f; cudaEventElapsedTime(&ms,s,e); cudaEventDestroy(s); cudaEventDestroy(e);
        total_kernel_ms += ms;

        auto counts_fp = counts_dev.to(torch::kFloat64);
        auto weighted = counts_fp.mul(query_weights.view({1, Sa}));
        auto slice_sums = weighted.sum(1);
        auto raw_tensor = slice_sums.mul(key_weights).sum();
        int totalDecimals = bsi_q->decimals + km.decimals;
        double scale = (totalDecimals > 0) ? std::pow(10.0, totalDecimals) : 1.0;
        auto scaled_tensor = raw_tensor / scale;
        out.index_put_({r}, scaled_tensor);
    }

    size_t mem_q = bsi_q->getSizeInMemory();
    delete bsi_q;
    uint64_t dot_ns = (uint64_t)(total_kernel_ms * 1.0e6);
    uint64_t total_ns = build_ns + dot_ns;
    return pybind11::make_tuple(out, total_ns, build_ns, dot_ns, (uint64_t)mem_q);
}

static pybind11::tuple debug_bsi_query_cuda(pybind11::capsule query_cap) {
    auto* query = capsule_to_query_cuda(query_cap);
    TORCH_CHECK(query != nullptr, "Invalid BSI CUDA query capsule");
    auto words_cpu = query->dev_words.to(torch::kCPU).clone();
    return pybind11::make_tuple(
        words_cpu,
        query->device_view.rows,
        query->device_view.offset,
        query->device_view.decimals,
        query->device_view.twos_complement);
}

static torch::Tensor debug_quantize_int64_cuda(torch::Tensor x, int decimal_places) {
    auto device = x.device().is_cuda() ? x.device() : torch::kCUDA;
    return bsi_cuda_quantize_to_int64(x, decimal_places, device);
}

static pybind11::tuple debug_quantize_details_cuda(torch::Tensor x, int decimal_places, int64_t k) {
    auto device = x.device().is_cuda() ? x.device() : torch::kCUDA;
    auto tup = bsi_cuda_quantize_debug(x, decimal_places, device, k);
    return pybind11::make_tuple(std::get<0>(tup), std::get<1>(tup), std::get<2>(tup));
}

void register_bsi_cuda(pybind11::module& m) {
    m.attr("CUDA_BUILDER_VERSION") = cuda_builder_version();

    // Version info
    m.def("cuda_builder_version",
        &cuda_builder_version);

    // Simple dot product (for testing)
    m.def("dot_product_decimal_cuda",
        &dot_product_decimal_cuda,
        "BSI dot product with decimal quantization on CUDA");

    // Legacy batch dot product (used by tests)
    m.def("batch_dot_product_prebuilt_cuda",
        &batch_dot_product_prebuilt_cuda,
        pybind11::arg("q"),
        pybind11::arg("keyset_cap"),
        pybind11::arg("query_threshold") = -1.0f,
        "Batch dot product using CPU-built BSI");

    // Query builder
    m.def("build_bsi_query_cuda",
        &build_bsi_query_cuda,
        pybind11::arg("q"),
        pybind11::arg("decimal_places"),
        pybind11::arg("compress_threshold") = 0.2f,
        "Build BSI query vector on CUDA");

    // Hybrid compressed query builder
    m.def("build_bsi_query_cuda_hybrid",
        &build_bsi_query_cuda_hybrid,
        pybind11::arg("q"),
        pybind11::arg("decimal_places"),
        pybind11::arg("compress_threshold") = 0.2,
        "Build BSI query with EWAH compression");

    // Debug helper for hybrid queries
    m.def("debug_bsi_query_hybrid_decompress",
        &debug_bsi_query_hybrid_decompress,
        pybind11::arg("query_cap"),
        "Decompress hybrid query to CPU for debugging");

    // Legacy batch dot product with capsules (used by tests)
    m.def("batch_dot_product_prebuilt_cuda_caps",
        &batch_dot_product_prebuilt_cuda_caps,
        pybind11::arg("query_cap"),
        pybind11::arg("keyset_cuda_cap"),
        "Batch dot product with prebuilt capsules");

    // OPTIMIZED: Multi-query batched dot product with fused kernel
    m.def("batch_dot_product_multiquery_cuda_caps",
        &batch_dot_product_multiquery_cuda_caps,
        pybind11::arg("query_caps_list"),
        pybind11::arg("keyset_cuda_cap"),
        "Multi-query batch dot product with fused kernel (28% faster)");

    m.def("batch_dot_product_two_stage_cuda_caps",
        &batch_dot_product_two_stage_cuda_caps,
        pybind11::arg("query_caps_list"),
        pybind11::arg("keyset_cuda_cap"),
        "Multi-query batch dot product with two-stage approach (popcount + weighted reduction)");

    // Key builder for CUDA
    m.def("build_bsi_keys_cuda",
        &build_bsi_keys_cuda,
        pybind11::arg("K"),
        pybind11::arg("decimal_places"),
        pybind11::arg("compress_threshold") = 0.2f,
        "Build BSI keys and prepack to CUDA");

    // Legacy batch dot product with keys (used by tests)
    m.def("batch_dot_product_prebuilt_cuda_keys",
        &batch_dot_product_prebuilt_cuda_keys,
        pybind11::arg("q"),
        pybind11::arg("keyset_cuda_cap"),
        pybind11::arg("query_threshold") = -1.0f,
        "Batch dot product with prepacked keys");

    // Debug helpers
    m.def("debug_bsi_query_cuda",
        &debug_bsi_query_cuda,
        pybind11::arg("query_cap"),
        "Debug: Extract query words and metadata");

    m.def("debug_quantize_int64_cuda",
        &debug_quantize_int64_cuda,
        pybind11::arg("x"),
        pybind11::arg("decimal_places"),
        "Debug: Quantize to int64 with parity rounding");

    m.def("debug_quantize_details_cuda",
        &debug_quantize_details_cuda,
        pybind11::arg("x"),
        pybind11::arg("decimal_places"),
        pybind11::arg("k") = 8,
        "Debug: Return quantization internals");
}
