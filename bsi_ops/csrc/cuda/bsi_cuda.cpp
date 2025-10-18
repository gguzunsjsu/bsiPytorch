// CUDA host wrappers and PyBind registrations for BSI GPU operations
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <cstdint>
#include <unordered_map>
#include <c10/cuda/CUDAFunctions.h>
#include <cstdlib>

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

extern "C" void launch_ewah_decompress(
    const unsigned long long* in,
    int in_len,
    int W,
    unsigned long long* out,
    cudaStream_t stream);

// no multi-slice EWAH decompress in Phase-2

extern "C" void launch_popcount_weighted(
    const unsigned long long* A,
    const double* Aw,
    int Sa, int W,
    const unsigned long long* B,
    const double* Bw,
    int Sb,
    double* out,
    cudaStream_t stream);

// Batched weighted popcount for a group of keys with identical Sb
extern "C" void launch_popcount_weighted_batch(
    const unsigned long long* A,
    const double* Aw,
    int Sa, int W,
    const unsigned long long* B,
    const double* Bw,
    int Sb,
    int R,
    double* out,
    cudaStream_t stream);

extern "C" void launch_popcount_weighted_keys(
    const unsigned long long* A,
    const double* Aw,
    int Sa, int W,
    const unsigned long long* B,
    const double* Bw,
    int Sb,
    int R,
    double* out,
    cudaStream_t stream);

extern "C" void launch_popcount_weighted_keys_tiled(
    const unsigned long long* A,
    const double* Aw,
    int Sa, int W,
    const unsigned long long* B,
    const double* Bw,
    int Sb,
    int R,
    int tiles,
    int tile_size,
    double* out,
    cudaStream_t stream);

extern "C" void launch_scatter_set_double(
    const long long* idx,
    const double* src,
    int n,
    double* out,
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

    cudaEvent_t start_evt, end_evt;
    cudaEventCreate(&start_evt);
    cudaEventCreate(&end_evt);

    // Correctness-first: per-key pairwise counts and CPU-style accumulation
    float total_kernel_ms = 0.0f;
    for (const auto& kv : by_Sb) {
        int Sb = kv.first;
        const auto& idxs = kv.second;
        int R = static_cast<int>(idxs.size());

        std::vector<at::Tensor> words_list; words_list.reserve(R);
        std::vector<at::Tensor> weights_list; weights_list.reserve(R);
        for (int i = 0; i < R; ++i) {
            int64_t r = idxs[i];
            words_list.push_back(keys->dev_words[r]);             // [Sb, W]
            weights_list.push_back(keys->slice_weights[r].view({1, Sb})); // [1, Sb]
        }
        auto B_stacked = torch::stack(words_list, 0).contiguous(); // [R, Sb, W]
        auto Bw_stacked = torch::cat(weights_list, 0).contiguous(); // [R, Sb]
        auto out_slice = torch::zeros({R}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
        // Precompute scale factors on device: divide by 10^(dec_q + dec_k)
        auto scales = torch::empty({R}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
        {
            std::vector<double> h_scales(R, 1.0);
            for (int i = 0; i < R; ++i) {
                int64_t r = idxs[i];
                const auto& km = keys->metas[r];
                int totalDecimals = query->device_view.decimals + km.decimals;
                h_scales[i] = (totalDecimals > 0) ? std::pow(10.0, totalDecimals) : 1.0;
            }
            auto tmp = torch::from_blob(h_scales.data(), {R}, torch::TensorOptions().dtype(torch::kFloat64)).clone();
            scales.copy_(tmp.to(torch::kCUDA));
        }

        auto stream = at::cuda::getCurrentCUDAStream();
        cudaEventRecord(start_evt, stream.stream());
        if (bsi_cuda_use_tiled()) {
            int tiles = 1; int tile_size = query->W;
            if (query->W > 1024) { tiles = std::min(16, (query->W + 1023) / 1024); tile_size = (query->W + tiles - 1) / tiles; }
            launch_popcount_weighted_keys_tiled(
                query_words,
                query_weights,
                query->S,
                query->W,
                reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(B_stacked)),
                tensor_data_ptr<double>(Bw_stacked),
                Sb,
                R,
                tiles,
                tile_size,
                tensor_data_ptr<double>(out_slice),
                stream.stream());
        } else {
            launch_popcount_weighted_keys(
                query_words,
                query_weights,
                query->S,
                query->W,
                reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(B_stacked)),
                tensor_data_ptr<double>(Bw_stacked),
                Sb,
                R,
                tensor_data_ptr<double>(out_slice),
                stream.stream());
        }
        cudaEventRecord(end_evt, stream.stream());
        cudaEventSynchronize(end_evt);
        float kernel_ms = 0.0f; cudaEventElapsedTime(&kernel_ms, start_evt, end_evt);
        total_kernel_ms += kernel_ms;

        // Scale and scatter to out_dev on device
        out_slice = out_slice / scales;
        auto idx_cpu = torch::from_blob(const_cast<int64_t*>(idxs.data()), {R}, torch::TensorOptions().dtype(torch::kInt64)).clone();
        auto idx_dev = idx_cpu.to(torch::kCUDA).contiguous();
        auto stream3 = at::cuda::getCurrentCUDAStream();
        launch_scatter_set_double(
            reinterpret_cast<const long long*>(tensor_data_ptr<int64_t>(idx_dev)),
            tensor_data_ptr<double>(out_slice),
            R,
            tensor_data_ptr<double>(out_dev),
            stream3.stream());
    }

    cudaEventDestroy(start_evt);
    cudaEventDestroy(end_evt);

    // Return GPU scores directly
    uint64_t dot_ns = static_cast<uint64_t>(total_kernel_ms * 1.0e6);
    uint64_t total_ns = dot_ns;
    return pybind11::make_tuple(out_dev, total_ns, static_cast<uint64_t>(0), dot_ns, static_cast<uint64_t>(query->mem_bytes));
}

void register_bsi_cuda(pybind11::module& m) {
    // Version/signature to verify the loaded .so
    const std::string version = std::string("CUDA_BUILDER_PHASE2_ROUNDING_V2 ") + __DATE__ + " " + __TIME__;
    m.attr("CUDA_BUILDER_VERSION") = version;
    m.def("cuda_builder_version", [version]() { return version; });
    m.def("dot_product_decimal_cuda", &dot_product_decimal_cuda, "BSI dot (decimal) computed on CUDA");
    m.def("batch_dot_product_prebuilt_cuda", &batch_dot_product_prebuilt_cuda,
          pybind11::arg("q"), pybind11::arg("keyset_cap"), pybind11::arg("query_threshold") = -1.0f,
          "Batch dot using CUDA popcount kernel with CPU-built BSI");
    m.def("build_bsi_query_cuda", &build_bsi_query_cuda,
          pybind11::arg("q"), pybind11::arg("decimal_places"), pybind11::arg("compress_threshold") = 0.2f,
          "Build a BSI query vector and prepack it to CUDA words");
    m.def("batch_dot_product_prebuilt_cuda_caps", &batch_dot_product_prebuilt_cuda_caps,
          pybind11::arg("query_cap"), pybind11::arg("keyset_cuda_cap"),
          "Batch dot using CUDA popcount kernel with prebuilt query and key capsules");
    // Build prepacked GPU keys
    m.def("build_bsi_keys_cuda", [](torch::Tensor K, int decimalPlaces, float compress_threshold) {
        TORCH_CHECK(K.dim() == 2, "K must be 2D [num_keys, d]");
        auto Kc = K.contiguous();
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

        holder->dev_words.reserve(num_keys);
        holder->slice_weights.reserve(num_keys);
        holder->metas.reserve(num_keys);
        holder->device_views.reserve(num_keys);
        size_t total_mem = 0;
        for (int64_t r=0;r<num_keys;++r) {
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
        TORCH_CHECK(Wb == holder->W,
                    "word count mismatch while building CUDA keys");
        holder->device_views.push_back(dev_view);
        holder->dev_words.push_back(dev_view.words);
        holder->slice_weights.push_back(make_slice_weights_cuda(Sb, bsi_k->offset, bsi_k->twosComplement));
            KeyMeta meta; meta.S = Sb; meta.offset = bsi_k->offset; meta.twos = bsi_k->twosComplement; meta.decimals = bsi_k->decimals;
            holder->metas.push_back(meta);
            total_mem += bsi_k->getSizeInMemory();
            delete bsi_k;
        }
        // Group contiguous tensors by Sb once
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
            }
        }
        pybind11::capsule cap(holder, "PrebuiltBSIKeysCUDA",
            [](PyObject* capsule){
                auto* p = reinterpret_cast<PrebuiltBSIKeysCUDA*>(PyCapsule_GetPointer(capsule, "PrebuiltBSIKeysCUDA"));
                if (p) delete p;
            }
        );
        return pybind11::make_tuple(cap, (uint64_t)total_mem, num_keys, d, holder->W);
    }, pybind11::arg("K"), pybind11::arg("decimal_places"), pybind11::arg("compress_threshold") = 0.2f,
       "Build BSI keys on CPU; prepack to CUDA words and return capsule");

    m.def("batch_dot_product_prebuilt_cuda_keys", [](torch::Tensor q, pybind11::capsule keyset_cuda_cap, float query_threshold){
        TORCH_CHECK(q.dim() == 1, "q must be 1D [d]");
        auto* keys = capsule_to_keys_cuda(keyset_cuda_cap);
        TORCH_CHECK(keys != nullptr, "Invalid CUDA keys capsule");
        TORCH_CHECK(q.size(0) == keys->d, "q.size(0) must match keys d");
        int64_t d = keys->d;
        int decimals = keys->decimals;
        float thr = (query_threshold >= 0.0f) ? query_threshold : keys->threshold;

        // Build query BSI on CPU
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
    }, pybind11::arg("q"), pybind11::arg("keyset_cuda_cap"), pybind11::arg("query_threshold") = -1.0f,
       "Batch dot using prepacked CUDA keys and CUDA popcount kernel");

    m.def("debug_bsi_query_cuda", [](pybind11::capsule query_cap) {
        auto* query = capsule_to_query_cuda(query_cap);
        TORCH_CHECK(query != nullptr, "Invalid BSI CUDA query capsule");
        auto words_cpu = query->dev_words.to(torch::kCPU).clone();
        return pybind11::make_tuple(
            words_cpu,
            query->device_view.rows,
            query->device_view.offset,
            query->device_view.decimals,
            query->device_view.twos_complement);
    }, pybind11::arg("query_cap"),
       "Return words and metadata for a GPU-built BSI query capsule (words copied to CPU)");

    // Phase-3 compressed view intentionally omitted

    // Expose quantizer for debugging parity (returns int64 tensor on device, contiguous)
    m.def("debug_quantize_int64_cuda", [](torch::Tensor x, int decimal_places) {
        auto device = x.device().is_cuda() ? x.device() : torch::kCUDA;
        return bsi_cuda_quantize_to_int64(x, decimal_places, device);
    }, pybind11::arg("x"), pybind11::arg("decimal_places"),
       "Quantize input tensor to int64 using GPU parity rounding (half-away-from-zero)");

    // Expose detailed quantizer internals (first k elements): (scaled_fp, rounded_fp, staged_int)
    m.def("debug_quantize_details_cuda", [](torch::Tensor x, int decimal_places, int64_t k) {
        auto device = x.device().is_cuda() ? x.device() : torch::kCUDA;
        auto tup = bsi_cuda_quantize_debug(x, decimal_places, device, k);
        return pybind11::make_tuple(std::get<0>(tup), std::get<1>(tup), std::get<2>(tup));
    }, pybind11::arg("x"), pybind11::arg("decimal_places"), pybind11::arg("k") = 8,
       "Return (scaled_fp, rounded_fp, staged_int) heads to verify rounding/parity");
}
