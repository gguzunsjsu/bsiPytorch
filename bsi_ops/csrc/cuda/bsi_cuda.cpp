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
extern "C" void launch_ewah_decompress(
    const unsigned long long* in,
    int in_len,
    int W,
    unsigned long long* out,
    cudaStream_t stream);

// no multi-slice EWAH decompress in Phase-2

extern "C" void launch_popcount_weighted_keys_literal_fused_multiq(
    const unsigned long long* A,
    const float* Aw,
    const float* A_chunk_scales,
    int A_scale_stride,
    int Sa,
    int W,
    const unsigned long long* B,
    const float* Bw,
    int Sb,
    int R,
    int Q,
    int q_tile,
    int r_tile,
    const long long* indices_r,
    const long long* indices_q,
    float scale_inv,
    int R_total,
    float* out_global,
    cudaStream_t stream);

template <typename T>
inline T* tensor_data_ptr(torch::Tensor& t) {
    return t.data_ptr<T>();
}

template <typename T>
inline const T* tensor_data_ptr(const torch::Tensor& t) {
    auto& nc = const_cast<torch::Tensor&>(t);
    return const_cast<const T*>(nc.data_ptr<T>());
}

static int bsi_cuda_q_tile() {
    static int cached = 0;
    if (cached > 0) return cached;
    int v = 8; // default tile size across queries
    if (const char* s = std::getenv("BSI_Q_TILE")) {
        int t = std::atoi(s);
        if (t > 0) v = t;
    }
    if (v < 1) v = 1;
    cached = v;
    return cached;
}

static int bsi_cuda_r_tile() {
    static int cached = 0;
    if (cached > 0) return cached;
    int v = 4; // default tile size across keys
    if (const char* s = std::getenv("BSI_R_TILE")) {
        int t = std::atoi(s);
        if (t > 0) v = t;
    }
    if (v < 1) v = 1;
    cached = v;
    return cached;
}

static bool bsi_cuda_query_batch() {
    static int cached = -1;
    if (cached >= 0) return cached != 0;
    int v = 1; // default on
    if (const char* s = std::getenv("BSI_QUERY_BATCH")) {
        int t = std::atoi(s);
        v = (t != 0) ? 1 : 0;
    }
    cached = v;
    return cached != 0;
}

static int bsi_cuda_fixed_bits_keys_env() {
    static int cached = -1;
    if (cached >= 0) return cached;
    const char* s = std::getenv("BSI_FIXED_BITS_KEYS");
    if (s == nullptr) {
        s = std::getenv("BSI_FIXED_BITS");
    }
    int v = (s != nullptr) ? std::atoi(s) : 0;
    if (v <= 0) v = 0;
    if (v > 0 && v < 2) v = 2;
    if (v > 63) v = 63;
    cached = v;
    return cached;
}

static bool bsi_cuda_profile_enabled() {
    static int cached = -1;
    if (cached >= 0) return cached != 0;
    int v = 0; // default off: do not synchronize inside every dot call
    if (const char* s = std::getenv("BSI_PROFILE")) {
        v = (std::atoi(s) != 0) ? 1 : 0;
    }
    cached = v;
    return cached != 0;
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
    std::vector<at::Tensor> slice_weights; // each [S_k], float32, cuda
    std::vector<KeyMeta> metas;
    std::vector<BsiVectorCudaData> device_views;
    // Grouped, contiguous views by Sb to avoid per-call stacking
    std::unordered_map<int, at::Tensor> grouped_words;   // Sb -> [R_sb, Sb, W]
    std::unordered_map<int, at::Tensor> grouped_weights; // Sb -> [R_sb, Sb]
    std::unordered_map<int, std::vector<int64_t>> grouped_indices; // Sb -> original key indices
    std::unordered_map<int, at::Tensor> grouped_indices_dev; // Sb -> [R_sb] int64 cuda
    std::unordered_map<int, bool> grouped_indices_identity; // Sb -> true when idx[i] == i
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
    at::Tensor chunk_scales;   // [chunks] or undefined
    int S = 0;
    int W = 0;
    size_t mem_bytes = 0;
};

static PrebuiltBSIQueryCUDA* capsule_to_query_cuda(const pybind11::capsule& cap) {
    return reinterpret_cast<PrebuiltBSIQueryCUDA*>(cap.get_pointer());
}

// Packed/batched query representation (no per-row capsules).
// This avoids:
// - creating Q Python capsules per layer, and
// - re-stacking Q small tensors inside the dot-product path.
struct PrebuiltBSIQueryBatchCUDA {
    at::Tensor words;         // [Q, Sa, W] int64 cuda
    at::Tensor slice_weights; // [Q, Sa] float32 cuda
    at::Tensor chunk_scales;  // [Q, chunks] float32 cuda or undefined
    int Sa = 0;
    int W = 0;
    int decimals = 0;
    int64_t Q = 0;
    size_t mem_bytes = 0;
};

static PrebuiltBSIQueryBatchCUDA* capsule_to_query_batch_cuda(const pybind11::capsule& cap) {
    return reinterpret_cast<PrebuiltBSIQueryBatchCUDA*>(cap.get_pointer());
}

static inline long double weight_for_meta(int offset, int idx, bool twos, int S) {
    int shift = offset + idx;
    long double w = (shift >= 0) ? std::ldexp(1.0L, shift) : 0.0L;
    if (twos && idx == S - 1) w = -w;
    return w;
}

static inline at::Tensor make_slice_weights_cuda(int S, int offset, bool twos) {
    std::vector<float> host(S);
    for (int i = 0; i < S; ++i) {
        long double w = weight_for_meta(offset, i, twos, S);
        host[i] = static_cast<float>(w);
    }
    return torch::from_blob(
               host.data(),
               {S},
               torch::TensorOptions().dtype(torch::kFloat32))
        .clone()
        .to(torch::kCUDA);
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
    if (holder->device_view.scale != 1.0f) {
        holder->slice_weights = holder->slice_weights * holder->device_view.scale;
    }
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

static pybind11::list build_bsi_queries_cuda_batch(torch::Tensor q2d, int decimalPlaces, float compress_threshold = 0.2f) {
    TORCH_CHECK(q2d.dim() == 2, "q must be 2D [Q, d]");
    (void)compress_threshold;
    auto device = torch::Device(torch::kCUDA, c10::cuda::current_device());
    bool verbose = bsi_cuda_should_log();
    const auto Q = q2d.size(0);
    pybind11::list out;
    if (bsi_cuda_query_batch()) {
        auto batch = build_bsi_queries_cuda_batch_data(q2d.detach(), decimalPlaces, device, verbose, /*for_keys=*/false);
        const int S = batch.slices;
        const int W = batch.words_per_slice;
        auto words = batch.words.contiguous();
        auto weights = batch.slice_weights.contiguous();
        auto chunk_scales = batch.chunk_scales;
        for (int64_t qi = 0; qi < Q; ++qi) {
            auto* holder = new PrebuiltBSIQueryCUDA();
            holder->vec = nullptr;
            holder->S = S;
            holder->W = W;
            holder->dev_words = words[qi];
            holder->slice_weights = weights[qi];
            if (chunk_scales.defined() && chunk_scales.numel() > 0) {
                holder->chunk_scales = chunk_scales[qi].contiguous();
            }
            holder->mem_bytes = static_cast<size_t>(holder->dev_words.numel() * holder->dev_words.element_size());
            holder->device_view.rows = static_cast<int64_t>(q2d.size(1));
            holder->device_view.slices = S;
            holder->device_view.words_per_slice = W;
            holder->device_view.offset = batch.offset;
            holder->device_view.decimals = decimalPlaces;
            holder->device_view.twos_complement = false;
            holder->device_view.words = holder->dev_words;

            pybind11::capsule cap(holder, "PrebuiltBSIQueryCUDA",
                [](PyObject* capsule) {
                    auto* p = reinterpret_cast<PrebuiltBSIQueryCUDA*>(PyCapsule_GetPointer(capsule, "PrebuiltBSIQueryCUDA"));
                    if (p) {
                        delete p->vec;
                        delete p;
                    }
                }
            );
            out.append(cap);
        }
        return out;
    }

    for (int64_t qi = 0; qi < Q; ++qi) {
        auto row = q2d[qi].detach();
        auto* holder = new PrebuiltBSIQueryCUDA();
        holder->vec = nullptr;
        holder->device_view = build_bsi_vector_from_float_tensor(row, decimalPlaces, device, verbose);
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
        out.append(cap);
    }
    return out;
}

static pybind11::capsule build_bsi_queries_cuda_batch_packed(torch::Tensor q2d,
                                                             int decimalPlaces,
                                                             float compress_threshold = 0.2f) {
    TORCH_CHECK(q2d.dim() == 2, "q must be 2D [Q, d]");
    (void)compress_threshold;
    auto device = torch::Device(torch::kCUDA, c10::cuda::current_device());
    bool verbose = bsi_cuda_should_log();

    auto batch = build_bsi_queries_cuda_batch_data(q2d.detach(), decimalPlaces, device, verbose, /*for_keys=*/false);
    auto* holder = new PrebuiltBSIQueryBatchCUDA();
    holder->words = batch.words.contiguous();
    holder->slice_weights = batch.slice_weights.contiguous();
    if (batch.chunk_scales.defined() && batch.chunk_scales.numel() > 0) {
        holder->chunk_scales = batch.chunk_scales.contiguous();
    }
    holder->Sa = batch.slices;
    holder->W = batch.words_per_slice;
    holder->decimals = int(decimalPlaces);
    holder->Q = q2d.size(0);
    holder->mem_bytes = static_cast<size_t>(holder->words.numel() * holder->words.element_size()) +
        static_cast<size_t>(holder->slice_weights.numel() * holder->slice_weights.element_size());
    if (holder->chunk_scales.defined() && holder->chunk_scales.numel() > 0) {
        holder->mem_bytes += static_cast<size_t>(holder->chunk_scales.numel() * holder->chunk_scales.element_size());
    }

    pybind11::capsule cap(holder, "PrebuiltBSIQueryBatchCUDA",
        [](PyObject* capsule) {
            auto* p = reinterpret_cast<PrebuiltBSIQueryBatchCUDA*>(
                PyCapsule_GetPointer(capsule, "PrebuiltBSIQueryBatchCUDA"));
            if (p) delete p;
        }
    );
    return cap;
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
    if (holder->device_view.scale != 1.0f) {
        holder->slice_weights = holder->slice_weights * holder->device_view.scale;
    }
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

static pybind11::tuple batch_dot_product_multiquery_cuda_caps(pybind11::list query_caps_list, pybind11::capsule keyset_cuda_cap) {
    auto* keys = capsule_to_keys_cuda(keyset_cuda_cap);
    TORCH_CHECK(keys != nullptr, "Invalid CUDA keys capsule");
    const int64_t R = keys->num_keys;
    const int64_t Q = static_cast<int64_t>(pybind11::len(query_caps_list));
    TORCH_CHECK(R > 0 && Q > 0, "Empty keys or queries");

struct TempGroup {
        int S = 0;
        std::vector<PrebuiltBSIQueryCUDA*> queries;
        std::vector<int64_t> indices;
    };
    std::unordered_map<int64_t, TempGroup> groups;
    int common_decimals = -1;

    for (int64_t qi = 0; qi < Q; ++qi) {
        auto cap_obj = query_caps_list[qi];
        TORCH_CHECK(pybind11::isinstance<pybind11::capsule>(cap_obj), "Each item must be a capsule");
        auto cap = pybind11::cast<pybind11::capsule>(cap_obj);
        auto* query = capsule_to_query_cuda(cap);
        TORCH_CHECK(query != nullptr, "Invalid BSI query capsule at index ", qi);
        TORCH_CHECK(query->W == keys->W, "Word count mismatch between query and keys");

        if (common_decimals < 0) {
            common_decimals = query->device_view.decimals;
        } else {
            TORCH_CHECK(query->device_view.decimals == common_decimals, "Decimal mismatch across queries not supported");
        }

        int64_t key = (static_cast<int64_t>(query->S) << 1) | (query->device_view.twos_complement ? 1LL : 0LL);
        auto& g = groups[key];
        if (g.queries.empty()) {
            g.S = query->S;
        }
        g.queries.push_back(query);
        g.indices.push_back(qi);
    }

    struct PreparedGroup {
        int S = 0;
        int64_t Qcount = 0;
        torch::Tensor words;      // [Q, S, W]
        torch::Tensor weights;    // [Q, S]
        torch::Tensor chunk_scales; // [Q, chunks] or undefined
        torch::Tensor q_indices;  // [Q]
    };
    std::vector<PreparedGroup> prepared;
    prepared.reserve(groups.size());
    for (auto& kv : groups) {
        auto& g = kv.second;
        PreparedGroup pg;
        pg.S = g.S;
        pg.Qcount = static_cast<int64_t>(g.queries.size());

        std::vector<torch::Tensor> word_stack;
        std::vector<torch::Tensor> weight_stack;
        std::vector<torch::Tensor> scale_stack;
        word_stack.reserve(g.queries.size());
        weight_stack.reserve(g.queries.size());
        scale_stack.reserve(g.queries.size());
        const bool have_scales =
            (!g.queries.empty() && g.queries[0]->chunk_scales.defined() && g.queries[0]->chunk_scales.numel() > 0);
        for (auto* qptr : g.queries) {
            word_stack.push_back(qptr->dev_words);
            weight_stack.push_back(qptr->slice_weights);
            if (have_scales) {
                TORCH_CHECK(qptr->chunk_scales.defined() && qptr->chunk_scales.numel() > 0,
                            "Mixed chunk-scale and non-chunk-scale queries in the same group");
                scale_stack.push_back(qptr->chunk_scales);
            } else {
                TORCH_CHECK(!(qptr->chunk_scales.defined() && qptr->chunk_scales.numel() > 0),
                            "Mixed chunk-scale and non-chunk-scale queries in the same group");
            }
        }
        pg.words = torch::stack(word_stack).contiguous();
        pg.weights = torch::stack(weight_stack).contiguous();
        if (have_scales) {
            pg.chunk_scales = torch::stack(scale_stack).contiguous();
        }
        pg.q_indices = torch::from_blob(
            g.indices.data(),
            {static_cast<int64_t>(g.indices.size())},
            torch::TensorOptions().dtype(torch::kInt64))
            .clone()
            .to(torch::kCUDA)
            .contiguous();
        prepared.push_back(std::move(pg));
    }

    auto out_all = torch::zeros({Q, R}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    TORCH_CHECK(common_decimals == keys->decimals, "Decimal mismatch between queries and keys");
    const int totalDecimals = common_decimals + keys->decimals;
    const double scale_inv = (totalDecimals > 0) ? (1.0 / std::pow(10.0, totalDecimals)) : 1.0;

    auto stream = at::cuda::getCurrentCUDAStream();
    float kernel_ms = 0.0f;
    const bool profile = bsi_cuda_profile_enabled();
    cudaEvent_t start_evt = nullptr;
    cudaEvent_t end_evt = nullptr;
    if (profile) {
        cudaEventCreate(&start_evt);
        cudaEventCreate(&end_evt);
        cudaEventRecord(start_evt, stream.stream());
    }

    for (const auto& pg : prepared) {
        const auto* A = reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(pg.words));
        const auto* Aw = tensor_data_ptr<float>(pg.weights);
        const float* A_chunk_scales = nullptr;
        int A_scale_stride = 0;
        if (pg.chunk_scales.defined() && pg.chunk_scales.numel() > 0) {
            // Chunk-scale mode requires the SM90+ BMMA path for correctness.
            {
                int dev = 0;
                cudaGetDevice(&dev);
                int major = 0;
                cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
                TORCH_CHECK(major >= 9, "Chunk scales require SM90+ (Hopper) tensor-core path");
            }
            TORCH_CHECK(keys->W % 4 == 0, "Chunk scales require W64 multiple of 4");
            TORCH_CHECK(pg.chunk_scales.dim() == 2, "Expected [Q, chunks] chunk scales");
            TORCH_CHECK(pg.chunk_scales.size(1) == (keys->W / 4),
                        "Chunk scale stride mismatch: got ", pg.chunk_scales.size(1),
                        " expected ", (keys->W / 4));
            A_chunk_scales = tensor_data_ptr<float>(pg.chunk_scales);
            A_scale_stride = static_cast<int>(pg.chunk_scales.size(1));
        }
        const auto* q_idx = reinterpret_cast<const long long*>(tensor_data_ptr<int64_t>(pg.q_indices));

        for (const auto& kv2 : keys->grouped_indices) {
            int Sb = kv2.first;
            const auto& idxs = kv2.second;
            const int Rg = static_cast<int>(idxs.size());

            const auto& words = keys->grouped_words.at(Sb);
            const auto& Bw_stacked = keys->grouped_weights.at(Sb);
            const auto* B_words = reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(words));
            const auto* Bw_ptr = tensor_data_ptr<float>(Bw_stacked);
            const bool identity_r = (keys->grouped_indices_identity.count(Sb) > 0)
                ? keys->grouped_indices_identity.at(Sb)
                : false;
            const long long* r_idx_ptr = nullptr;
            if (!identity_r) {
                const auto& idx_dev = keys->grouped_indices_dev.at(Sb);
                r_idx_ptr = reinterpret_cast<const long long*>(tensor_data_ptr<int64_t>(idx_dev));
            }
            launch_popcount_weighted_keys_literal_fused_multiq(
                A,
                Aw,
                A_chunk_scales,
                A_scale_stride,
                pg.S,
                keys->W,
                B_words,
                Bw_ptr,
                Sb,
                Rg,
                static_cast<int>(pg.Qcount),
                bsi_cuda_q_tile(),
                bsi_cuda_r_tile(),
                r_idx_ptr,
                q_idx,
                static_cast<float>(scale_inv),
                static_cast<int>(R),
                tensor_data_ptr<float>(out_all),
                stream.stream());
        }
    }

    if (profile) {
        cudaEventRecord(end_evt, stream.stream());
        cudaEventSynchronize(end_evt);
        cudaEventElapsedTime(&kernel_ms, start_evt, end_evt);
        cudaEventDestroy(start_evt);
        cudaEventDestroy(end_evt);
    }

    // Total GPU time (ns) spent inside the fused dot kernels launched in this call.
    // This covers computing the full output matrix of shape [Q, R].
    const uint64_t dot_kernel_ns_total = static_cast<uint64_t>(kernel_ms * 1.0e6);

    // Average GPU time per query vector (one input row producing an output vector of length R).
    const double dot_kernel_ns_per_query =
        (Q > 0) ? (static_cast<double>(dot_kernel_ns_total) / static_cast<double>(Q)) : 0.0;

    // Average GPU time per scalar dot (one output element out_all[q, r]).
    const double dot_kernel_ns_per_scalar =
        (Q > 0 && R > 0)
            ? (static_cast<double>(dot_kernel_ns_total) / (static_cast<double>(Q) * static_cast<double>(R)))
            : 0.0;

    return pybind11::make_tuple(out_all, dot_kernel_ns_total, dot_kernel_ns_per_query, dot_kernel_ns_per_scalar);
}

static pybind11::tuple batch_dot_product_multiquery_cuda_batch_caps(pybind11::capsule query_batch_cap,
                                                                    pybind11::capsule keyset_cuda_cap) {
    auto* keys = capsule_to_keys_cuda(keyset_cuda_cap);
    TORCH_CHECK(keys != nullptr, "Invalid CUDA keys capsule");
    auto* qb = capsule_to_query_batch_cuda(query_batch_cap);
    TORCH_CHECK(qb != nullptr, "Invalid CUDA query-batch capsule");

    TORCH_CHECK(qb->words.defined() && qb->slice_weights.defined(),
                "Query batch tensors are not defined");
    TORCH_CHECK(qb->words.is_cuda() && qb->slice_weights.is_cuda(),
                "Query batch tensors must be CUDA tensors");
    TORCH_CHECK(qb->words.dim() == 3, "Query batch words must be [Q, Sa, W]");
    TORCH_CHECK(qb->slice_weights.dim() == 2, "Query batch slice_weights must be [Q, Sa]");

    const int64_t Q = qb->words.size(0);
    const int64_t Sa = qb->words.size(1);
    const int64_t W = qb->words.size(2);
    TORCH_CHECK(Q > 0 && Sa > 0 && W > 0, "Empty query batch");
    TORCH_CHECK(W == keys->W, "Word count mismatch between query batch and keys");
    TORCH_CHECK(qb->slice_weights.size(0) == Q && qb->slice_weights.size(1) == Sa,
                "slice_weights shape mismatch: expected [", Q, ", ", Sa, "]");

    const int64_t R = keys->num_keys;
    TORCH_CHECK(R > 0, "Empty keys");

    // If chunk scales are provided, validate (and enforce SM90 path for correctness).
    const float* A_chunk_scales = nullptr;
    int A_scale_stride = 0;
    if (qb->chunk_scales.defined() && qb->chunk_scales.numel() > 0) {
        TORCH_CHECK(qb->chunk_scales.is_cuda(), "chunk_scales must be a CUDA tensor");
        TORCH_CHECK(qb->chunk_scales.dim() == 2, "Expected [Q, chunks] chunk scales");
        TORCH_CHECK(qb->chunk_scales.size(0) == Q, "chunk_scales Q mismatch");
        {
            int dev = 0;
            cudaGetDevice(&dev);
            int major = 0;
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
            TORCH_CHECK(major >= 9, "Chunk scales require SM90+ (Hopper) tensor-core path");
        }
        TORCH_CHECK(keys->W % 4 == 0, "Chunk scales require W64 multiple of 4");
        TORCH_CHECK(qb->chunk_scales.size(1) == (keys->W / 4),
                    "Chunk scale stride mismatch: got ", qb->chunk_scales.size(1),
                    " expected ", (keys->W / 4));
        A_chunk_scales = tensor_data_ptr<float>(qb->chunk_scales);
        A_scale_stride = static_cast<int>(qb->chunk_scales.size(1));
    }

    // Output: [Q, R_total] (full keyset width).
    auto out_all = torch::zeros({Q, R}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    TORCH_CHECK(qb->decimals == keys->decimals,
                "Decimal mismatch between queries and keys: q=", qb->decimals, " k=", keys->decimals);
    const int totalDecimals = qb->decimals + keys->decimals;
    const double scale_inv = (totalDecimals > 0) ? (1.0 / std::pow(10.0, totalDecimals)) : 1.0;

    auto stream = at::cuda::getCurrentCUDAStream();
    float kernel_ms = 0.0f;
    const bool profile = bsi_cuda_profile_enabled();
    cudaEvent_t start_evt = nullptr;
    cudaEvent_t end_evt = nullptr;
    if (profile) {
        cudaEventCreate(&start_evt);
        cudaEventCreate(&end_evt);
        cudaEventRecord(start_evt, stream.stream());
    }

    const auto* A = reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(qb->words));
    const auto* Aw = tensor_data_ptr<float>(qb->slice_weights);
    const long long* indices_q = nullptr; // identity mapping in packed batch mode

    for (const auto& kv2 : keys->grouped_indices) {
        int Sb = kv2.first;
        const auto& idxs = kv2.second;
        const int Rg = static_cast<int>(idxs.size());

        const auto& words = keys->grouped_words.at(Sb);
        const auto& Bw_stacked = keys->grouped_weights.at(Sb);
        const auto* B_words = reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(words));
        const auto* Bw_ptr = tensor_data_ptr<float>(Bw_stacked);
        const bool identity_r = (keys->grouped_indices_identity.count(Sb) > 0)
            ? keys->grouped_indices_identity.at(Sb)
            : false;
        const long long* r_idx_ptr = nullptr;
        if (!identity_r) {
            const auto& idx_dev = keys->grouped_indices_dev.at(Sb);
            r_idx_ptr = reinterpret_cast<const long long*>(tensor_data_ptr<int64_t>(idx_dev));
        }

        launch_popcount_weighted_keys_literal_fused_multiq(
            A,
            Aw,
            A_chunk_scales,
            A_scale_stride,
            static_cast<int>(Sa),
            keys->W,
            B_words,
            Bw_ptr,
            Sb,
            Rg,
            static_cast<int>(Q),
            bsi_cuda_q_tile(),
            bsi_cuda_r_tile(),
            r_idx_ptr,
            indices_q,
            static_cast<float>(scale_inv),
            static_cast<int>(R),
            tensor_data_ptr<float>(out_all),
            stream.stream());
    }

    if (profile) {
        cudaEventRecord(end_evt, stream.stream());
        cudaEventSynchronize(end_evt);
        cudaEventElapsedTime(&kernel_ms, start_evt, end_evt);
        cudaEventDestroy(start_evt);
        cudaEventDestroy(end_evt);
    }

    const uint64_t dot_kernel_ns_total = static_cast<uint64_t>(kernel_ms * 1.0e6);
    const double dot_kernel_ns_per_query =
        (Q > 0) ? (static_cast<double>(dot_kernel_ns_total) / static_cast<double>(Q)) : 0.0;
    const double dot_kernel_ns_per_scalar =
        (Q > 0 && R > 0)
            ? (static_cast<double>(dot_kernel_ns_total) / (static_cast<double>(Q) * static_cast<double>(R)))
            : 0.0;

    return pybind11::make_tuple(out_all, dot_kernel_ns_total, dot_kernel_ns_per_query, dot_kernel_ns_per_scalar);
}

static pybind11::dict bsi_keys_cuda_stats(pybind11::capsule keyset_cuda_cap) {
    auto* keys = capsule_to_keys_cuda(keyset_cuda_cap);
    TORCH_CHECK(keys != nullptr, "Invalid CUDA keys capsule");
    pybind11::dict sb_counts;
    for (const auto& kv : keys->grouped_indices) {
        sb_counts[pybind11::int_(kv.first)] = pybind11::int_(kv.second.size());
    }
    pybind11::dict out;
    out["W"] = pybind11::int_(keys->W);
    out["d"] = pybind11::int_(keys->d);
    out["num_keys"] = pybind11::int_(keys->num_keys);
    out["decimals"] = pybind11::int_(keys->decimals);
    out["threshold"] = pybind11::float_(keys->threshold);
    out["Sb_counts"] = sb_counts;
    return out;
}

static pybind11::dict bsi_query_caps_stats(pybind11::list query_caps_list) {
    const int64_t Q = static_cast<int64_t>(pybind11::len(query_caps_list));
    TORCH_CHECK(Q > 0, "Empty query capsule list");
    std::unordered_map<int, int64_t> counts;
    int W = -1;
    for (int64_t qi = 0; qi < Q; ++qi) {
        auto cap_obj = query_caps_list[qi];
        TORCH_CHECK(pybind11::isinstance<pybind11::capsule>(cap_obj), "Each item must be a capsule");
        auto cap = pybind11::cast<pybind11::capsule>(cap_obj);
        auto* query = capsule_to_query_cuda(cap);
        TORCH_CHECK(query != nullptr, "Invalid BSI query capsule at index ", qi);
        counts[query->S] += 1;
        if (W < 0) {
            W = query->W;
        } else {
            TORCH_CHECK(query->W == W, "Word count mismatch across queries");
        }
    }
    pybind11::dict s_counts;
    for (const auto& kv : counts) {
        s_counts[pybind11::int_(kv.first)] = pybind11::int_(kv.second);
    }
    pybind11::dict out;
    out["Q"] = pybind11::int_(Q);
    out["W"] = pybind11::int_(W);
    out["S_counts"] = s_counts;
    return out;
}

static pybind11::tuple build_bsi_keys_cuda(torch::Tensor K, int decimalPlaces, float compress_threshold) {
    TORCH_CHECK(K.dim() == 2, "K must be 2D [num_keys, d]");
    // Fixed-bit mode: build key bitplanes directly on CUDA to ensure
    // keys and queries use the same quantization/scaling behavior.
    if (bsi_cuda_fixed_bits_keys_env() > 0) {
        const int64_t num_keys = K.size(0);
        const int64_t d = K.size(1);
        auto device = torch::Device(torch::kCUDA, c10::cuda::current_device());
        bool verbose = bsi_cuda_should_log();

        auto batch = build_bsi_queries_cuda_batch_data(K.detach(), decimalPlaces, device, verbose, /*for_keys=*/true);
        TORCH_CHECK(batch.rows == num_keys, "CUDA fixed-bit key build row mismatch");
        TORCH_CHECK(batch.words_per_slice == static_cast<int>((d + 63) / 64),
                    "CUDA fixed-bit key build word count mismatch");

        auto* holder = new PrebuiltBSIKeysCUDA();
        holder->num_keys = num_keys;
        holder->d = d;
        holder->W = batch.words_per_slice;
        holder->decimals = decimalPlaces;
        holder->threshold = compress_threshold;

        const int Sb = batch.slices;
        holder->grouped_words[Sb] = batch.words.contiguous();
        holder->grouped_weights[Sb] = batch.slice_weights.contiguous();

        std::vector<int64_t> idxs(static_cast<size_t>(num_keys));
        for (int64_t i = 0; i < num_keys; ++i) idxs[static_cast<size_t>(i)] = i;
        holder->grouped_indices[Sb] = idxs;
        holder->grouped_indices_identity[Sb] = true;

        const uint64_t total_mem_bytes = static_cast<uint64_t>(
            holder->grouped_words[Sb].numel() * holder->grouped_words[Sb].element_size());

        pybind11::capsule cap(holder, "PrebuiltBSIKeysCUDA",
            [](PyObject* capsule){
                auto* p = reinterpret_cast<PrebuiltBSIKeysCUDA*>(PyCapsule_GetPointer(capsule, "PrebuiltBSIKeysCUDA"));
                if (p) delete p;
            }
        );
        return pybind11::make_tuple(cap, total_mem_bytes, num_keys, d, holder->W);
    }

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
            holder->slice_weights.push_back(make_slice_weights_cuda(Sb, dev_view.offset, dev_view.twos_complement));
            KeyMeta meta; meta.S = Sb; meta.offset = dev_view.offset; meta.twos = dev_view.twos_complement; meta.decimals = bsi_k->decimals;
            holder->metas.push_back(meta);
            total_mem += bsi_k->getSizeInMemory();
            delete bsi_k;
    }
    {
        std::unordered_map<int, std::vector<int64_t>> groups;
        for (int64_t r=0; r<num_keys; ++r) groups[ holder->metas[r].S ].push_back(r);
        for (auto& kv : groups) {
            int Sb = kv.first; const auto& idxs = kv.second; int R = static_cast<int>(idxs.size());
            auto gw = torch::zeros({R, Sb, holder->W}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
            auto gwt = torch::zeros({R, Sb}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            for (int i=0; i<R; ++i) {
                int64_t r = idxs[i];
                int Sb_actual = holder->metas[r].S;
                gw[i].narrow(0, 0, Sb_actual).copy_(holder->dev_words[r]);
                gwt[i].narrow(0, 0, Sb_actual).copy_(holder->slice_weights[r]);
            }
            holder->grouped_words[Sb] = gw.contiguous();
            holder->grouped_weights[Sb] = gwt.contiguous();
            holder->grouped_indices[Sb] = idxs;
            bool identity = true;
            for (int i = 0; i < R; ++i) {
                if (idxs[static_cast<size_t>(i)] != static_cast<int64_t>(i)) {
                    identity = false;
                    break;
                }
            }
            holder->grouped_indices_identity[Sb] = identity;
            if (!identity) {
                auto idx_cpu = torch::from_blob(
                    const_cast<int64_t*>(idxs.data()),
                    {R},
                    torch::TensorOptions().dtype(torch::kInt64)).clone();
                holder->grouped_indices_dev[Sb] = idx_cpu.to(torch::kCUDA).contiguous();
            }
        }
    }
    pybind11::capsule cap(holder, "PrebuiltBSIKeysCUDA",
        [](PyObject* capsule){
            auto* p = reinterpret_cast<PrebuiltBSIKeysCUDA*>(PyCapsule_GetPointer(capsule, "PrebuiltBSIKeysCUDA"));
            if (p) delete p;
        }
    );
    return pybind11::make_tuple(cap, (uint64_t)total_mem, num_keys, d, holder->W);
}

void register_bsi_cuda(pybind11::module& m) {
    m.attr("CUDA_BUILDER_VERSION") = cuda_builder_version();

    // Version info
    m.def("cuda_builder_version",
        &cuda_builder_version);

    // Query builder
    m.def("build_bsi_query_cuda",
        &build_bsi_query_cuda,
        pybind11::arg("q"),
        pybind11::arg("decimal_places"),
        pybind11::arg("compress_threshold") = 0.2f,
        "Build BSI query vector on CUDA");
    m.def("build_bsi_queries_cuda_batch",
        &build_bsi_queries_cuda_batch,
        pybind11::arg("q2d"),
        pybind11::arg("decimal_places"),
        pybind11::arg("compress_threshold") = 0.2f,
        "Build BSI queries for a batch on CUDA");
    m.def("build_bsi_queries_cuda_batch_packed",
        &build_bsi_queries_cuda_batch_packed,
        pybind11::arg("q2d"),
        pybind11::arg("decimal_places"),
        pybind11::arg("compress_threshold") = 0.2f,
        "Build a packed (batched) BSI query object on CUDA (no per-row capsules)");
    m.def("get_last_query_build_profile_cuda",
        &bsi_cuda_get_last_query_build_profile,
        "Return (quantize_ns, pack_ns, total_ns) for the last CUDA query-batch build");
    m.def("reset_last_query_build_profile_cuda",
        &bsi_cuda_reset_last_query_build_profile,
        "Reset last CUDA query-batch build profile counters to zero");

    // Hybrid compressed query builder
    m.def("build_bsi_query_cuda_hybrid",
        &build_bsi_query_cuda_hybrid,
        pybind11::arg("q"),
        pybind11::arg("decimal_places"),
        pybind11::arg("compress_threshold") = 0.2,
        "Build BSI query with EWAH compression");

    // OPTIMIZED: Multi-query batched dot product with fused kernel
    m.def("batch_dot_product_multiquery_cuda_caps",
        &batch_dot_product_multiquery_cuda_caps,
        pybind11::arg("query_caps_list"),
        pybind11::arg("keyset_cuda_cap"),
        "Multi-query batch dot product with fused kernel (28% faster)");
    m.def("batch_dot_product_multiquery_cuda_batch_caps",
        &batch_dot_product_multiquery_cuda_batch_caps,
        pybind11::arg("query_batch_cap"),
        pybind11::arg("keyset_cuda_cap"),
        "Packed-batch multi-query dot product with fused kernel (avoids stacking + Python capsules)");
    m.def("bsi_keys_cuda_stats",
        &bsi_keys_cuda_stats,
        pybind11::arg("keyset_cuda_cap"),
        "Return shape/group stats for a CUDA BSI keyset");
    m.def("bsi_query_caps_stats",
        &bsi_query_caps_stats,
        pybind11::arg("query_caps_list"),
        "Return shape stats for a list of CUDA BSI query capsules");

    // Key builder for CUDA
    m.def("build_bsi_keys_cuda",
        &build_bsi_keys_cuda,
        pybind11::arg("K"),
        pybind11::arg("decimal_places"),
        pybind11::arg("compress_threshold") = 0.2f,
        "Build BSI keys and prepack to CUDA");
}
