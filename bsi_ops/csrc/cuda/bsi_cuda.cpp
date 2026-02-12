#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <iostream>
#include <cctype>
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

extern "C" void launch_popcount_weighted_keys_literal_fused_multiq_hybrid_keys(
    const unsigned long long* A,
    const float* Aw,
    int Sa,
    int W,
    const unsigned long long* B,
    const float* Bw,
    int Sb,
    int R,
    int Q,
    int q_tile,
    int r_tile,
    const unsigned char* B_comp_flags,
    const unsigned long long* B_comp_words,
    const long long* B_comp_off,
    const int* B_comp_len,
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
    // Hybrid key representation per Sb group (optional).
    std::unordered_map<int, at::Tensor> grouped_comp_words; // Sb -> [u64_total], int64 cuda
    std::unordered_map<int, at::Tensor> grouped_comp_off;   // Sb -> [R_sb, Sb], int64 cuda
    std::unordered_map<int, at::Tensor> grouped_comp_len;   // Sb -> [R_sb, Sb], int32 cuda
    std::unordered_map<int, at::Tensor> grouped_comp_flags; // Sb -> [R_sb, Sb], uint8 cuda
    std::unordered_map<int, bool> grouped_has_compressed;   // Sb -> any compressed slice
    std::unordered_map<int, float> grouped_comp_frac;       // Sb -> compressed slice fraction [0,1]
    int W = 0;
    int64_t d = 0;
    int64_t num_keys = 0;
    int decimals = 0;
    float threshold = 0.2f;
    std::string storage_mode = "verbatim";
    int64_t total_slices = 0;
    int64_t compressed_slices = 0;
    int64_t verbatim_slices = 0;
    int64_t compressed_words_u64 = 0;
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

static std::string normalize_key_storage_mode(const std::string& raw_mode) {
    std::string mode = raw_mode;
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (mode.empty()) return "verbatim";
    if (mode == "verbatim") return "verbatim";
    if (mode == "hybrid") return "hybrid";
    TORCH_CHECK(false, "Unsupported key storage mode: ", raw_mode,
                ". Expected one of {'verbatim','hybrid'}.");
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

static pybind11::list build_bsi_queries_cuda_batch(torch::Tensor q2d, int decimalPlaces, float compress_threshold = 0.2f) {
    TORCH_CHECK(q2d.dim() == 2, "q must be 2D [Q, d]");
    (void)compress_threshold;
    auto device = torch::Device(torch::kCUDA, c10::cuda::current_device());
    bool verbose = bsi_cuda_should_log();
    const auto Q = q2d.size(0);
    pybind11::list out;
    if (bsi_cuda_query_batch()) {
        auto batch = build_bsi_queries_cuda_batch_data(q2d.detach(), decimalPlaces, device, verbose);
        const int S = batch.slices;
        const int W = batch.words_per_slice;
        auto words = batch.words.contiguous();
        auto weights = batch.slice_weights.contiguous();
        for (int64_t qi = 0; qi < Q; ++qi) {
            auto* holder = new PrebuiltBSIQueryCUDA();
            holder->vec = nullptr;
            holder->S = S;
            holder->W = W;
            holder->dev_words = words[qi];
            holder->slice_weights = weights[qi];
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
        word_stack.reserve(g.queries.size());
        weight_stack.reserve(g.queries.size());
        for (auto* qptr : g.queries) {
            word_stack.push_back(qptr->dev_words);
            weight_stack.push_back(qptr->slice_weights);
        }
        pg.words = torch::stack(word_stack).contiguous();
        pg.weights = torch::stack(weight_stack).contiguous();
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

    cudaEvent_t start_evt, end_evt;
    cudaEventCreate(&start_evt);
    cudaEventCreate(&end_evt);
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaEventRecord(start_evt, stream.stream());
    bool use_hybrid_dot = (keys->storage_mode == "hybrid");
    if (const char* s = std::getenv("BSI_HYBRID_DOT")) {
        use_hybrid_dot = (std::atoi(s) != 0);
    }
    static float hybrid_min_comp_frac = -1.0f;
    if (hybrid_min_comp_frac < 0.0f) {
        hybrid_min_comp_frac = 0.35f;
        if (const char* s = std::getenv("BSI_HYBRID_MIN_COMP_FRAC")) {
            hybrid_min_comp_frac = static_cast<float>(std::atof(s));
        }
        if (hybrid_min_comp_frac < 0.0f) hybrid_min_comp_frac = 0.0f;
        if (hybrid_min_comp_frac > 1.0f) hybrid_min_comp_frac = 1.0f;
    }

    for (const auto& pg : prepared) {
        const auto* A = reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(pg.words));
        const auto* Aw = tensor_data_ptr<float>(pg.weights);
        const auto* q_idx = reinterpret_cast<const long long*>(tensor_data_ptr<int64_t>(pg.q_indices));

        for (const auto& kv2 : keys->grouped_indices) {
            int Sb = kv2.first;
            const auto& idxs = kv2.second;
            const int Rg = static_cast<int>(idxs.size());

            const auto& words = keys->grouped_words.at(Sb);
            const auto& Bw_stacked = keys->grouped_weights.at(Sb);
            const auto* B_words = reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(words));
            const auto* Bw_ptr = tensor_data_ptr<float>(Bw_stacked);
            const auto& idx_dev = keys->grouped_indices_dev.at(Sb);
            bool group_has_compressed = false;
            auto has_it = keys->grouped_has_compressed.find(Sb);
            if (has_it != keys->grouped_has_compressed.end()) {
                group_has_compressed = has_it->second;
            }
            float group_comp_frac = 0.0f;
            auto frac_it = keys->grouped_comp_frac.find(Sb);
            if (frac_it != keys->grouped_comp_frac.end()) {
                group_comp_frac = frac_it->second;
            }

            if (use_hybrid_dot && group_has_compressed && group_comp_frac >= hybrid_min_comp_frac) {
                const auto& comp_flags = keys->grouped_comp_flags.at(Sb);
                const auto& comp_words = keys->grouped_comp_words.at(Sb);
                const auto& comp_off = keys->grouped_comp_off.at(Sb);
                const auto& comp_len = keys->grouped_comp_len.at(Sb);
                launch_popcount_weighted_keys_literal_fused_multiq_hybrid_keys(
                    A,
                    Aw,
                    pg.S,
                    keys->W,
                    B_words,
                    Bw_ptr,
                    Sb,
                    Rg,
                    static_cast<int>(pg.Qcount),
                    bsi_cuda_q_tile(),
                    bsi_cuda_r_tile(),
                    tensor_data_ptr<uint8_t>(comp_flags),
                    reinterpret_cast<const unsigned long long*>(tensor_data_ptr<int64_t>(comp_words)),
                    reinterpret_cast<const long long*>(tensor_data_ptr<int64_t>(comp_off)),
                    tensor_data_ptr<int>(comp_len),
                    reinterpret_cast<const long long*>(tensor_data_ptr<int64_t>(idx_dev)),
                    q_idx,
                    static_cast<float>(scale_inv),
                    static_cast<int>(R),
                    tensor_data_ptr<float>(out_all),
                    stream.stream());
            } else {
                launch_popcount_weighted_keys_literal_fused_multiq(
                    A,
                    Aw,
                    pg.S,
                    keys->W,
                    B_words,
                    Bw_ptr,
                    Sb,
                    Rg,
                    static_cast<int>(pg.Qcount),
                    bsi_cuda_q_tile(),
                    bsi_cuda_r_tile(),
                    reinterpret_cast<const long long*>(tensor_data_ptr<int64_t>(idx_dev)),
                    q_idx,
                    static_cast<float>(scale_inv),
                    static_cast<int>(R),
                    tensor_data_ptr<float>(out_all),
                    stream.stream());
            }
        }
    }

    cudaEventRecord(end_evt, stream.stream());
    cudaEventSynchronize(end_evt);
    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start_evt, end_evt);
    cudaEventDestroy(start_evt);
    cudaEventDestroy(end_evt);

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

static pybind11::dict bsi_keys_cuda_stats(pybind11::capsule keyset_cuda_cap) {
    auto* keys = capsule_to_keys_cuda(keyset_cuda_cap);
    TORCH_CHECK(keys != nullptr, "Invalid CUDA keys capsule");
    pybind11::dict sb_counts;
    pybind11::dict sb_hybrid_counts;
    pybind11::dict sb_hybrid_frac;
    for (const auto& kv : keys->grouped_indices) {
        sb_counts[pybind11::int_(kv.first)] = pybind11::int_(kv.second.size());
        int64_t compressed = 0;
        auto fit = keys->grouped_comp_flags.find(kv.first);
        if (fit != keys->grouped_comp_flags.end() && fit->second.defined() && fit->second.numel() > 0) {
            compressed = fit->second.to(torch::kCPU).to(torch::kInt64).sum().item<int64_t>();
        }
        sb_hybrid_counts[pybind11::int_(kv.first)] = pybind11::int_(compressed);
        auto ff = keys->grouped_comp_frac.find(kv.first);
        sb_hybrid_frac[pybind11::int_(kv.first)] = pybind11::float_((ff != keys->grouped_comp_frac.end()) ? ff->second : 0.0f);
    }
    pybind11::dict out;
    out["W"] = pybind11::int_(keys->W);
    out["d"] = pybind11::int_(keys->d);
    out["num_keys"] = pybind11::int_(keys->num_keys);
    out["decimals"] = pybind11::int_(keys->decimals);
    out["threshold"] = pybind11::float_(keys->threshold);
    out["storage_mode"] = pybind11::str(keys->storage_mode);
    out["total_slices"] = pybind11::int_(keys->total_slices);
    out["compressed_slices"] = pybind11::int_(keys->compressed_slices);
    out["verbatim_slices"] = pybind11::int_(keys->verbatim_slices);
    out["compressed_words_u64"] = pybind11::int_(keys->compressed_words_u64);
    out["Sb_counts"] = sb_counts;
    out["Sb_compressed_slices"] = sb_hybrid_counts;
    out["Sb_compressed_frac"] = sb_hybrid_frac;
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

static pybind11::tuple build_bsi_keys_cuda(torch::Tensor K,
                                           int decimalPlaces,
                                           float compress_threshold,
                                           std::string storage_mode_raw = "verbatim") {
    TORCH_CHECK(K.dim() == 2, "K must be 2D [num_keys, d]");
    const std::string storage_mode = normalize_key_storage_mode(storage_mode_raw);
    auto Kc = K.detach().to(torch::kCPU, /*non_blocking=*/false).contiguous();
    const float* Kd = static_cast<const float*>(Kc.data_ptr());
    int64_t num_keys = Kc.size(0);
    int64_t d = Kc.size(1);
    PrebuiltBSIKeysCUDA* holder = new PrebuiltBSIKeysCUDA();
    holder->num_keys = num_keys;
    holder->d = d;
    holder->decimals = decimalPlaces;
    holder->threshold = compress_threshold;
    holder->storage_mode = storage_mode;
    holder->W = (d > 0) ? static_cast<int>((d + 63) / 64) : 1;

    holder->dev_words.clear(); holder->dev_words.reserve(num_keys);
    holder->slice_weights.clear(); holder->slice_weights.reserve(num_keys);
    holder->metas.clear(); holder->metas.reserve(num_keys);
    holder->device_views.clear(); holder->device_views.reserve(num_keys);

    struct KeyHybridHost {
        std::vector<u64> words;
        std::vector<int64_t> off;
        std::vector<int32_t> len;
        std::vector<uint8_t> flags;
    };
    std::vector<KeyHybridHost> hybrid_host;
    if (storage_mode == "hybrid") {
        hybrid_host.resize(static_cast<size_t>(num_keys));
    }

    auto device = torch::Device(torch::kCUDA, c10::cuda::current_device());
    for (int64_t r = 0; r < num_keys; ++r) {
        std::vector<double> kv;
        kv.reserve(d);
        const float* rowp = Kd + r * d;
        for (int64_t c = 0; c < d; ++c) kv.push_back(static_cast<double>(rowp[c]));
        BsiSigned<u64> b;
        BsiVector<u64>* bsi_k = b.buildBsiVector(kv, decimalPlaces, compress_threshold);
        bsi_k->setPartitionID(0);
        bsi_k->setFirstSliceFlag(true);
        bsi_k->setLastSliceFlag(true);

        auto dev_view = create_bsi_vector_cuda_from_cpu(*bsi_k, device, bsi_cuda_should_log());
        int Sb = dev_view.slices;
        int Wb = dev_view.words_per_slice;
        TORCH_CHECK(Wb == holder->W, "word count mismatch while building CUDA keys");
        holder->device_views.push_back(dev_view);
        holder->dev_words.push_back(dev_view.words);
        holder->slice_weights.push_back(make_slice_weights_cuda(Sb, dev_view.offset, dev_view.twos_complement));
        KeyMeta meta;
        meta.S = Sb;
        meta.offset = dev_view.offset;
        meta.twos = dev_view.twos_complement;
        meta.decimals = bsi_k->decimals;
        holder->metas.push_back(meta);

        holder->total_slices += Sb;
        if (storage_mode == "hybrid") {
            auto& hk = hybrid_host[static_cast<size_t>(r)];
            hk.off.resize(static_cast<size_t>(Sb), 0);
            hk.len.resize(static_cast<size_t>(Sb), 0);
            hk.flags.resize(static_cast<size_t>(Sb), 0);
            for (int s = 0; s < Sb; ++s) {
                const auto& hb = bsi_k->bsi[static_cast<size_t>(s)];
                const bool is_compressed = !hb.verbatim;
                hk.flags[static_cast<size_t>(s)] = static_cast<uint8_t>(is_compressed ? 1 : 0);
                if (is_compressed) {
                    std::vector<u64> stream_words;
                    hb_to_ewah_stream_helper<u64>(hb, bsi_k->getNumberOfRows(), stream_words);
                    hk.off[static_cast<size_t>(s)] = static_cast<int64_t>(hk.words.size());
                    hk.len[static_cast<size_t>(s)] = static_cast<int32_t>(stream_words.size());
                    hk.words.insert(hk.words.end(), stream_words.begin(), stream_words.end());
                    holder->compressed_slices += 1;
                    holder->compressed_words_u64 += static_cast<int64_t>(stream_words.size());
                } else {
                    holder->verbatim_slices += 1;
                }
            }
        } else {
            holder->verbatim_slices += Sb;
        }
        delete bsi_k;
    }
    {
        std::unordered_map<int, std::vector<int64_t>> groups;
        for (int64_t r = 0; r < num_keys; ++r) groups[holder->metas[r].S].push_back(r);
        for (auto& kv : groups) {
            int Sb = kv.first;
            const auto& idxs = kv.second;
            int R = static_cast<int>(idxs.size());
            auto gw = torch::zeros({R, Sb, holder->W}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
            auto gwt = torch::zeros({R, Sb}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
            std::vector<uint8_t> comp_flags_host(static_cast<size_t>(R) * static_cast<size_t>(Sb), 0);
            std::vector<int64_t> comp_off_host(static_cast<size_t>(R) * static_cast<size_t>(Sb), 0);
            std::vector<int32_t> comp_len_host(static_cast<size_t>(R) * static_cast<size_t>(Sb), 0);
            std::vector<u64> comp_words_host;
            bool group_has_compressed = false;
            int64_t group_compressed_slices = 0;
            for (int i=0; i<R; ++i) {
                int64_t r = idxs[i];
                int Sb_actual = holder->metas[r].S;
                gw[i].narrow(0, 0, Sb_actual).copy_(holder->dev_words[r]);
                gwt[i].narrow(0, 0, Sb_actual).copy_(holder->slice_weights[r]);
                if (storage_mode == "hybrid") {
                    const auto& hk = hybrid_host[static_cast<size_t>(r)];
                    TORCH_CHECK(static_cast<int>(hk.flags.size()) == Sb_actual, "Hybrid key flags shape mismatch");
                    for (int s = 0; s < Sb_actual; ++s) {
                        const size_t idx = static_cast<size_t>(i) * static_cast<size_t>(Sb) + static_cast<size_t>(s);
                        const uint8_t flag = hk.flags[static_cast<size_t>(s)];
                        comp_flags_host[idx] = flag;
                        if (flag != 0) {
                            group_compressed_slices += 1;
                            group_has_compressed = true;
                            const int64_t src_off = hk.off[static_cast<size_t>(s)];
                            const int32_t len = hk.len[static_cast<size_t>(s)];
                            TORCH_CHECK(src_off >= 0 && len >= 0 &&
                                            src_off + static_cast<int64_t>(len) <= static_cast<int64_t>(hk.words.size()),
                                        "Hybrid key compressed span out of bounds");
                            const int64_t dst_off = static_cast<int64_t>(comp_words_host.size());
                            comp_off_host[idx] = dst_off;
                            comp_len_host[idx] = len;
                            comp_words_host.insert(
                                comp_words_host.end(),
                                hk.words.begin() + src_off,
                                hk.words.begin() + src_off + static_cast<int64_t>(len));
                        }
                    }
                }
            }
            holder->grouped_words[Sb] = gw.contiguous();
            holder->grouped_weights[Sb] = gwt.contiguous();
            holder->grouped_indices[Sb] = idxs;
            auto idx_cpu = torch::from_blob(
                const_cast<int64_t*>(idxs.data()),
                {R},
                torch::TensorOptions().dtype(torch::kInt64)).clone();
            holder->grouped_indices_dev[Sb] = idx_cpu.to(torch::kCUDA).contiguous();
            if (storage_mode == "hybrid") {
                at::Tensor comp_words;
                if (comp_words_host.empty()) {
                    comp_words = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
                } else {
                    comp_words = torch::from_blob(
                        comp_words_host.data(),
                        {static_cast<int64_t>(comp_words_host.size())},
                        torch::TensorOptions().dtype(torch::kInt64)).clone().to(torch::kCUDA).contiguous();
                }
                auto comp_off = torch::from_blob(
                    comp_off_host.data(),
                    {R, Sb},
                    torch::TensorOptions().dtype(torch::kInt64)).clone().to(torch::kCUDA).contiguous();
                auto comp_len = torch::from_blob(
                    comp_len_host.data(),
                    {R, Sb},
                    torch::TensorOptions().dtype(torch::kInt32)).clone().to(torch::kCUDA).contiguous();
                auto comp_flags = torch::from_blob(
                    comp_flags_host.data(),
                    {R, Sb},
                    torch::TensorOptions().dtype(torch::kUInt8)).clone().to(torch::kCUDA).contiguous();
                holder->grouped_comp_words[Sb] = comp_words;
                holder->grouped_comp_off[Sb] = comp_off;
                holder->grouped_comp_len[Sb] = comp_len;
                holder->grouped_comp_flags[Sb] = comp_flags;
                holder->grouped_has_compressed[Sb] = group_has_compressed;
                const int64_t total_group_slices = static_cast<int64_t>(R) * static_cast<int64_t>(Sb);
                holder->grouped_comp_frac[Sb] =
                    (total_group_slices > 0)
                        ? static_cast<float>(static_cast<double>(group_compressed_slices) / static_cast<double>(total_group_slices))
                        : 0.0f;
            } else {
                holder->grouped_comp_words[Sb] = torch::empty({0}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
                holder->grouped_comp_off[Sb] = torch::zeros({R, Sb}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA));
                holder->grouped_comp_len[Sb] = torch::zeros({R, Sb}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
                holder->grouped_comp_flags[Sb] = torch::zeros({R, Sb}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));
                holder->grouped_has_compressed[Sb] = false;
                holder->grouped_comp_frac[Sb] = 0.0f;
            }
        }
    }

    // Release per-key tensors after grouped packing to avoid duplicated storage.
    holder->dev_words.clear();
    holder->slice_weights.clear();
    holder->metas.clear();
    holder->device_views.clear();

    uint64_t total_mem = 0;
    for (const auto& kv : holder->grouped_words) {
        total_mem += static_cast<uint64_t>(kv.second.numel() * kv.second.element_size());
    }
    for (const auto& kv : holder->grouped_weights) {
        total_mem += static_cast<uint64_t>(kv.second.numel() * kv.second.element_size());
    }
    for (const auto& kv : holder->grouped_comp_words) {
        total_mem += static_cast<uint64_t>(kv.second.numel() * kv.second.element_size());
    }
    for (const auto& kv : holder->grouped_comp_off) {
        total_mem += static_cast<uint64_t>(kv.second.numel() * kv.second.element_size());
    }
    for (const auto& kv : holder->grouped_comp_len) {
        total_mem += static_cast<uint64_t>(kv.second.numel() * kv.second.element_size());
    }
    for (const auto& kv : holder->grouped_comp_flags) {
        total_mem += static_cast<uint64_t>(kv.second.numel() * kv.second.element_size());
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
        pybind11::arg("storage_mode") = "verbatim",
        "Build BSI keys and prepack to CUDA (storage_mode: verbatim|hybrid)");
}
