// CUDA host wrappers and PyBind registrations for BSI GPU operations
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>

// bring in BSI core (CPU) to build and access slices
#include "../../../bsiCPP/bsi/BsiVector.hpp"
#include "../../../bsiCPP/bsi/BsiSigned.hpp"
#include "../../../bsiCPP/bsi/BsiUnsigned.hpp"
#include "../../../bsiCPP/bsi/hybridBitmap/hybridbitmap.h"
#include "../../../bsiCPP/bsi/hybridBitmap/runninglengthword.h"

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

static inline uint64_t now_ns() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

// Flatten one HybridBitmap (compressed or verbatim) to verbatim word buffer of length W words
static void hb_to_verbatim_words(const HybridBitmap<u64>& hb, int64_t rows, std::vector<u64>& out_words) {
    const int word_bits = 8 * sizeof(u64); // 64
    const int64_t W = (rows + word_bits - 1) / word_bits;
    out_words.clear();
    out_words.reserve(W);

    HybridBitmapRawIterator<u64> it = hb.raw_iterator();
    HybridBitmap<u64> tmp(true);
    size_t written = 0;
    while (it.hasNext() && written < (size_t)W) {
        auto& brlw = it.next();
        size_t before = tmp.buffer.size();
        size_t just = brlw.dischargeDecompressed(tmp, (size_t)W - written);
        written += just;
        // if nothing written (should not happen), break to avoid infinite loop
        if (tmp.buffer.size() == before && just == 0) break;
    }
    // pad to W
    out_words.assign(tmp.buffer.begin(), tmp.buffer.end());
    if (out_words.size() < (size_t)W) out_words.resize((size_t)W, 0ULL);
}

// Flatten full BsiVector into [S, W] verbatim words
static void bsi_flatten_words(const BsiVector<u64>& v, std::vector<u64>& out, int& S, int& W) {
    const int word_bits = 8 * (int)sizeof(u64);
    const int64_t rows = v.getNumberOfRows();
    W = (int)((rows + word_bits - 1) / word_bits);
    S = v.getNumberOfSlices();
    out.clear();
    out.resize((size_t)S * (size_t)W, 0ULL);

    std::vector<u64> tmp;
    for (int s = 0; s < S; ++s) {
        hb_to_verbatim_words(v.bsi[s], rows, tmp);
        std::copy(tmp.begin(), tmp.end(), out.begin() + (size_t)s * (size_t)W);
    }
}

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
    auto ca = counts_cpu.accessor<long long, 2>(); // [Sb, Sa]
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
    std::vector<KeyMeta> metas;
    int W = 0;
    int64_t d = 0;
    int64_t num_keys = 0;
    int decimals = 0;
    float threshold = 0.2f;
};

static PrebuiltBSIKeysCUDA* capsule_to_keys_cuda(const pybind11::capsule& cap) {
    return reinterpret_cast<PrebuiltBSIKeysCUDA*>(cap.get_pointer());
}

static inline long double weight_for_meta(int offset, int idx, bool twos, int S) {
    int shift = offset + idx;
    long double w = (shift >= 0) ? std::ldexp(1.0L, shift) : 0.0L;
    if (twos && idx == S - 1) w = -w;
    return w;
}

static double accumulate_weighted_dot_meta(
    int Sa, int offset_a, bool twos_a, int decimals_a,
    const at::Tensor& counts_cpu,
    const KeyMeta& kb)
{
    auto ca = counts_cpu.accessor<long long, 2>(); // [Sb, Sa]
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
    bsi_flatten_words(*bsi_q, A_host, Sa, Wa);
    bsi_flatten_words(*bsi_k, B_host, Sb, Wb);
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
        reinterpret_cast<const unsigned long long*>(A_dev.data_ptr()),
        reinterpret_cast<const unsigned long long*>(B_dev.data_ptr()),
        Sa, Sb, Wa,
        reinterpret_cast<unsigned long long*>(counts_dev.data_ptr()),
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
    bsi_flatten_words(*bsi_q, A_host, Sa, Wa);
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
        bsi_flatten_words(*bsi_k, B_host, Sb, Wb);
        TORCH_CHECK(Wb == Wa, "word count mismatch");
        auto B_dev = torch::from_blob(B_host.data(), {Sb, Wa}, torch::dtype(torch::kInt64)).clone().to(torch::kCUDA);
        auto counts_dev = torch::empty({Sb, Sa}, torch::device(torch::kCUDA).dtype(torch::kInt64));

        dim3 grid(Sa, Sb);
        dim3 block(256);
        cudaEvent_t start, end; cudaEventCreate(&start); cudaEventCreate(&end);
        cudaEventRecord(start);
        auto stream1 = at::cuda::getCurrentCUDAStream();
        launch_popcount_pairwise(
            reinterpret_cast<const unsigned long long*>(A_dev.data_ptr()),
            reinterpret_cast<const unsigned long long*>(B_dev.data_ptr()),
            Sa, Sb, Wa,
            reinterpret_cast<unsigned long long*>(counts_dev.data_ptr()),
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

void register_bsi_cuda(pybind11::module& m) {
    m.def("dot_product_decimal_cuda", &dot_product_decimal_cuda, "BSI dot (decimal) computed on CUDA");
    m.def("batch_dot_product_prebuilt_cuda", &batch_dot_product_prebuilt_cuda,
          pybind11::arg("q"), pybind11::arg("keyset_cap"), pybind11::arg("query_threshold") = -1.0f,
          "Batch dot using CUDA popcount kernel with CPU-built BSI");
    // Build prepacked GPU keys
    m.def("build_bsi_keys_cuda", [](torch::Tensor K, int decimalPlaces, float compress_threshold) {
        TORCH_CHECK(K.dim() == 2, "K must be 2D [num_keys, d]");
        auto Ka = K.accessor<float,2>();
        int64_t num_keys = Ka.size(0);
        int64_t d = Ka.size(1);
        PrebuiltBSIKeysCUDA* holder = new PrebuiltBSIKeysCUDA();
        holder->num_keys = num_keys;
        holder->d = d;
        holder->decimals = decimalPlaces;
        holder->threshold = compress_threshold;

        // Build one to get W
        {
            std::vector<double> tmp_row; tmp_row.reserve(d);
            for (int64_t c=0;c<d;++c) tmp_row.push_back(static_cast<double>(Ka[0][c]));
            BsiSigned<u64> b; BsiVector<u64>* t = b.buildBsiVector(tmp_row, decimalPlaces, compress_threshold);
            int S0, W0; std::vector<u64> tmp_words; bsi_flatten_words(*t, tmp_words, S0, W0);
            holder->W = W0; delete t;
        }

        holder->dev_words.reserve(num_keys);
        holder->metas.reserve(num_keys);
        size_t total_mem = 0;
        for (int64_t r=0;r<num_keys;++r) {
            std::vector<double> kv; kv.reserve(d);
            for (int64_t c=0;c<d;++c) kv.push_back(static_cast<double>(Ka[r][c]));
            BsiSigned<u64> b;
            BsiVector<u64>* bsi_k = b.buildBsiVector(kv, decimalPlaces, compress_threshold);
            bsi_k->setPartitionID(0); bsi_k->setFirstSliceFlag(true); bsi_k->setLastSliceFlag(true);
            // Allocate device matrix [S, W]
            int Sb = bsi_k->getNumberOfSlices();
            int Wb = holder->W;
            auto B_dev = torch::zeros({Sb, Wb}, torch::device(torch::kCUDA).dtype(torch::kInt64));
            // Fill each slice
            for (int s=0; s<Sb; ++s) {
                const auto& hb = bsi_k->bsi[s];
                auto* row_ptr = reinterpret_cast<unsigned long long*>(B_dev.data_ptr() + (size_t)s * Wb);
                if (hb.isVerbatim()) {
                    // verbatim: copy host words -> device row
                    size_t n = hb.buffer.size();
                    if (n > (size_t)Wb) n = (size_t)Wb;
                    if (n > 0) {
                        auto stream0 = at::cuda::getCurrentCUDAStream();
                        cudaMemcpyAsync(
                            row_ptr,
                            hb.buffer.data(),
                            n * sizeof(unsigned long long),
                            cudaMemcpyHostToDevice,
                            stream0.stream());
                    }
                    // remaining elements already zeros in B_dev
                } else {
                    // compressed: upload buffer and decompress on device to row
                    int in_len = static_cast<int>(hb.buffer.size());
                    if (in_len > 0) {
                        at::Tensor in_dev = torch::from_blob((void*)hb.buffer.data(), {(long long)in_len}, torch::TensorOptions().dtype(torch::kInt64)).clone().to(torch::kCUDA);
                        auto stream = at::cuda::getCurrentCUDAStream();
                        launch_ewah_decompress(
                            reinterpret_cast<const unsigned long long*>(in_dev.data_ptr()),
                            in_len,
                            Wb,
                            row_ptr,
                            stream.stream());
                        // a cudaDeviceSynchronize is not strictly needed here; rely on later ops
                    }
                }
            }
            holder->dev_words.push_back(B_dev);
            KeyMeta meta; meta.S = Sb; meta.offset = bsi_k->offset; meta.twos = bsi_k->twosComplement; meta.decimals = bsi_k->decimals;
            holder->metas.push_back(meta);
            total_mem += bsi_k->getSizeInMemory();
            delete bsi_k;
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
        int Sa, Wa; std::vector<u64> A_host; bsi_flatten_words(*bsi_q, A_host, Sa, Wa);
        TORCH_CHECK(Wa == keys->W, "word count mismatch (q vs keys)");
        auto A_dev = torch::from_blob(A_host.data(), {Sa, Wa}, torch::dtype(torch::kInt64)).clone().to(torch::kCUDA);

        auto out = torch::empty({keys->num_keys}, torch::TensorOptions().dtype(torch::kFloat64));
        auto out_a = out.accessor<double,1>();
        float total_kernel_ms = 0.0f;

        for (int64_t r=0;r<keys->num_keys;++r) {
            const auto& B_dev = keys->dev_words[r];
            const auto& km = keys->metas[r];
            auto counts_dev = torch::empty({km.S, Sa}, torch::device(torch::kCUDA).dtype(torch::kInt64));
            dim3 grid(Sa, km.S); dim3 block(256);
            cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
            cudaEventRecord(s);
            auto stream2 = at::cuda::getCurrentCUDAStream();
            launch_popcount_pairwise(
                reinterpret_cast<const unsigned long long*>(A_dev.data_ptr()),
                reinterpret_cast<const unsigned long long*>(B_dev.data_ptr()),
                Sa, km.S, Wa,
                reinterpret_cast<unsigned long long*>(counts_dev.data_ptr()),
                stream2.stream());
            cudaEventRecord(e); cudaEventSynchronize(e);
            float ms=0.0f; cudaEventElapsedTime(&ms,s,e); cudaEventDestroy(s); cudaEventDestroy(e);
            total_kernel_ms += ms;

            auto counts_cpu = counts_dev.to(torch::kCPU).contiguous();
            double score = accumulate_weighted_dot_meta(Sa, bsi_q->offset, bsi_q->twosComplement, bsi_q->decimals, counts_cpu, km);
            out_a[r] = score;
        }

        delete bsi_q;
        uint64_t dot_ns = (uint64_t)(total_kernel_ms * 1.0e6);
        uint64_t total_ns = build_ns + dot_ns;
        return pybind11::make_tuple(out, total_ns, build_ns, dot_ns, (uint64_t)0);
    }, pybind11::arg("q"), pybind11::arg("keyset_cuda_cap"), pybind11::arg("query_threshold") = -1.0f,
       "Batch dot using prepacked CUDA keys and CUDA popcount kernel");
}
