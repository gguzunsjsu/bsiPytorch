#include "../bsiCPP/bsi/BsiVector.hpp"
#include "../bsiCPP/bsi/BsiUnsigned.hpp"
#include "../bsiCPP/bsi/BsiSigned.hpp"
#include <fstream>
#include <cmath>
#include <torch/extension.h>
#include <filesystem>
// #include <ATen/ATen.h>

#include <vector>

#include <iostream>
#include <chrono>

using std::cout;
using std::endl;

// function to return time since epoch
// for measuring how long certain tasks take
uint64_t timeSinceEpoch() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
}

struct BsiSliceStats {
    size_t total_slices = 0;
    size_t verbatim_slices = 0;

    size_t compressed_slices() const {
        return (total_slices >= verbatim_slices) ? (total_slices - verbatim_slices) : 0;
    }

    double compressed_pct() const {
        return total_slices ? (static_cast<double>(compressed_slices()) * 100.0 / total_slices) : 0.0;
    }

    double verbatim_pct() const {
        return total_slices ? (static_cast<double>(verbatim_slices) * 100.0 / total_slices) : 0.0;
    }

    void accumulate(const BsiSliceStats &other) {
        total_slices += other.total_slices;
        verbatim_slices += other.verbatim_slices;
    }
};

BsiSliceStats compute_bsi_slice_stats(const BsiVector<uint64_t> &vec) {
    BsiSliceStats stats;
    stats.total_slices = vec.bsi.size();
    for (const auto &slice : vec.bsi) {
        if (slice.verbatim) {
            ++stats.verbatim_slices;
        }
    }
    return stats;
}

// Function to log information to a text file
void logToFile(size_t size, size_t verbatimCount) {
    std::ofstream outFile("extract_tensors/log.txt", std::ios::app);  // Open file in append mode
    if (outFile.is_open()) {
        double ratio = static_cast<double>(verbatimCount) / size;
        outFile << "Number of BSI slices: " << size << ", Number of verbatim slices: " << verbatimCount << ", Ratio: " << ratio << "\n";
        outFile.close();
    } else {
        std::cerr << "Unable to open log file for writing." << std::endl;
    }
}

// Define a structure to hold the result and time
struct DotProductResult {
    double result;
    uint64_t timeTaken;
    size_t sizeOfBsi1;
    size_t sizeOfBsi2;


};

struct VectorDotProductResult{
    long result;
    uint64_t timeTaken;
    size_t memoryUsedVec1;
    size_t memoryUsedVec2;
    size_t bitsUsedVec1;
    size_t bitsUsedVec2;
};

struct RandomNumberDotProduct{
    long result;
    uint64_t timeTaken;
};

RandomNumberDotProduct random_number_dot_vector(torch::Tensor m, torch::Tensor n){
    RandomNumberDotProduct result;
    std::vector<long> vec1 = {};
    std::vector<long> vec2 = {};
    auto m_a = m.accessor<float, 1>();
    auto n_a = n.accessor<float, 1>();

    for(auto i=0; i<m_a.size(0); i++){
        vec1.push_back(static_cast<long>(m_a[i]));
    }
    for(auto i=0; i<n_a.size(0); i++){
        vec2.push_back(static_cast<long>(n_a[i]));
    }
    long res=0;
    auto start = timeSinceEpoch();
    for(auto i=0; i<vec1.size(); i++){
        res += vec1[i]*vec2[i];
    }
    auto end = timeSinceEpoch();
    auto random_vector_duration = (end-start);
    result.result = res;
    result.timeTaken = random_vector_duration;
    return result;

}

RandomNumberDotProduct random_number_dot_bsi(torch::Tensor m, torch::Tensor n){
    RandomNumberDotProduct result;
    std::vector<long> m_v;
    std::vector<long> n_v;
    auto m_a = m.accessor<float, 1>();
    auto n_a = n.accessor<float, 1>();

    for(auto i=0; i<m_a.size(0); i++){
        m_v.push_back(static_cast<long>(m_a[i]));
    }
    for(auto i=0; i<n_a.size(0); i++){
        n_v.push_back(static_cast<long>(n_a[i]));
    }

    BsiSigned<uint64_t> bsi;
    BsiVector<uint64_t>* bsi_1;
    BsiVector<uint64_t>* bsi_2;
    bsi_1 = bsi.buildBsiVector(m_v, 1);
    bsi_1->setPartitionID(0);
    bsi_1->setFirstSliceFlag(true);
    bsi_1->setLastSliceFlag(true);
    bsi_2 = bsi.buildBsiVector(n_v, 1);
    bsi_2->setPartitionID(0);
    bsi_2->setFirstSliceFlag(true);
    bsi_2->setLastSliceFlag(true);

    auto start = timeSinceEpoch();
    double res = bsi_1->dot(bsi_2);
    auto end = timeSinceEpoch();
    auto duration = (end-start);
    result.result=res;
    result.timeTaken=duration;
    return result;

}

VectorDotProductResult dot_product_vector(torch::Tensor m, torch::Tensor n, float precision_factor){
    VectorDotProductResult result;
    int64_t precision_factor_long = static_cast<int64_t>(precision_factor);
    std::vector<int64_t> vec1 = {};
    std::vector<int64_t> vec2 = {};
    auto m_a = m.accessor<float, 1>();
    auto n_a = n.accessor<float, 1>();

    //creating vectors
    for(auto i=0; i<m_a.size(0); i++){
        vec1.push_back(static_cast<int64_t>(m_a[i]*precision_factor_long));
    }
    for(auto i=0; i<n_a.size(0); i++){
        vec2.push_back(static_cast<int64_t>(n_a[i]*precision_factor_long));
    }

//    cout << "Size of vectors: vec1 size " << vec1.size() << " and vec2 size: " << vec2.size() << endl;
    //doing dot product and logging time
    uint64_t start = timeSinceEpoch();
    auto res = 0;
    for(int i=0;i<vec1.size(); i++){
        res += vec1[i]*vec2[i];
    }
    uint64_t end = timeSinceEpoch();

    //For checking
//    size_t memory_for_elements = vec1.capacity() * sizeof(int64_t);
//    cout << "Memory for elements is: " << memory_for_elements/(1024*1024) << " MB" << endl;

    std::byte* vec1_start = reinterpret_cast<std::byte*>(vec1.data());
    std::byte* vec1_end = vec1_start+(vec1.capacity() * sizeof(int64_t));
    size_t vec1_memory = reinterpret_cast<uintptr_t>(vec1_end)-reinterpret_cast<uintptr_t>(vec1_start);

    std::byte* vec2_start = reinterpret_cast<std::byte*>(vec2.data());
    std::byte* vec2_end = vec2_start+(vec2.capacity() * sizeof(int64_t));
    size_t vec2_memory = reinterpret_cast<uintptr_t>(vec2_end)-reinterpret_cast<uintptr_t>(vec2_start);

//    size_t vec1_metadata = sizeof(std::vector<int16_t>);
//    size_t vec2_metadata = sizeof(std::vector<int16_t>);

    result.result = res;
    result.timeTaken = end-start;
    result.memoryUsedVec1 = vec1_memory;  // Memory in bytes
    result.memoryUsedVec2 = vec2_memory;  // Memory in bytes
    result.bitsUsedVec1 = sizeof(int64_t) * 8;     // bytes to bits
    result.bitsUsedVec2 = sizeof(int64_t) * 8;     // bytes to bits

//    cout << "Vec1 bits" << result.bitsUsedVec1 << endl;

    return result;
    
}

VectorDotProductResult dot_product_vector_noPrecision(torch::Tensor m, torch::Tensor n){
    VectorDotProductResult result;
    std::vector<int16_t> vec1 = {};
    std::vector<int16_t> vec2 = {};
    auto m_a = m.accessor<float, 1>();
    auto n_a = n.accessor<float, 1>();

    //creating vectors
    for(auto i=0; i<m_a.size(0); i++){
        vec1.push_back(static_cast<int16_t>(m_a[i]));
    }
    for(auto i=0; i<n_a.size(0); i++){
        vec2.push_back(static_cast<int16_t>(n_a[i]));
    }

//    cout << "Size of vectors: vec1 size " << vec1.size() << " and vec2 size: " << vec2.size() << endl;
    //doing dot product and logging time
    uint64_t start = timeSinceEpoch();
    long res = 0;
    for(int i=0;i<vec1.size(); i++){
        res += vec1[i]*vec2[i];
    }
    uint64_t end = timeSinceEpoch();

    cout << "Vector result: " << res << endl;

    result.result = res;
    result.timeTaken = end-start;
    result.memoryUsedVec1 = vec1.size() * sizeof(int16_t);  // Memory in bytes
    result.memoryUsedVec2 = vec2.size() * sizeof(int16_t);  // Memory in bytes
    result.bitsUsedVec1 = sizeof(int16_t) * 8;     // bytes to bits
    result.bitsUsedVec2 = sizeof(int16_t) * 8;     // bytes to bits

    cout << "Vec1 bits" << result.bitsUsedVec1 << endl;

    return result;

}

DotProductResult dot_product(torch::Tensor m, torch::Tensor n, float precision_factor) {
    long precision_factor_long = static_cast<long>(precision_factor);
    uint64_t start = timeSinceEpoch();

    // to be used if we are converting from tensor long to long
    // std::vector<long> m_v(m.data_ptr<long>(), m.data_ptr<long>() + m.numel());
    // std::vector<long> n_v(n.data_ptr<long>(), n.data_ptr<long>() + n.numel());

    //convert float tensor to vector float
    std::vector<long> m_v = {};
    std::vector<long> n_v = {};
    auto m_a = m.accessor<float, 1>();
    auto n_a = n.accessor<float, 1>();

//    std::cout << "[C++]" << "Got tensors of size " << m_a.size(0) << " and " << n_a.size(0) << std::endl; this would come as 10420224

    for(auto i=0; i<m_a.size(0); i++) {
        m_v.push_back(static_cast<long>(m_a[i] * precision_factor_long));
        // std::cout << "Scaled weights: " << static_cast<long>(m_a[i] * precision_factor_long) << std::endl;
    }
    for(auto i=0; i<n_a.size(0); i++) {
        n_v.push_back(static_cast<long>(n_a[i] * precision_factor_long));
    }
    u_int64_t end = timeSinceEpoch();

    // Finding size of m and v vectors
    size_t m_v_bytes = sizeof(m_v) + (m_v.size() * sizeof(long));
    size_t n_v_bytes = sizeof(n_v) + (n_v.size() * sizeof(long));

    double m_v_mb = static_cast<double>(m_v_bytes) / (1<<20);
    double n_v_mb = static_cast<double>(n_v_bytes) / (1<<20);

//    std::cout << "m_v size in mb: " << m_v_mb << std::endl;
//    std::cout << "n_v size in mb: " << n_v_mb << std::endl;

    BsiUnsigned<uint64_t> ubsi;
    BsiVector<uint64_t>* bsi_1;
    BsiVector<uint64_t>* bsi_2;
    bsi_1 = ubsi.buildBsiVector(m_v, 1);
    // std::cout << "BSI1 slices: " << bsi_1->getNumberOfSlices() << std::endl;
//    std::cout << "----------- added bits login -----------" << std::endl;
    bsi_1->setPartitionID(0);
    bsi_1->setFirstSliceFlag(true);
    bsi_1->setLastSliceFlag(true);
    bsi_2 = ubsi.buildBsiVector(n_v, 1);
    // std::cout << "BSI2 slices: " << bsi_2->getNumberOfSlices() << std::endl;
//    std::cout << "*********** second bits logic ************" << std::endl;
    bsi_2->setPartitionID(0);
    bsi_2->setFirstSliceFlag(true);
    bsi_2->setLastSliceFlag(true);

//    std::cout << "Not logging in this run" << std::endl;
//    logToFile(bsi1Info.first, bsi1Info.second);
//    logToFile(bsi1Info.first, bsi1Info.second);


    /*
    std::cout << "Printing out the bsi vector arrays (x 10^3 for conversion factor)" << std::endl;
    for(int i=0; i<m_a.size(0); i++) {
        std::cout << bsi_1->getValue(i) << " " << bsi_2->getValue(i) << std::endl;
    }
    std::cout << "Printing bsi vector done" << std::endl;
    */
    // torch::Tensor result = torch::zeros({1}, torch::kFloat64);
    uint64_t start_dot_product = timeSinceEpoch();
    double res = bsi_1->dot(bsi_2);
    uint64_t end_dot_product = timeSinceEpoch();
//    std::cout<<"res: "<<res<<std::endl;
    // divide by conversion factor twice because mutiplication
    double result = res/float(precision_factor * precision_factor);
//    std::cout<<"result after division: "<<result<<std::endl;
//    cout << "dot product completed" << endl;
    DotProductResult resultStruct;
    resultStruct.result = result;
    resultStruct.timeTaken = end_dot_product - start_dot_product;
    uint64_t bsiMemory = 0;
//    for(auto i=0; i<bsi_1->bsi.size(); i++){
//        bsiMemory += bsi_1[i].getSizeInMemory();
//    }
//    cout << "BSI_1 memory " << bsiMemory/(1024*1024) << endl;
    resultStruct.sizeOfBsi1 = bsi_1->getSizeInMemory();
    resultStruct.sizeOfBsi2 = bsi_2->getSizeInMemory();
//    cout << "Fixed by returning total" << endl;
//    cout << "bsi1 size: " << resultStruct.sizeOfBsi1/(1024*1024) << endl;
//    cout << "bsi2 size: " << resultStruct.sizeOfBsi2/(1024*1024) << endl;
    delete bsi_1;
    delete bsi_2;


    return resultStruct;

}

DotProductResult dot_product_decimal(torch::Tensor m, torch::Tensor n, int decimalPlaces){
    std::vector<double> m_v = {};
    std::vector<double> n_v = {};
    auto m_a = m.accessor<float, 1>();
    auto n_a = n.accessor<float, 1>();

    for(auto i=0; i<m_a.size(0); i++){
        m_v.push_back(m_a[i]);
    }
    for(auto i=0; i<n_a.size(0); i++){
        n_v.push_back(n_a[i]);
    }

    BsiSigned<uint64_t> bsi;
    BsiVector<uint64_t>* bsi_1;
    BsiVector<uint64_t>* bsi_2;
    bsi_1 = bsi.buildBsiVector(m_v, decimalPlaces, 0.2f);
    bsi_1->setPartitionID(0); 
    bsi_1->setFirstSliceFlag(true); 
    bsi_1->setLastSliceFlag(true);

    bsi_2 = bsi.buildBsiVector(n_v, decimalPlaces, 0.2f);
    bsi_2->setPartitionID(0); 
    bsi_2->setFirstSliceFlag(true); 
    bsi_2->setLastSliceFlag(true);

    uint64_t start = timeSinceEpoch();
    double raw = static_cast<double>(bsi_1->dot(bsi_2));
    uint64_t end = timeSinceEpoch();

    const int totalDecimals = bsi_1->decimals + bsi_2->decimals;
    double scale = (totalDecimals > 0) ? std::pow(10.0, totalDecimals) : 1.0;
    double res = raw / scale;

    DotProductResult result;
    result.result = res;
    result.timeTaken = end-start;
    result.sizeOfBsi1 = bsi_1->getSizeInMemory();
    result.sizeOfBsi2 = bsi_2->getSizeInMemory();
    delete bsi_1;
    delete bsi_2;
    return result;
}

pybind11::tuple dot_product_with_time(torch::Tensor m, torch::Tensor n, float precision_factor) {
    DotProductResult result = dot_product(m, n, precision_factor);
    return pybind11::make_tuple(result.result, result.timeTaken, result.sizeOfBsi1,result.sizeOfBsi2);
}

pybind11::tuple dot_product_with_decimal(torch::Tensor m, torch::Tensor n, int decimalPlaces){
    DotProductResult result = dot_product_decimal(m, n, decimalPlaces);
    return pybind11::make_tuple(result.result, result.timeTaken, result.sizeOfBsi1,result.sizeOfBsi2);
}

pybind11::tuple random_number_dot_product_bsi(torch::Tensor m, torch::Tensor n){
    RandomNumberDotProduct result = random_number_dot_bsi(m, n);
    return pybind11::make_tuple(result.result, result.timeTaken);
}

pybind11::tuple random_number_dot_product_vector(torch::Tensor m, torch::Tensor n){
    RandomNumberDotProduct result = random_number_dot_vector(m, n);
    return pybind11::make_tuple(result.result, result.timeTaken);
}

pybind11::tuple vector_dot_product_no_precison(torch::Tensor m, torch::Tensor n){
    VectorDotProductResult result = dot_product_vector_noPrecision(m, n);
    return pybind11::make_tuple(result.result,
                                result.timeTaken,
                                result.memoryUsedVec1,
                                result.memoryUsedVec2,
                                result.bitsUsedVec1,
                                result.bitsUsedVec2);
}

pybind11::tuple vector_dot_product(torch::Tensor m, torch::Tensor n, float precision_factor){
    VectorDotProductResult result = dot_product_vector(m, n, precision_factor);
    return pybind11::make_tuple(result.result,
                                result.timeTaken,
                                result.memoryUsedVec1,
                                result.memoryUsedVec2,
                                result.bitsUsedVec1,
                                result.bitsUsedVec2 );
}

pybind11::tuple batch_dot_product(torch::Tensor q, torch::Tensor K, int decimalPlaces, 
                                 float threshold = 0.2f) {
    TORCH_CHECK(q.dim() == 1, "q must be 1D");
    TORCH_CHECK(K.dim() == 2, "K must be 2D");
    TORCH_CHECK(q.size(0) == K.size(1), "q.size(0) must equal K.size(1)");

    // int64_t pf = static_cast<int64_t>(precision_factor);

    auto q_a = q.accessor<float, 1>();
    auto K_a = K.accessor<float, 2>();
    const int64_t d = q_a.size(0);
    const int64_t num_keys = K_a.size(0);

    std::vector<double> q_v; q_v.reserve(d);
    for (int64_t i = 0; i < d; ++i) q_v.push_back(static_cast<double>(q_a[i]));

    BsiSigned<uint64_t> bsi;
    BsiVector<uint64_t>* bsi_q = bsi.buildBsiVector(q_v, decimalPlaces, threshold);
    bsi_q->setPartitionID(0); 
    bsi_q->setFirstSliceFlag(true); 
    bsi_q->setLastSliceFlag(true);
    size_t mem_q = bsi_q->getSizeInMemory();

    auto out = torch::empty({num_keys}, torch::TensorOptions().dtype(torch::kFloat64));
    auto out_a = out.accessor<double, 1>();
    size_t mem_k_first = 0;

    uint64_t total_ns = 0;
    for (int64_t r = 0; r < num_keys; ++r) {
        std::vector<double> k_v; 
        k_v.reserve(d);
        for (int64_t c = 0; c < d; ++c) k_v.push_back(static_cast<double>(K_a[r][c]));
        BsiVector<uint64_t>* bsi_k = bsi.buildBsiVector(k_v, decimalPlaces, threshold);
        bsi_k->setPartitionID(0); bsi_k->setFirstSliceFlag(true); bsi_k->setLastSliceFlag(true);
        if (r == 0) mem_k_first = bsi_k->getSizeInMemory();

        uint64_t t0 = timeSinceEpoch();
        double raw = static_cast<double>(bsi_q->dot(bsi_k));
        const int totalDecimals = bsi_q->decimals + bsi_k->decimals;
        double scale = (totalDecimals > 0) ? std::pow(10.0, totalDecimals) : 1.0;
        double score = raw / scale;
        uint64_t t1 = timeSinceEpoch();
        total_ns += (t1 - t0);

        // double score = raw / double(precision_factor * precision_factor);
        out_a[r] = score;

        delete bsi_k;
    }

    delete bsi_q;
    return pybind11::make_tuple(out, total_ns, mem_q, mem_k_first);
}

struct PrebuiltBSIKeys {
    std::vector<BsiVector<uint64_t>*> keys;
    int decimalPlaces = 0;
    int64_t d = 0;
    int64_t num_keys = 0;
    float threshold = 0.2f;
};

static PrebuiltBSIKeys* capsule_to_keys(const pybind11::capsule& cap) {
    return reinterpret_cast<PrebuiltBSIKeys*>(cap.get_pointer());
}

pybind11::dict keyset_slice_stats(pybind11::capsule keyset_cap) {
    auto* keys = capsule_to_keys(keyset_cap);
    TORCH_CHECK(keys != nullptr, "Invalid BSI keys capsule");

    BsiSliceStats aggregate;
    for (auto* key : keys->keys) {
        if (key != nullptr) {
            aggregate.accumulate(compute_bsi_slice_stats(*key));
        }
    }

    const size_t compressed = aggregate.compressed_slices();

    pybind11::dict result;
    result["num_vectors"] = static_cast<size_t>(keys->num_keys);
    result["total_slices"] = aggregate.total_slices;
    result["verbatim_slices"] = aggregate.verbatim_slices;
    result["compressed_slices"] = compressed;
    result["compressed_pct"] = aggregate.compressed_pct();
    result["verbatim_pct"] = aggregate.verbatim_pct();
    return result;
}

pybind11::dict tensor_slice_stats(torch::Tensor tensor, int decimalPlaces, float compress_threshold = 0.2f, bool signed_input = true) {
    TORCH_CHECK(tensor.dim() == 1, "tensor must be 1D [n]");

    auto tensor_cpu = tensor.detach().to(torch::kFloat32).cpu().contiguous();
    auto accessor = tensor_cpu.accessor<float, 1>();
    const int64_t length = accessor.size(0);

    std::vector<double> values;
    values.reserve(length);
    for (int64_t i = 0; i < length; ++i) {
        values.push_back(static_cast<double>(accessor[i]));
    }

    BsiVector<uint64_t>* bsi_vec = nullptr;
    if (signed_input) {
        BsiSigned<uint64_t> builder;
        bsi_vec = builder.buildBsiVector(values, decimalPlaces, compress_threshold);
    } else {
        BsiUnsigned<uint64_t> builder;
        bsi_vec = builder.buildBsiVector(values, decimalPlaces, compress_threshold);
    }

    TORCH_CHECK(bsi_vec != nullptr, "Failed to build BSI vector for slice stats");

    BsiSliceStats stats = compute_bsi_slice_stats(*bsi_vec);
    const size_t compressed = stats.compressed_slices();
    const size_t memory_bytes = bsi_vec->getSizeInMemory();
    delete bsi_vec;

    pybind11::dict result;
    result["total_slices"] = stats.total_slices;
    result["verbatim_slices"] = stats.verbatim_slices;
    result["compressed_slices"] = compressed;
    result["compressed_pct"] = stats.compressed_pct();
    result["verbatim_pct"] = stats.verbatim_pct();
    result["memory_bytes"] = memory_bytes;
    result["decimal_places"] = decimalPlaces;
    result["compress_threshold"] = compress_threshold;
    result["signed_input"] = signed_input;
    return result;
}

pybind11::tuple build_bsi_keys(torch::Tensor K, int decimalPlaces, float compress_threshold = 0.2f) {
    TORCH_CHECK(K.dim() == 2, "K must be 2D [num_keys, d]");
    // int64_t pf = static_cast<int64_t>(precision_factor);

    auto K_a = K.accessor<float, 2>();
    const int64_t num_keys = K_a.size(0);
    const int64_t d = K_a.size(1);

    BsiSigned<uint64_t> bsi;

    auto* holder = new PrebuiltBSIKeys();
    holder->decimalPlaces = decimalPlaces;
    holder->d = d;
    holder->num_keys = num_keys;
    holder->threshold = compress_threshold;

    size_t total_mem = 0;
    holder->keys.reserve(num_keys);

    for (int64_t r = 0; r < num_keys; ++r) {
        std::vector<double> k_v; k_v.reserve(d);
        for (int64_t c = 0; c < d; ++c) {
            k_v.push_back(static_cast<double>(K_a[r][c]));
        }
        BsiVector<uint64_t>* bsi_k = bsi.buildBsiVector(k_v, decimalPlaces, compress_threshold);
        bsi_k->setPartitionID(0);
        bsi_k->setFirstSliceFlag(true);
        bsi_k->setLastSliceFlag(true);
        total_mem += bsi_k->getSizeInMemory();
        holder->keys.push_back(bsi_k);
    }

    pybind11::capsule cap(holder, "PrebuiltBSIKeys",
        [](PyObject* capsule) {
            auto* p = reinterpret_cast<PrebuiltBSIKeys*>(PyCapsule_GetPointer(capsule, "PrebuiltBSIKeys"));
            if (p) {
                for (auto* k : p->keys) { delete k; }
                delete p;
            }
        }
    );

    return pybind11::make_tuple(cap, total_mem, num_keys, d);
}

pybind11::tuple batch_dot_product_prebuilt(torch::Tensor q, pybind11::capsule keyset_cap, 
                                          float threshold = 0.2f) {
    TORCH_CHECK(q.dim() == 1, "q must be 1D [d]");
    auto* keys = capsule_to_keys(keyset_cap);
    TORCH_CHECK(keys != nullptr, "Invalid BSI keys capsule");

    const int64_t d = keys->d;
    TORCH_CHECK(q.size(0) == d, "q.size(0) must equal keys' dimension");

    const int decimalPlaces = keys->decimalPlaces;
    float threshold_val = threshold >= 0.0f ? threshold : keys->threshold;
    auto q_a = q.accessor<float, 1>();

    std::vector<double> q_v; q_v.reserve(d);
    for (int64_t i = 0; i < d; ++i) q_v.push_back(static_cast<double>(q_a[i]));

    BsiSigned<uint64_t> bsi;
    BsiVector<uint64_t>* bsi_q = bsi.buildBsiVector(q_v, decimalPlaces, threshold_val);
    bsi_q->setPartitionID(0);
    bsi_q->setFirstSliceFlag(true);
    bsi_q->setLastSliceFlag(true);
    size_t mem_q = bsi_q->getSizeInMemory();

    auto out = torch::empty({keys->num_keys}, torch::TensorOptions().dtype(torch::kFloat64));
    auto out_a = out.accessor<double, 1>();

    uint64_t total_ns = 0;
    for (int64_t r = 0; r < keys->num_keys; ++r) {
        uint64_t t0 = timeSinceEpoch();
        double raw = static_cast<double>(bsi_q->dot(keys->keys[r]));
        uint64_t t1 = timeSinceEpoch();
        total_ns += (t1 - t0);
        const int totalDecimals = bsi_q->decimals + keys->keys[r]->decimals;
        // if (r == 0) {
        //     std::cout << "[batch_dot_product_prebuilt] q decimals=" << bsi_q->decimals
        //               << " k decimals=" << keys->keys[r]->decimals << " total=" << totalDecimals
        //               << " raw=" << raw << std::endl;
        // }
        double scale = (totalDecimals > 0) ? std::pow(10.0, totalDecimals) : 1.0;
        double score = raw / scale;
        out_a[r] = score;
    }

    delete bsi_q;
    return pybind11::make_tuple(out, total_ns, mem_q);
}

pybind11::tuple keyset_size_on_disk(pybind11::capsule keyset_cap) {
    auto* keys = capsule_to_keys(keyset_cap);
    TORCH_CHECK(keys != nullptr, "Invalid BSI keys capsule");
    size_t mem_in_memory = 0;
    size_t bytes_on_disk = 0;
    for (auto* key : keys->keys) {
        if (!key) continue;
        mem_in_memory += key->getSizeInMemory(); // current behaviour you already use
        // sum file size for each slice if serialized with write(..., savesizeinbits=true)
        for (const auto& slice : key->bsi) {
            bytes_on_disk += slice.sizeOnDisk(true);
        }
    }
    return pybind11::make_tuple(mem_in_memory, bytes_on_disk);
}

pybind11::none save_keyset(pybind11::capsule keyset_cap, const std::string& out_dir) {
    auto* keys = capsule_to_keys(keyset_cap);
    TORCH_CHECK(keys != nullptr, "Invalid BSI keys capsule");
    std::filesystem::create_directories(out_dir);
    for (size_t r = 0; r < keys->keys.size(); ++r) {
        const auto& vec = *keys->keys[r];
        // write each slice as its own file: key_<r>_slice_<s>.hb
        for (size_t s = 0; s < vec.bsi.size(); ++s) {
            const auto& hb = vec.bsi[s];
            std::string fname = out_dir + "/key_" + std::to_string(r) + "_slice_" + std::to_string(s) + ".hb";
            std::ofstream ofs(fname, std::ios::binary);
            TORCH_CHECK(ofs.good(), "Failed to open file: ", fname);
            hb.write(ofs, true); // format described in HybridBitmap::write
        }
    }
    return pybind11::none();
}

PYBIND11_MODULE(bsi_ops, m) {
    m.def("dot_product", &dot_product_with_time, "Dot product using BSI (Non-CUDA)");
    m.def("dot_product_decimal", &dot_product_with_decimal, "Dot product using BSI with decimal places (Non-CUDA)");
    m.def("vector_dot_product", &vector_dot_product, "Dot product using c++ vectors");
    m.def("random_number_dot_product_vector",  &random_number_dot_product_vector, "Dot product of random numbers using C++ vectors");
    m.def("random_number_dot_product_bsi", &random_number_dot_product_bsi, "Dot product of random numbers using bsi");
    m.def("vector_dot_product_no_precison", &vector_dot_product_no_precison, "Dot product using c++ vector without precison");
    m.def("batch_dot_product", &batch_dot_product, "Batch dot product: one query vs many keys using BSI");
    m.def("build_bsi_keys", &build_bsi_keys, pybind11::arg("K"), pybind11::arg("decimal_places"), pybind11::arg("compress_threshold") = 0.2f,
         "Prebuild BSI keys for a weight matrix; returns a capsule and total memory in bytes");
    m.def("batch_dot_product_prebuilt", &batch_dot_product_prebuilt, pybind11::arg("q"), pybind11::arg("keyset_cap"), pybind11::arg("query_threshold") = -1.0f,
         "Batch dot product using prebuilt BSI keys with optional query compression threshold");
    m.def("keyset_slice_stats", &keyset_slice_stats, pybind11::arg("keyset_cap"),
         "Aggregate slice statistics for a prebuilt BSI key capsule");
    m.def("tensor_slice_stats", &tensor_slice_stats, pybind11::arg("tensor"), pybind11::arg("decimal_places"),
         pybind11::arg("compress_threshold") = 0.2f, pybind11::arg("signed_input") = true,
         "Build a BSI vector for a 1D tensor and return basic slice compression statistics");
    m.def("keyset_size_on_disk", &keyset_size_on_disk, "Return (mem_in_memory, bytes_on_disk)");
    m.def("save_keyset", &save_keyset, "Serialize keyset to directory of .hb slice files");
}
