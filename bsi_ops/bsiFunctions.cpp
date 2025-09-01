#include "../bsiCPP/bsi/BsiVector.hpp"
#include "../bsiCPP/bsi/BsiUnsigned.hpp"
#include "../bsiCPP/bsi/BsiSigned.hpp"
#include <fstream>
#include <torch/extension.h>
// #include <ATen/ATen.h>

#include <vector>

// for printing and timing related reasons
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

// Function to get the size of the bsi vector and count the verbatim elements
std::pair<size_t, size_t> getBsiInfo(const BsiVector<uint64_t>& BsiVector) {
    size_t size = BsiVector.bsi.size();
    size_t verbatimCount = 0;

    // Iterate through the vector
    for (const auto& element : BsiVector.bsi) {
        // Check if the verbatim field is true
        if (element.verbatim) {
            verbatimCount++;
        }
    }

    return std::make_pair(size, verbatimCount);
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

    BsiUnsigned<uint64_t> ubsi;
    BsiVector<uint64_t>* bsi_1;
    BsiVector<uint64_t>* bsi_2;
    bsi_1 = ubsi.buildBsiVector(m_v, 1);
    bsi_1->setPartitionID(0);
    bsi_1->setFirstSliceFlag(true);
    bsi_1->setLastSliceFlag(true);
    bsi_2 = ubsi.buildBsiVector(n_v, 1);
    bsi_2->setPartitionID(0);
    bsi_2->setFirstSliceFlag(true);
    bsi_2->setLastSliceFlag(true);

    auto start = timeSinceEpoch();
    long res = bsi_1->dot(bsi_2);
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
    //long precision_factor = 10000;  // 10^4
    //long precision_factor = 100000000;  // 10^7
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
//    std::cout << "Scaling completed and converted tensors to vectors" << std::endl;
    // std::cout << "[C++] Converted tensors to vectors" << std::endl;
    // std::cout << "[C++] Time Taken to convert tensors to vectors: " << end - start << "ns" << std::endl;

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
//    std::cout << "----------- added bits login -----------" << std::endl;
    bsi_1->setPartitionID(0);
    bsi_1->setFirstSliceFlag(true);
    bsi_1->setLastSliceFlag(true);
    bsi_2 = ubsi.buildBsiVector(n_v, 1);
//    std::cout << "*********** second bits logic ************" << std::endl;
    bsi_2->setPartitionID(0);
    bsi_2->setFirstSliceFlag(true);
    bsi_2->setLastSliceFlag(true);

    std::pair<size_t, size_t> bsi1Info = getBsiInfo(*bsi_1);
    std::pair<size_t, size_t> bsi2Info = getBsiInfo(*bsi_2);
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

// DotProductResult dot_product_without_compression(torch::Tensor m, torch::Tensor n, float precision_factor) {
//     //long precision_factor = 10000;  // 10^4
//     //long precision_factor = 100000000;  // 10^7
//     long precision_factor_long = static_cast<long>(precision_factor);

//     uint64_t start = timeSinceEpoch();

//     // to be used if we are converting from tensor long to long
//     // std::vector<long> m_v(m.data_ptr<long>(), m.data_ptr<long>() + m.numel());
//     // std::vector<long> n_v(n.data_ptr<long>(), n.data_ptr<long>() + n.numel());

//     //convert float tensor to vector float
//     std::vector<long> m_v = {};
//     std::vector<long> n_v = {};
//     auto m_a = m.accessor<float, 1>();
//     auto n_a = n.accessor<float, 1>();

// //    std::cout << "[C++]" << "Got tensors of size " << m_a.size(0) << " and " << n_a.size(0) << std::endl;

//     for(auto i=0; i<m_a.size(0); i++) {
//         m_v.push_back(static_cast<long>(m_a[i] * precision_factor_long));
//         // std::cout << "Scaled weights: " << static_cast<long>(m_a[i] * precision_factor_long) << std::endl;
//     }
//     for(auto i=0; i<n_a.size(0); i++) {
//         n_v.push_back(static_cast<long>(n_a[i] * precision_factor_long));
//     }
//     u_int64_t end = timeSinceEpoch();
//     std::cout << "Scaling completed and converted tensors to vectors" << std::endl;
//     // std::cout << "[C++] Converted tensors to vectors" << std::endl;
//     // std::cout << "[C++] Time Taken to convert tensors to vectors: " << end - start << "ns" << std::endl;

//     // Finding size of m and v vectors
//     size_t m_v_bytes = sizeof(m_v) + (m_v.size() * sizeof(long));
//     size_t n_v_bytes = sizeof(n_v) + (n_v.size() * sizeof(long));

//     double m_v_mb = static_cast<double>(m_v_bytes) / (1<<20);
//     double n_v_mb = static_cast<double>(n_v_bytes) / (1<<20);

// //    std::cout << "m_v size in mb: " << m_v_mb << std::endl;
// //    std::cout << "n_v size in mb: " << n_v_mb << std::endl;

//     BsiUnsigned<uint64_t> ubsi;
//     BsiVector<uint64_t>* bsi_1;
//     BsiVector<uint64_t>* bsi_2;
//     bsi_1 = ubsi.buildBsiVector_without_compression(m_v);
//     bsi_1->setPartitionID(0);
//     bsi_1->setFirstSliceFlag(true);
//     bsi_1->setLastSliceFlag(true);
//     bsi_2 = ubsi.buildBsiVector_without_compression(n_v);
//     bsi_2->setPartitionID(0);
//     bsi_2->setFirstSliceFlag(true);
//     bsi_2->setLastSliceFlag(true);

//     std::pair<size_t, size_t> bsi1Info = getBsiInfo(*bsi_1);
//     std::pair<size_t, size_t> bsi2Info = getBsiInfo(*bsi_2);
//     logToFile(bsi1Info.first, bsi1Info.second);
//     logToFile(bsi1Info.first, bsi1Info.second);


//     /*
//     std::cout << "Printing out the bsi vector arrays (x 10^3 for conversion factor)" << std::endl;
//     for(int i=0; i<m_a.size(0); i++) {
//         std::cout << bsi_1->getValue(i) << " " << bsi_2->getValue(i) << std::endl;
//     }
//     std::cout << "Printing bsi vector done" << std::endl;
//     */
//     // torch::Tensor result = torch::zeros({1}, torch::kFloat64);
//     uint64_t start_dot_product = timeSinceEpoch();
//     double res = bsi_1->dot_withoutCompression(bsi_2);
//     uint64_t end_dot_product = timeSinceEpoch();
//     std::cout<<"res: "<<res<<std::endl;
//     // divide by conversion factor twice because mutiplication
//     double result = res/float(precision_factor * precision_factor);
//     std::cout<<"result after division: "<<result<<std::endl;

//     DotProductResult resultStruct;
//     resultStruct.result = result;
//     resultStruct.timeTaken = end_dot_product - start_dot_product;
//     resultStruct.sizeOfBsi1 = bsi_1->getSizeInMemory();;
//     resultStruct.sizeOfBsi2 = bsi_2->getSizeInMemory();;
//     delete bsi_1;
//     delete bsi_2;


//     return resultStruct;

// }

// Modify the PyTorch binding to specify the return type as a tuple
pybind11::tuple dot_product_with_time(torch::Tensor m, torch::Tensor n, float precision_factor) {
    DotProductResult result = dot_product(m, n, precision_factor);
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

// pybind11::tuple dot_product_without_compression_with_time(torch::Tensor m, torch::Tensor n, float precision_factor){
//     DotProductResult result = dot_product_without_compression(m, n, precision_factor);
//     return pybind11::make_tuple(result.result, result.timeTaken, result.sizeOfBsi1,result.sizeOfBsi2);
// }

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

PYBIND11_MODULE(bsi_ops, m) {
   m.def("dot_product", &dot_product_with_time, "Dot product using BSI (Non-CUDA)");
   m.def("vector_dot_product", &vector_dot_product, "Dot product using c++ vectors");
//    m.def("dot_product_without_compression", &dot_product_without_compression_with_time, "Dot product using non-compressed BSI (Non-CUDA)");
   m.def("random_number_dot_product_vector",  &random_number_dot_product_vector, "Dot product of random numbers using C++ vectors");
   m.def("random_number_dot_product_bsi", &random_number_dot_product_bsi, "Dot product of random numbers using bsi");
   m.def("vector_dot_product_no_precison", &vector_dot_product_no_precison, "Dot product using c++ vector without precison");
}


