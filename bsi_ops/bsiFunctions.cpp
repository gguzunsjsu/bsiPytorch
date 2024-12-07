#include "../bsiCPP/bsi/BsiAttribute.hpp"
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
std::pair<size_t, size_t> getBsiInfo(const BsiAttribute<uint64_t>& bsiAttribute) {
    size_t size = bsiAttribute.bsi.size();
    size_t verbatimCount = 0;

    // Iterate through the vector
    for (const auto& element : bsiAttribute.bsi) {
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
};

VectorDotProductResult dot_product_vector(torch::Tensor m, torch::Tensor n, float precision_factor){
    VectorDotProductResult result;
    long precision_factor_long = static_cast<long>(precision_factor);
    std::vector<long> vec1 = {};
    std::vector<long> vec2 = {};
    auto m_a = m.accessor<float, 1>();
    auto n_a = n.accessor<float, 1>();

    //creating vectors
    for(auto i=0; i<m_a.size(0); i++){
        vec1.push_back(static_cast<long>(m_a[i]*precision_factor_long));
    }
    for(auto i=0; i<n_a.size(0); i++){
        vec2.push_back(static_cast<long>(n_a[i]*precision_factor_long));
    }
    cout << "Size of vectors: vec1 size " << vec1.size() << " and vec2 size: " << vec2.size() << endl;
    //doing dot product and logging time
    uint64_t start = timeSinceEpoch();
    long res = 0;
    for(int i=0;i<vec1.size(); i++){
        res += vec1[i]*vec2[i];
    }
    uint64_t end = timeSinceEpoch();

    result.result = res;
    result.timeTaken = end-start;
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

    std::cout << "[C++]" << "Got tensors of size " << m_a.size(0) << " and " << n_a.size(0) << std::endl;

    for(auto i=0; i<m_a.size(0); i++) {
        m_v.push_back(static_cast<long>(m_a[i] * precision_factor_long));
        // std::cout << "Scaled weights: " << static_cast<long>(m_a[i] * precision_factor_long) << std::endl;
    }
    for(auto i=0; i<n_a.size(0); i++) {
        n_v.push_back(static_cast<long>(n_a[i] * precision_factor_long));
    }
    u_int64_t end = timeSinceEpoch();
    std::cout << "Scaling completed and converted tensors to vectors" << std::endl;
    // std::cout << "[C++] Converted tensors to vectors" << std::endl;
    // std::cout << "[C++] Time Taken to convert tensors to vectors: " << end - start << "ns" << std::endl;

    // Finding size of m and v vectors
    size_t m_v_bytes = sizeof(m_v) + (m_v.size() * sizeof(long));
    size_t n_v_bytes = sizeof(n_v) + (n_v.size() * sizeof(long));

    double m_v_mb = static_cast<double>(m_v_bytes) / (1<<20);
    double n_v_mb = static_cast<double>(n_v_bytes) / (1<<20);

    std::cout << "m_v size in mb: " << m_v_mb << std::endl;
    std::cout << "n_v size in mb: " << n_v_mb << std::endl;

    BsiUnsigned<uint64_t> ubsi;
    BsiAttribute<uint64_t>* bsi_1;
    BsiAttribute<uint64_t>* bsi_2;
    bsi_1 = ubsi.buildBsiAttributeFromVector(m_v, 1);
    bsi_1->setPartitionID(0);
    bsi_1->setFirstSliceFlag(true);
    bsi_1->setLastSliceFlag(true);
    bsi_2 = ubsi.buildBsiAttributeFromVector(n_v, 1);
    bsi_2->setPartitionID(0);
    bsi_2->setFirstSliceFlag(true);
    bsi_2->setLastSliceFlag(true);

    std::pair<size_t, size_t> bsi1Info = getBsiInfo(*bsi_1);
    std::pair<size_t, size_t> bsi2Info = getBsiInfo(*bsi_2);
    logToFile(bsi1Info.first, bsi1Info.second);
    logToFile(bsi1Info.first, bsi1Info.second);


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
    std::cout<<"res: "<<res<<std::endl;
    // divide by conversion factor twice because mutiplication
    double result = res/float(precision_factor * precision_factor);
    std::cout<<"result after division: "<<result<<std::endl;

    DotProductResult resultStruct;
    resultStruct.result = result;
    resultStruct.timeTaken = end_dot_product - start_dot_product;
    resultStruct.sizeOfBsi1 = bsi_1->getSizeInMemory();;
    resultStruct.sizeOfBsi2 = bsi_2->getSizeInMemory();;
    delete bsi_1;
    delete bsi_2;


    return resultStruct;

}

DotProductResult dot_product_without_compression(torch::Tensor m, torch::Tensor n, float precision_factor) {
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

    std::cout << "[C++]" << "Got tensors of size " << m_a.size(0) << " and " << n_a.size(0) << std::endl;

    for(auto i=0; i<m_a.size(0); i++) {
        m_v.push_back(static_cast<long>(m_a[i] * precision_factor_long));
        // std::cout << "Scaled weights: " << static_cast<long>(m_a[i] * precision_factor_long) << std::endl;
    }
    for(auto i=0; i<n_a.size(0); i++) {
        n_v.push_back(static_cast<long>(n_a[i] * precision_factor_long));
    }
    u_int64_t end = timeSinceEpoch();
    std::cout << "Scaling completed and converted tensors to vectors" << std::endl;
    // std::cout << "[C++] Converted tensors to vectors" << std::endl;
    // std::cout << "[C++] Time Taken to convert tensors to vectors: " << end - start << "ns" << std::endl;

    // Finding size of m and v vectors
    size_t m_v_bytes = sizeof(m_v) + (m_v.size() * sizeof(long));
    size_t n_v_bytes = sizeof(n_v) + (n_v.size() * sizeof(long));

    double m_v_mb = static_cast<double>(m_v_bytes) / (1<<20);
    double n_v_mb = static_cast<double>(n_v_bytes) / (1<<20);

    std::cout << "m_v size in mb: " << m_v_mb << std::endl;
    std::cout << "n_v size in mb: " << n_v_mb << std::endl;

    BsiUnsigned<uint64_t> ubsi;
    BsiAttribute<uint64_t>* bsi_1;
    BsiAttribute<uint64_t>* bsi_2;
    bsi_1 = ubsi.buildBsiAttributeFromVector_without_compression(m_v);
    bsi_1->setPartitionID(0);
    bsi_1->setFirstSliceFlag(true);
    bsi_1->setLastSliceFlag(true);
    bsi_2 = ubsi.buildBsiAttributeFromVector_without_compression(n_v);
    bsi_2->setPartitionID(0);
    bsi_2->setFirstSliceFlag(true);
    bsi_2->setLastSliceFlag(true);

    std::pair<size_t, size_t> bsi1Info = getBsiInfo(*bsi_1);
    std::pair<size_t, size_t> bsi2Info = getBsiInfo(*bsi_2);
    logToFile(bsi1Info.first, bsi1Info.second);
    logToFile(bsi1Info.first, bsi1Info.second);


    /*
    std::cout << "Printing out the bsi vector arrays (x 10^3 for conversion factor)" << std::endl;
    for(int i=0; i<m_a.size(0); i++) {
        std::cout << bsi_1->getValue(i) << " " << bsi_2->getValue(i) << std::endl;
    }
    std::cout << "Printing bsi vector done" << std::endl;
    */
    // torch::Tensor result = torch::zeros({1}, torch::kFloat64);
    uint64_t start_dot_product = timeSinceEpoch();
    double res = bsi_1->dot_withoutCompression(bsi_2);
    uint64_t end_dot_product = timeSinceEpoch();
    std::cout<<"res: "<<res<<std::endl;
    // divide by conversion factor twice because mutiplication
    double result = res/float(precision_factor * precision_factor);
    std::cout<<"result after division: "<<result<<std::endl;

    DotProductResult resultStruct;
    resultStruct.result = result;
    resultStruct.timeTaken = end_dot_product - start_dot_product;
    resultStruct.sizeOfBsi1 = bsi_1->getSizeInMemory();;
    resultStruct.sizeOfBsi2 = bsi_2->getSizeInMemory();;
    delete bsi_1;
    delete bsi_2;


    return resultStruct;

}

// Modify the PyTorch binding to specify the return type as a tuple
pybind11::tuple dot_product_with_time(torch::Tensor m, torch::Tensor n, float precision_factor) {
    DotProductResult result = dot_product(m, n, precision_factor);
    return pybind11::make_tuple(result.result, result.timeTaken, result.sizeOfBsi1,result.sizeOfBsi2);
}

pybind11::tuple dot_product_without_compression_with_time(torch::Tensor m, torch::Tensor n, float precision_factor){
    DotProductResult result = dot_product_without_compression(m, n, precision_factor);
    return pybind11::make_tuple(result.result, result.timeTaken, result.sizeOfBsi1,result.sizeOfBsi2);
}

pybind11::tuple vector_dot_product(torch::Tensor m, torch::Tensor n, float precision_factor){
    VectorDotProductResult result = dot_product_vector(m, n, precision_factor);
    return pybind11::make_tuple(result.result, result.timeTaken);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   m.def("dot_product", &dot_product_with_time, "Dot product using BSI (Non-CUDA)");
   m.def("vector_dot_product", &vector_dot_product, "Dot product using c++ vectors");
   m.def("dot_product_without_compression", &dot_product_without_compression_with_time, "Dot product using non-compressed BSI (Non-CUDA)");
}


