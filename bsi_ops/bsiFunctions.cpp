#include "../bsiCPP/bsi/BsiAttribute.hpp"
#include "../bsiCPP/bsi/BsiUnsigned.hpp"
#include "../bsiCPP/bsi/BsiSigned.hpp"
#include <fstream>
#include <torch/extension.h>

#include <vector>

// for printing and timing related reasons
#include <iostream>
#include <chrono>

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
};

DotProductResult dot_product(torch::Tensor m, torch::Tensor n, float conversion_factor) {
    //long CONVERSION_FACTOR = 10000;  // 10^4
    //long CONVERSION_FACTOR = 100000000;  // 10^7
    long CONVERSION_FACTOR = static_cast<long>(conversion_factor);

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
        m_v.push_back(static_cast<long>(m_a[i] * CONVERSION_FACTOR));
    }
    for(auto i=0; i<n_a.size(0); i++) {
        n_v.push_back(static_cast<long>(n_a[i] * CONVERSION_FACTOR));
    }
    u_int64_t end = timeSinceEpoch();

    // std::cout << "[C++] Converted tensors to vectors" << std::endl;
    // std::cout << "[C++] Time Taken to convert tensors to vectors: " << end - start << "ns" << std::endl;

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
    double result = res/float(CONVERSION_FACTOR * CONVERSION_FACTOR);
    std::cout<<"result after division: "<<result<<std::endl;
    delete bsi_1;
    delete bsi_2;
    DotProductResult resultStruct;
    resultStruct.result = result;
    resultStruct.timeTaken = end_dot_product - start_dot_product;


    return resultStruct;

}
// Modify the PyTorch binding to specify the return type as a tuple
pybind11::tuple dot_product_with_time(torch::Tensor m, torch::Tensor n, float conversion_factor) {
    DotProductResult result = dot_product(m, n, conversion_factor);
    return pybind11::make_tuple(result.result, result.timeTaken);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   m.def("dot_product", &dot_product_with_time, "Dot product using BSI (Non-CUDA)");
}


