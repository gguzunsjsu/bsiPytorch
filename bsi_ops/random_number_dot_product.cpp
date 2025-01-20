//
// Created by poorna on 1/14/25.
//
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <random>

#include "../bsiCPP/bsi/BsiUnsigned.hpp"
#include "../bsiCPP/bsi/BsiSigned.hpp"
#include "../bsiCPP/bsi/BsiAttribute.hpp"
#include "../bsiCPP/bsi/hybridBitmap/hybridbitmap.h"

using std::cout;
using std::endl;

int main(){
    int range = 100;
    long length = 10000000;

    std::vector<uint8_t> vec1;
    std::vector<uint8_t> vec2;

    for(auto i=0; i<length; i++){
        vec1.push_back(std::rand()%range);
        vec2.push_back(std::rand()%range);
    }

    std::byte* vec1_start = reinterpret_cast<std::byte*>(vec1.data());
    std::byte* vec1_end = vec1_start+vec1.capacity();
    size_t vec1_memory = reinterpret_cast<uintptr_t>(vec1_end)-reinterpret_cast<uintptr_t>(vec1_start);

    std::byte* vec2_start = reinterpret_cast<std::byte*>(vec2.data());
    std::byte* vec2_end = vec2_start+vec2.capacity();
    size_t vec2_memory = reinterpret_cast<uintptr_t>(vec2_end)-reinterpret_cast<uintptr_t>(vec2_start);


//    size_t vec1_memory = sizeof(uint64_t) * vec1.capacity();
//    size_t vec2_memory = sizeof(uint64_t) * vec2.capacity();

    cout << "vec1 memory: " << vec1_memory/(1024.0 * 1024.0) << " mb" << endl;
    cout << "vec2 memory: " << vec2_memory/(1024.0 * 1024.0) << " mb" << endl;

    cout << "size of vec1: " << vec1.size() << endl;
    cout << "Size of vec2: " << vec2.size() << endl;

    /*
     * Performing dot product using c++ vectors
     */
    long res_vec = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for(auto i=0; i<length; i++){
        res_vec += vec1[i]*vec2[i];
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto vec_duration = (std::chrono::duration_cast<std::chrono::milliseconds>(end-start)).count();

    cout << "Bits used by vector values: " << sizeof(uint8_t)*8 << " bits" << endl;


    /*
     * Performing dot product using bsi data structures
     */
    std::vector<long> m_v;
    std::vector<long> n_v;
    for(auto i=0; i<vec1.size(); i++){
        m_v.push_back(static_cast<long>(vec1[i]));
        n_v.push_back(static_cast<long>(vec2[i]));
    }

    //bsi run
    BsiUnsigned<uint64_t> ubsi;
    BsiAttribute<uint64_t>* bsi_1;
    BsiAttribute<uint64_t>* bsi_2;
    bsi_1 = ubsi.buildBsiAttributeFromVector(m_v, 1);
    bsi_2 = ubsi.buildBsiAttributeFromVector(n_v, 1);
    bsi_1->setPartitionID(0);
    bsi_1->setFirstSliceFlag(true);
    bsi_1->setLastSliceFlag(true);
    bsi_2->setPartitionID(0);
    bsi_2->setFirstSliceFlag(true);
    bsi_2->setLastSliceFlag(true);

    cout << "bsi_1 size: " << bsi_1->getSizeInMemory()/(1024*1024) << " mb" << endl;
    cout << "bsi_2 size: " << bsi_2->getSizeInMemory()/(1024*1024) << " mb" << endl;


    long res_bsi = 0;
    auto bsi_start = std::chrono::high_resolution_clock::now();
    res_bsi = bsi_1->dot(bsi_2);
    auto bsi_end = std::chrono::high_resolution_clock::now();
    int bsi_duration = std::chrono::duration_cast<std::chrono::milliseconds>(bsi_end-bsi_start).count();

    cout << "C++ Vector result: " << res_vec << endl;
    cout << "bsi result: " << res_bsi << endl;


    std::cout << "C++ vector result " << res_vec << " time taken " << vec_duration << std::endl;
    std::cout << "bsi result " << res_bsi << " time taken " << bsi_duration << std::endl;

}