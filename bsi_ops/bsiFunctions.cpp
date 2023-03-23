#include "../bsiCPP/bsi/BsiAttribute.hpp"
#include "../bsiCPP/bsi/BsiUnsigned.hpp"
#include "../bsiCPP/bsi/BsiSigned.hpp"
#include <torch/extension.h>

#include <vector>
// #include <iostream>


long dot_product(torch::Tensor m, torch::Tensor n) {
    std::vector<long> m_v(m.data_ptr<long>(), m.data_ptr<long>() + m.numel());
    std::vector<long> n_v(n.data_ptr<long>(), n.data_ptr<long>() + n.numel());

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

    BsiAttribute<uint64_t>* res = bsi_1->multiplyBSI(bsi_2);
    return res->sumOfBsi();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
   m.def("dot_product", &dot_product, "Dot product using BSI (Non-CUDA)");
}


