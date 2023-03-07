#include "../bsiCopy/BsiAttribute.cpp"
#include "../bsiCopy/BsiUnsigned.cpp"
#include "../bsiCopy/BsiSigned.cpp"
#include <torch/extension.h>

long dot_product(torch::Tensor m, torch::Tensor n) {
    BsiUnsigned<uint64_t> ubsi;
    BsiAttribute<uint64_t>* bsi_1;
    BsiAttribute<uint64_t>* bsi_2;

    bsi_1 = ubsi.buildBsiAttributeFromTensor(m, 1);
    bsi_1->setPartitionID(0);
    bsi_1->setFirstSliceFlag(true);
    bsi_1->setLastSliceFlag(true);
    bsi_2 = ubsi.buildBsiAttributeFromTensor(n, 1);
    bsi_2->setPartitionID(0);
    bsi_2->setFirstSliceFlag(true);
    bsi_2->setLastSliceFlag(true); 

    BsiAttribute<uint64_t>* bsi_res;
    bsi_res = bsi_1->multiplication(bsi_2);

    return bsi_res->sumOfBsi();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dot_product", &dot_product, "Dot product using BSI (Non-CUDA)");
}


