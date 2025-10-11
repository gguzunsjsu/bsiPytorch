import bsi_ops
import torch

q = torch.randn(8, device='cuda')
k = torch.randn(8, device='cuda')

out, kernel_ns, build_ns, dot_ns = bsi_ops.dot_product_decimal_cuda(q, k, 2)
print('dot_product_decimal_cuda OK:', out, kernel_ns, build_ns, dot_ns)
