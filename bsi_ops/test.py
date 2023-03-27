import torch
import bsi_ops

print('import works')  # just to verify against import errors

TENSOR_SIZE = 300
m = torch.rand(TENSOR_SIZE)
n = torch.rand(TENSOR_SIZE)
res = bsi_ops.dot_product(m, n)
print('bsi:',res,'normal:',torch.dot(m, n))
print(res == torch.dot(m, n))