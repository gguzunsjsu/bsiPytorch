import torch
import bsi_ops

print('import works')  # just to verify against import errors
m = torch.tensor([1,2,3,4], dtype=torch.int64)
n = torch.tensor([5,6,7,8], dtype=torch.int64)
res = bsi_ops.dot_product(m, n)
print(res)