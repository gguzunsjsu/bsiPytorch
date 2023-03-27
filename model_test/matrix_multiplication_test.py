import torch
import bsi_ops
from personal_net.matrix_multiplication import matrix_multiplication

m = torch.tensor(
    [[-1,-2,-3],
     [4,5,6]],
     dtype = torch.float32
)

n = torch.tensor(
    [[1, -2],
     [3, -4],
     [5, -6]],
     dtype = torch.float32
)

print('res: ', matrix_multiplication(m, n, bsi_ops.dot_product))