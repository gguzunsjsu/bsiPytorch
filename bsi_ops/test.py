import torch
import bsi_ops
from extract_resnet_tensors import get_fc_weight_row_one

print('import works')  # just to verify against import errors

# small test 
m = torch.tensor([1,2,3], dtype=torch.float32)
n = torch.tensor([4,5,6], dtype=torch.float32)

print('small stuff:: bsi:', bsi_ops.topk(m), 'torch.dot:', torch.topk(m))
print('small stuff:: bsi:', bsi_ops.topk(n), 'torch.dot:', torch.topk(n))

#Use the tensor from resnet extracted using the script to check the result
m = torch.tensor(get_fc_weight_row_one())
n = torch.tensor(get_fc_weight_row_one())
print("Shape of the two tensors: ", m.shape)
res = bsi_ops.topk(m)
print('resnet fc layer 1 topk::: bsi:',res,'normal:',torch.topk(m))
res = bsi_ops.topk(n)
print('resnet fc layer 1 topk::: bsi:',res,'normal:',torch.topk(n))