import numpy as np
import torch
# loss function and its derivative
def mse(y_true, y_pred):
    # print(y_true)
    # print(y_pred)
    return torch.mean(torch.pow(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    # print(y_true)
    # print(y_true.size())
    # print(y_pred)

    return 2*(y_pred-y_true)/y_true.size()[0]