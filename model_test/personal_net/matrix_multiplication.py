import torch
import bsi_ops

class InvalidDimensionsException(Exception):
    "Raise when dimensions of matrices are not suitable for matrix multiplication"
    pass

def matrix_multiplication(A, B, dot_product_function = torch.dot):
    A_shape = A.shape
    B_shape = B.shape

    if len(A_shape) != 2 or len(B_shape) != 2 or A_shape[1] != B_shape[0] or 0 in A_shape or 0 in B_shape:
        print('Issues with dimensions')
        raise InvalidDimensionsException
    
    result = torch.zeros((A_shape[0], B_shape[1]), dtype=torch.float32)
    for row in range(A_shape[0]):
        row_ten = A[row, :]
        for col in range(B_shape[1]):
            col_ten = B[:, col]
            result[row, col] = dot_product_function(row_ten, col_ten)
    # result = result.astype(torch.float32)
    return result
