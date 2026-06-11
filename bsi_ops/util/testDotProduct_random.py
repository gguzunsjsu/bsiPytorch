import torch
import bsi_ops
import random
import pandas as pd

#checking import working
print("Import works")

random.seed(42)

num_numbers = 10_000_000

vec1 = torch.randint(0, 100, (num_numbers, ), dtype=torch.float32)
vec2 = torch.randint(0, 100, (num_numbers, ), dtype=torch.float32)

# print(vec1[:50])
decimal_places = 2
#bsi vector product
result_bsi, timeTaken_bsi, qbsisize, kbsisize = bsi_ops.dot_product_decimal(vec1, vec2, decimal_places)

print(f"q bsi size {qbsisize/(1024*1024)}")
print(f"k bsi size {kbsisize/(1024*1024)}")
#c++ vector product
# result_cPlus, timeTaken_cPlus = bsi_ops.random_number_dot_product_vector(vec1, vec2)

print(f"BSI result: result-{result_bsi} timeTaken-{timeTaken_bsi}")
# print(f"C++ result: result-{result_cPlus} timeTaken-{timeTaken_cPlus}")
