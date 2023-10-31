import torch
import bsi_ops
from extract_resnet_tensors import get_fc_weight_row_one
import pickle

print('import works')  # just to verify against import errors

# small test 
m = torch.tensor([1,2,3], dtype=torch.float32)
n = torch.tensor([4,5,6], dtype=torch.float32)

print('small stuff:: bsi:', bsi_ops.dot_product(m,n,1000), 'torch.dot:', torch.dot(m, n))

#Use the tensor from resnet extracted using the script to check the result
#m = torch.tensor(get_fc_weight_row_one())
#n = torch.tensor(get_fc_weight_row_one())
print("Shape of the two tensors: ", m.shape)
#res = bsi_ops.dot_product(m, n)
#print('resnet fc layer 1 dot product::: bsi:',res,'normal:',torch.dot(m, n))

# Specify the file path from which to load the weight pairs
pickle_file_path = 'weight_pairs.pkl'
pickle_file_path = 'normalized_weight_pairs.pkl'

# Load the weight pairs from the pickle file
with open(pickle_file_path, 'rb') as file:
    loaded_weight_pairs = pickle.load(file)
print("LSTM vectors loaded")
# Print the shapes of the loaded tensor pairs
for i, (input_weights, hidden_weights) in enumerate(loaded_weight_pairs):
    max_decimal_places = 0
    print(f"Pair {i + 1}: Input Weights Shape - {input_weights.shape}, Hidden Weights Shape - {hidden_weights.shape}")
    #An attempt to calculate a good conversion factor
    #max_value = min(torch.abs(input_weights).min(), torch.abs(hidden_weights).min())
    #max_decimal_places = max(max_decimal_places, -int(torch.floor(torch.log10(max_value))))
    # Calculate the conversion factor based on the maximum decimal places
    conversion_factor = 10.0 ** max_decimal_places
    # Try to find an optimal conversion factor
    #conversion_factor = 10000000.0
    conversion_factor = 1000.0;
    print(conversion_factor)
    res = bsi_ops.dot_product(input_weights, hidden_weights,conversion_factor)
    print('LSTM normalized input and hidden layer dot product::: bsi:',res,'normal:',torch.dot(input_weights, hidden_weights))