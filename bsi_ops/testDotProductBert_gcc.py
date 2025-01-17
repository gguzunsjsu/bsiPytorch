import torch
import bsi_ops
import pickle
import matplotlib.pyplot as plt
import time
import sys
import os
import pandas as pd

print('import works')  # just to verify against import errors

# Load the triplets from the saved pickle file
# pickle_file_weights_stored_path = './hpcBERTTrainDataDotProduct/output_39882/bertVectors/bertVectors_9.pkl'
with open('/home/poorna/Desktop/RA BSI/bsi_pytorch/bsiPytorch/bsi_ops/extract_tensors/Weight_Processing/bert_imdb_pickle_store/bert_imdb45.pkl', 'rb') as f:
    triplets = pickle.load(f)
print("BERT triplets loaded from the pickle file")

weight_epoch_using = 45
vector_dotProduct_times = []
vector_memory_usage_data = []

# Number of runs for averaging
num_runs = 5

# Create a text file for saving the results
output_text_file = './hpcBERTTrainDataDotProduct/results/imdb_e45/vector/bert_imdb_e45_pf31_6bit.txt'
os.makedirs('./hpcBERTTrainDataDotProduct/results/imdb_e45/vector/', exist_ok=True)
os.makedirs(os.path.dirname(output_text_file), exist_ok=True)

with open(output_text_file, 'w') as text_file:
    # Iterate through each layer's triplets
    for i, triplet in enumerate(triplets, 1):
        Q, K, V = triplet
        # Flatten the tensors to 1D using reshape
        Q_flat = Q.reshape(-1)
        K_flat = K.reshape(-1)
        V_flat = V.reshape(-1)
        # Convert the NumPy array to a PyTorch tensor
        Q_flat = torch.tensor(Q_flat, dtype=torch.float32)
        K_flat = torch.tensor(K_flat, dtype=torch.float32)
        V_flat = torch.tensor(V_flat, dtype=torch.float32)

        Q_bits_used = Q_flat.element_size() * 8 # element_size() return size of an element in bytes
        K_bits_used = K_flat.element_size() * 8
        V_bits_used = V_flat.element_size() * 8
        print(f"Bits used by Q_flat {Q_bits_used}, K_flat {K_bits_used}, V_flat {V_bits_used}")


        # Print the shape and size of  of the flattened tensors
        print(f"Layer {i} - Q shape: {Q_flat.shape}, K shape: {K_flat.shape}, V shape: {V_flat.shape}")
        # Calculate the total size of each tensor in bytes using sys.getsizeof
        Q_size = sys.getsizeof(Q_flat.untyped_storage()) + sys.getsizeof(Q_flat) #storage() is being deprecated. so used untyped_storage()
        K_size = sys.getsizeof(K_flat.untyped_storage()) + sys.getsizeof(K_flat)
        V_size = sys.getsizeof(V_flat.untyped_storage()) + sys.getsizeof(V_flat)

        # Convert sizes to kilobytes (optional)
        Q_size_kb = Q_size / 1024
        K_size_kb = K_size / 1024
        V_size_kb = V_size / 1024

        precision_factor = 31

        vector_exec_times = []
        for _ in range(num_runs):
            # print(bsi_ops.vector_dot_product(Q_flat, K_flat, precision_factor))
            vector_result, vector_dot_product_timeTaken, memVec1, memVec2, bitsVec1, bitsVec2 = bsi_ops.vector_dot_product(Q_flat, K_flat, precision_factor) #c++ vector dot product
            vector_exec_times.append(vector_dot_product_timeTaken/1e9)
            print(f"Layer {i} Vector1Memory {memVec1} Vector2Memory {memVec2} Vector1bits {bitsVec1} Vector2bits {bitsVec2}")
        vector_dot_product_avg_timeTaken = sum(vector_exec_times)/num_runs

        vector_dotProduct_times.append(vector_dot_product_avg_timeTaken*1000)
    vector_memory_usage_data.append({
        "Operation": "Vector",
        "Vector1_Size": memVec1/(2**20),
        "Vector2_size": memVec2/(2**20),
        "Vector1_bits": bitsVec1,
        "Vector2_bits": bitsVec2,
        "Epoch": weight_epoch_using
    })

        # print(f"Length of layer numbers {len(vector_exec_times)}")

        # text_file.write(f"Layer {i} - Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}\n")
        # text_file.write(f"Bits used by Q_flat: {Q_bits_used}, K_flat: {K_bits_used}, V_flat: {V_bits_used}\n")
        # text_file.write(f"Q size: {Q_size} bytes\n")
        # text_file.write(f"K size: {K_size} bytes\n")
        # text_file.write(f"Q size in MB: {Q_size/(1024*1024)} MB\n")
        # text_file.write(f"K size in MB: {K_size/(1024*1024)} MB\n")
        #bsiSizeQ = sys.getsizeof(bsiQ)
        #bsiSizeK = sys.getsizeof(bsiK)
        #bsiSizeK = 0
        #bsiSizeQ = 0
        # text_file.write(f"BSI Q size: {bsiSizeQ} bytes\n")
        # text_file.write(f"BSI K size: {bsiSizeK} bytes\n")
        # text_file.write(f"BSI Q size in MB: {bsiSizeQ/(2**20)}MB\n")imdb_e45
        # text_file.write(f"BSI K size in MB: {bsiSizeK/(2**20)}MB\n")
        # dtype = Q_flat.dtype
        # precision = torch.finfo(dtype).bits
        # text_file.write(f"Precision of the K tensor: {precision} bits\n")
        # text_file.write(f"Data Type of the K tensor: {dtype} \n")
        # dtype = K_flat.dtype
        # precision = torch.finfo(dtype).bits
        # text_file.write(f"Precision of the K tensor: {precision} bits\n")
        # text_file.write(f"Data Type of the K tensor: {dtype} \n")
        # text_file.write(f'BERT normalized Q and K dot product::: bsi: {res}, normal: {torch_res}, '
        #                 f'percentage error: {percentage_error}%\n')
        # text_file.write(f"Time taken for BSI operation: {custom_avg_time}\n Time taken for torch operation: {torch_avg_time}\n")
        # text_file.write('\n')
print(f"Results saved to {output_text_file}")

output_figures_save_folder = './hpcBERTTrainDataDotProduct/results/imdb_e45/vector/'
os.makedirs(output_figures_save_folder, exist_ok=True)

memory_usage_df = pd.DataFrame(vector_memory_usage_data)
csv_file_path = './hpcBERTTrainDataDotProduct/results/vector_memory_usage/'
os.makedirs(csv_file_path, exist_ok=True)
csv_file_name = os.path.join(csv_file_path, "tensor_memory_usage.csv")

if os.path.isdir(csv_file_name):
    raise ValueError(f"Conflict: '{csv_file_name}' is a directory. Please rename or remove it.")
file_exists = os.path.isfile(csv_file_name)

memory_usage_df.to_csv(csv_file_name, mode='a', header=not file_exists, index=False)
print(f"memory usage data saved to CSV file")

#Create visualization
# layer_numbers = list(range(1, 7))
layer_numbers = list(range(1, len(triplets) + 1))

# Plot the time results
plt.figure(figsize=(12, 6))
plt.plot(layer_numbers, vector_dotProduct_times, marker='o', label='C++ vector Dot Product')
plt.xlabel('Layer Number')
plt.ylabel('Average Execution Time (milliseconds)')
plt.legend()
plt.title('Average Execution Time Comparison (5 Runs)')
plt.grid(True)
plt.savefig('./hpcBERTTrainDataDotProduct/results/imdb_e45/vector/bert_time_visualization_e45_pf31_6bit.png')
plt.show()


# Plot histograms for Q_flat and K_flat
# plt.figure(figsize=(10, 6))
#
# plt.subplot(1, 2, 1)
# plt.hist(q_flat_histograms, bins=50, alpha=0.7, label='Query Tensors')
# plt.title('Histogram for Query tensors')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.legend()
#
# plt.subplot(1, 2, 2)
# plt.hist(k_flat_histograms, bins=50, alpha=0.7, label='Key Tensors')
# plt.title('Histogram for Key tensors')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.legend()
#
# plt.tight_layout()
# plt.savefig('./hpcBERTTrainDataDotProduct/results/imdb_initial/torch_32/all/bert_tensor_distribution_e45_pf31_6bit.png')
# plt.show()
