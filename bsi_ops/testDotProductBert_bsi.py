import math

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
# print("BERT triplets loaded from the pickle file")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Available device is {device}")

# Lists to store execution times
custom_times = []
bsi_memory_usage_data = []

q_flat_histograms = []
k_flat_histograms = []


# Number of runs for averaging
num_runs = 5

# Create a text file for saving the results
output_text_file = ('./hpcBERTTrainDataDotProduct/results/imdb_e30/bsi_runs/bert_imdb_e45_pf31_6bit.txt')
weight_epoch_using = 45
os.makedirs(os.path.dirname(output_text_file), exist_ok=True)
bsi_values = []

with open(output_text_file, 'w') as text_file:
    # Iterate through each layer's triplets
    for i, triplet in enumerate(triplets, 1):
        Q, K, V = triplet
        # Flatten the tensors to 1D using reshape
        Q_flat = Q.reshape(-1)
        K_flat = K.reshape(-1)
        V_flat = V.reshape(-1)
        # Convert the NumPy array to a PyTorch tensor
        Q_flat = torch.tensor(Q_flat, dtype=torch.float32, device=device)
        K_flat = torch.tensor(K_flat, dtype=torch.float32, device=device)
        V_flat = torch.tensor(V_flat, dtype=torch.float32, device=device)

        Q_bits_used = Q_flat.element_size() * 8 # element_size() return size of an element in bytes
        K_bits_used = K_flat.element_size() * 8
        V_bits_used = V_flat.element_size() * 8
        print(f"Bits used by Q_flat {Q_bits_used}, K_flat {K_bits_used}, V_flat {V_bits_used}")

        # Store histogram data
        q_flat_histograms.append(Q_flat.detach().cpu().numpy())
        k_flat_histograms.append(K_flat.detach().cpu().numpy())


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

        # precision_factor = 38; #changed name from conversion_factor to precision_factor. Changed value to 10^31 -- Initially it is 31 -> 6bits
        precision_factor = (2**31)-1
        bits_using = math.log2((precision_factor+1))+1
        # print(f"Bits using by the bsi is {bits_using}")
        custom_exec_times = []

        for _ in range(num_runs):
            #res, time_taken, bsiQ, bsiK = bsi_ops.dot_product(Q_flat, K_flat, precision_factor)
            res, time_taken, bsiSizeQ, bsiSizeK = bsi_ops.dot_product(Q_flat, K_flat, precision_factor) #bsi dot product
            # print(f"Layer {i} bsiSizeQ {bsiSizeQ} bsiSizeK {bsiSizeK}")
            # res, time_taken, bsiSizeQ, bsiSizeK     = bsi_ops.dot_product_without_compression(Q_flat, K_flat, precision_factor)/ #bsi dot product without compression
            custom_exec_times.append(time_taken/1e9)
            # start_time = time.time()

        custom_avg_time = sum(custom_exec_times) / num_runs

        custom_times.append(custom_avg_time*1000)

        bsi_values.append(res)

        print('BERT normalized Q and K dot product::: bsi:', res)
        bsi_Q_size_mb = bsiSizeQ/(2**20)
        bsi_k_size_mb = bsiSizeK/(2**20)
        # print(f"Q size in mb: {bsi_Q_size_mb}")
        # print(f"K size in mb: {bsi_k_size_mb}")

        bsi_memory_usage_data.append({
            "Operation": "bsi",
            # "Layer" : i,
            # "Run": run,
            "Q size in MB": bsi_Q_size_mb,
            "K size in MB": bsi_k_size_mb,
            "Q Bits used": bits_using,
            "K bits used": bits_using,
            "epoch" : weight_epoch_using
        })

        print(f"layer {i} bsiQsize {bsi_Q_size_mb} bsiKsize {bsi_k_size_mb}")

        # text_file.write(f"Layer {i} - Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}\n")
        # text_file.write(f"Bits used by Q_flat: {Q_bits_used}, K_flat: {K_bits_used}, V_flat: {V_bits_used}\n")
        # text_file.write(f"Q size: {Q_size} bytes\n")
        # text_file.write(f"K size: {K_size} bytes\n")
        #bsiSizeQ = sys.getsizeof(bsiQ)
        #bsiSizeK = sys.getsizeof(bsiK)
        #bsiSizeK = 0
        #bsiSizeQ = 0
        # text_file.write(f"BSI Q size: {bsiSizeQ} bytes\n")
        # text_file.write(f"BSI K size: {bsiSizeK} bytes\n")
        # text_file.write(f"BSI Q size in MB: {bsiSizeQ/(2**20)}MB\n")
        # text_file.write(f"BSI K size in MB: {bsiSizeK/(2**20)}MB\n")
        # dtype = Q_flat.dtype
        # precision = torch.finfo(dtype).bits
        # text_file.write(f"Precision of the K tensor: {precision} bits\n")
        # text_file.write(f"Data Type of the K tensor: {dtype} \n")
        # dtype = K_flat.dtype
        # precision = torch.finfo(dtype).bits
        # text_file.write(f"Precision of the K tensor: {precision} bits\n")
        # text_file.write(f"Data Type of the K tensor: {dtype} \n")
        # text_file.write(f"Time taken for BSI operation: {custom_avg_time}")
        # text_file.write('\n')
# print(f"Results saved to {output_text_file}")

#Create visualization
# layer_numbers = list(range(1, 7))
layer_numbers = list(range(1, len(triplets) + 1))

output_figures_save_folder = './hpcBERTTrainDataDotProduct/results/imdb_e45/bsi_runs/'
os.makedirs(output_figures_save_folder, exist_ok=True)

# memory_usage_df = pd.DataFrame(bsi_memory_usage_data)
# csv_file_path = './hpcBERTTrainDataDotProduct/results/bsi_memory_usage/'
# os.makedirs(csv_file_path, exist_ok=True)
# csv_file_name = os.path.join(csv_file_path, "bsi_memory_usage.csv")
#
# if os.path.isdir(csv_file_name):
#     raise ValueError(f"Conflict: '{csv_file_name}' is a directory. Please rename or remove it.")
# file_exists = os.path.isfile(csv_file_name)
#
# memory_usage_df.to_csv(csv_file_name, mode='a', header=not file_exists, index=False)
# print(f"memory usage data saved to CSV file")

# Plot the time results
plt.figure(figsize=(12, 6))
plt.plot(layer_numbers, custom_times, marker='o', label='BSI Dot Product')
plt.xlabel('Layer Number')
plt.ylabel('Average Execution Time (milliseconds)')
plt.legend()
plt.title('Average Execution Time Comparison (5 Runs)')
plt.grid(True)
plt.savefig('./hpcBERTTrainDataDotProduct/results/imdb_e45/bsi_runs/bert_time_visualization_e45_pf31_6bit.png')
plt.show()


# Plot histograms for Q_flat and K_flat
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.hist(q_flat_histograms, bins=50, alpha=0.7, label='Query Tensors')
plt.title('Histogram for Query tensors')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(k_flat_histograms, bins=50, alpha=0.7, label='Key Tensors')
plt.title('Histogram for Key tensors')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.savefig('./hpcBERTTrainDataDotProduct/results/imdb_e45/bsi_runs/bert_tensor_distribution_e45_pf31_6bit.png')
plt.show()
