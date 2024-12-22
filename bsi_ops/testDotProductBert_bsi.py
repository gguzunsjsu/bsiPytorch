import torch
import bsi_ops
import pickle
import matplotlib.pyplot as plt
import time
import sys
import os

print('import works')  # just to verify against import errors

# Load the triplets from the saved pickle file
# pickle_file_weights_stored_path = './hpcBERTTrainDataDotProduct/output_39882/bertVectors/bertVectors_9.pkl'
with open('/home/poorna/Desktop/RA BSI/bsi_pytorch/bsiPytorch/bsi_ops/extract_tensors/Weight_Processing/bert_imdb_pickle_store/bert_imdb0.pkl', 'rb') as f:
    triplets = pickle.load(f)
print("BERT triplets loaded from the pickle file")

# Lists to store execution times
custom_times = []

q_flat_histograms = []
k_flat_histograms = []


# Number of runs for averaging
num_runs = 5

# Create a text file for saving the results
output_text_file = ('./hpcBERTTrainDataDotProduct/results/imdb_initial/torch_32/torch_runs/bert_imdb_e0_pf31_6bit.txt')
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
        Q_flat = torch.tensor(Q_flat, dtype=torch.float32)
        K_flat = torch.tensor(K_flat, dtype=torch.float32)
        V_flat = torch.tensor(V_flat, dtype=torch.float32)

        Q_bits_used = Q_flat.element_size() * 8 # element_size() return size of an element in bytes
        K_bits_used = K_flat.element_size() * 8
        V_bits_used = V_flat.element_size() * 8
        print(f"Bits used by Q_flat {Q_bits_used}, K_flat {K_bits_used}, V_flat {V_bits_used}")

        # Store histogram data
        q_flat_histograms.append(Q_flat.detach().numpy())
        k_flat_histograms.append(K_flat.detach().numpy())


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
        precision_factor = 31
        custom_exec_times = []

        for _ in range(num_runs):
            #res, time_taken, bsiQ, bsiK = bsi_ops.dot_product(Q_flat, K_flat, precision_factor)
            res, time_taken, bsiSizeQ, bsiSizeK = bsi_ops.dot_product(Q_flat, K_flat, precision_factor) #bsi dot product
            # res, time_taken, bsiSizeQ, bsiSizeK     = bsi_ops.dot_product_without_compression(Q_flat, K_flat, precision_factor)/ #bsi dot product without compression
            custom_exec_times.append(time_taken/1e9)
            start_time = time.time()

        custom_avg_time = sum(custom_exec_times) / num_runs

        custom_times.append(custom_avg_time*1000)

        bsi_values.append(res)

        print('BERT normalized Q and K dot product::: bsi:', res)

        text_file.write(f"Layer {i} - Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}\n")
        text_file.write(f"Bits used by Q_flat: {Q_bits_used}, K_flat: {K_bits_used}, V_flat: {V_bits_used}\n")
        text_file.write(f"Q size: {Q_size} bytes\n")
        text_file.write(f"K size: {K_size} bytes\n")
        #bsiSizeQ = sys.getsizeof(bsiQ)
        #bsiSizeK = sys.getsizeof(bsiK)
        #bsiSizeK = 0
        #bsiSizeQ = 0
        text_file.write(f"BSI Q size: {bsiSizeQ} bytes\n")
        text_file.write(f"BSI K size: {bsiSizeK} bytes\n")
        text_file.write(f"BSI Q size in MB: {bsiSizeQ/(2**20)}MB\n")
        text_file.write(f"BSI K size in MB: {bsiSizeK/(2**20)}MB\n")
        dtype = Q_flat.dtype
        precision = torch.finfo(dtype).bits
        text_file.write(f"Precision of the K tensor: {precision} bits\n")
        text_file.write(f"Data Type of the K tensor: {dtype} \n")
        dtype = K_flat.dtype
        precision = torch.finfo(dtype).bits
        text_file.write(f"Precision of the K tensor: {precision} bits\n")
        text_file.write(f"Data Type of the K tensor: {dtype} \n")
        text_file.write(f"Time taken for BSI operation: {custom_avg_time}")
        text_file.write('\n')
print(f"Results saved to {output_text_file}")

#Create visualization
# layer_numbers = list(range(1, 7))
layer_numbers = list(range(1, len(triplets) + 1))

output_figures_save_folder = './hpcBERTTrainDataDotProduct/results/imdb_initial/torch_32/torch_runs/'
os.makedirs(output_figures_save_folder, exist_ok=True)

# Plot the time results
plt.figure(figsize=(12, 6))
plt.plot(layer_numbers, custom_times, marker='o', label='BSI Dot Product')
plt.xlabel('Layer Number')
plt.ylabel('Average Execution Time (milliseconds)')
plt.legend()
plt.title('Average Execution Time Comparison (5 Runs)')
plt.grid(True)
plt.savefig('./hpcBERTTrainDataDotProduct/results/imdb_initial/torch_32/torch_runs/bert_time_visualization_e0_pf31_6bit.png')
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
plt.savefig('./hpcBERTTrainDataDotProduct/results/imdb_initial/torch_32/torch_runs/bert_tensor_distribution_e0_pf31_6bit.png')
plt.show()
