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
with open('extract_tensors/Weight_Processing/bert_imdb_pickle_store/bert_imdb45.pkl', 'rb') as f:
    triplets = pickle.load(f)
print("BERT triplets loaded from the pickle file")
# List to store dot products for each layer
dot_products = []
# Lists to store execution times
custom_times = []
torch_times = []
vector_dotProduct_times = []
# Lists to store histogram data of the tensors
q_flat_histograms = []
k_flat_histograms = []


# Number of runs for averaging
num_runs = 5

# Number of decimal places for BSI dot product
decimal_places = 1

# Create a text file for saving the results
output_text_file = './hpcBERTTrainDataDotProduct/results/imdb_initial/torch_32/all/bert_imdb_e0_pf31_6bit.txt'
os.makedirs('./hpcBERTTrainDataDotProduct/results/imdb_initial/torch_32/all/', exist_ok=True)
os.makedirs(os.path.dirname(output_text_file), exist_ok=True)
bsi_values = []
normal_values = []
percentage_error_values = []
layer_compression_summary = []
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

        compress_threshold = 0.2
        q_stats = bsi_ops.tensor_slice_stats(Q_flat, decimal_places, compress_threshold)
        k_stats = bsi_ops.tensor_slice_stats(K_flat, decimal_places, compress_threshold)
        v_stats = bsi_ops.tensor_slice_stats(V_flat, decimal_places, compress_threshold)

        q_summary = (q_stats['compressed_slices'], q_stats['total_slices'])
        k_summary = (k_stats['compressed_slices'], k_stats['total_slices'])
        v_summary = (v_stats['compressed_slices'], v_stats['total_slices'])
        layer_total_compressed = q_summary[0] + k_summary[0] + v_summary[0]
        layer_total_slices = q_summary[1] + k_summary[1] + v_summary[1]
        layer_total_pct = (layer_total_compressed * 100.0 / layer_total_slices) if layer_total_slices else 0.0

        print(
            f"  Q compressed: {q_stats['compressed_pct']:.2f}% "
            f"({q_summary[0]}/{q_summary[1]})"
        )
        print(
            f"  K compressed: {k_stats['compressed_pct']:.2f}% "
            f"({k_summary[0]}/{k_summary[1]})"
        )
        print(
            f"  V compressed: {v_stats['compressed_pct']:.2f}% "
            f"({v_summary[0]}/{v_summary[1]})"
        )
        print(
            f"  Layer total compressed: {layer_total_pct:.2f}% "
            f"({layer_total_compressed}/{layer_total_slices})"
        )
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
        torch_exec_times = []
        vector_exec_times = []
        for _ in range(num_runs):
            #res, time_taken, bsiQ, bsiK = bsi_ops.dot_product(Q_flat, K_flat, precision_factor)
            # res, time_taken, bsiSizeQ, bsiSizeK     = bsi_ops.dot_product(Q_flat, K_flat, precision_factor) #bsi dot product
            res, time_taken, bsiSizeQ, bsiSizeK = bsi_ops.dot_product_decimal(Q_flat, K_flat, decimal_places) #bsi dot product with decimal places
            # res, time_taken, bsiSizeQ, bsiSizeK     = bsi_ops.dot_product_without_compression(Q_flat, K_flat, precision_factor)/ #bsi dot product without compression
            custom_exec_times.append(time_taken/1e9)
            start_time = time.time()
            torch_res = torch.dot(Q_flat, K_flat) # torch dot product
            torch_exec_time = time.time() - start_time
            torch_exec_times.append(torch_exec_time)
            vector_result, vector_dot_product_timeTaken, memoryUsedVec1, memoryUsedVec2, bitsUsedVec1, bitsUsedVec2 = bsi_ops.vector_dot_product(Q_flat, K_flat, precision_factor) #c++ vector dot product
            vector_exec_times.append(vector_dot_product_timeTaken/1e9)
        custom_avg_time = sum(custom_exec_times) / num_runs
        torch_avg_time = sum(torch_exec_times) / num_runs
        vector_dot_product_avg_timeTaken = sum(vector_exec_times)/num_runs

        custom_times.append(custom_avg_time*1000)
        torch_times.append(torch_avg_time*1000)
        vector_dotProduct_times.append(vector_dot_product_avg_timeTaken*1000)
        percentage_error = (abs(res - torch_res) / torch_res) * 100
        bsi_values.append(res)
        normal_values.append(torch_res.detach().numpy())
        percentage_error_values.append(percentage_error.detach().numpy())
        print(f"Length of layer numbers {len(vector_exec_times)}")
        print('BERT normalized Q and K dot product::: bsi:', res, 'normal:',torch_res)

        text_file.write(f"Layer {i} - Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}\n")
        text_file.write(f"Bits used by Q_flat: {Q_bits_used}, K_flat: {K_bits_used}, V_flat: {V_bits_used}\n")
        text_file.write(f"Q size: {Q_size} bytes\n")
        text_file.write(f"K size: {K_size} bytes\n")
        text_file.write(f"Q size in MB: {Q_size/(1024*1024)} MB\n")
        text_file.write(f"K size in MB: {K_size/(1024*1024)} MB\n")
        text_file.write(
            f"Q compressed: {q_stats['compressed_slices']}/{q_stats['total_slices']} "
            f"({q_stats['compressed_pct']:.2f}%)\n"
        )
        text_file.write(
            f"K compressed: {k_stats['compressed_slices']}/{k_stats['total_slices']} "
            f"({k_stats['compressed_pct']:.2f}%)\n"
        )
        text_file.write(
            f"V compressed: {v_stats['compressed_slices']}/{v_stats['total_slices']} "
            f"({v_stats['compressed_pct']:.2f}%)\n"
        )
        text_file.write(
            f"Layer total compressed: {layer_total_compressed}/{layer_total_slices}"
            f" ({layer_total_pct:.2f}%)\n"
        )
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
        text_file.write(f'BERT normalized Q and K dot product::: bsi: {res}, normal: {torch_res}, '
                        f'percentage error: {percentage_error}%\n')
        text_file.write(f"Time taken for BSI operation: {custom_avg_time}\n Time taken for torch operation: {torch_avg_time}\n")
        text_file.write('\n')

        layer_compression_summary.append({
            "layer": i,
            "q": q_summary,
            "k": k_summary,
            "v": v_summary,
            "total": (layer_total_compressed, layer_total_slices)
        })
print(f"Results saved to {output_text_file}")

if layer_compression_summary:
    total_compressed = sum(entry["total"][0] for entry in layer_compression_summary)
    total_slices = sum(entry["total"][1] for entry in layer_compression_summary)
    overall_pct = (total_compressed * 100.0 / total_slices) if total_slices else 0.0
    print("\n=== Compression Summary ===")
    for entry in layer_compression_summary:
        layer = entry["layer"]
        compressed, total = entry["total"]
        pct = (compressed * 100.0 / total) if total else 0.0
        print(
            f"Layer {layer}: {compressed}/{total} ({pct:.2f}%) | "
            f"Q {entry['q'][0]}/{entry['q'][1]}, "
            f"K {entry['k'][0]}/{entry['k'][1]}, V {entry['v'][0]}/{entry['v'][1]}"
        )
    print(
        f"Overall compressed slices: {total_compressed}/{total_slices} "
        f"({overall_pct:.2f}%)"
    )
    with open(output_text_file, 'a') as summary_file:
        summary_file.write("=== Compression Summary ===\n")
        for entry in layer_compression_summary:
            layer = entry["layer"]
            compressed, total = entry["total"]
            pct = (compressed * 100.0 / total) if total else 0.0
            summary_file.write(
                f"Layer {layer}: {compressed}/{total} ({pct:.2f}%) | "
                f"Q {entry['q'][0]}/{entry['q'][1]}, "
                f"K {entry['k'][0]}/{entry['k'][1]}, V {entry['v'][0]}/{entry['v'][1]}\n"
            )
        summary_file.write(
            f"Overall compressed slices: {total_compressed}/{total_slices} "
            f"({overall_pct:.2f}%)\n\n"
        )

#Create visualization
# layer_numbers = list(range(1, 7))
layer_numbers = list(range(1, len(triplets) + 1))
# Create subplots for bsi, normal, and percentage error
fig, ax = plt.subplots(3, 1, figsize=(10, 6))

# Plot BSI values
ax[0].plot(layer_numbers, bsi_values, marker='o', linestyle='-', color='b')
ax[0].set_title('BSI Values')
ax[0].set_xlabel('Layer')
ax[0].set_ylabel('Value')

# Plot Normal values
ax[1].plot(layer_numbers, normal_values, marker='o', linestyle='-', color='g')
ax[1].set_title('Normal Values')
ax[1].set_xlabel('Layer')
ax[1].set_ylabel('Value')

# Plot Percentage Error values
ax[2].plot(layer_numbers, percentage_error_values, marker='o', linestyle='-', color='r')
ax[2].set_title('Percentage Error')
ax[2].set_xlabel('Layer')
ax[2].set_ylabel('Error (%)')

# Add a common x-axis label
fig.text(0.5, 0.04, 'Layer', ha='center')

# Adjust spacing between subplots
plt.tight_layout()

# Save the plot as an image (e.g., PNG)
plt.savefig('./hpcBERTTrainDataDotProduct/results/imdb_initial/torch_32/all/bert_visualization_e0_pf31_6bit.png')

# Show the plot (optional)
plt.show()


# Plot the time results
plt.figure(figsize=(12, 6))
plt.plot(layer_numbers, custom_times, marker='o', label='BSI Dot Product')
plt.plot(layer_numbers, torch_times, marker='o', label='Torch Dot Product')
plt.plot(layer_numbers, vector_dotProduct_times, marker='o', label='C++ vector Dot Product')
plt.xlabel('Layer Number')
plt.ylabel('Average Execution Time (milliseconds)')
plt.legend()
plt.title('Average Execution Time Comparison (5 Runs)')
plt.grid(True)
plt.savefig('./hpcBERTTrainDataDotProduct/results/imdb_initial/torch_32/all/bert_time_visualization_e0_pf31_6bit.png')
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
plt.savefig('./hpcBERTTrainDataDotProduct/results/imdb_initial/torch_32/all/bert_tensor_distribution_e0_pf31_6bit.png')
plt.show()
