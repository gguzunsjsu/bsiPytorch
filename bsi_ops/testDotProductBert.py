import torch
import bsi_ops
import pickle
import matplotlib.pyplot as plt
import time
print('import works')  # just to verify against import errors

# Load the triplets from the saved pickle file
with open('extract_tensors/bert_triplets.pkl', 'rb') as f:
    triplets = pickle.load(f)
print("BERT triplets loaded from the pickle file")
# List to store dot products for each layer
dot_products = []
# Lists to store execution times
custom_times = []
torch_times = []
# Lists to store histogram data of the tensors
q_flat_histograms = []
k_flat_histograms = []


# Number of runs for averaging
num_runs = 5

# Create a text file for saving the results
output_text_file = 'extract_tensors/dot_product_results.txt'
bsi_values = []
normal_values = []
percentage_error_values = []
with open(output_text_file, 'w') as text_file:
    # Iterate through each layer's triplets
    for i, triplet in enumerate(triplets, 1):
        Q, K, V = triplet
        # Flatten the tensors to 1D using reshape
        Q_flat = Q.reshape(-1)
        K_flat = K.reshape(-1)
        V_flat = V.reshape(-1)
        # Store histogram data
        q_flat_histograms.append(Q_flat.detach().numpy())
        k_flat_histograms.append(K_flat.detach().numpy())
        # Print the shape of the flattened tensors
        print(f"Layer {i} - Q shape: {Q_flat.shape}, K shape: {K_flat.shape}, V shape: {V_flat.shape}")
        conversion_factor = 1000.0;

        custom_exec_times = []
        torch_exec_times = []
        for _ in range(num_runs):
            res, time_taken = bsi_ops.dot_product(Q_flat, K_flat, conversion_factor)
            custom_exec_times.append(time_taken/1e9)
            start_time = time.time()
            torch_res = torch.dot(Q_flat, K_flat)
            torch_exec_time = time.time() - start_time
            torch_exec_times.append(torch_exec_time)
        custom_avg_time = sum(custom_exec_times) / num_runs
        torch_avg_time = sum(torch_exec_times) / num_runs

        custom_times.append(custom_avg_time)
        torch_times.append(torch_avg_time)
        percentage_error = (abs(res - torch_res) / res) * 100
        bsi_values.append(res)
        normal_values.append(torch_res.detach().numpy())
        percentage_error_values.append(percentage_error.detach().numpy())
        print('BERT normalized Q and K dot product::: bsi:', res, 'normal:',torch_res)

        text_file.write(f"Layer {i} - Q shape: {Q.shape}, K shape: {K.shape}, V shape: {V.shape}\n")
        text_file.write(f'BERT normalized Q and K dot product::: bsi: {res}, normal: {torch_res}, '
                            f'percentage error: {percentage_error}%\n')
        text_file.write(f"Time taken for BSI operation: {custom_avg_time}\n Time taken for torch operation: {torch_avg_time}\n")
        text_file.write('\n')
print(f"Results saved to {output_text_file}")

#Create visualization
layer_numbers = list(range(1, 13))
# Create subplots for bsi, normal, and percentage error
fig, ax = plt.subplots(3, 1, figsize=(10, 12))

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
plt.savefig('extract_tensors/bert_visualization.png')

# Show the plot (optional)
plt.show()


# Plot the time results
plt.figure(figsize=(10, 6))
plt.plot(layer_numbers, custom_times, marker='o', label='Custom Dot Product')
plt.plot(layer_numbers, torch_times, marker='o', label='Torch Dot Product')
plt.xlabel('Layer Number')
plt.ylabel('Average Execution Time (seconds)')
plt.legend()
plt.title('Average Execution Time Comparison (5 Runs)')
plt.grid(True)
plt.savefig('extract_tensors/bert_time_visualization.png')
plt.show()
# Plot histograms for Q_flat and K_flat
plt.figure(figsize=(12, 6))

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
plt.savefig('extract_tensors/bert_tensor_distribution.png')
plt.show()
