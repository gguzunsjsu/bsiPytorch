import torch
import bsi_ops
import pickle
import matplotlib.pyplot as plt
import time
'''m = torch.tensor([1,2,3], dtype=torch.float32)
n = torch.tensor([4,5,6], dtype=torch.float32)

print('small stuff:: bsi:', bsi_ops.dot_product(m,n), 'torch.dot:', torch.dot(m, n))
'''

#Use the tensor to check the result
m = torch.tensor([1.0,2.0,3.0,5.0])
res = bsi_ops.topKMax(m,2)
print('resnet fc layer 1 topk::: bsi:',res,'normal:',torch.topk(m,2))

#Use pickle file to check result
with open('pkl_files/resnet50_data_vectors.pkl', 'rb') as f:
    data = pickle.load(f)
print("Resnet50 data vectors loaded from the pickle file")
print(data)

# List to store topKMax for each layer
topkMax = []
# Lists to store execution times
custom_times = []
torch_times = []

# Number of runs for averaging
num_runs = 5

# Create a text file for saving the results
output_text_file = 'topKMax_results.txt'
bsi_values = []
normal_values = []
with open(output_text_file, 'w') as text_file:
    # Iterate through each layer's triplets
    for i,d in enumerate(data,1):
        # Flatten the tensors to 1D using reshape
        print("flattening vector")
        print(d)
        d_flat = d.reshape(-1)
        print("flattened vector")
        print(d)
'''
        # Print the shape of the flattened tensors
        print(f"Layer {i} - Q shape: {d_flat.shape}")

        custom_exec_times = []
        torch_exec_times = []
        for _ in range(num_runs):
            start_time = time.time()
            res = bsi_ops.topK(d_flat)
            custom_exec_time = time.time() - start_time
            custom_exec_times.append(custom_exec_time)

            start_time = time.time()
            torch_res = torch.topk(d_flat)
            torch_exec_time = time.time() - start_time
            torch_exec_times.append(torch_exec_time)
        custom_avg_time = sum(custom_exec_times) / num_runs
        torch_avg_time = sum(torch_exec_times) / num_runs

        custom_times.append(custom_avg_time)
        torch_times.append(torch_avg_time)
        bsi_values.append(res)
        normal_values.append(torch_res.detach().numpy())

        text_file.write(f"Time taken for BSI operation: {custom_avg_time}\n Time taken for torch operation: {torch_avg_time}\n\n")

#Create visualization
layer_numbers = list(range(1, 13))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(layer_numbers, custom_times, marker='o', label='Custom TopKMax')
plt.plot(layer_numbers, torch_times, marker='o', label='Torch TopKMax')
plt.xlabel('Layer Number')
plt.ylabel('Average Execution Time (seconds)')
plt.legend()
plt.title('Average Execution Time Comparison (5 Runs)')
plt.grid(True)
plt.savefig('fig/topK_time_visualization.png')
plt.show()'''