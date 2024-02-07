import torch
import bsi_ops
import pickle
import matplotlib.pyplot as plt
import time
import numpy as np
'''m = torch.tensor([1,2,3], dtype=torch.float32)
n = torch.tensor([4,5,6], dtype=torch.float32)

print('small stuff:: bsi:', bsi_ops.dot_product(m,n), 'torch.dot:', torch.dot(m, n))
'''

#Use the tensor to check the result
'''m = torch.tensor([1.0,2.0,3.0,5.0])
res = bsi_ops.topKMax(m,2)
print('resnet fc layer 1 topk::: bsi:',res,'normal:',torch.topk(m,2))'''

#Use pickle file to check result
with open('pkl_files/resnet50_data_vectors.pkl', 'rb') as f:
    data = pickle.load(f)
'''print("Resnet50 data vectors loaded from the pickle file")
print(data)'''

# Lists to store execution times
bsi_topkmax_times = []
bsi_topkmin_times = []
torch_topkmax_times = []
torch_topkmin_times = []

# Number of runs for averaging
num_runs = 5

# Create a text file for saving the results
output_text_file = 'topKMax_results.txt'
with open(output_text_file, 'w') as text_file:
    # Iterate through each layer's triplets
    #count = 2
    for i,d in enumerate(data,1):
        #if count != 1:
        #    count -= 1
        #    continue
        # Flatten the tensors to 1D using reshape
        d = torch.tensor(d)
        '''print("flattening vector")
        print(d)'''
        d_flat = d.flatten()
        '''print("flattened vector")
        print(d_flat)'''

        # Print the shape of the flattened tensors
        print(f"Layer {i} - Q shape: {d_flat.shape}")

        bsi_topkmax_times = []
        bsi_topkmin_times = []
        torch_topkmax_times = []
        torch_topkmin_times = []
        for _ in range(num_runs):
            start_time = time.time()
            res = bsi_ops.topKMax(d_flat, 100)
            custom_exec_time = time.time() - start_time
            bsi_topkmax_times.append(custom_exec_time)

            start_time = time.time()
            torch_res = torch.topk(d_flat, 100)
            torch_exec_time = time.time() - start_time
            torch_topkmax_times.append(torch_exec_time)

            start_time = time.time()
            res = bsi_ops.topKMin(d_flat, 100)
            custom_exec_time = time.time() - start_time
            bsi_topkmax_times.append(custom_exec_time)

            start_time = time.time()
            torch_res = torch.topk(d_flat, 100, largest=False)
            torch_exec_time = time.time() - start_time
            torch_topkmin_times.append(torch_exec_time)

            # write testcase to file
            input_vector = bsi_ops.convertTensor(d_flat, 10000)
            string = ""
            for i in range(len(input_vector)):
                string += str(input_vector[i].item())+"\n"
            with open("testcase.txt","w") as t:
                t.write(string)
        '''custom_avg_time = sum(custom_exec_times) / num_runs'''
        torch_topkmax_avg_time = sum(torch_topkmax_times) / num_runs
        torch_topkmin_avg_time = sum(torch_topkmin_times) / num_runs
        bsi_topkmax_avg_time = sum(bsi_topkmax_times) / num_runs
        bsi_topkmin_avg_time = sum(bsi_topkmin_times) / num_runs

        '''custom_times.append(custom_avg_time)
        torch_times.append(torch_avg_time)'''

        text_file.write(f"Time taken for torch topkmax operation: {torch_topkmax_avg_time*10**6}\n Time taken for torch topkmin operation: {torch_topkmin_avg_time*10**6}\n\n")
        text_file.write(f"Time taken for bsi topkmax operation: {bsi_topkmax_avg_time*10**6}\n Time taken for bsi topkmin operation: {bsi_topkmin_avg_time*10**6}\n\n")
        #break
'''
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