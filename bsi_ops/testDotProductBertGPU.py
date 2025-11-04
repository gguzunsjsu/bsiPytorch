import torch
import bsi_ops
import pickle
import time
import sys
import os

print('Import works - testing BSI GPU accuracy')

# Load the triplets from the saved pickle file
pickle_file = 'extract_tensors/Weight_Processing/bert_imdb_pickle_store/bert_imdb45.pkl'
with open(pickle_file, 'rb') as f:
    triplets = pickle.load(f)
print(f"BERT triplets loaded from {pickle_file}")

# Test parameters
decimal_places = 2
compress_threshold = 0.2
num_runs = 5

# Create output file
output_file = './testResults/bert_gpu_accuracy_test.txt'
os.makedirs('./testResults/', exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    sys.exit(1)

with open(output_file, 'w') as f:
    f.write(f"BSI GPU Accuracy Test\n")
    f.write(f"Decimal places: {decimal_places}\n")
    f.write(f"Compress threshold: {compress_threshold}\n")
    f.write(f"Device: {device}\n")
    f.write("="*80 + "\n\n")

    # Iterate through each layer's triplets
    for i, triplet in enumerate(triplets, 1):
        Q, K, V = triplet

        # Flatten to 1D
        Q_flat = torch.tensor(Q.reshape(-1), dtype=torch.float32, device=device)
        K_flat = torch.tensor(K.reshape(-1), dtype=torch.float32, device=device)

        print(f"\nLayer {i} - Q shape: {Q_flat.shape}, K shape: {K_flat.shape}")
        f.write(f"Layer {i}\n")
        f.write(f"Q shape: {Q_flat.shape}, K shape: {K_flat.shape}\n")

        # Ground truth: PyTorch dot product
        torch_result = torch.dot(Q_flat, K_flat).item()

        # BUILD ONCE (outside timing)
        print("  Building BSI structures (one-time)...")
        cpu_compat_result, dot_ns, q_mem, k_mem = bsi_ops.dot_product_decimal_cuda(
            Q_flat, K_flat, decimal_places
        )

        q_capsule, q_mem_fused, q_slices, q_words = bsi_ops.build_bsi_query_cuda(
            Q_flat, decimal_places, compress_threshold
        )
        k_capsule, k_mem_fused, k_num_keys, k_dim, k_words = bsi_ops.build_bsi_keys_cuda(
            K_flat.unsqueeze(0),
            decimal_places,
            compress_threshold
        )

        # NOW TIME ONLY THE DOT KERNEL
        cpu_compat_times = []
        two_stage_times = []
        fused_times = []
        torch_times = []

        for run in range(num_runs):
            # Method 1: GPU popcount-only - ONLY popcount kernel time (from CUDA events)
            cpu_compat_result, dot_ns, _, _ = bsi_ops.dot_product_decimal_cuda(
                Q_flat, K_flat, decimal_places
            )
            cpu_compat_times.append(dot_ns / 1e6)  # ns to ms

            # Method 2: GPU two-stage (NEW) - ONLY kernel time (from CUDA events)
            result_tensor, dot_ns_two_stage = bsi_ops.batch_dot_product_two_stage_cuda_caps(
                [q_capsule], k_capsule
            )
            two_stage_result = result_tensor[0, 0].item()
            two_stage_times.append(dot_ns_two_stage / 1e6)  # ns to ms

            # Method 3: GPU fused kernel (OLD, used in stable) - ONLY kernel time
            result_tensor_fused, dot_ns_fused = bsi_ops.batch_dot_product_multiquery_cuda_caps(
                [q_capsule], k_capsule
            )
            fused_result = result_tensor_fused[0, 0].item()
            fused_times.append(dot_ns_fused / 1e6)  # ns to ms

            # Method 4: PyTorch baseline
            start = time.perf_counter()
            torch_res = torch.dot(Q_flat, K_flat).item()
            torch_times.append((time.perf_counter() - start) * 1000)

        # Calculate statistics
        cpu_compat_avg_time = sum(cpu_compat_times) / num_runs
        two_stage_avg_time = sum(two_stage_times) / num_runs
        fused_avg_time = sum(fused_times) / num_runs
        torch_avg_time = sum(torch_times) / num_runs

        cpu_compat_abs_error = abs(cpu_compat_result - torch_result)
        cpu_compat_rel_error = (cpu_compat_abs_error / abs(torch_result)) * 100 if torch_result != 0 else 0

        two_stage_abs_error = abs(two_stage_result - torch_result)
        two_stage_rel_error = (two_stage_abs_error / abs(torch_result)) * 100 if torch_result != 0 else 0

        fused_abs_error = abs(fused_result - torch_result)
        fused_rel_error = (fused_abs_error / abs(torch_result)) * 100 if torch_result != 0 else 0

        # Print results
        print(f"  Torch result:        {torch_result:.10f}")
        print(f"  Popcount-only:       {cpu_compat_result:.10f}  (error: {cpu_compat_rel_error:.4f}%, time: {cpu_compat_avg_time:.4f}ms)")
        print(f"  Two-stage (NEW):     {two_stage_result:.10f}  (error: {two_stage_rel_error:.4f}%, time: {two_stage_avg_time:.4f}ms)")
        print(f"  Fused (STABLE):      {fused_result:.10f}  (error: {fused_rel_error:.4f}%, time: {fused_avg_time:.4f}ms)")
        print(f"  Torch time:          {torch_avg_time:.4f} ms")

        # Write to file
        f.write(f"  Torch result (ground truth): {torch_result:.10f}\n\n")
        f.write(f"  Method 1 - Popcount-only:\n")
        f.write(f"    Result:          {cpu_compat_result:.10f}\n")
        f.write(f"    Absolute error:  {cpu_compat_abs_error:.6e}\n")
        f.write(f"    Relative error:  {cpu_compat_rel_error:.6f}%\n")
        f.write(f"    Avg time:        {cpu_compat_avg_time:.4f} ms\n\n")

        f.write(f"  Method 2 - Two-Stage (NEW):\n")
        f.write(f"    Result:          {two_stage_result:.10f}\n")
        f.write(f"    Absolute error:  {two_stage_abs_error:.6e}\n")
        f.write(f"    Relative error:  {two_stage_rel_error:.6f}%\n")
        f.write(f"    Avg time:        {two_stage_avg_time:.4f} ms\n\n")

        f.write(f"  Method 3 - Fused (STABLE):\n")
        f.write(f"    Result:          {fused_result:.10f}\n")
        f.write(f"    Absolute error:  {fused_abs_error:.6e}\n")
        f.write(f"    Relative error:  {fused_rel_error:.6f}%\n")
        f.write(f"    Avg time:        {fused_avg_time:.4f} ms\n\n")

        f.write(f"  PyTorch baseline time: {torch_avg_time:.4f} ms\n")
        f.write(f"  Query memory: {q_mem} bytes\n")
        f.write(f"  Keys memory:  {k_mem} bytes\n")
        f.write("-"*80 + "\n\n")

        # Alert if error is high
        if cpu_compat_rel_error > 1.0 or two_stage_rel_error > 1.0 or fused_rel_error > 1.0:
            print(f"  WARNING: High error on layer {i}!")
            if cpu_compat_rel_error > 1.0:
                f.write(f"  WARNING: Popcount-only method has high error!\n")
            if two_stage_rel_error > 1.0:
                f.write(f"  WARNING: Two-stage method has high error!\n")
            if fused_rel_error > 1.0:
                f.write(f"  WARNING: Fused method has high error!\n")
            f.write("\n")

print(f"\nResults saved to {output_file}")
