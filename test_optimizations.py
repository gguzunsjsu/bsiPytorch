#!/usr/bin/env python3
"""
Quick test script to compare old vs new optimized BSI CUDA kernels.
Run this after building to verify optimizations work correctly.
"""

import os
import sys
import time
import torch
import numpy as np

# Import bsi_ops
try:
    import bsi_ops
    print(f"✓ bsi_ops imported successfully")
    print(f"  Version: {bsi_ops.cuda_builder_version()}")
except ImportError as e:
    print(f"✗ Failed to import bsi_ops: {e}")
    print("  Make sure you've built the extension with: python setup.py build_ext --inplace")
    sys.exit(1)

def test_dot_product_correctness():
    """Test that old and new kernels produce identical results."""
    print("\n" + "="*80)
    print("TEST 1: Dot Product Correctness")
    print("="*80)

    torch.manual_seed(42)
    d = 1024
    q = torch.randn(d, device='cuda')
    k = torch.randn(d, device='cuda')

    # Test with old kernel
    os.environ['BSI_OPTIMIZED'] = '0'
    result_old, time_old, _, _ = bsi_ops.dot_product_decimal_cuda(q, k, 2)

    # Test with new kernel
    os.environ['BSI_OPTIMIZED'] = '1'
    result_new, time_new, _, _ = bsi_ops.dot_product_decimal_cuda(q, k, 2)

    # Compare
    diff = abs(result_new - result_old)
    relative_error = diff / (abs(result_old) + 1e-10)

    print(f"\nOld kernel result: {result_old:.6f} (time: {time_old/1e6:.2f} ms)")
    print(f"New kernel result: {result_new:.6f} (time: {time_new/1e6:.2f} ms)")
    print(f"Absolute difference: {diff:.2e}")
    print(f"Relative error: {relative_error:.2e}")

    if relative_error < 1e-4:
        print("✓ PASS: Results match!")
        speedup = time_old / max(time_new, 1)
        print(f"  Speedup: {speedup:.1f}x")
        return True
    else:
        print("✗ FAIL: Results differ significantly!")
        return False

def test_batch_performance():
    """Test batched query building and dot products."""
    print("\n" + "="*80)
    print("TEST 2: Batch Performance")
    print("="*80)

    torch.manual_seed(42)
    d = 1024
    num_keys = 128
    batch_size = 16

    # Build keys
    print(f"\nBuilding {num_keys} keys (dimension={d})...")
    K = torch.randn(num_keys, d, device='cuda', dtype=torch.float32)
    keys_cap, mem, nk, kd, W = bsi_ops.build_bsi_keys_cuda(K, 2, 0.2)
    print(f"✓ Keys built: {nk} keys, dimension={kd}, words={W}, memory={mem/1e6:.2f}MB")

    # Test old approach (individual query building)
    print(f"\n--- Old Approach (Individual Queries) ---")
    os.environ['BSI_OPTIMIZED'] = '0'

    queries = torch.randn(batch_size, d, device='cuda', dtype=torch.float32)

    start = time.perf_counter()
    build_time_old = 0
    dot_time_old = 0

    for i in range(batch_size):
        q = queries[i]
        # Build query
        t0 = time.perf_counter()
        q_cap, q_mem, q_meta, _ = bsi_ops.build_bsi_query_cuda(q, 2, 0.2)
        build_time_old += (time.perf_counter() - t0)

        # Dot product
        t0 = time.perf_counter()
        scores, total_ns, _, dot_ns, _ = bsi_ops.batch_dot_product_prebuilt_cuda_caps(q_cap, keys_cap)
        dot_time_old += (time.perf_counter() - t0)

    total_old = time.perf_counter() - start

    print(f"Build time: {build_time_old*1000:.2f} ms ({build_time_old*1000/batch_size:.2f} ms/query)")
    print(f"Dot time:   {dot_time_old*1000:.2f} ms ({dot_time_old*1000/batch_size:.2f} ms/query)")
    print(f"Total time: {total_old*1000:.2f} ms")

    # Test new approach (optimized kernels)
    print(f"\n--- New Approach (Optimized Kernels) ---")
    os.environ['BSI_OPTIMIZED'] = '1'

    start = time.perf_counter()
    build_time_new = 0
    dot_time_new = 0

    for i in range(batch_size):
        q = queries[i]
        # Build query
        t0 = time.perf_counter()
        q_cap, q_mem, q_meta, _ = bsi_ops.build_bsi_query_cuda(q, 2, 0.2)
        build_time_new += (time.perf_counter() - t0)

        # Dot product
        t0 = time.perf_counter()
        scores_new, total_ns, _, dot_ns, _ = bsi_ops.batch_dot_product_prebuilt_cuda_caps(q_cap, keys_cap)
        dot_time_new += (time.perf_counter() - t0)

    total_new = time.perf_counter() - start

    print(f"Build time: {build_time_new*1000:.2f} ms ({build_time_new*1000/batch_size:.2f} ms/query)")
    print(f"Dot time:   {dot_time_new*1000:.2f} ms ({dot_time_new*1000/batch_size:.2f} ms/query)")
    print(f"Total time: {total_new*1000:.2f} ms")

    # Summary
    print(f"\n--- Speedup Summary ---")
    build_speedup = build_time_old / max(build_time_new, 1e-9)
    dot_speedup = dot_time_old / max(dot_time_new, 1e-9)
    total_speedup = total_old / max(total_new, 1e-9)

    print(f"Build speedup: {build_speedup:.2f}x")
    print(f"Dot speedup:   {dot_speedup:.2f}x")
    print(f"Total speedup: {total_speedup:.2f}x")

    if total_speedup > 5.0:
        print("✓ PASS: Significant speedup achieved!")
        return True
    elif total_speedup > 1.2:
        print("⚠ WARN: Some speedup but less than expected")
        return True
    else:
        print("✗ FAIL: No significant speedup")
        return False

def test_gpu_utilization():
    """Check GPU memory and compute utilization."""
    print("\n" + "="*80)
    print("TEST 3: GPU Utilization Check")
    print("="*80)

    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu',
                               '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, check=True)
        print("\nGPU Status:")
        for line in result.stdout.strip().split('\n'):
            parts = line.split(', ')
            if len(parts) >= 4:
                print(f"  GPU: {parts[0]}")
                print(f"  Memory: {parts[2]} / {parts[1]} MB")
                print(f"  Utilization: {parts[3]}%")
    except Exception as e:
        print(f"Could not query GPU status: {e}")

    print("\n✓ If you see this, CUDA is working")
    return True

def main():
    print("BSI CUDA Optimizations - Quick Test Suite")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("✗ ERROR: CUDA not available!")
        print("  Make sure you're running on a GPU node.")
        sys.exit(1)

    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA version: {torch.version.cuda}")

    # Run tests
    results = []

    results.append(("Correctness", test_dot_product_correctness()))
    results.append(("Performance", test_batch_performance()))
    results.append(("GPU Check", test_gpu_utilization()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n🎉 All tests passed! Optimizations are working correctly.")
        print("\nNext steps:")
        print("1. Run full benchmark: python benchmarks/benchmark_performance_bsi.py ...")
        print("2. Compare against baseline (with BSI_OPTIMIZED=0)")
        print("3. Check OPTIMIZATION_TESTING.md for detailed instructions")
        return 0
    else:
        print("\n⚠ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
