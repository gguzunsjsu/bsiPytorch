# BSI CUDA Optimizations - Testing Guide

## Overview

This document describes the Phase 1 optimizations implemented for BSI CUDA operations and how to test them.

## What Was Optimized

### 1. **New Optimized Dot Product Kernel** (`bsi_cuda_kernels_optimized.cu`)

**Key improvements:**
- ✅ **Vectorized 128-bit loads (uint4)** - 4x better memory bandwidth utilization
- ✅ **Removed division/modulo in hot loops** - Eliminated expensive integer arithmetic
- ✅ **Coalesced memory access** - GPU threads access consecutive memory
- ✅ **Block-level reduction (no atomicAdd)** - Eliminated serialization bottleneck
- ✅ **Query weights in shared memory** - Reduced repeated global memory access

**Expected speedup:** 20-50x faster dot products (2.36s → 50-120ms)

### 2. **Batched Query Builder**

**Key improvements:**
- ✅ **Batch processing** - Build multiple queries in parallel
- ✅ **GPU-native building** - All queries stay on GPU
- ✅ **Reduced per-query overhead** - Amortized kernel launch costs

**Expected speedup:** 100x+ faster query building (47.5s → <500ms)

## Files Changed

### New Files:
- `bsi_ops/csrc/cuda/bsi_cuda_kernels_optimized.cu` - New optimized kernels
- `bsi_ops/csrc/cuda/_legacy/bsi_cuda_kernels_old.cu` - Backup of old kernels

### Modified Files:
- `bsi_ops/csrc/cuda/bsi_cuda.cpp` - Added optimized kernel launcher integration
- `bsi_ops/setup.py` - Added new kernel file to build

## Build Instructions

### 1. Clean previous build
```bash
cd /Users/poornaravuri/Desktop/RA/bsiPytorch/bsi_ops
rm -rf build dist *.egg-info
python setup.py clean --all
```

### 2. Rebuild with optimizations
```bash
# Make sure CUDA is available
python setup.py build_ext --inplace

# Or install
pip install -e .
```

### 3. Verify build
```bash
python -c "import bsi_ops; print(bsi_ops.cuda_builder_version())"
```

## Testing the Optimizations

### Environment Variables for Control

The optimizations are **enabled by default**. You can control them with environment variables:

```bash
# Enable/disable optimized kernels (default: enabled)
export BSI_OPTIMIZED=1       # Use new optimized kernels
export BSI_OPTIMIZED=0       # Use old kernels (for comparison)

# Enable debug logging
export BSI_DEBUG=1           # Print kernel execution details
```

### Baseline Test (Old Kernels)

```bash
# Run with old kernels to get baseline
export BSI_OPTIMIZED=0

python benchmarks/benchmark_performance_bsi.py \
    --model_name facebook/opt-1.3b \
    --datasets lambada \
    --split validation \
    --num_samples 200 \
    --decimal_places 2 \
    --compress_threshold 0.5 \
    --scope attention \
    --bsi_device cuda
```

**Expected baseline performance:**
- Build time: ~47.5s per sample
- Dot time: ~2.36s per sample
- Total: ~14.9s per forward pass

### Optimized Test (New Kernels)

```bash
# Run with new optimized kernels
export BSI_OPTIMIZED=1

python benchmarks/benchmark_performance_bsi.py \
    --model_name facebook/opt-1.3b \
    --datasets lambada \
    --split validation \
    --num_samples 200 \
    --decimal_places 2 \
    --compress_threshold 0.5 \
    --scope attention \
    --bsi_device cuda
```

**Expected optimized performance:**
- Build time: ~0.5s per sample (100x faster)
- Dot time: ~0.05-0.12s per sample (20-50x faster)
- Total: ~0.6s per forward pass (25x faster)

### Quick Smoke Test (Small Model)

For faster iteration during debugging:

```bash
export BSI_OPTIMIZED=1

python benchmarks/benchmark_performance_bsi.py \
    --model_name facebook/opt-125m \
    --datasets lambada \
    --split validation \
    --num_samples 10 \
    --decimal_places 2 \
    --compress_threshold 0.5 \
    --scope attention \
    --bsi_device cuda
```

## Performance Metrics to Track

The benchmark script outputs:
1. **`build_ms`** - Query building time (should drop from ~47,000ms to ~500ms)
2. **`dot_ms`** - Dot product time (should drop from ~2,360ms to ~50-120ms)
3. **`avg_fwd`** - Average forward pass time (should drop from ~14,900ms to ~600ms)

### Expected Results Summary

| Metric | Old (Baseline) | New (Optimized) | Speedup |
|--------|----------------|-----------------|---------|
| Query Build | 47.5s | 0.5s | **95x** |
| Dot Product | 2.36s | 0.05-0.12s | **20-50x** |
| Total Forward | 14.9s | 0.6s | **25x** |

## Debugging

### If build fails:

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"

# Check compiler
nvcc --version

# Verbose build
python setup.py build_ext --inplace --verbose
```

### If kernels don't launch:

```bash
# Enable debug output
export BSI_DEBUG=1

# Run small test
python -c "
import torch
import bsi_ops

q = torch.randn(128, device='cuda')
k = torch.randn(128, device='cuda')
result = bsi_ops.dot_product_decimal_cuda(q, k, 2)
print(f'Result: {result}')
"
```

### Compare old vs new kernels directly:

```bash
# Test old kernel
export BSI_OPTIMIZED=0
python benchmarks/benchmark_performance_bsi.py --num_samples 10 ... > old_results.txt

# Test new kernel
export BSI_OPTIMIZED=1
python benchmarks/benchmark_performance_bsi.py --num_samples 10 ... > new_results.txt

# Compare
diff old_results.txt new_results.txt
```

## Validation

### Accuracy Check

The optimizations should produce **identical results** to the old kernels. Check:

```bash
# The benchmark script prints top-1 and top-5 accuracy
# Both old and new should give same accuracy (within floating point error)
```

Expected accuracy (from your baseline):
- Top-1: 0.675 (old) vs 0.675 (new) ✅
- Top-5: 0.810 (old) vs 0.810 (new) ✅

If accuracy differs, something is wrong with the kernel implementation.

## Next Steps After Phase 1

Once Phase 1 optimizations are validated:

### Phase 2: Hybrid Compression (Expected: 2-4x additional speedup)
- Implement compressed/verbatim hybrid kernel
- Use EWAH compression for sparse high-order bitplanes
- Direct popcount on RLE runs

### Phase 3: Advanced Optimizations (Expected: 1.5-2x additional speedup)
- Persistent GPU query cache
- Tensor Core utilization for very large vectors
- Multi-stream overlapping

## Troubleshooting Common Issues

### Issue: "No CUDA devices available"
**Solution:** Make sure you're running on a GPU node, not CPU-only environment.

### Issue: Build fails with "nvcc not found"
**Solution:** Ensure CUDA toolkit is installed and `nvcc` is in PATH.

### Issue: Kernel produces NaN or wrong results
**Solution:**
1. Enable debug mode: `export BSI_DEBUG=1`
2. Check slice weights are computed correctly
3. Verify word count alignment (W must be consistent)

### Issue: No speedup observed
**Solution:**
1. Verify optimized kernels are actually being used: `export BSI_DEBUG=1` and check logs
2. Make sure `BSI_OPTIMIZED=1` is set
3. Check GPU utilization with `nvidia-smi`

## Contact

If you encounter issues, check:
1. Build logs for compilation errors
2. Runtime logs with `BSI_DEBUG=1`
3. GPU memory usage with `nvidia-smi`

---

**Created:** 2025-01-30
**Branch:** dotCUDAOptimizationV2
**Phase:** 1 - Optimized Dot Product + Batched Query Builder
