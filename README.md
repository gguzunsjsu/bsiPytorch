# bsiPytorch

PyTorch + CUDA extension for Bit-Sliced Indexing (BSI) quantization/compression experiments and profiling on GPUs.

This repo focuses on:
- Building the `bsi_ops` C++/CUDA extension
- Running correctness + microbenchmarks for the BSI tensor-dot kernels
- Running end-to-end LLM eval benchmarks (e.g., OPT on LAMBADA)

## 0) GPU + Network Setup (Cluster)

If you're running on a GPU cluster that requires a proxy for outbound traffic (e.g., to download HuggingFace models/datasets),
source the proxy script first:

```bash
source bsi_ops/network_connection.sh
```

Load the CUDA compiler/module (example module name used on our cluster):

```bash
module load nvhpc-hpcx-cuda12/24.11
```

Then set CUDA toolchain env vars:

```bash
source gpu_env_setup.sh
```

## 1) Repo Setup

Clone and initialize submodules:

```bash
git clone https://github.com/gguzunsjsu/bsiPytorch.git
cd bsiPytorch
git submodule update --init --recursive
```

Create/activate your Python environment (conda/venv) and install a CUDA-enabled PyTorch build that matches your system.

## 2) Build / Rebuild the `bsi_ops` Extension

We rebuild and reinstall the extension using:

```bash
cd bsi_ops
bash rebuild_local.sh
```

Notes:
- This uninstalls any existing `bsi_ops`, cleans build artifacts, and reinstalls via `pip install . -v`.
- Build logs are written to `bsi_ops/install.log`.

## 3) End-to-End LLM Benchmark (BSI in OPT Linear Layers)

Run LAMBADA next-token eval + timing breakdown (examples):

> NOTE: The commands in sections 3-6 assume you are running from the `bsi_ops/` directory (i.e., `cd bsi_ops`).

```bash
# OPT-125M
BSI_TC_DOT=1 BSI_WARP_OUT=1 BSI_CK_BLOCK=128 BSI_Q_TILE=8 BSI_R_TILE=4 \
  python benchmarks/benchmark_performance_bsi.py \
    --model_name facebook/opt-125m \
    --datasets lambada --split validation --num_samples 200 \
    --decimal_places 2 --compress_threshold 0.5 \
    --scope all --bsi_device cuda

# OPT-1.3B
BSI_TC_DOT=1 BSI_WARP_OUT=1 BSI_CK_BLOCK=128 BSI_Q_TILE=8 BSI_R_TILE=4 \
  python benchmarks/benchmark_performance_bsi.py \
    --model_name facebook/opt-1.3b \
    --datasets lambada --split validation --num_samples 200 \
    --decimal_places 2 --compress_threshold 0.5 \
    --scope all --bsi_device cuda

# OPT-6.7B
BSI_TC_DOT=1 BSI_WARP_OUT=1 BSI_CK_BLOCK=128 BSI_Q_TILE=8 BSI_R_TILE=4 \
  python benchmarks/benchmark_performance_bsi.py \
    --model_name facebook/opt-6.7b \
    --datasets lambada --split validation --num_samples 200 \
    --decimal_places 2 --compress_threshold 0.5 \
    --scope all --bsi_device cuda
```

## 4) Tensor-Core Dot Correctness (TC vs Baseline)

```bash
BSI_TC_DOT=1 BSI_TC_TM=32 BSI_TC_TN=32 \
  python benchmarks/verify_tc_dot_correctness.py \
    --Q 64 --R 256 --D 2048 \
    --decimal_places 2 --compress_threshold 0.5
```

## 5) Microbenchmarks (BSI Tensor Dot)

FC1-like shape:

```bash
BSI_TC_DOT=1 BSI_TC_TM=32 BSI_TC_TN=32 \
  python benchmarks/benchmark_dot_kernel_micro.py \
    --Q 512 --R 8192 --D 2048 \
    --decimal_places 2 --compress_threshold 0.5 \
    --warmup 10 --iters 50 --report_stats
```

FC2-like shape:

```bash
BSI_TC_DOT=1 BSI_TC_TM=32 BSI_TC_TN=32 \
  python benchmarks/benchmark_dot_kernel_micro.py \
    --Q 512 --R 2048 --D 8192 \
    --decimal_places 2 --compress_threshold 0.5 \
    --warmup 10 --iters 50 --report_stats
```

## 6) Nsight Compute Profiling (NCU) for Tensor Dot

Example NCU command for the tensor-core kernel (single launch):

```bash
BSI_TC_DOT=1 BSI_TC_TM=32 BSI_TC_TN=32 \
  ncu --set full --target-processes all -f -o ncu_tc_tm32_full \
    -k popcount_weighted_keys_literal_fused_bmma_tc_kernel_tm32 \
    --launch-count 1 \
  python benchmarks/benchmark_dot_kernel_micro.py \
    --Q 512 --R 8192 --D 2048 \
    --decimal_places 2 --compress_threshold 0.5 \
    --warmup 0 --iters 1
```

View the report:

```bash
ncu -i ncu_tc_tm32_full.ncu-rep
```
