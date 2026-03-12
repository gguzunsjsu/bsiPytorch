# BSI-Native Tensor-Core Inference Baseline

This project implements BSI-native linear layers and CUDA dot kernels for compressed LLM inference. The current stable operating point is a fixed-bit, tensor-core, TM32, `R_sweep=4`, chunk-scaled path on SM90/H100-class GPUs.

## 0) GPU + Network Setup (Cluster)
If the cluster requires a proxy for outbound traffic, source the network helper first from the repository root:

```bash
source bsi_ops/network_connection.sh
```

Load the CUDA compiler/module used on the cluster:

```bash
module load nvhpc-hpcx-cuda12/24.11
```

If your cluster provides an additional GPU connection helper, source it before CUDA environment setup:

```bash
source gpu_connection.sh
```

Then set the CUDA toolchain environment variables:

```bash
source gpu_env_setup.sh
```

## 1) Repo Setup
Clone the repository and initialize submodules:

```bash
git clone https://github.com/gguzunsjsu/bsiPytorch.git
cd bsiPytorch
git submodule update --init --recursive
```

Create and activate a Python environment with a CUDA-enabled PyTorch build that matches your system.

## 2) Build / Rebuild the `bsi_ops` Extension
Move into the extension directory and rebuild:

```bash
cd bsi_ops
bash rebuild_local.sh
```

Notes:
- This uninstalls any existing `bsi_ops` installation.
- It cleans previous build artifacts.
- It reinstalls the extension through `pip install . -v`.
- Build logs are written to `bsi_ops/install.log`.

## 3) Stable Same-Bit Baseline Configuration
Use the following environment variables for the stable same-bit baseline:

```bash
export BSI_TC_DOT=1
export BSI_TC_FIXED_INT=1
export BSI_TC_CPASYNC=1
export BSI_TC_TM=32
export BSI_FIXED_BITS_KEYS=6
export BSI_FIXED_BITS_QUERIES=7
export BSI_FIXED_CHUNK_SCALE=1
export BSI_TC_R_SWEEP=4
export BSI_TC_TMA=1
export BSI_DOT_DEBUG=1   # optional
```

Notes:
- `BSI_PROFILE=1` is useful when the goal is to attribute build time and dot time.
- `BSI_PROFILE=0` is better when the goal is fairer total runtime timing.
- `BSI_DOT_DEBUG=1` prints the selected kernel path and shape metadata.

## 4) Apples-to-Apples Benchmark Modes
The main harness is `benchmarks/benchmark_apples_to_apples_bsi.py`. It supports three modes:

- `kernel_only`: builds keys and queries once and measures only the BSI dot kernel against `torch.matmul`
- `linear_e2e`: includes per-call query building and approximates a single BSI linear layer end to end
- `model_e2e`: shells into the full LAMBADA evaluator and measures end-to-end model behavior

Use `kernel_only` to study the pure kernel, `linear_e2e` to include runtime query construction, and `model_e2e` to study real model behavior.

## 5) Kernel-Only Checks
FC1-like 6.7B shape:

```bash
python benchmarks/benchmark_apples_to_apples_bsi.py --modes kernel_only \
  --Q 512 --R 16384 --D 4096 \
  --decimal_places 2 --compress_threshold 0.5 \
  --torch_dtype fp16 --bsi_profile 0 --base_dtype fp16
```

Very-wide output shape:

```bash
python benchmarks/benchmark_apples_to_apples_bsi.py --modes kernel_only \
  --Q 512 --R 50272 --D 4096 \
  --decimal_places 2 --compress_threshold 0.5 \
  --torch_dtype fp16 --bsi_profile 0 --base_dtype fp16
```

Single-layer end-to-end timing:

```bash
python benchmarks/benchmark_apples_to_apples_bsi.py --modes linear_e2e \
  --Q 512 --R 16384 --D 4096 \
  --decimal_places 2 --compress_threshold 0.5 \
  --torch_dtype fp16 --bsi_profile 0 --base_dtype fp16
```

## 6) Model End-to-End
`opt-6.7b`, `lambada`, `num_samples=200`, attribution mode:

```bash
python benchmarks/benchmark_apples_to_apples_bsi.py --modes model_e2e \
  --model_name facebook/opt-6.7b \
  --dataset lambada --split validation --num_samples 200 \
  --decimal_places 2 --compress_threshold 0.5 \
  --scope all --bsi_device cuda --bsi_profile 1 --base_dtype fp16
```

The evaluator also supports `scope=attention` and `scope=mlp` when layer-group sensitivity needs to be measured separately.

## 7) Interpretation Notes
- Use `kernel_only` when the goal is to isolate CUDA dot-kernel behavior.
- Use `linear_e2e` when the goal is to include runtime query construction in a single-layer measurement.
- Use `model_e2e` when the goal is end-to-end LLM behavior on LAMBADA.
- Compare `scope=all` and `scope=attention` separately. They correspond to different speed and accuracy regimes.
