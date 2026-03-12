# BSI-Native Tensor-Core Inference Baseline

This codebase implements BSI-native linear layers and CUDA dot kernels for compressed LLM inference. The current stable operating point is a fixed-bit, tensor-core, TM32, `R_sweep=4`, chunk-scaled path on SM90/H100-class GPUs.

## 0) GPU + Network Setup (Cluster)
These commands assume you are at the repository root after cloning. If you are running on a GPU cluster that requires a proxy for outbound traffic, source the network helper first:

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
- `BSI_PROFILE=1` is useful for build/dot attribution.
- `BSI_PROFILE=0` is better for fairer end-to-end runtime measurement.
- `BSI_DOT_DEBUG=1` prints the selected kernel path and shape metadata.

## 4) Kernel-Only Checks
The apples-to-apples benchmark harness in `benchmarks/benchmark_apples_to_apples_bsi.py` supports three modes:

- `kernel_only`: builds keys and queries once and measures only the BSI dot kernel against `torch.matmul`
- `linear_e2e`: includes per-call query building and approximates a single BSI linear layer end to end
- `model_e2e`: shells into the full LAMBADA evaluator and measures end-to-end model behavior

Use `kernel_only` to study the pure kernel, `linear_e2e` to include runtime query construction, and `model_e2e` to study real model behavior.

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

## 5) Model End-to-End
`opt-6.7b`, `lambada`, `num_samples=200`, attribution mode:

```bash
python benchmarks/benchmark_apples_to_apples_bsi.py --modes model_e2e \
  --model_name facebook/opt-6.7b \
  --dataset lambada --split validation --num_samples 200 \
  --decimal_places 2 --compress_threshold 0.5 \
  --scope all --bsi_device cuda --bsi_profile 1 --base_dtype fp16
```

## 6) Nsight Compute Profiling
The preferred kernel for the current stable path on SM90/H100 is:

- `popcount_weighted_keys_literal_fused_bmma_tc_kernel_tm32_fixed76_chunkscale_rsweep4_tma_tensorB`

Fallback kernels are:

- `popcount_weighted_keys_literal_fused_bmma_tc_kernel_tm32_fixed76_chunkscale_rsweep4`
- `popcount_weighted_keys_literal_fused_bmma_tc_kernel_tm32_fixed76_chunkscale`

### NCU: FC1-like 6.7B Shape
```bash
ncu --set full --target-processes all -f -o ncu_rsweep4_tma_fc1 \
  -k popcount_weighted_keys_literal_fused_bmma_tc_kernel_tm32_fixed76_chunkscale_rsweep4_tma_tensorB \
  --launch-count 1 \
  python benchmarks/benchmark_apples_to_apples_bsi.py --modes kernel_only \
    --Q 512 --R 16384 --D 4096 \
    --decimal_places 2 --compress_threshold 0.5 \
    --warmup 0 --iters 1 --torch_dtype fp16 --bsi_profile 0 --base_dtype fp16
```

### NCU: Very-Wide Output Shape
```bash
ncu --set full --target-processes all -f -o ncu_rsweep4_tma_vocab \
  -k popcount_weighted_keys_literal_fused_bmma_tc_kernel_tm32_fixed76_chunkscale_rsweep4_tma_tensorB \
  --launch-count 1 \
  python benchmarks/benchmark_apples_to_apples_bsi.py --modes kernel_only \
    --Q 512 --R 50272 --D 4096 \
    --decimal_places 2 --compress_threshold 0.5 \
    --warmup 0 --iters 1 --torch_dtype fp16 --bsi_profile 0 --base_dtype fp16
```

### NCU: Fallback Non-TMA Kernel
If the TMA descriptor path is unavailable or you want the cp.async fallback explicitly:

```bash
BSI_TC_TMA=0 ncu --set full --target-processes all -f -o ncu_rsweep4_no_tma \
  -k popcount_weighted_keys_literal_fused_bmma_tc_kernel_tm32_fixed76_chunkscale_rsweep4 \
  --launch-count 1 \
  python benchmarks/benchmark_apples_to_apples_bsi.py --modes kernel_only \
    --Q 512 --R 16384 --D 4096 \
    --decimal_places 2 --compress_threshold 0.5 \
    --warmup 0 --iters 1 --torch_dtype fp16 --bsi_profile 0 --base_dtype fp16
```

Open a saved report with:

```bash
ncu -i ncu_rsweep4_tma_fc1.ncu-rep
```

## 7) Figure Workflow
Generate the benchmark plots used by the LaTeX report:

```bash
python docs/latex/scripts/generate_benchmark_plots.py
```

If you export Nsight screenshots or other manual images, place them in `docs/latex/figures/raw/` and stage them into the filenames expected by the LaTeX document:

```bash
bash docs/latex/scripts/stage_ncu_screenshots.sh docs/latex/figures/raw docs/latex/figures/manual
```

## 8) Notes on Interpretation
- Use `kernel_only` to study the pure BSI dot kernel.
- Use `linear_e2e` to include per-call query building.
- Use `model_e2e` for end-to-end LLM behavior.
- Use `scope=attention` and `scope=all` separately. They capture different speed/accuracy regimes.
