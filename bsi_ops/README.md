# BSI-Native Tensor-Core Inference Baseline

This codebase implements BSI-native linear layers and CUDA dot kernels for compressed LLM inference. The current stable operating point is a fixed-bit, tensor-core, TM32, `R_sweep=4`, chunk-scaled path on SM90/H100-class GPUs.

The main goals of this code are:

- store linear weights in BSI form
- build BSI queries at runtime from activations
- execute dot products natively from bit-sliced data
- benchmark kernel-only, layer-level, and model-level behavior
- compare dense FP16 and BSI on OPT-family models

## Layout
- `../bsiCPP`: CPU-side BSI and bitmap implementation
- `benchmarks/`: kernel and model evaluation scripts
- `csrc/cuda/`: PyTorch CUDA extension and dot kernels

## Environment
If you are on the cluster and need outbound access for HuggingFace downloads:

```bash
source network_connection.sh
```

Activate the Python environment and rebuild the extension:

```bash
source /home/017510883/miniconda3/bin/activate bsiPytorch
bash rebuild_local.sh
```

## Stable Same-Bit Baseline Configuration
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

## Baseline Test Commands
### Kernel-Only Checks
FC1-like 6.7B shape:

```bash
python benchmarks/benchmark_apples_to_apples_bsi.py --modes kernel_only \
  --Q 512 --R 16384 --D 4096 \
  --decimal_places 2 --compress_threshold 0.5 \
  --torch_dtype fp16 --bsi_profile 0 --base_dtype fp16
```

LM-head / very-wide output shape:

```bash
python benchmarks/benchmark_apples_to_apples_bsi.py --modes kernel_only \
  --Q 512 --R 50272 --D 4096 \
  --decimal_places 2 --compress_threshold 0.5 \
  --torch_dtype fp16 --bsi_profile 0 --base_dtype fp16
```

### Model End-to-End
`opt-6.7b`, `lambada`, `num_samples=200`, attribution mode:

```bash
python benchmarks/benchmark_apples_to_apples_bsi.py --modes model_e2e \
  --model_name facebook/opt-6.7b \
  --dataset lambada --split validation --num_samples 200 \
  --decimal_places 2 --compress_threshold 0.5 \
  --scope all --bsi_device cuda --bsi_profile 1 --base_dtype fp16
```

## Nsight Compute Profiling
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

## Notes on Interpretation
- Use `kernel_only` to study the pure BSI dot kernel.
- Use `linear_e2e` to include per-call query building.
- Use `model_e2e` for end-to-end LLM behavior.
- Use `scope=attention` and `scope=all` separately. They capture different speed/accuracy regimes.
