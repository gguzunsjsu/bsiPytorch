#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if ! command -v python >/dev/null 2>&1; then
  echo "python is not on PATH"
  exit 1
fi

export BSI_TC_DOT=1
export BSI_TC_FIXED_INT=1
export BSI_TC_CPASYNC=1
export BSI_FIXED_BITS_KEYS=6
export BSI_FIXED_BITS_QUERIES=7
export BSI_FIXED_CHUNK_SCALE=1
export BSI_PROFILE=1
export BSI_QUERY_PACKED=1
export BSI_TC_POLICY="${BSI_TC_POLICY:-auto}"
export BSI_TC_TMA="${BSI_TC_TMA:-2}"
unset BSI_TC_TM
unset BSI_TC_X_REPEAT

echo "[Env]"
echo "  BSI_TC_POLICY=${BSI_TC_POLICY}"
echo "  BSI_TC_TMA=${BSI_TC_TMA}"
echo "  BSI_TC_STRICT=${BSI_TC_STRICT:-0}"
echo "  SKIP_BUILD=${SKIP_BUILD:-0}"

run_rebuild() {
  if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
    echo
    echo "[Rebuild] skipped"
    return
  fi
  echo
  echo "[Rebuild]"
  bash rebuild_local.sh
  tail -n 50 install.log
}

run_correctness() {
  echo
  echo "[Correctness]"
  if [[ "${BSI_TC_POLICY}" == "sm90_splitk" ]]; then
    BSI_QUERY_BATCH=0 python benchmarks/verify_tc_dot_correctness.py --Q 64  --R 2048  --D 2048
    BSI_QUERY_BATCH=0 python benchmarks/verify_tc_dot_correctness.py --Q 128 --R 4096  --D 4096
    BSI_QUERY_BATCH=0 python benchmarks/verify_tc_dot_correctness.py --Q 512 --R 16384 --D 4096
    BSI_QUERY_BATCH=0 python benchmarks/verify_tc_dot_correctness.py --Q 512 --R 4096  --D 16384
  else
    BSI_QUERY_BATCH=0 python benchmarks/verify_tc_dot_correctness.py --Q 32  --R 768   --D 768
    BSI_QUERY_BATCH=0 python benchmarks/verify_tc_dot_correctness.py --Q 64  --R 2048  --D 2048
    BSI_QUERY_BATCH=0 python benchmarks/verify_tc_dot_correctness.py --Q 63  --R 4096  --D 4096
    BSI_QUERY_BATCH=0 python benchmarks/verify_tc_dot_correctness.py --Q 65  --R 4096  --D 4096
    BSI_QUERY_BATCH=0 python benchmarks/verify_tc_dot_correctness.py --Q 128 --R 4096  --D 4096
    BSI_QUERY_BATCH=0 python benchmarks/verify_tc_dot_correctness.py --Q 512 --R 16384 --D 4096
    BSI_QUERY_BATCH=0 python benchmarks/verify_tc_dot_correctness.py --Q 512 --R 4096  --D 16384
    BSI_QUERY_BATCH=0 python benchmarks/verify_tc_dot_correctness.py --Q 96  --R 16416 --D 4096
    BSI_QUERY_BATCH=0 python benchmarks/verify_tc_dot_correctness.py --Q 96  --R 16512 --D 4096
  fi
}

run_microbench() {
  echo
  echo "[Microbench]"
  python benchmarks/benchmark_dot_kernel_micro.py --shape_manifest fixed76 --warmup 10 --iters 50
}

run_dense_baseline() {
  echo
  echo "[Dense Baseline]"
  python benchmarks/benchmark_dense_matmul_micro.py --shape_manifest fixed76 --dtype fp16 --warmup 10 --iters 50
}

run_apples_kernel() {
  echo
  echo "[Apples Kernel]"
  python benchmarks/benchmark_apples_to_apples_bsi.py \
    --modes kernel_only \
    --shape_manifest fixed76 \
    --decimal_places 2 --compress_threshold 0.5 \
    --scope all --bsi_device cuda --bsi_profile 1 --base_dtype fp16
}

run_model_125m() {
  echo
  echo "[Model 125m]"
  python benchmarks/benchmark_apples_to_apples_bsi.py \
    --modes model_e2e \
    --model_name facebook/opt-125m \
    --dataset lambada --split validation --num_samples 200 \
    --decimal_places 2 --compress_threshold 0.5 \
    --scope all --bsi_device cuda --bsi_profile 1 --base_dtype fp16 \
    --report_bsi_layers 10
}

run_model_13b() {
  echo
  echo "[Model 1.3b]"
  python benchmarks/benchmark_apples_to_apples_bsi.py \
    --modes model_e2e \
    --model_name facebook/opt-1.3b \
    --dataset lambada --split validation --num_samples 200 \
    --decimal_places 2 --compress_threshold 0.5 \
    --scope all --bsi_device cuda --bsi_profile 1 --base_dtype fp16 \
    --report_bsi_layers 10
}

run_model_67b() {
  echo
  echo "[Model 6.7b]"
  python benchmarks/benchmark_apples_to_apples_bsi.py \
    --modes model_e2e \
    --model_name facebook/opt-6.7b \
    --dataset lambada --split validation --num_samples 200 \
    --decimal_places 2 --compress_threshold 0.5 \
    --scope all --bsi_device cuda --bsi_profile 1 --base_dtype fp16 \
    --report_bsi_layers 10
}

run_ncu() {
  echo
  echo "[NCU]"
  ncu --set full --target-processes all --import-source yes \
    -o ncu_4096x4096 \
    python benchmarks/benchmark_dot_kernel_micro.py --Q 512 --R 4096 --D 4096 --warmup 10 --iters 50

  ncu --set full --target-processes all --import-source yes \
    -o ncu_16384x4096 \
    python benchmarks/benchmark_dot_kernel_micro.py --Q 512 --R 16384 --D 4096 --warmup 10 --iters 50

  ncu --set full --target-processes all --import-source yes \
    -o ncu_4096x16384 \
    python benchmarks/benchmark_dot_kernel_micro.py --Q 512 --R 4096 --D 16384 --warmup 10 --iters 50

  ncu --set full --target-processes all --import-source yes \
    -o ncu_8192x2048 \
    python benchmarks/benchmark_dot_kernel_micro.py --Q 512 --R 8192 --D 2048 --warmup 10 --iters 50

  ncu --set full --target-processes all --import-source yes \
    -o ncu_2048x8192 \
    python benchmarks/benchmark_dot_kernel_micro.py --Q 512 --R 2048 --D 8192 --warmup 10 --iters 50
}

case "${MODE}" in
  quick)
    run_rebuild
    run_correctness
    run_microbench
    run_apples_kernel
    ;;
  perf)
    run_rebuild
    run_microbench
    run_dense_baseline
    run_apples_kernel
    ;;
  model_125m)
    run_rebuild
    run_model_125m
    ;;
  model_13b)
    run_rebuild
    run_model_13b
    ;;
  model_67b)
    run_rebuild
    run_model_67b
    ;;
  all_models)
    run_rebuild
    run_model_125m
    run_model_13b
    run_model_67b
    ;;
  full)
    run_rebuild
    run_correctness
    run_microbench
    run_dense_baseline
    run_apples_kernel
    run_model_125m
    run_model_13b
    run_model_67b
    ;;
  ncu)
    run_ncu
    ;;
  *)
    echo "Usage: $0 {quick|perf|model_125m|model_13b|model_67b|all_models|full|ncu}"
    exit 1
    ;;
esac
