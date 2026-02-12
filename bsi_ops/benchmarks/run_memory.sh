#!/usr/bin/env bash
set -euo pipefail

MODELS="opt-125m,opt-1.3b,opt-6.7b"
DECIMALS=(2)
THRS=(0.5)

for THR in "${THRS[@]}"; do
  echo "=== MEMORY-ONLY | thr=${THR} ==="
  python benchmarks/benchmark_performance_bsi.py \
    --models "${MODELS}" \
    --memory_only \
    --decimal_places "${DECIMALS[@]}" \
    --compress_threshold "${THR}" \
    --scope all \
    --base_dtype fp32 \
    --simple_report_txt "reports/simple_memory_thr${THR}.txt" \
    --report_dir "reports/memory_runs_thr${THR}"
done