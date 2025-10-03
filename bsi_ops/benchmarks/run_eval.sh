#!/usr/bin/env bash
set -euo pipefail

MODELS="opt-125m,opt-1.3b,opt-6.7b"
DECIMALS=(2)
THRS=(0.5)

for THR in "${THRS[@]}"; do
  echo "=== EVAL | thr=${THR} ==="
  python benchmarks/benchmark_performance_bsi.py \
    --models "${MODELS}" \
    --num_samples 100 --max_seq_len 128 \
    --decimal_places "${DECIMALS[@]}" \
    --compress_threshold "${THR}" \
    --scope attention \
    --base_dtype fp32 \
    --simple_report_txt "reports/simple_eval_thr${THR}.txt" \
    --report_dir "reports/eval_runs_thr${THR}"
done