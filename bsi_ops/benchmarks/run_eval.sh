#!/usr/bin/env bash
set -euo pipefail

MODELS="opt-125m,opt-1.3b,opt-6.7b"
DECIMALS=(1 2 3)
THRS=(0 0.2 0.4 0.5 0.7)

for THR in "${THRS[@]}"; do
  echo "=== EVAL | thr=${THR} ==="
  python bsi_ops/benchmarks/benchmark_performance_bsi.py \
    --models "${MODELS}" \
    --num_samples 1000 --max_seq_len 512 \
    --decimal_places "${DECIMALS[@]}" \
    --compress_threshold "${THR}" \
    --scope all \
    --base_dtype fp32 \
    --simple_report_txt "reports/simple_eval_thr${THR}.txt" \
    --report_dir "reports/eval_runs_thr${THR}"
done