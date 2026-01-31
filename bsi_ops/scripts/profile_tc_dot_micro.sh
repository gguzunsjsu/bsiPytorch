#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/_common.sh"

Q="${1:-512}"
R="${2:-8192}"
D="${3:-2048}"
OUT="${4:-ncu_tc_full}"

KERNEL_NAME="popcount_weighted_keys_literal_fused_bmma_tc_kernel"
if [[ "${BSI_TC_TM}" == "32" && "${BSI_TC_TN}" == "64" ]]; then
  KERNEL_NAME="popcount_weighted_keys_literal_fused_bmma_tc_kernel_tm32_tn64"
elif [[ "${BSI_TC_TM}" == "32" ]]; then
  KERNEL_NAME="popcount_weighted_keys_literal_fused_bmma_tc_kernel_tm32"
elif [[ "${BSI_TC_TN}" == "64" ]]; then
  KERNEL_NAME="popcount_weighted_keys_literal_fused_bmma_tc_kernel_tn64"
fi

BSI_TC_DOT=1 bsi_run ncu --set full --target-processes all \
  -k "${KERNEL_NAME}" \
  --launch-skip-before-match 0 --launch-count 1 \
  -f \
  -o "${OUT}" \
python benchmarks/benchmark_dot_kernel_micro.py \
  --Q "${Q}" --R "${R}" --D "${D}" \
  --decimal_places 2 --compress_threshold 0.5 \
  --warmup 0 --iters 1

echo "NCU report: ${OUT}.ncu-rep"
echo "View: ncu -i ${OUT}.ncu-rep"
