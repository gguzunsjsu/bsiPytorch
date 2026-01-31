#!/usr/bin/env bash
set -euo pipefail

Q="${1:-512}"
R="${2:-8192}"
D="${3:-2048}"
OUT="${4:-ncu_tc_full}"

: "${BSI_TC_TN:=32}"
: "${BSI_WARP_OUT:=1}"
: "${BSI_CK_BLOCK:=128}"
: "${BSI_Q_TILE:=8}"
: "${BSI_R_TILE:=4}"

KERNEL_NAME="popcount_weighted_keys_literal_fused_bmma_tc_kernel"
if [[ "${BSI_TC_TN}" == "64" ]]; then
  KERNEL_NAME="popcount_weighted_keys_literal_fused_bmma_tc_kernel_tn64"
fi

BSI_TC_DOT=1 \
BSI_TC_TN="${BSI_TC_TN}" \
BSI_WARP_OUT="${BSI_WARP_OUT}" \
BSI_CK_BLOCK="${BSI_CK_BLOCK}" \
BSI_Q_TILE="${BSI_Q_TILE}" \
BSI_R_TILE="${BSI_R_TILE}" \
ncu --set full --target-processes all \
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
