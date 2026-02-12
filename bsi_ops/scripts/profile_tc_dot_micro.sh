#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/_common.sh"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BSI_OPS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${BSI_OPS_DIR}"

Q="${1:-512}"
R="${2:-8192}"
D="${3:-2048}"
OUT="${4:-ncu_tc_tm32_full}"
KERNEL_NAME="${5:-${KERNEL_NAME:-popcount_weighted_keys_literal_fused_bmma_tc_kernel_tm32}}"

NCU_SET="${NCU_SET:-full}"
DECIMAL_PLACES="${DECIMAL_PLACES:-2}"
COMPRESS_THRESHOLD="${COMPRESS_THRESHOLD:-0.5}"
WARMUP="${WARMUP:-0}"
ITERS="${ITERS:-1}"
LAUNCH_COUNT="${LAUNCH_COUNT:-1}"
LAUNCH_SKIP_BEFORE_MATCH="${LAUNCH_SKIP_BEFORE_MATCH:-0}"

BSI_TC_DOT=1 bsi_run ncu --set "${NCU_SET}" --target-processes all \
  -k "${KERNEL_NAME}" \
  --launch-skip-before-match "${LAUNCH_SKIP_BEFORE_MATCH}" --launch-count "${LAUNCH_COUNT}" \
  -f \
  -o "${OUT}" \
python benchmarks/benchmark_dot_kernel_micro.py \
  --Q "${Q}" --R "${R}" --D "${D}" \
  --decimal_places "${DECIMAL_PLACES}" --compress_threshold "${COMPRESS_THRESHOLD}" \
  --warmup "${WARMUP}" --iters "${ITERS}"

echo "NCU report: ${OUT}.ncu-rep"
echo "View: ncu -i ${OUT}.ncu-rep"
