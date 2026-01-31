#!/usr/bin/env bash
set -euo pipefail

NUM_SAMPLES="${1:-200}"

# can override these by exporting them before running the script.
: "${BSI_TC_DOT:=0}"
: "${BSI_TC_TN:=32}"
: "${BSI_WARP_OUT:=1}"
: "${BSI_CK_BLOCK:=128}"
: "${BSI_Q_TILE:=8}"
: "${BSI_R_TILE:=4}"

BSI_TC_DOT="${BSI_TC_DOT}" \
BSI_TC_TN="${BSI_TC_TN}" \
BSI_WARP_OUT="${BSI_WARP_OUT}" \
BSI_CK_BLOCK="${BSI_CK_BLOCK}" \
BSI_Q_TILE="${BSI_Q_TILE}" \
BSI_R_TILE="${BSI_R_TILE}" \
python benchmarks/benchmark_performance_bsi.py \
  --model_name facebook/opt-1.3b \
  --datasets lambada \
  --split validation \
  --num_samples "${NUM_SAMPLES}" \
  --decimal_places 2 \
  --compress_threshold 0.5 \
  --scope all \
  --bsi_device cuda
