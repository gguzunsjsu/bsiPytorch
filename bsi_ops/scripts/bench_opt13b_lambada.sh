#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/_common.sh"

NUM_SAMPLES="${1:-200}"
KEY_STORAGE_MODE="${2:-verbatim}"

bsi_run python benchmarks/benchmark_performance_bsi.py \
  --model_name facebook/opt-1.3b \
  --datasets lambada \
  --split validation \
  --num_samples "${NUM_SAMPLES}" \
  --decimal_places 2 \
  --compress_threshold 0.5 \
  --scope all \
  --bsi_device cuda \
  --key_storage_mode "${KEY_STORAGE_MODE}"
