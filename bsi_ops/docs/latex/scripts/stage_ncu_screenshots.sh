#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="${1:-docs/latex/figures/raw}"
OUT_DIR="${2:-docs/latex/figures/manual}"

mkdir -p "$OUT_DIR"

copy_if_present() {
  local src="$1"
  local dst="$2"
  if [[ -f "$src" ]]; then
    cp "$src" "$dst"
    echo "copied: $src -> $dst"
  else
    echo "missing: $src" >&2
  fi
}

copy_if_present "$RAW_DIR/ncu_rsweep4_tma_fc1.png" "$OUT_DIR/ncu_summary_fc1.png"
copy_if_present "$RAW_DIR/ncu_rsweep4_tma_vocab.png" "$OUT_DIR/ncu_summary_vocab.png"
copy_if_present "$RAW_DIR/hybridbitmap_structure.png" "$OUT_DIR/hybridbitmap_structure.png"
copy_if_present "$RAW_DIR/bsi_build_pipeline.png" "$OUT_DIR/bsi_build_pipeline.png"
