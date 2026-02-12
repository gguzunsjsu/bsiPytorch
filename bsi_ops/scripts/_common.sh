#!/usr/bin/env bash
set -euo pipefail

# Shared defaults for bsi_ops scripts. Callers can override by exporting vars.
: "${BSI_TC_DOT:=0}"
: "${BSI_WARP_OUT:=1}"
: "${BSI_CK_BLOCK:=128}"
: "${BSI_Q_TILE:=8}"
: "${BSI_R_TILE:=4}"
: "${BSI_HYBRID_DOT:=1}"
: "${BSI_HYBRID_MIN_COMP_FRAC:=0.35}"
: "${BSI_AUTO_TILES:=0}"

bsi_run() {
  BSI_TC_DOT="${BSI_TC_DOT}" \
  BSI_WARP_OUT="${BSI_WARP_OUT}" \
  BSI_CK_BLOCK="${BSI_CK_BLOCK}" \
  BSI_Q_TILE="${BSI_Q_TILE}" \
  BSI_R_TILE="${BSI_R_TILE}" \
  BSI_HYBRID_DOT="${BSI_HYBRID_DOT}" \
  BSI_HYBRID_MIN_COMP_FRAC="${BSI_HYBRID_MIN_COMP_FRAC}" \
  BSI_AUTO_TILES="${BSI_AUTO_TILES}" \
  "$@"
}
