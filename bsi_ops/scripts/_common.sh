#!/usr/bin/env bash
set -euo pipefail

# Shared defaults for bsi_ops scripts. Callers can override by exporting vars.
: "${BSI_TC_DOT:=0}"
: "${BSI_TC_TM:=0}"
: "${BSI_TC_LOWREG:=1}"
: "${BSI_WARP_OUT:=1}"
: "${BSI_CK_BLOCK:=128}"
: "${BSI_Q_TILE:=8}"
: "${BSI_R_TILE:=4}"
: "${BSI_DOT_SLICE_MASKS:=0}"

bsi_run() {
  BSI_TC_DOT="${BSI_TC_DOT}" \
  BSI_TC_TM="${BSI_TC_TM}" \
  BSI_TC_LOWREG="${BSI_TC_LOWREG}" \
  BSI_WARP_OUT="${BSI_WARP_OUT}" \
  BSI_CK_BLOCK="${BSI_CK_BLOCK}" \
  BSI_Q_TILE="${BSI_Q_TILE}" \
  BSI_R_TILE="${BSI_R_TILE}" \
  BSI_DOT_SLICE_MASKS="${BSI_DOT_SLICE_MASKS}" \
  "$@"
}
