#!/usr/bin/env bash
set -euo pipefail

# Shared defaults for bsi_ops scripts. Callers can override by exporting vars.
: "${BSI_TC_DOT:=0}"
: "${BSI_Q_TILE:=8}"
: "${BSI_R_TILE:=4}"

bsi_run() {
  BSI_TC_DOT="${BSI_TC_DOT}" \
  BSI_Q_TILE="${BSI_Q_TILE}" \
  BSI_R_TILE="${BSI_R_TILE}" \
  "$@"
}
