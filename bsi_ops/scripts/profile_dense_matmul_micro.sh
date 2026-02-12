#!/usr/bin/env bash
set -euo pipefail

# Nsight Compute profiling wrapper for the dense torch baseline:
#   out = queries[Q,D] @ keys[R,D]^T -> [Q,R]
#
# This mirrors `profile_tc_dot_micro.sh` but profiles the dense matmul path
# (cuBLAS/cuBLASLt kernels launched by `torch.matmul`).

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BSI_OPS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${BSI_OPS_DIR}"

Q="${1:-512}"
R="${2:-8192}"
D="${3:-2048}"
OUT="${4:-ncu_dense_mm_fp16}"
DTYPE="${5:-fp16}" # fp16 | bf16 | fp32

NCU_SET="${NCU_SET:-full}"
WARMUP="${WARMUP:-0}"
ITERS="${ITERS:-1}"
LAUNCH_COUNT="${LAUNCH_COUNT:-1}"
LAUNCH_SKIP_BEFORE_MATCH="${LAUNCH_SKIP_BEFORE_MATCH:-0}"

# Use NVTX to focus NCU on the timed matmul loop (dramatically reduces noise).
NCU_NVTX="${NCU_NVTX:-1}" # 1=enable, 0=disable
NVTX_RANGE="${NVTX_RANGE:-dense_matmul_timed}"

ALLOW_TF32="${ALLOW_TF32:-0}" # only relevant for DTYPE=fp32

NCU_NVTX_FLAGS=()
PY_NVTX_FLAGS=()
if [[ "${NCU_NVTX}" == "1" ]]; then
  # `benchmark_dense_matmul_micro.py --nvtx` emits this range around the timed loop.
  NCU_NVTX_FLAGS+=(--nvtx --nvtx-include "${NVTX_RANGE}")
  PY_NVTX_FLAGS+=(--nvtx)
fi

PY_TF32_FLAGS=()
if [[ "${DTYPE}" == "fp32" && "${ALLOW_TF32}" == "1" ]]; then
  PY_TF32_FLAGS+=(--allow_tf32)
fi

ncu --set "${NCU_SET}" --target-processes all \
  "${NCU_NVTX_FLAGS[@]}" \
  --launch-skip-before-match "${LAUNCH_SKIP_BEFORE_MATCH}" --launch-count "${LAUNCH_COUNT}" \
  -f \
  -o "${OUT}" \
python benchmarks/benchmark_dense_matmul_micro.py \
  --Q "${Q}" --R "${R}" --D "${D}" \
  --dtype "${DTYPE}" \
  "${PY_TF32_FLAGS[@]}" \
  --warmup "${WARMUP}" --iters "${ITERS}" \
  "${PY_NVTX_FLAGS[@]}"

echo "NCU report: ${OUT}.ncu-rep"
echo "View: ncu -i ${OUT}.ncu-rep"

