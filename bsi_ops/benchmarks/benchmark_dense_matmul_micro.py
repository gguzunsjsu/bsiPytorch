import argparse
import os
import sys
import time

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from shape_manifest import resolve_shape_manifest


def _parse_dtype(s: str) -> torch.dtype:
    s = s.lower().strip()
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp32", "float32", "float"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Microbench torch dot Q@K.T"
    )
    parser.add_argument("--Q", type=int, default=512, help="number of query vectors (rows)")
    parser.add_argument("--R", type=int, default=8192, help="number of keys / output columns")
    parser.add_argument("--D", type=int, default=2048, help="input dimension")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--allow_tf32", action="store_true",
                        help="enable TF32 for fp32 matmul (ignored for fp16/bf16)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--shape_manifest", type=str, default="",
                        help='Optional shape manifest name (e.g. "fixed76"). When set, ignores --Q/--R/--D and runs all listed shapes.')
    parser.add_argument("--nvtx", action="store_true",
                        help="emit NVTX ranges around the timed matmul loop (useful for Nsight/NCU filtering)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this microbench")

    shapes = resolve_shape_manifest(args.shape_manifest)
    if shapes is None:
        shapes = [(args.Q, args.R, args.D)]

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = _parse_dtype(args.dtype)

    if dtype == torch.float32:
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)

    for shape_idx, (q_rows, r_rows, d_cols) in enumerate(shapes):
        if shape_idx > 0:
            print()

        queries = torch.randn(q_rows, d_cols, device=device, dtype=dtype)
        keys = torch.randn(r_rows, d_cols, device=device, dtype=dtype)

        for _ in range(max(0, args.warmup)):
            _ = torch.matmul(queries, keys.t())
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        nvtx_handle = None
        if args.nvtx:
            nvtx_handle = torch.cuda.nvtx.range_start("dense_matmul_timed")
        t0 = time.perf_counter()
        start.record()
        out = None
        for _ in range(max(1, args.iters)):
            out = torch.matmul(queries, keys.t())
        end.record()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if args.nvtx:
            torch.cuda.nvtx.range_end(nvtx_handle)

        if out is None:
            raise RuntimeError("matmul did not run")

        total_ms = float(start.elapsed_time(end))
        iters = max(1, int(args.iters))
        avg_ms = total_ms / iters
        per_query_us = (avg_ms * 1e3) / q_rows if q_rows > 0 else 0.0
        per_scalar_ns = (avg_ms * 1e6) / (q_rows * r_rows) if (q_rows > 0 and r_rows > 0) else 0.0

        prefix = "[Dense Microbench]"
        if args.shape_manifest:
            prefix = f"[Dense Microbench {shape_idx + 1}/{len(shapes)}]"
        print(f"{prefix} Q={q_rows} R={r_rows} D={d_cols} dtype={args.dtype} iters={iters}")
        print(f"[Kernel] avg_mm_ms={avg_ms:.3f}  mm_q_us={per_query_us:.3f}  mm_s_ns={per_scalar_ns:.3f}")
        print(f"[Wall] total_elapsed_s={t1 - t0:.3f}")


if __name__ == "__main__":
    main()
