import argparse
import os
import sys
import time

import torch

import bsi_ops

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from shape_manifest import resolve_shape_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Microbench BSI dot kernel (CUDA)")
    parser.add_argument("--Q", type=int, default=128, help="number of query vectors")
    parser.add_argument("--R", type=int, default=256, help="number of keys")
    parser.add_argument("--D", type=int, default=2048, help="input dimension")
    parser.add_argument("--decimal_places", type=int, default=2)
    parser.add_argument("--compress_threshold", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--shape_manifest", type=str, default="",
                        help='Optional shape manifest name (e.g. "fixed76"). When set, ignores --Q/--R/--D and runs all listed shapes.')
    parser.add_argument("--report_stats", action="store_true",
                        help="print Sb and S distributions from capsules")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this microbench")

    shapes = resolve_shape_manifest(args.shape_manifest)
    if shapes is None:
        shapes = [(args.Q, args.R, args.D)]

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    for shape_idx, (q_rows, r_rows, d_cols) in enumerate(shapes):
        if shape_idx > 0:
            print()

        # Build random keys on CPU (builder moves to CUDA internally)
        K = torch.randn(r_rows, d_cols, dtype=torch.float32, device="cpu")
        keys_cap, _, _, _, _ = bsi_ops.build_bsi_keys_cuda(
            K, args.decimal_places, float(args.compress_threshold)
        )

        # Build random queries on CUDA
        Q = torch.randn(q_rows, d_cols, dtype=torch.float32, device=device)
        query_caps = bsi_ops.build_bsi_queries_cuda_batch(
            Q, args.decimal_places, float(args.compress_threshold)
        )

        if args.report_stats:
            key_stats = bsi_ops.bsi_keys_cuda_stats(keys_cap)
            qry_stats = bsi_ops.bsi_query_caps_stats(query_caps)
            print("[Keys]", dict(key_stats))
            print("[Queries]", dict(qry_stats))

        for _ in range(args.warmup):
            bsi_ops.batch_dot_product_multiquery_cuda_caps(query_caps, keys_cap)

        total_ns = 0
        t0 = time.perf_counter()
        for _ in range(args.iters):
            _, dot_ns_total, _, _ = bsi_ops.batch_dot_product_multiquery_cuda_caps(
                query_caps, keys_cap
            )
            total_ns += int(dot_ns_total)
        t1 = time.perf_counter()

        avg_ns = total_ns / max(1, args.iters)
        avg_ms = avg_ns / 1e6
        per_query_us = (avg_ns / q_rows) / 1e3 if q_rows > 0 else 0.0
        per_scalar_ns = (avg_ns / (q_rows * r_rows)) if (q_rows > 0 and r_rows > 0) else 0.0

        prefix = "[Microbench]"
        if args.shape_manifest:
            prefix = f"[Microbench {shape_idx + 1}/{len(shapes)}]"
        print(f"{prefix} Q={q_rows} R={r_rows} D={d_cols} iters={args.iters}")
        print(f"[Kernel] avg_dot_ms={avg_ms:.3f}  dot_q_us={per_query_us:.3f}  dot_s_ns={per_scalar_ns:.3f}")
        if hasattr(bsi_ops, "get_last_dot_profile_cuda"):
            prof = bsi_ops.get_last_dot_profile_cuda()
            print(
                "[Engine] "
                f"engine={prof.get('engine', 'legacy')} "
                f"transport={prof.get('transport', 'legacy')} "
                f"split_k={int(prof.get('split_k', 1))} "
                f"reject={prof.get('reject_reason', 'none')} "
                f"fallback={int(bool(prof.get('fallback_used', False)))}"
            )
        print(f"[Wall] total_elapsed_s={t1 - t0:.3f}")


if __name__ == "__main__":
    main()
