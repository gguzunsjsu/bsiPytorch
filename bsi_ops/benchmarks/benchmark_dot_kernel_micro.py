import argparse
import time

import torch

import bsi_ops


def main() -> None:
    parser = argparse.ArgumentParser(description="Microbench BSI dot kernel (CUDA)")
    parser.add_argument("--Q", type=int, default=128, help="number of query vectors")
    parser.add_argument("--R", type=int, default=256, help="number of keys")
    parser.add_argument("--D", type=int, default=2048, help="input dimension")
    parser.add_argument("--decimal_places", type=int, default=2)
    parser.add_argument("--compress_threshold", type=float, default=0.5)
    parser.add_argument("--key_storage_mode", type=str, choices=["verbatim", "hybrid"], default="verbatim")
    parser.add_argument("--hybrid_dot", type=int, choices=[0, 1], default=None,
                        help="Override BSI_HYBRID_DOT (only meaningful with key_storage_mode=hybrid)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--report_stats", action="store_true",
                        help="print Sb and S distributions from capsules")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this microbench")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    # Build random keys on CPU (builder moves to CUDA internally)
    K = torch.randn(args.R, args.D, dtype=torch.float32, device="cpu")
    if args.hybrid_dot is not None:
        import os
        os.environ["BSI_HYBRID_DOT"] = str(int(args.hybrid_dot))
    keys_cap, _, _, _, _ = bsi_ops.build_bsi_keys_cuda(
        K,
        args.decimal_places,
        float(args.compress_threshold),
        args.key_storage_mode,
    )

    # Build random queries on CUDA
    Q = torch.randn(args.Q, args.D, dtype=torch.float32, device=device)
    query_caps = bsi_ops.build_bsi_queries_cuda_batch(
        Q, args.decimal_places, float(args.compress_threshold)
    )

    if args.report_stats:
        key_stats = bsi_ops.bsi_keys_cuda_stats(keys_cap)
        qry_stats = bsi_ops.bsi_query_caps_stats(query_caps)
        print("[Keys]", dict(key_stats))
        print("[Queries]", dict(qry_stats))

    # Warmup
    for _ in range(args.warmup):
        bsi_ops.batch_dot_product_multiquery_cuda_caps(query_caps, keys_cap)

    # Timed iterations: use kernel timings returned by the extension.
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
    per_query_us = (avg_ns / args.Q) / 1e3 if args.Q > 0 else 0.0
    per_scalar_ns = (avg_ns / (args.Q * args.R)) if (args.Q > 0 and args.R > 0) else 0.0

    print(f"[Microbench] Q={args.Q} R={args.R} D={args.D} iters={args.iters}")
    print(f"[Kernel] avg_dot_ms={avg_ms:.3f}  dot_q_us={per_query_us:.3f}  dot_s_ns={per_scalar_ns:.3f}")
    print(f"[Wall] total_elapsed_s={t1 - t0:.3f}")


if __name__ == "__main__":
    main()
