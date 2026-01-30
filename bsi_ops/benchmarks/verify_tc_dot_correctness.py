import argparse
import os

import torch

import bsi_ops


def _run_dot(*, use_tc: bool, query_caps, keys_cap):
    # bsi_ops reads this env var inside the C++ dispatch. We intentionally set it
    # in-process so we can A/B compare without rebuilding.
    os.environ["BSI_TC_DOT"] = "1" if use_tc else "0"

    out, dot_ns_total, dot_ns_per_query, dot_ns_per_scalar = (
        bsi_ops.batch_dot_product_multiquery_cuda_caps(query_caps, keys_cap)
    )
    torch.cuda.synchronize()
    return out, int(dot_ns_total), float(dot_ns_per_query), float(dot_ns_per_scalar)


def main() -> None:
    p = argparse.ArgumentParser(description="Verify TC BMMA dot matches baseline dot")
    p.add_argument("--Q", type=int, default=64)
    p.add_argument("--R", type=int, default=256)
    p.add_argument("--D", type=int, default=2048)
    p.add_argument("--decimal_places", type=int, default=2)
    p.add_argument("--compress_threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--warmup", type=int, default=2)
    # TC and baseline accumulate in different orders/instructions; expect small fp32 diffs.
    p.add_argument("--rtol", type=float, default=1e-2)
    p.add_argument("--atol", type=float, default=5e-3)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this check")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    # Keys are built on CPU (builder moves to CUDA internally).
    K = torch.randn(args.R, args.D, dtype=torch.float32, device="cpu")
    keys_cap, *_ = bsi_ops.build_bsi_keys_cuda(
        K, args.decimal_places, float(args.compress_threshold)
    )

    # Queries are built on CUDA.
    Q = torch.randn(args.Q, args.D, dtype=torch.float32, device=device)
    query_caps = bsi_ops.build_bsi_queries_cuda_batch(
        Q, args.decimal_places, float(args.compress_threshold)
    )

    # Warmup both paths to avoid one-time overhead in the timings.
    for _ in range(max(0, args.warmup)):
        _run_dot(use_tc=False, query_caps=query_caps, keys_cap=keys_cap)
        _run_dot(use_tc=True, query_caps=query_caps, keys_cap=keys_cap)

    ref, ref_ns, *_ = _run_dot(use_tc=False, query_caps=query_caps, keys_cap=keys_cap)
    tc, tc_ns, *_ = _run_dot(use_tc=True, query_caps=query_caps, keys_cap=keys_cap)

    diff = (tc - ref).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())

    denom = ref.abs().clamp_min(1.0e-6)
    max_rel = float((diff / denom).max().item())

    print(f"[Shape] Q={args.Q} R={args.R} D={args.D}")
    print(f"[Timing] baseline_ms={ref_ns/1e6:.3f} tc_ms={tc_ns/1e6:.3f}")
    print(f"[Diff] max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} max_rel={max_rel:.6e}")
    print(f"[NaN] baseline={bool(torch.isnan(ref).any().item())} tc={bool(torch.isnan(tc).any().item())}")

    # A hard fail in CI-style usage.
    if not torch.allclose(ref, tc, rtol=float(args.rtol), atol=float(args.atol)):
        raise SystemExit("FAIL: TC output does not match baseline within tolerance")

    print("OK: TC output matches baseline within tolerance")


if __name__ == "__main__":
    main()
