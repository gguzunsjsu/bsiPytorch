import argparse
import os
from contextlib import contextmanager

import torch

import bsi_ops


@contextmanager
def _temporary_env(**updates):
    saved = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_dot(*, query_caps=None, query_batch=None, keys_cap, tc_policy: str, tc_dot: int = 1):
    with _temporary_env(BSI_TC_POLICY=tc_policy, BSI_TC_DOT=str(tc_dot)):
        if query_batch is not None:
            out, dot_ns_total, dot_ns_per_query, dot_ns_per_scalar = (
                bsi_ops.batch_dot_product_multiquery_cuda_batch_caps(query_batch, keys_cap)
            )
        else:
            out, dot_ns_total, dot_ns_per_query, dot_ns_per_scalar = (
                bsi_ops.batch_dot_product_multiquery_cuda_caps(query_caps, keys_cap)
            )
        torch.cuda.synchronize()
        return out, int(dot_ns_total), float(dot_ns_per_query), float(dot_ns_per_scalar)


def main() -> None:
    p = argparse.ArgumentParser(description="Verify selected dot policy matches the legacy fixed76 path")
    p.add_argument("--Q", type=int, default=64)
    p.add_argument("--R", type=int, default=256)
    p.add_argument("--D", type=int, default=2048)
    p.add_argument("--decimal_places", type=int, default=2)
    p.add_argument("--compress_threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--check_packed", action=argparse.BooleanOptionalAction, default=True,
                   help="compare packed-batch output against the capsule path")
    p.add_argument("--baseline_policy", type=str, default="legacy")
    p.add_argument("--test_policy", type=str, default=os.environ.get("BSI_TC_POLICY", "auto"))
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
        K, args.decimal_places, float(args.compress_threshold), enable_hot_layout=True
    )

    # Queries are built on CUDA.
    Q = torch.randn(args.Q, args.D, dtype=torch.float32, device=device)
    query_caps = bsi_ops.build_bsi_queries_cuda_batch(
        Q, args.decimal_places, float(args.compress_threshold)
    )
    query_batch = bsi_ops.build_bsi_queries_cuda_batch_packed(
        Q, args.decimal_places, float(args.compress_threshold)
    )

    # Warmup both paths to avoid one-time overhead in the timings.
    for _ in range(max(0, args.warmup)):
        _run_dot(query_caps=query_caps, keys_cap=keys_cap, tc_policy=args.baseline_policy)
        _run_dot(query_caps=query_caps, keys_cap=keys_cap, tc_policy=args.test_policy)

    ref, ref_ns, *_ = _run_dot(query_caps=query_caps, keys_cap=keys_cap, tc_policy=args.baseline_policy)
    tc, tc_ns, *_ = _run_dot(query_caps=query_caps, keys_cap=keys_cap, tc_policy=args.test_policy)

    diff = (tc - ref).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())

    denom = ref.abs().clamp_min(1.0e-6)
    max_rel = float((diff / denom).max().item())

    print(f"[Shape] Q={args.Q} R={args.R} D={args.D}")
    print(f"[Policy] baseline={args.baseline_policy} test={args.test_policy}")
    print(f"[Timing] baseline_ms={ref_ns/1e6:.3f} tc_ms={tc_ns/1e6:.3f}")
    print(f"[Diff] max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} max_rel={max_rel:.6e}")
    print(f"[NaN] baseline={bool(torch.isnan(ref).any().item())} tc={bool(torch.isnan(tc).any().item())}")

    # A hard fail in CI-style usage.
    if not torch.allclose(ref, tc, rtol=float(args.rtol), atol=float(args.atol)):
        raise SystemExit("FAIL: TC output does not match baseline within tolerance")

    if args.check_packed:
        packed, packed_ns, *_ = _run_dot(
            query_batch=query_batch,
            keys_cap=keys_cap,
            tc_policy=args.test_policy,
        )
        packed_diff = (packed - tc).abs()
        packed_max_abs = float(packed_diff.max().item())
        packed_mean_abs = float(packed_diff.mean().item())
        packed_max_rel = float((packed_diff / tc.abs().clamp_min(1.0e-6)).max().item())
        print(
            f"[Packed] packed_ms={packed_ns/1e6:.3f} "
            f"max_abs={packed_max_abs:.6e} mean_abs={packed_mean_abs:.6e} max_rel={packed_max_rel:.6e}"
        )
        if not torch.allclose(tc, packed, rtol=float(args.rtol), atol=float(args.atol)):
            raise SystemExit("FAIL: packed-batch output does not match capsule path within tolerance")

    print("OK: TC output matches baseline within tolerance")


if __name__ == "__main__":
    main()
