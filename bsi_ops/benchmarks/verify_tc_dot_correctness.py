import argparse
import os
from typing import Optional

import torch

import bsi_ops


def _run_dot(*, use_tc: bool, hybrid_dot: Optional[int], query_caps, keys_cap):
    # bsi_ops reads env vars inside CUDA dispatch.
    os.environ["BSI_TC_DOT"] = "1" if use_tc else "0"
    if hybrid_dot is not None:
        os.environ["BSI_HYBRID_DOT"] = "1" if int(hybrid_dot) != 0 else "0"

    out, dot_ns_total, dot_ns_per_query, dot_ns_per_scalar = (
        bsi_ops.batch_dot_product_multiquery_cuda_caps(query_caps, keys_cap)
    )
    torch.cuda.synchronize()
    return out, int(dot_ns_total), float(dot_ns_per_query), float(dot_ns_per_scalar)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Verify CUDA dot correctness. "
            "For key_storage_mode=verbatim: compare BMMA TC vs baseline. "
            "For key_storage_mode=hybrid: compare hybrid compressed-key dot vs literal fallback."
        )
    )
    p.add_argument("--Q", type=int, default=64)
    p.add_argument("--R", type=int, default=256)
    p.add_argument("--D", type=int, default=2048)
    p.add_argument("--decimal_places", type=int, default=2)
    p.add_argument("--compress_threshold", type=float, default=0.5)
    p.add_argument("--key_storage_mode", type=str, choices=["verbatim", "hybrid"], default="verbatim")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--warmup", type=int, default=2)
    # Different paths accumulate in different orders/instructions; allow small fp32 diffs.
    p.add_argument("--rtol", type=float, default=1e-2)
    p.add_argument("--atol", type=float, default=5e-3)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this check")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    # Keys are built on CPU (builder pre-packs to CUDA internally).
    K = torch.randn(args.R, args.D, dtype=torch.float32, device="cpu")
    keys_cap, *_ = bsi_ops.build_bsi_keys_cuda(
        K,
        args.decimal_places,
        float(args.compress_threshold),
        args.key_storage_mode,
    )

    # Queries are built on CUDA.
    Q = torch.randn(args.Q, args.D, dtype=torch.float32, device=device)
    query_caps = bsi_ops.build_bsi_queries_cuda_batch(
        Q, args.decimal_places, float(args.compress_threshold)
    )

    if args.key_storage_mode == "hybrid":
        for _ in range(max(0, args.warmup)):
            _run_dot(use_tc=False, hybrid_dot=0, query_caps=query_caps, keys_cap=keys_cap)
            _run_dot(use_tc=False, hybrid_dot=1, query_caps=query_caps, keys_cap=keys_cap)

        ref, ref_ns, *_ = _run_dot(use_tc=False, hybrid_dot=0, query_caps=query_caps, keys_cap=keys_cap)
        opt, opt_ns, *_ = _run_dot(use_tc=False, hybrid_dot=1, query_caps=query_caps, keys_cap=keys_cap)
        lhs_name = "literal_fallback"
        rhs_name = "hybrid_compressed"
    else:
        for _ in range(max(0, args.warmup)):
            _run_dot(use_tc=False, hybrid_dot=None, query_caps=query_caps, keys_cap=keys_cap)
            _run_dot(use_tc=True, hybrid_dot=None, query_caps=query_caps, keys_cap=keys_cap)

        ref, ref_ns, *_ = _run_dot(use_tc=False, hybrid_dot=None, query_caps=query_caps, keys_cap=keys_cap)
        opt, opt_ns, *_ = _run_dot(use_tc=True, hybrid_dot=None, query_caps=query_caps, keys_cap=keys_cap)
        lhs_name = "baseline"
        rhs_name = "tc"

    diff = (opt - ref).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())

    denom = ref.abs().clamp_min(1.0e-6)
    max_rel = float((diff / denom).max().item())

    print(f"[Shape] Q={args.Q} R={args.R} D={args.D} mode={args.key_storage_mode}")
    print(f"[Timing] {lhs_name}_ms={ref_ns/1e6:.3f} {rhs_name}_ms={opt_ns/1e6:.3f}")
    print(f"[Diff] max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} max_rel={max_rel:.6e}")
    print(f"[NaN] {lhs_name}={bool(torch.isnan(ref).any().item())} {rhs_name}={bool(torch.isnan(opt).any().item())}")

    if not torch.allclose(ref, opt, rtol=float(args.rtol), atol=float(args.atol)):
        raise SystemExit("FAIL: outputs do not match within tolerance")

    print("OK: outputs match within tolerance")


if __name__ == "__main__":
    main()
