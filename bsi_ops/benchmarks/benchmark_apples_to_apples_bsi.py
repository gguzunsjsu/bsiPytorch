import argparse
import os
import subprocess
import sys
from typing import Callable, Dict, List

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bsi_ops


def _parse_modes(raw: str) -> List[str]:
    out: List[str] = []
    for mode in raw.split(","):
        m = mode.strip().lower()
        if m:
            out.append(m)
    return out


def _torch_dtype_from_flag(flag: str) -> torch.dtype:
    f = flag.strip().lower()
    if f == "fp16":
        return torch.float16
    if f == "bf16":
        return torch.bfloat16
    if f == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {flag}")


def _time_cuda_ms(fn: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(max(0, warmup)):
        fn()
    torch.cuda.synchronize()

    total_ms = 0.0
    for _ in range(max(1, iters)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        total_ms += float(start.elapsed_time(end))
    return total_ms / float(max(1, iters))


def _print_row(name: str, avg_ms: float, q: int, r: int) -> None:
    qps = (q * 1000.0 / avg_ms) if avg_ms > 0.0 else 0.0
    ns_per_scalar = (avg_ms * 1.0e6 / float(q * r)) if (q > 0 and r > 0 and avg_ms > 0.0) else 0.0
    us_per_query = (avg_ms * 1000.0 / float(q)) if (q > 0 and avg_ms > 0.0) else 0.0
    print(
        f"{name:18s} avg_ms={avg_ms:9.4f}  "
        f"dot_q_us={us_per_query:8.4f}  dot_s_ns={ns_per_scalar:8.4f}  qps={qps:10.2f}"
    )


def run_kernel_only(args: argparse.Namespace, k_cpu: torch.Tensor, q_fp32: torch.Tensor, torch_dtype: torch.dtype) -> Dict[str, float]:
    print("\n[Mode] kernel_only")
    keys_cap, _, _, _, _ = bsi_ops.build_bsi_keys_cuda(
        k_cpu, args.decimal_places, float(args.compress_threshold)
    )
    query_batch = bsi_ops.build_bsi_queries_cuda_batch_packed(
        q_fp32, args.decimal_places, float(args.compress_threshold)
    )

    k_torch = k_cpu.to(device="cuda", dtype=torch_dtype, non_blocking=False)
    q_torch = q_fp32.to(dtype=torch_dtype)

    def run_bsi() -> None:
        bsi_ops.batch_dot_product_multiquery_cuda_batch_caps(query_batch, keys_cap)

    def run_torch() -> None:
        torch.matmul(q_torch, k_torch.t())

    bsi_ms = _time_cuda_ms(run_bsi, args.warmup, args.iters)
    torch_ms = _time_cuda_ms(run_torch, args.warmup, args.iters)

    _print_row("BSI kernel", bsi_ms, args.Q, args.R)
    _print_row(f"Torch {args.torch_dtype}", torch_ms, args.Q, args.R)
    speedup = (torch_ms / bsi_ms) if bsi_ms > 0.0 else 0.0
    print(f"speedup_vs_torch = {speedup:.4f}x")
    return {"bsi_ms": bsi_ms, "torch_ms": torch_ms, "speedup_vs_torch": speedup}


def run_linear_e2e(args: argparse.Namespace, k_cpu: torch.Tensor, q_fp32: torch.Tensor, torch_dtype: torch.dtype) -> Dict[str, float]:
    print("\n[Mode] linear_e2e")
    keys_cap, _, _, _, _ = bsi_ops.build_bsi_keys_cuda(
        k_cpu, args.decimal_places, float(args.compress_threshold)
    )
    k_torch = k_cpu.to(device="cuda", dtype=torch_dtype, non_blocking=False)
    q_torch = q_fp32.to(dtype=torch_dtype)

    def run_bsi() -> None:
        query_batch = bsi_ops.build_bsi_queries_cuda_batch_packed(
            q_fp32, args.decimal_places, float(args.compress_threshold)
        )
        bsi_ops.batch_dot_product_multiquery_cuda_batch_caps(query_batch, keys_cap)

    def run_torch() -> None:
        torch.matmul(q_torch, k_torch.t())

    bsi_ms = _time_cuda_ms(run_bsi, args.warmup, args.iters)
    torch_ms = _time_cuda_ms(run_torch, args.warmup, args.iters)

    _print_row("BSI e2e", bsi_ms, args.Q, args.R)
    _print_row(f"Torch {args.torch_dtype}", torch_ms, args.Q, args.R)
    speedup = (torch_ms / bsi_ms) if bsi_ms > 0.0 else 0.0
    print(f"speedup_vs_torch = {speedup:.4f}x")
    return {"bsi_ms": bsi_ms, "torch_ms": torch_ms, "speedup_vs_torch": speedup}


def run_model_e2e(args: argparse.Namespace) -> None:
    print("\n[Mode] model_e2e")
    this_dir = os.path.dirname(os.path.abspath(__file__))
    script = os.path.join(this_dir, "benchmark_performance_bsi.py")
    cmd = [
        sys.executable,
        script,
        "--model_name", args.model_name,
        "--datasets", args.dataset,
        "--split", args.split,
        "--num_samples", str(args.num_samples),
        "--max_seq_len", str(args.max_seq_len),
        "--decimal_places", str(args.decimal_places),
        "--compress_threshold", str(args.compress_threshold),
        "--scope", args.scope,
        "--bsi_device", args.bsi_device,
        "--bsi_profile", str(args.bsi_profile),
    ]
    env = dict(os.environ)
    env["BSI_PROFILE"] = "1" if int(args.bsi_profile) != 0 else "0"
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apples-to-apples BSI vs Torch timing (kernel_only, linear_e2e, model_e2e)"
    )
    parser.add_argument("--modes", type=str, default="kernel_only,linear_e2e",
                        help="Comma-separated modes: kernel_only,linear_e2e,model_e2e")
    parser.add_argument("--Q", type=int, default=128, help="Query rows")
    parser.add_argument("--R", type=int, default=256, help="Key rows / output features")
    parser.add_argument("--D", type=int, default=2048, help="Input features")
    parser.add_argument("--decimal_places", type=int, default=2)
    parser.add_argument("--compress_threshold", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--torch_dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--bsi_profile", type=int, default=0,
                        help="Set BSI_PROFILE env (0 recommended for apples-to-apples runtime timing)")
    parser.add_argument("--model_name", type=str, default="facebook/opt-125m")
    parser.add_argument("--dataset", type=str, default="lambada")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--scope", type=str, default="all")
    parser.add_argument("--bsi_device", type=str, default="cuda")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    os.environ["BSI_PROFILE"] = "1" if int(args.bsi_profile) != 0 else "0"
    torch_dtype = _torch_dtype_from_flag(args.torch_dtype)
    modes = _parse_modes(args.modes)
    allowed = {"kernel_only", "linear_e2e", "model_e2e"}
    invalid = [m for m in modes if m not in allowed]
    if invalid:
        raise ValueError(f"Unsupported mode(s): {invalid}. Allowed: {sorted(allowed)}")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    k_cpu = torch.randn(args.R, args.D, dtype=torch.float32, device="cpu")
    q_fp32 = torch.randn(args.Q, args.D, dtype=torch.float32, device=device)

    print("[Config]")
    print(f"  modes={modes}")
    print(f"  shape=(Q={args.Q}, R={args.R}, D={args.D})")
    print(f"  decimal_places={args.decimal_places}  compress_threshold={args.compress_threshold}")
    print(f"  warmup={args.warmup}  iters={args.iters}")
    print(f"  torch_dtype={args.torch_dtype}  BSI_PROFILE={os.environ.get('BSI_PROFILE', '0')}")

    results: Dict[str, Dict[str, float]] = {}
    if "kernel_only" in modes:
        results["kernel_only"] = run_kernel_only(args, k_cpu, q_fp32, torch_dtype)
    if "linear_e2e" in modes:
        results["linear_e2e"] = run_linear_e2e(args, k_cpu, q_fp32, torch_dtype)
    if "model_e2e" in modes:
        run_model_e2e(args)

    print("\n[Summary]")
    for mode in modes:
        if mode in results:
            row = results[mode]
            print(
                f"  {mode:12s}  bsi_ms={row['bsi_ms']:.4f}  "
                f"torch_ms={row['torch_ms']:.4f}  speedup_vs_torch={row['speedup_vs_torch']:.4f}x"
            )
        elif mode == "model_e2e":
            print(f"  {mode:12s}  completed (see benchmark_performance_bsi.py output above)")


if __name__ == "__main__":
    main()
