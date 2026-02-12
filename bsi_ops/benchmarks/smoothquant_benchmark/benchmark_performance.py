import argparse
import gc
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, OPTForCausalLM


THIS_FILE = Path(__file__).resolve()
BENCH_ROOT = THIS_FILE.parents[1]
if str(BENCH_ROOT) not in sys.path:
    sys.path.append(str(BENCH_ROOT))

from benchmark_performance_bsi import Evaluator, get_device, normalize_model_name  # noqa: E402
try:
    from smoothquant.opt import Int8OPTForCausalLM
except Exception:  # pragma: no cover - optional dependency
    Int8OPTForCausalLM = None


def _auto_smoothquant_name(model_name: str) -> str:
    # facebook/opt-1.3b -> mit-han-lab/opt-1.3b-smoothquant
    base = model_name.split("/")[-1]
    return f"mit-han-lab/{base}-smoothquant"


def _load_baseline(model_name: str, device: str, base_dtype: str):
    dtype = torch.float16 if (device == "cuda" and base_dtype == "fp16") else torch.float32
    device_map = "auto" if device == "cuda" else "cpu"
    return OPTForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map=device_map)


def _print_result(tag: str, top1: float, top5: float, fwd_ms: float, dot_ms: float, dot_q_us: float, dot_s_ns: float, peak_mb: float):
    print(
        f"-> {tag}: top1={top1:.4f}, top5={top5:.4f}, avg_fwd={fwd_ms:.3f}ms, "
        f"dot={dot_ms:.3f}ms, dot_q={dot_q_us:.3f}us, dot_s={dot_s_ns:.3f}ns, peak_mem={peak_mb:.2f}MB"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Apples-to-apples FP baseline vs SmoothQuant benchmark")
    p.add_argument("--model_name", type=str, default="facebook/opt-1.3b")
    p.add_argument("--smoothquant_model_name", type=str, default="",
                   help="HF model id for SmoothQuant INT8. Default auto-maps from --model_name.")
    p.add_argument("--dataset", type=str, default="lambada")
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--base_dtype", type=str, choices=["fp16", "fp32"], default="fp16")
    p.add_argument("--skip_baseline", action="store_true")
    p.add_argument("--run_bsi", action="store_true", help="Also run BSI with the same evaluator path")
    p.add_argument("--decimal_places", type=int, default=2)
    p.add_argument("--compress_threshold", type=float, default=0.5)
    p.add_argument("--scope", type=str, choices=["all", "attention", "mlp"], default="all")
    p.add_argument("--bsi_device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--key_storage_mode", type=str, choices=["verbatim", "hybrid"], default="verbatim")
    args = p.parse_args()

    model_name = normalize_model_name(args.model_name)
    sq_model_name = args.smoothquant_model_name.strip() or _auto_smoothquant_name(model_name)
    device = get_device() if args.device == "auto" else args.device

    split_expr = f"{args.split}[:{args.num_samples}]" if args.num_samples > 0 else args.split
    print(f"Dataset: {args.dataset} | Split: {split_expr} | Device: {device}")
    print(f"Baseline model: {model_name}")
    print(f"SmoothQuant model: {sq_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset(args.dataset, split=split_expr)
    evaluator = Evaluator(dataset, tokenizer, device=device, max_seq_len=args.max_seq_len)

    if not args.skip_baseline:
        print(f"\n--- Benchmarking FP baseline: {model_name} ---")
        model_fp = _load_baseline(model_name, device=device, base_dtype=args.base_dtype)
        (
            acc_fp,
            top5_fp,
            latency_fp,
            mem_fp,
            dot_fp_ms,
            dot_fp_q_us,
            dot_fp_s_ns,
        ) = evaluator.evaluate(model_fp, scope="all")
        _print_result("FP baseline", acc_fp, top5_fp, latency_fp, dot_fp_ms, dot_fp_q_us, dot_fp_s_ns, mem_fp)
        del model_fp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n--- Benchmarking SmoothQuant INT8: {sq_model_name} ---")
    if Int8OPTForCausalLM is None:
        raise RuntimeError(
            "smoothquant package is unavailable. Install SmoothQuant dependencies before running this benchmark."
        )

    model_sq = Int8OPTForCausalLM.from_pretrained(
        sq_model_name,
        device_map="auto" if device == "cuda" else "cpu",
    )
    (
        acc_sq,
        top5_sq,
        latency_sq,
        mem_sq,
        dot_sq_ms,
        dot_sq_q_us,
        dot_sq_s_ns,
    ) = evaluator.evaluate(model_sq, scope="all")
    _print_result("SmoothQuant INT8", acc_sq, top5_sq, latency_sq, dot_sq_ms, dot_sq_q_us, dot_sq_s_ns, mem_sq)

    if args.run_bsi:
        print("\n--- Benchmarking BSI (same evaluator setup) ---")
        model_bsi = OPTForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if args.base_dtype == "fp16" else torch.float32,
            device_map="cpu",
        )
        (
            acc_bsi,
            fwd_bsi,
            build_bsi,
            dot_bsi,
            dot_q_bsi,
            dot_s_bsi,
            top5_bsi,
            _summary,
            _layer_stats,
        ) = evaluator.evaluate_with_bsi(
            model_bsi,
            decimal_cfg=args.decimal_places,
            scope=args.scope,
            threshold_cfg=float(args.compress_threshold),
            bsi_device=args.bsi_device,
            key_storage_mode=args.key_storage_mode,
        )
        print(
            f"-> BSI: top1={acc_bsi:.4f}, top5={top5_bsi:.4f}, avg_fwd={fwd_bsi:.3f}ms, "
            f"build={build_bsi:.3f}ms, dot={dot_bsi:.3f}ms, dot_q={dot_q_bsi:.3f}us, dot_s={dot_s_bsi:.3f}ns"
        )


if __name__ == "__main__":
    main()
