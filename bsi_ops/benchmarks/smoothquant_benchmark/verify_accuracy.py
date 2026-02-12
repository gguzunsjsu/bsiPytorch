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
    from smoothquant.smooth import smooth_lm
    from smoothquant.fake_quant import quantize_model
except Exception:  # pragma: no cover - optional dependency
    smooth_lm = None
    quantize_model = None

try:
    from smoothquant.opt import Int8OPTForCausalLM
except Exception:  # pragma: no cover - optional dependency
    Int8OPTForCausalLM = None


def _load_fp_model(model_name: str, device: str):
    return OPTForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else "cpu",
    )


def _print_acc(tag: str, top1: float, top5: float):
    print(f"-> {tag}: top1={top1:.4f}, top5={top5:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Apples-to-apples SmoothQuant accuracy checks")
    p.add_argument("--model_name", type=str, default="facebook/opt-1.3b")
    p.add_argument("--smoothquant_int8_model_name", type=str, default="",
                   help="Optional HF id for pre-exported SmoothQuant INT8 checkpoint")
    p.add_argument("--act_scales", type=str, default="",
                   help="Optional activation scales path for smooth_lm + fake quant flow")
    p.add_argument("--dataset", type=str, default="lambada")
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--run_naive_w8a8", action="store_true",
                   help="Run fake-quant naive W8A8 flow using smoothquant.fake_quant.quantize_model")
    p.add_argument("--run_smooth_w8a8", action="store_true",
                   help="Run smooth_lm + fake quant W8A8 flow (requires --act_scales)")
    args = p.parse_args()

    model_name = normalize_model_name(args.model_name)
    device = get_device() if args.device == "auto" else args.device
    split_expr = f"{args.split}[:{args.num_samples}]" if args.num_samples > 0 else args.split

    print(f"Dataset: {args.dataset} | Split: {split_expr} | Device: {device}")
    print(f"Model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset(args.dataset, split=split_expr)
    evaluator = Evaluator(dataset, tokenizer, device=device, max_seq_len=args.max_seq_len)

    print("\n--- FP baseline ---")
    model_fp = _load_fp_model(model_name, device)
    top1_fp, top5_fp, *_ = evaluator.evaluate(model_fp, scope="all")
    _print_acc("FP baseline", top1_fp, top5_fp)
    del model_fp
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.run_naive_w8a8:
        if quantize_model is None:
            raise RuntimeError("smoothquant.fake_quant is unavailable.")
        print("\n--- Naive W8A8 (fake quant) ---")
        model_naive = _load_fp_model(model_name, device)
        quantize_model(model_naive)
        top1_n, top5_n, *_ = evaluator.evaluate(model_naive, scope="all")
        _print_acc("Naive W8A8", top1_n, top5_n)
        del model_naive
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.run_smooth_w8a8:
        if smooth_lm is None or quantize_model is None:
            raise RuntimeError("smoothquant.smooth/fake_quant is unavailable.")
        if not args.act_scales:
            raise ValueError("--run_smooth_w8a8 requires --act_scales")
        print("\n--- SmoothQuant W8A8 (smooth_lm + fake quant) ---")
        model_smooth = _load_fp_model(model_name, device)
        act_scales = torch.load(args.act_scales)
        smooth_lm(model_smooth, act_scales, 0.5)
        model_smooth_w8a8 = quantize_model(model_smooth)
        top1_s, top5_s, *_ = evaluator.evaluate(model_smooth_w8a8, scope="all")
        _print_acc("SmoothQuant W8A8", top1_s, top5_s)
        del model_smooth
        del model_smooth_w8a8
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.smoothquant_int8_model_name.strip():
        if Int8OPTForCausalLM is None:
            raise RuntimeError("smoothquant.opt.Int8OPTForCausalLM is unavailable.")
        print(f"\n--- SmoothQuant INT8 checkpoint ({args.smoothquant_int8_model_name}) ---")
        model_int8 = Int8OPTForCausalLM.from_pretrained(
            args.smoothquant_int8_model_name.strip(),
            device_map="auto" if device == "cuda" else "cpu",
        )
        top1_i8, top5_i8, *_ = evaluator.evaluate(model_int8, scope="all")
        _print_acc("SmoothQuant INT8 checkpoint", top1_i8, top5_i8)


if __name__ == "__main__":
    main()
