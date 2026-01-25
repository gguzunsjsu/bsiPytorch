import argparse
from collections import Counter, defaultdict

import torch
from transformers import AutoModelForCausalLM

import bsi_ops
from bsi_ops.benchmarks.verify_accuracy_bsi import BSIQuantizedLinear
from bsi_ops.benchmarks.benchmark_performance_bsi import quantize_model_bsi


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect BSI shape stats (Sb/W) per layer")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--decimal_places", type=int, default=2)
    parser.add_argument("--compress_threshold", type=float, default=0.5)
    parser.add_argument("--scope", type=str, default="all")
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()

    # Quantize in-place to BSI
    model = quantize_model_bsi(
        model,
        decimalPlaces=args.decimal_places,
        scope=args.scope,
        compress_threshold=args.compress_threshold,
    )

    sb_hist = Counter()
    w_hist = Counter()
    layer_rows = []

    for name, module in model.named_modules():
        if isinstance(module, BSIQuantizedLinear):
            stats = bsi_ops.bsi_keys_cuda_stats(module._bsi_keys_cuda)
            w = int(stats["W"])
            w_hist[w] += 1
            sb_counts = stats["Sb_counts"]
            for sb, cnt in sb_counts.items():
                sb_hist[int(sb)] += int(cnt)
            layer_rows.append((name, w, dict(sb_counts)))

    print("=== W (words per slice) distribution across layers ===")
    for w, cnt in sorted(w_hist.items()):
        print(f"W={w}: {cnt} layers")

    print("\n=== Sb (slices per key) distribution across all keys ===")
    for sb, cnt in sorted(sb_hist.items()):
        print(f"Sb={sb}: {cnt} keys")

    print("\n=== Per-layer breakdown (name, W, Sb_counts) ===")
    for name, w, sb_counts in layer_rows:
        print(f"{name}: W={w}, Sb_counts={sb_counts}")


if __name__ == "__main__":
    main()
