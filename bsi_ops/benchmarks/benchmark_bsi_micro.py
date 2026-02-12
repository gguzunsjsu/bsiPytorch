#!/usr/bin/env python3
import argparse
import math
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import bsi_ops

def raw_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()

def to_cpu_contig(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to("cpu").contiguous()

def main():
    parser = argparse.ArgumentParser(description="BSI vs Torch dot micro-benchmark (GPU baseline, CPU BSI).")
    parser.add_argument("--pickle", type=str,
                        default="../extract_tensors/Weight_Processing/bert_imdb_pickle_store/bert_imdb45.pkl")
    parser.add_argument("--decimal_places", type=int, default=2, help="Number of decimal places for BSI encoding.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Device for Torch baseline; BSI always runs on CPU.")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize Q,K by their max-abs and rescale result.")
    parser.add_argument("--max_layers", type=int, default=50, help="Limit for speed.")
    parser.add_argument("--out_csv", type=str, default="bsi_ops/results/bsi_micro_benchmark_gpu.csv")
    args = parser.parse_args()

    use_cuda = (args.device == "cuda") and torch.cuda.is_available()
    print(f"Torch baseline device: {'cuda' if use_cuda else 'cpu'}; BSI device: cpu")
    scale_factor = 10 ** args.decimal_places

    with open(args.pickle, "rb") as f:
        triplets = pickle.load(f)

    rows = []
    acc_hits = 0
    total_pairs = 0
    bsi_lat_total_ms = 0.0
    torch_lat_total_ms = 0.0

    for layer_idx, (Q, K, _) in enumerate(triplets, start=1):
        if layer_idx > args.max_layers:
            break

        Q_flat = torch.tensor(np.asarray(Q).reshape(-1), dtype=torch.float32)
        K_flat = torch.tensor(np.asarray(K).reshape(-1), dtype=torch.float32)

        scale_q = float(Q_flat.abs().max().item()) or 1.0
        scale_k = float(K_flat.abs().max().item()) or 1.0
        if args.normalize:
            Q_in = (Q_flat / scale_q)
            K_in = (K_flat / scale_k)
        else:
            Q_in, K_in = Q_flat, K_flat

        q_tensor_bytes = raw_bytes(Q_flat)
        k_tensor_bytes = raw_bytes(K_flat)

        bsi_ms_list = []
        torch_ms_list = []
        bsi_res = None
        torch_res = None
        bsi_q_bytes = None
        bsi_k_bytes = None

        for _ in range(args.runs):
            q_cpu = to_cpu_contig(Q_in)
            k_cpu = to_cpu_contig(K_in)
            res, time_ns, q_b, k_b = bsi_ops.dot_product_decimal(q_cpu, k_cpu, args.decimal_places)
            if args.normalize:
                res *= (scale_q * scale_k)
            bsi_ms_list.append(time_ns / 1e6)
            bsi_res = res
            bsi_q_bytes = q_b
            bsi_k_bytes = k_b

            if use_cuda:
                Q_dev = Q_flat.to("cuda", non_blocking=True)
                K_dev = K_flat.to("cuda", non_blocking=True)
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                t_res = torch.dot(Q_dev, K_dev).item()
                end.record()
                torch.cuda.synchronize()
                torch_ms = start.elapsed_time(end)
            else:
                t0 = time.perf_counter_ns()
                t_res = torch.dot(Q_flat, K_flat).item()
                t1 = time.perf_counter_ns()
                torch_ms = (t1 - t0) / 1e6

            torch_ms_list.append(torch_ms)
            torch_res = t_res

        denom = max(abs(torch_res), 1e-12)
        rel_err = abs(bsi_res - torch_res) / denom
        accurate = int(rel_err < args.eps)

        total_pairs += 1
        acc_hits += accurate

        bsi_avg = float(np.mean(bsi_ms_list))
        torch_avg = float(np.mean(torch_ms_list))
        bsi_lat_total_ms += bsi_avg
        torch_lat_total_ms += torch_avg

        rows.append({
            "layer": layer_idx,
            "decimal_places": args.decimal_places,
            "normalize": int(args.normalize),
            "scale_factor": scale_factor,
            "bsi_result": bsi_res,
            "torch_result": torch_res,
            "rel_err": rel_err,
            "accurate_eps": args.eps,
            "accurate_flag": accurate,
            "bsi_latency_ms": bsi_avg,
            "torch_latency_ms": torch_avg,
            "bsi_q_mb": (bsi_q_bytes or 0) / (1024 ** 2),
            "bsi_k_mb": (bsi_k_bytes or 0) / (1024 ** 2),
            "tensor_q_mb": q_tensor_bytes / (1024 ** 2),
            "tensor_k_mb": k_tensor_bytes / (1024 ** 2),
        })

    df = pd.DataFrame(rows)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    avg_acc = acc_hits / max(total_pairs, 1)
    avg_bsi_ms = bsi_lat_total_ms / max(total_pairs, 1)
    avg_torch_ms = torch_lat_total_ms / max(total_pairs, 1)
    avg_bsi_mem_mb = float(df[["bsi_q_mb", "bsi_k_mb"]].sum(axis=1).mean()) if not df.empty else 0.0
    avg_tensor_mem_mb = float(df[["tensor_q_mb", "tensor_k_mb"]].sum(axis=1).mean()) if not df.empty else 0.0

    print("\n--- Benchmarking BSI micro (GPU baseline) ---")
    print(f"Pickle: {args.pickle}")
    print(f"decimal_places={args.decimal_places} (scale=10^{args.decimal_places}), runs={args.runs}, eps={args.eps:g}, normalize={args.normalize}")
    print(f"-> BSI Accuracy (rel_err<eps): {avg_acc:.4f}, Avg Latency: {avg_bsi_ms:.3f}ms, Avg BSI Mem (Q+K): {avg_bsi_mem_mb:.2f}MB")
    print(f"-> Torch baseline Avg Latency: {avg_torch_ms:.3f}ms, Avg Tensor Mem (Q+K): {avg_tensor_mem_mb:.2f}MB")
    print(f"Per-layer results saved to: {out}")

if __name__ == "__main__":
    main()
