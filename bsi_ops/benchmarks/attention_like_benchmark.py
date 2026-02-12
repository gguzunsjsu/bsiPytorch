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


def main():
    parser = argparse.ArgumentParser(description="Attention-like benchmark (one query vs many keys).")
    parser.add_argument("--keys", type=int, default=64)
    parser.add_argument("--pf", type=int, default=3)
    parser.add_argument("--pickle", type=str,
                        default="../extract_tensors/Weight_Processing/bert_imdb_pickle_store/bert_imdb45.pkl")
    parser.add_argument("--eps", type=float, default=1e-2)
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    with open(args.pickle, "rb") as f:
        triplets = pickle.load(f)
    Q0, K0, _ = triplets[0]
    d = int(np.asarray(Q0).reshape(-1).shape[0])

    rng = np.random.default_rng(1234)
    q = torch.tensor(rng.standard_normal(d).astype(np.float32))
    K = torch.tensor(rng.standard_normal((args.keys, d)).astype(np.float32))

    if args.normalize:
        sq = float(q.abs().max().item()) or 1.0
        sk = float(K.abs().max().item()) or 1.0
        q_in = q / sq
        K_in = K / sk
    else:
        sq = sk = 1.0
        q_in, K_in = q, K

    # Torch baseline (CPU for simplicity here)
    t0 = time.perf_counter_ns()
    baseline = (K @ q).to(torch.float64)
    t1 = time.perf_counter_ns()
    torch_ms = (t1 - t0) / 1e6

    # BSI batch (CPU)
    scores, ns, mem_q, mem_k = bsi_ops.batch_dot_product(q_in, K_in, float(args.pf))
    scores = scores * (sq * sk)
    bsi_ms = ns / 1e6

    denom = torch.maximum(baseline.abs(), torch.tensor(1e-12, dtype=baseline.dtype))
    rel_err = (scores - baseline).abs() / denom
    acc = float((rel_err < args.eps).to(torch.float32).mean().item())

    sc = scores / math.sqrt(d)
    bl = baseline / math.sqrt(d)
    top1_agree = int(sc.argmax().item() == bl.argmax().item())

    out = {
        "keys": args.keys, "pf": args.pf, "eps": args.eps, "normalize": int(args.normalize),
        "acc_rel_err<eps": acc, "top1_agree": top1_agree,
        "bsi_ms": bsi_ms, "torch_ms": torch_ms,
        "bsi_mem_k_mb": mem_k / 1e6, "bsi_mem_q_mb": mem_q / 1e6,
    }
    print("--- Attention-like (one query vs many keys) ---")
    print(f"-> BSI acc(rel_err<eps): {acc:.4f}, Lat: {bsi_ms:.3f}ms, "
          f"Mem(Q+K): {(mem_q + mem_k) / 1e6:.2f}MB")
    print(f"-> Torch Lat: {torch_ms:.3f}ms, Top-1 agree: {top1_agree}")

    df = pd.DataFrame([out])
    out_path = Path("results/attention_like_benchmark.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, mode="a", header=not out_path.exists())
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()