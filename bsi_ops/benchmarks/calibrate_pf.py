#!/usr/bin/env python3
import argparse
import math
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import bsi_ops

def to_cpu_contig(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to("cpu").contiguous()

def main():
    parser = argparse.ArgumentParser(description="Calibrate precision_factor (pf) for BSI dot product.")
    parser.add_argument("--pickle", type=str,
                        default="../extract_tensors/Weight_Processing/bert_imdb_pickle_store/bert_imdb45.pkl")
    parser.add_argument("--pf_list", type=str, default="31,63,127,255",
                        help="Comma-separated list of pf values.")
    parser.add_argument("--max_layers", type=int, default=10, help="Limit layers for speed.")
    parser.add_argument("--eps", type=float, default=1e-3, help="Relative error threshold.")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize Q and K by their max-abs, then rescale result.")
    parser.add_argument("--out_csv", type=str, default="results/bsi_pf_calibration.csv")
    args = parser.parse_args()

    pfs = [float(s) for s in args.pf_list.split(",")]
    with open(args.pickle, "rb") as f:
        triplets = pickle.load(f)

    rows = []
    for pf in pfs:
        hits, total = 0, 0
        rel_errs = []
        for idx, (Q, K, _) in enumerate(triplets, start=1):
            if idx > args.max_layers:
                break
            Q_flat = torch.tensor(np.asarray(Q).reshape(-1), dtype=torch.float32)
            K_flat = torch.tensor(np.asarray(K).reshape(-1), dtype=torch.float32)

            # Optional normalization
            scale_q = float(Q_flat.abs().max().item()) or 1.0
            scale_k = float(K_flat.abs().max().item()) or 1.0
            if args.normalize:
                Q_in = (Q_flat / scale_q).to(torch.float32)
                K_in = (K_flat / scale_k).to(torch.float32)
            else:
                Q_in, K_in = Q_flat, K_flat

            # BSI on CPU
            q_cpu = to_cpu_contig(Q_in)
            k_cpu = to_cpu_contig(K_in)
            bsi_res, _, _, _ = bsi_ops.dot_product(q_cpu, k_cpu, pf)
            if args.normalize:
                bsi_res *= (scale_q * scale_k)

            torch_res = torch.dot(Q_flat, K_flat).item()

            denom = max(abs(torch_res), 1e-12)
            rel_err = abs(bsi_res - torch_res) / denom
            rel_errs.append(rel_err)
            hits += int(rel_err < args.eps)
            total += 1

        acc = hits / max(total, 1)
        med_rel_err = float(np.median(rel_errs)) if rel_errs else float("nan")
        rows.append({"pf": pf, "normalize": int(args.normalize), "eps": args.eps,
                     "layers": total, "acc_rel_err<eps": acc, "median_rel_err": med_rel_err})

    df = pd.DataFrame(rows)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    best = df.sort_values(by=["acc_rel_err<eps", "median_rel_err"], ascending=[False, True]).head(1)
    if not best.empty:
        bpf = best.iloc[0]
        print(f"\nSuggested pf: {bpf['pf']} (normalize={bool(bpf['normalize'])}) -> "
              f"acc={bpf['acc_rel_err<eps']:.4f}, median_rel_err={bpf['median_rel_err']:.2e}")
    print(f"Saved to {out}")

if __name__ == "__main__":
    main()