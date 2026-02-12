#!/usr/bin/env python
"""
inspect_opt_rounding.py  --decimal 2

Loads facebook/opt-125m, runs a short prompt through the first decoder layer
and compares the exact q·k dot with the dot after rounding both operands to N
decimal places.
"""

import argparse
import torch
from transformers import AutoTokenizer, OPTForCausalLM

def main(decimal_places: int, text: str, seed: int) -> None:
    torch.manual_seed(seed)

    model_name = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = OPTForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    layer = model.model.decoder.layers[0]
    q_proj = layer.self_attn.q_proj
    k_proj = layer.self_attn.k_proj

    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded["input_ids"]
    embeddings = model.model.decoder.embed_tokens(input_ids)
    inputs = embeddings[0, :-1, :]          # drop the causal target token
    q_vec = q_proj(inputs)[0]               # query vector for position 0
    k_weights = k_proj.weight.detach()      # full key weight matrix

    exact = torch.matmul(k_weights, q_vec)

    scale = 10 ** decimal_places
    q_round = torch.round(q_vec * scale) / scale
    k_round = torch.round(k_weights * scale) / scale
    rounded = torch.matmul(k_round, q_round)

    diff = (exact - rounded).abs()
    print(f"\n=== decimal_places = {decimal_places} ===")
    print(f"input stats : min {inputs.min():.6f}, max {inputs.max():.6f}, mean {inputs.mean():.6f}")
    print(f"q stats     : min {q_vec.min():.6f}, max {q_vec.max():.6f}, mean {q_vec.mean():.6f}")
    print(f"k weights   : min {k_weights.min():.6f}, max {k_weights.max():.6f}, mean {k_weights.mean():.6f}")
    print(f"exact dot   : min {exact.min():.6f},   max {exact.max():.6f},   mean {exact.mean():.6f}")
    print(f"rounded dot : min {rounded.min():.6f}, max {rounded.max():.6f}, mean {rounded.mean():.6f}")
    print(f"mean |diff| : {diff.mean():.6f}, max |diff|: {diff.max():.6f}")

    sample_idx = torch.arange(5)
    print(" exact[:5]   :", exact[sample_idx].tolist())
    print(" rounded[:5] :", rounded[sample_idx].tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--decimal", type=int, default=2, help="decimal places for rounding")
    parser.add_argument("--text", type=str, default="Hello world", help="prompt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.decimal, args.text, args.seed)
