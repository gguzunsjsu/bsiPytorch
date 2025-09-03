import os
import json
import argparse
import time

import torch
from transformers import AutoTokenizer, OPTForCausalLM
from torch.nn.functional import pad

try:
    from .verify_accuracy_bsi import quantize_model_bsi, summarize_bsi_model
except ImportError:
    from verify_accuracy_bsi import quantize_model_bsi, summarize_bsi_model


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


class Evaluator:
    def __init__(self, dataset, tokenizer, max_seq_len=512):
        self.dataset = dataset.map(lambda ex: tokenizer(ex['text']), batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])
        self.max_seq_len = max_seq_len
        self.num_samples = len(self.dataset)

    @torch.no_grad()
    def evaluate_cpu(self, model):
        model.eval()
        total, hit = 0, 0
        latency_ms = 0.0

        device = next(model.parameters()).device
        assert str(device) == 'cpu'

        for batch in self.dataset:
            input_ids = batch['input_ids'].to(device).unsqueeze(0)
            label = input_ids[:, -1]
            pad_len = max(0, self.max_seq_len - input_ids.shape[1])
            input_ids = pad(input_ids, (0, pad_len), value=1)

            t0 = time.perf_counter()
            outputs = model(input_ids)
            t1 = time.perf_counter()
            latency_ms += (t1 - t0) * 1000

            last_token_logits = outputs.logits[:, -2 - pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()

        accuracy = hit / total if total else 0.0
        avg_latency = latency_ms / max(1, self.num_samples)
        return accuracy, avg_latency


def main():
    parser = argparse.ArgumentParser(description='Load a dense checkpoint + BSI spec, rebuild BSI keys, and optionally evaluate')
    parser.add_argument('--spec', type=str, required=True, help='Path to bsi_spec.json from export_bsi_checkpoint.py')
    parser.add_argument('--weights', type=str, default=None, help='Optional override to a state_dict .pt path')
    parser.add_argument('--num_samples', type=int, default=500, help='Validation samples (LAMBADA)')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Max sequence length for padding')
    parser.add_argument('--evaluate', action='store_true', help='Run a quick CPU evaluation to verify load')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        spec = json.load(f)

    model_name = spec['model_name']
    pf = int(spec['precision_factor'])
    scope = spec['scope']
    sd_path = args.weights or spec.get('dense_state_dict_path')

    print(f"Loading base model: {model_name}")
    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map='cpu')

    if sd_path and os.path.isfile(sd_path):
        print(f"Loading dense state_dict from {sd_path}")
        sd = torch.load(sd_path, map_location='cpu')
        model.load_state_dict(sd, strict=False)

    print(f"Quantizing with BSI (pf={pf}, scope={scope})...")
    model = quantize_model_bsi(model, precision_factor=pf, scope=scope)
    summary = summarize_bsi_model(model)
    to_mb = lambda b: b / (1024**2)
    print("BSI storage summary:")
    print(f"  BSI weights: {to_mb(summary['total_bsi_bytes']):.2f} MB | Bias: {to_mb(summary['total_bias_bytes']):.2f} MB | Total: {to_mb(summary['total_static_bytes']):.2f} MB")

    if args.evaluate:
        from datasets import load_dataset
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        dataset = load_dataset('lambada', split=f'validation[:{args.num_samples}]')
        evaluator = Evaluator(dataset, tokenizer, max_seq_len=args.max_seq_len)
        acc, lat = evaluator.evaluate_cpu(model)
        print(f"Accuracy: {acc:.4f} | Avg latency: {lat:.3f} ms")


if __name__ == '__main__':
    main()
