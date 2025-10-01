import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, OPTForCausalLM
import gc
from torch.nn.functional import pad
from datasets import load_dataset
import time
from verify_accuracy_bsi import (
    quantize_model_bsi, summarize_bsi_model,
    reset_bsi_dot_counters, sum_bsi_dot_counters,
    enable_bsi_error_stats, collect_bsi_error_stats,
    print_compression_summary
)
import argparse
from tqdm import tqdm

def get_device():
    """Auto-detect best available device"""
    if torch.cuda.is_available():
        return 'cuda' 
    else:
        return 'cpu'

class Evaluator:
    def __init__(self, dataset, tokenizer, device=None, max_seq_len=512, layer_stats_batches=0):
        if device is None:
            device = get_device()
        print(f"Using device: {device}")
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_len = max_seq_len
        self.dataset = self.dataset.map(self.tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])
        self.num_samples = len(self.dataset)
        self.layer_stats_batches = max(0, layer_stats_batches)
        # Choose a safe pad id for right-padding
        self.pad_id = getattr(self.tokenizer, 'pad_token_id', None)
        if self.pad_id is None:
            self.pad_id = getattr(self.tokenizer, 'eos_token_id', None)
        if self.pad_id is None:
            self.pad_id = 1

    def tokenize_function(self, examples):
        return self.tokenizer(examples['text'])

    @torch.no_grad()
    def evaluate_with_bsi(self, model, decimal_cfg=2, scope='all', threshold_cfg=0.2):
        """Evaluate model with actual BSI quantization on CPU"""
        model.eval()
        total, hit = 0, 0
        latency = 0

        print(f"Quantizing model with BSI (decimal_places={decimal_cfg}, scope={scope})...")
        model = quantize_model_bsi(model, decimalPlaces=decimal_cfg, scope=scope, compress_threshold=threshold_cfg)
        summary = summarize_bsi_model(model)
        reset_bsi_dot_counters(model)
        if summary["total_bsi_bytes"] == 0:
            print("Warning: BSI weight memory not found on modules. Did quantization run?")

        # Always run BSI model on CPU; move inputs to model's device
        model_device = next(model.parameters()).device
        assert str(model_device) == 'cpu', "BSI model should be on CPU for evaluation"

        # Enable layer-wise error stats for first N batches if requested.
        if self.layer_stats_batches > 0:
            enable_bsi_error_stats(model, True)
        else:
            enable_bsi_error_stats(model, False)

        print("Running evaluation...")
        dot_ns_prev = sum_bsi_dot_counters(model)
        top5_hit = 0
        if tqdm is not None:
            pbar = tqdm(total=self.num_samples, desc="BSI eval", dynamic_ncols=True)
            for batch_idx, batch in enumerate(self.dataset):
                input_ids = batch['input_ids'].to(model_device).unsqueeze(0)
                label = input_ids[:, -1]
                # Truncate or pad to max_seq_len
                if input_ids.shape[1] > self.max_seq_len:
                    input_ids = input_ids[:, -self.max_seq_len:]
                    pad_len = 0
                else:
                    pad_len = max(0, self.max_seq_len - input_ids.shape[1])
                    input_ids = pad(input_ids, (0, pad_len), value=self.pad_id)

                start_time = time.perf_counter()
                outputs = model(input_ids)
                end_time = time.perf_counter()
                batch_fwd_ms = (end_time - start_time) * 1000
                latency += batch_fwd_ms

                last_token_logits = outputs.logits[:, -2 - pad_len, :]
                pred = last_token_logits.argmax(dim=-1)
                total += label.size(0)
                hit += (pred == label).sum().item()
                # top-5 accuracy accumulation
                topk = torch.topk(last_token_logits, k=5, dim=-1)
                top_ids = [int(x) for x in topk.indices[0].tolist()]
                if int(label.item()) in top_ids:
                    top5_hit += 1

                if batch_idx < 3:
                    label_id = int(label.item())
                    pred_id = int(pred.item())
                    label_tok = self.tokenizer.convert_ids_to_tokens([label_id])[0] if hasattr(self.tokenizer, 'convert_ids_to_tokens') else str(label_id)
                    top_vals = [float(x) for x in topk.values[0].tolist()]
                    top_toks = self.tokenizer.convert_ids_to_tokens(top_ids) if hasattr(self.tokenizer, 'convert_ids_to_tokens') else list(map(str, top_ids))
                    print(
                        f"  dbg batch={batch_idx} len={int(input_ids.size(1) - pad_len)}, pad_len={int(pad_len)}, "
                        f"label_id={label_id}({label_tok}), pred_id={pred_id}, match={bool(label_id==pred_id)}"
                    )
                    print("    top5:", list(zip(top_ids, top_toks, [f"{v:.3f}" for v in top_vals])))

                dot_ns_cur = sum_bsi_dot_counters(model)
                batch_dot_ms = (dot_ns_cur - dot_ns_prev) / 1e6
                dot_ns_prev = dot_ns_cur

                if self.layer_stats_batches and (batch_idx + 1) == self.layer_stats_batches:
                    enable_bsi_error_stats(model, False)

                pbar.update(1)
                pbar.set_postfix(fwd_ms=f"{batch_fwd_ms:.1f}", dot_ms=f"{batch_dot_ms:.1f}")
            pbar.close()
        else:
            top5_hit = 0
            for batch_idx, batch in enumerate(self.dataset):
                input_ids = batch['input_ids'].to(model_device).unsqueeze(0)
                label = input_ids[:, -1]
                # Truncate or pad to max_seq_len
                if input_ids.shape[1] > self.max_seq_len:
                    input_ids = input_ids[:, -self.max_seq_len:]
                    pad_len = 0
                else:
                    pad_len = max(0, self.max_seq_len - input_ids.shape[1])
                    input_ids = pad(input_ids, (0, pad_len), value=self.pad_id)

                start_time = time.perf_counter()
                outputs = model(input_ids)
                end_time = time.perf_counter()
                batch_fwd_ms = (end_time - start_time) * 1000
                latency += batch_fwd_ms

                last_token_logits = outputs.logits[:, -2 - pad_len, :]
                pred = last_token_logits.argmax(dim=-1)
                total += label.size(0)
                hit += (pred == label).sum().item()
                # top-5 accuracy accumulation
                topk = torch.topk(last_token_logits, k=5, dim=-1)
                top_ids = [int(x) for x in topk.indices[0].tolist()]
                if int(label.item()) in top_ids:
                    top5_hit += 1

                if batch_idx < 3:
                    label_id = int(label.item())
                    pred_id = int(pred.item())
                    label_tok = self.tokenizer.convert_ids_to_tokens([label_id])[0] if hasattr(self.tokenizer, 'convert_ids_to_tokens') else str(label_id)
                    top_vals = [float(x) for x in topk.values[0].tolist()]
                    top_toks = self.tokenizer.convert_ids_to_tokens(top_ids) if hasattr(self.tokenizer, 'convert_ids_to_tokens') else list(map(str, top_ids))
                    print(
                        f"  dbg batch={batch_idx} len={int(input_ids.size(1) - pad_len)}, pad_len={int(pad_len)}, "
                        f"label_id={label_id}({label_tok}), pred_id={pred_id}, match={bool(label_id==pred_id)}"
                    )
                    print("    top5:", list(zip(top_ids, top_toks, [f"{v:.3f}" for v in top_vals])))

                dot_ns_cur = sum_bsi_dot_counters(model)
                batch_dot_ms = (dot_ns_cur - dot_ns_prev) / 1e6
                dot_ns_prev = dot_ns_cur

                print(
                    f"  Processed batch {batch_idx + 1}/{self.num_samples} "
                    f"(fwd_ms={batch_fwd_ms:.1f}, dot_ms={batch_dot_ms:.1f})"
                )

                if self.layer_stats_batches and (batch_idx + 1) == self.layer_stats_batches:
                    enable_bsi_error_stats(model, False)

        accuracy = hit / total if total else 0.0
        top5_acc = top5_hit / total if total else 0.0
        avg_forward_ms = latency / max(1, self.num_samples)
        dot_ns_total = sum_bsi_dot_counters(model)
        avg_dot_ms = (dot_ns_total / 1e6) / max(1, self.num_samples)
        layer_stats = collect_bsi_error_stats(model) if self.layer_stats_batches > 0 else []
        enable_bsi_error_stats(model, False)
        print(f"Completed BSI eval: top1_acc={accuracy:.4f}, top5_acc={top5_acc:.4f}")
        return accuracy, avg_forward_ms, avg_dot_ms, top5_acc, summary, layer_stats

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        total, hit = 0, 0
        latency = 0
        
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        for batch in self.dataset:
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            # Truncate or pad to max_seq_len
            if input_ids.shape[1] > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]
                pad_len = 0
            else:
                pad_len = max(0, self.max_seq_len - input_ids.shape[1])
                input_ids = pad(input_ids, (0, pad_len), value=self.pad_id)
            
            if self.device == 'cuda' and torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                
                start.record()
                outputs = model(input_ids)
                end.record()
                
                torch.cuda.synchronize()
                latency += start.elapsed_time(end)
            else:
                start_time = time.perf_counter()
                outputs = model(input_ids)
                end_time = time.perf_counter()
                latency += (end_time - start_time) * 1000 
            
            last_token_logits = outputs.logits[:, -2 - pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        
        accuracy = hit / total if total else 0.0
        avg_latency = latency / max(1, self.num_samples)
        peak_memory = torch.cuda.max_memory_allocated() / (1024**2) if self.device == 'cuda' and torch.cuda.is_available() else 0

        return accuracy, avg_latency, peak_memory

def print_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f'Model static size: {size_all_mb:.3f}MB')
    return param_size, buffer_size


def build_layer_config(default_value, attention_value, mlp_value):
    """Utility to construct per-scope configuration dictionaries."""
    if attention_value is None and mlp_value is None:
        return default_value
    config = {'default': default_value}
    if attention_value is not None:
        config['attention'] = attention_value
    if mlp_value is not None:
        config['mlp'] = mlp_value
    return config

def main():
    parser = argparse.ArgumentParser(description='Benchmark FP16 vs BSI (accuracy, latency, and/or static memory)')
    parser.add_argument('--model_name', type=str, default='facebook/opt-125m',
                    help='HuggingFace model name')
    parser.add_argument('--decimal_places', type=int, nargs='+', default=[2],
                    help='Decimal-place settings to test for BSI')
    parser.add_argument('--skip_baseline', action='store_true',
                    help='Skip evaluating FP baseline (useful on CPU)')
    parser.add_argument('--decimal_attention', type=int, default=None,
                    help='Optional decimal-place override for attention layers')
    parser.add_argument('--decimal_mlp', type=int, default=None,
                    help='Optional decimal-place override for MLP/FFN layers')
    parser.add_argument('--compress_threshold', type=float, default=0.2,
                    help='Base compression threshold for BSI slices (0 disables compression)')
    parser.add_argument('--threshold_attention', type=float, default=None,
                    help='Compression threshold override for attention layers')
    parser.add_argument('--threshold_mlp', type=float, default=None,
                    help='Compression threshold override for MLP/FFN layers')
    parser.add_argument('--layer_stats_batches', type=int, default=0,
                    help='Collect per-layer error metrics for the first N batches (0 disables)')

    parser.add_argument('--memory_only', action='store_true', help='Only compute and print static BSI weights memory (no evaluation)')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of validation samples from LAMBADA')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Max sequence length (padding/truncation length)')
    parser.add_argument('--scope', type=str, default='all', choices=['all', 'attention', 'mlp'], help='Layers to quantize with BSI')
    args = parser.parse_args()

    fp16_model_name = args.model_name
    device = get_device()
    print(f"Auto-detected device: {device}")

    if not args.memory_only:
        print("Initializing tokenizer and dataset...")
        tokenizer = AutoTokenizer.from_pretrained(fp16_model_name)
        print(f"Tokenizer pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")
        print("Note: Right-padding uses tokenizer.pad_token_id when available; fallback is eos or 1.")
        dataset = load_dataset('lambada', split=f'validation[:{args.num_samples}]')
        evaluator = Evaluator(dataset, tokenizer, device=device, max_seq_len=args.max_seq_len, layer_stats_batches=args.layer_stats_batches)
    
    print(f"\n--- Benchmarking FP baseline: {fp16_model_name} ---")
    if device == 'cuda':
        model_fp16 = OPTForCausalLM.from_pretrained(
            fp16_model_name,
            torch_dtype=torch.float16,
            device_map='auto'
        )
    else:
        model_fp16 = OPTForCausalLM.from_pretrained(
            fp16_model_name,
            torch_dtype=torch.float32,
            device_map='cpu'
        )
    param_size, buffer_size = print_model_size(model_fp16)
    fp16_total_static_mb = (param_size + buffer_size) / (1024**2)
    # Also compute FP16 linear weights size for apples-to-apples compression against BSI weights
    fp16_linear_weight_bytes = 0
    for name, module in model_fp16.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name:
            w = module.weight
            fp16_linear_weight_bytes += w.nelement() * w.element_size()
    fp16_linear_weight_mb = fp16_linear_weight_bytes / (1024**2)
    print(f"FP16 linear weights size: {fp16_linear_weight_mb:.2f}MB")
    if not args.memory_only and not args.skip_baseline:
        acc_fp16, latency_fp16, mem_fp16 = evaluator.evaluate(model_fp16)
        print(f"-> FP Accuracy: {acc_fp16:.4f}, Avg Forward Latency: {latency_fp16:.3f}ms, Peak Memory: {mem_fp16:.2f}MB")
    elif not args.memory_only:
        print("Skipping baseline evaluation (--skip_baseline)")
    
    # Decimal-place configurations to test
    bsi_configs = [(f'BSI-dec{dp}', dp) for dp in args.decimal_places]
    threshold_global_cfg = build_layer_config(args.compress_threshold, args.threshold_attention, args.threshold_mlp)

    for name, dec in bsi_configs:
        print(f"\n--- Benchmarking {name} (decimal_places={dec}) ---")

        model_bsi = OPTForCausalLM.from_pretrained(
            fp16_model_name, 
            torch_dtype=torch.float32,
            device_map='cpu'
        )
        decimal_cfg = build_layer_config(dec, args.decimal_attention, args.decimal_mlp)

        if args.memory_only:
            model_bsi = quantize_model_bsi(model_bsi, decimalPlaces=decimal_cfg, scope=args.scope, compress_threshold=threshold_global_cfg)
            summary = summarize_bsi_model(model_bsi)
            bsi_weight_mb = summary["total_bsi_bytes"] / (1024**2)
            bsi_bias_mb = summary["total_bias_bytes"] / (1024**2)
            bsi_total_mb = summary["total_static_bytes"] / (1024**2)
            dense_linear_mb = summary["total_dense_bytes"] / (1024**2)
            compression_vs_fp16_linear = fp16_linear_weight_mb / bsi_total_mb if bsi_total_mb > 0 else 0
            compression_vs_dense = dense_linear_mb / bsi_total_mb if bsi_total_mb > 0 else 0
            print(f"{name} Static Size (all linear layers):")
            print(f"  BSI Weights: {bsi_weight_mb:.2f}MB  |  Bias: {bsi_bias_mb:.2f}MB")
            print(f"  Total BSI Linear Storage: {bsi_total_mb:.2f}MB")
            print(f"  Dense Linear Storage (reference dtype): {dense_linear_mb:.2f}MB")
            print(f"  Compression vs FP16 Linear Weights: {compression_vs_fp16_linear:.2f}x")
            print(f"  Compression vs Dense Linear Weights: {compression_vs_dense:.2f}x")
            print(f"  Reference FP16 Full Model Static Size: {fp16_total_static_mb:.2f}MB")
            print_compression_summary(summary, heading=f"{name} Compression Summary")
        else:
            acc_bsi, fwd_ms, dot_ms, top5_acc, summary, layer_stats = evaluator.evaluate_with_bsi(
                model_bsi, decimal_cfg, scope=args.scope, threshold_cfg=threshold_global_cfg
            )
            bsi_weight_mb = summary["total_bsi_bytes"] / (1024**2)
            bsi_bias_mb = summary["total_bias_bytes"] / (1024**2)
            bsi_total_mb = summary["total_static_bytes"] / (1024**2)
            dense_linear_mb = summary["total_dense_bytes"] / (1024**2)
            print(f"\n{name} Results:")
            print(f"  Accuracy: {acc_bsi:.4f}")
            print(f"  Avg Forward Latency: {fwd_ms:.3f}ms")
            print(f"  Avg Dot-only Latency: {dot_ms:.3f}ms")
            print(f"  Top-5 Accuracy: {top5_acc:.4f}")
            compression_vs_fp16_linear = fp16_linear_weight_mb / bsi_total_mb if bsi_total_mb > 0 else 0
            compression_vs_dense = dense_linear_mb / bsi_total_mb if bsi_total_mb > 0 else 0
            print(f"  BSI Weights: {bsi_weight_mb:.2f}MB  |  Bias: {bsi_bias_mb:.2f}MB")
            print(f"  Total BSI Linear Storage: {bsi_total_mb:.2f}MB")
            print(f"  Dense Linear Storage (reference dtype): {dense_linear_mb:.2f}MB")
            print(f"  Compression vs FP16 Linear Weights: {compression_vs_fp16_linear:.2f}x")
            print(f"  Compression vs Dense Linear Weights: {compression_vs_dense:.2f}x")
            print(f"  Reference FP16 Full Model Static Size: {fp16_total_static_mb:.2f}MB")
            if layer_stats:
                print("  Worst layers by MSE (top 5):")
                for entry in layer_stats[:5]:
                    print(
                        "    {name}: decimal={dec:.0f}, thr={thr:.3f}, samples={samples:.0f}, "
                        "mse={mse:.6f}, mae={mae:.6f}, cosine={cos:.4f}, max_abs={mx:.6f}".format(
                            name=entry['name'],
                            dec=entry['decimalPlaces'],
                            thr=entry['compress_threshold'],
                            samples=entry['samples_tracked'],
                            mse=entry['mse'],
                            mae=entry['mae'],
                            cos=entry['cosine'],
                            mx=entry['max_abs']
                        )
                    )
            print_compression_summary(summary, heading=f"{name} Compression Summary")
         
        del model_bsi
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
