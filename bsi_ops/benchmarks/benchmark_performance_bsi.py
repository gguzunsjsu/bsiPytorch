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
    reset_bsi_dot_counters, sum_bsi_dot_counters, sum_bsi_build_counters,
    sum_bsi_dot_query_vectors, sum_bsi_dot_output_elements,
    enable_bsi_error_stats, collect_bsi_error_stats,
    print_compression_summary, save_bsi_model, bsi_full_model_static_bytes
)
import argparse
from tqdm import tqdm
import json
import csv
from datetime import datetime

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
    def evaluate_with_bsi(self, model, decimal_cfg=2, scope='all', threshold_cfg=0.2, bsi_device: str='auto'):
        """Evaluate model with actual BSI quantization."""
        model.eval()
        total, hit = 0, 0
        latency = 0

        print(f"Quantizing model with BSI (decimal_places={decimal_cfg}, scope={scope}, device={bsi_device})...")
        prefer_cuda = (bsi_device == 'cuda') or (bsi_device == 'auto' and torch.cuda.is_available())
        model = quantize_model_bsi(model, decimalPlaces=decimal_cfg, scope=scope, compress_threshold=threshold_cfg, prefer_cuda=prefer_cuda)
        summary = summarize_bsi_model(model)
        reset_bsi_dot_counters(model)
        if summary["total_bsi_bytes"] == 0:
            print("Warning: BSI weight memory not found on modules. Did quantization run?")

        if prefer_cuda and torch.cuda.is_available():
            model = model.to('cuda')
            model_device = torch.device('cuda')
        else:
            model = model.to('cpu')
            model_device = torch.device('cpu')

        # Enable layer-wise error stats for first N batches if requested.
        if self.layer_stats_batches > 0:
            enable_bsi_error_stats(model, True)
        else:
            enable_bsi_error_stats(model, False)

        print("Running evaluation...")
        dot_ns_prev = sum_bsi_dot_counters(model)
        build_ns_prev = sum_bsi_build_counters(model)
        dot_q_prev = sum_bsi_dot_query_vectors(model)
        dot_elem_prev = sum_bsi_dot_output_elements(model)
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

                if model_device.type == 'cuda':
                    torch.cuda.synchronize()
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    outputs = model(input_ids)
                    end_event.record()
                    torch.cuda.synchronize()
                    batch_fwd_ms = start_event.elapsed_time(end_event)
                    del start_event, end_event
                else:
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
                build_ns_cur = sum_bsi_build_counters(model)
                dot_q_cur = sum_bsi_dot_query_vectors(model)
                dot_elem_cur = sum_bsi_dot_output_elements(model)

                batch_dot_ns = int(dot_ns_cur - dot_ns_prev)
                batch_dot_ms = (dot_ns_cur - dot_ns_prev) / 1e6
                batch_build_ms = (build_ns_cur - build_ns_prev) / 1e6
                dot_ns_prev = dot_ns_cur
                build_ns_prev = build_ns_cur
                batch_dot_q = int(dot_q_cur - dot_q_prev)
                batch_dot_elem = int(dot_elem_cur - dot_elem_prev)
                dot_q_prev = dot_q_cur
                dot_elem_prev = dot_elem_cur

                # Dot-only averages derived from dot_ns + denominators.
                batch_dot_us_per_query = (batch_dot_ns / batch_dot_q) / 1e3 if batch_dot_q > 0 else 0.0
                batch_dot_ns_per_scalar = (batch_dot_ns / batch_dot_elem) if batch_dot_elem > 0 else 0.0

                if self.layer_stats_batches and (batch_idx + 1) == self.layer_stats_batches:
                    enable_bsi_error_stats(model, False)

                pbar.update(1)
                pbar.set_postfix(
                    fwd_ms=f"{batch_fwd_ms:.1f}",
                    build_ms=f"{batch_build_ms:.1f}",
                    dot_ms=f"{batch_dot_ms:.1f}",
                    dot_q_us=f"{batch_dot_us_per_query:.2f}",
                    dot_s_ns=f"{batch_dot_ns_per_scalar:.3f}",
                )
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

                if model_device.type == 'cuda':
                    torch.cuda.synchronize()
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    outputs = model(input_ids)
                    end_event.record()
                    torch.cuda.synchronize()
                    batch_fwd_ms = start_event.elapsed_time(end_event)
                    del start_event, end_event
                else:
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
                build_ns_cur = sum_bsi_build_counters(model)
                dot_q_cur = sum_bsi_dot_query_vectors(model)
                dot_elem_cur = sum_bsi_dot_output_elements(model)

                batch_dot_ns = int(dot_ns_cur - dot_ns_prev)
                batch_dot_ms = (dot_ns_cur - dot_ns_prev) / 1e6
                batch_build_ms = (build_ns_cur - build_ns_prev) / 1e6
                dot_ns_prev = dot_ns_cur
                build_ns_prev = build_ns_cur
                batch_dot_q = int(dot_q_cur - dot_q_prev)
                batch_dot_elem = int(dot_elem_cur - dot_elem_prev)
                dot_q_prev = dot_q_cur
                dot_elem_prev = dot_elem_cur

                batch_dot_us_per_query = (batch_dot_ns / batch_dot_q) / 1e3 if batch_dot_q > 0 else 0.0
                batch_dot_ns_per_scalar = (batch_dot_ns / batch_dot_elem) if batch_dot_elem > 0 else 0.0

                print(
                    f"  Processed batch {batch_idx + 1}/{self.num_samples} "
                    f"(fwd_ms={batch_fwd_ms:.1f}, build_ms={batch_build_ms:.1f}, dot_ms={batch_dot_ms:.1f}, "
                    f"dot_q_us={batch_dot_us_per_query:.2f}, dot_s_ns={batch_dot_ns_per_scalar:.3f})"
                )

                if self.layer_stats_batches and (batch_idx + 1) == self.layer_stats_batches:
                    enable_bsi_error_stats(model, False)

        accuracy = hit / total if total else 0.0
        top5_acc = top5_hit / total if total else 0.0
        avg_forward_ms = latency / max(1, self.num_samples)
        dot_ns_total = sum_bsi_dot_counters(model)
        build_ns_total = sum_bsi_build_counters(model)
        dot_q_total = sum_bsi_dot_query_vectors(model)
        dot_elem_total = sum_bsi_dot_output_elements(model)
        avg_dot_ms = (dot_ns_total / 1e6) / max(1, self.num_samples)
        avg_build_ms = (build_ns_total / 1e6) / max(1, self.num_samples)
        avg_dot_us_per_query = ((dot_ns_total / dot_q_total) / 1e3) if dot_q_total > 0 else 0.0
        avg_dot_ns_per_scalar = (dot_ns_total / dot_elem_total) if dot_elem_total > 0 else 0.0
        layer_stats = collect_bsi_error_stats(model) if self.layer_stats_batches > 0 else []
        enable_bsi_error_stats(model, False)
        print(f"Completed BSI eval: top1_acc={accuracy:.4f}, top5_acc={top5_acc:.4f}")
        return (
            accuracy,
            avg_forward_ms,
            avg_build_ms,
            avg_dot_ms,
            avg_dot_us_per_query,
            avg_dot_ns_per_scalar,
            top5_acc,
            summary,
            layer_stats,
        )

    @torch.no_grad()
    def evaluate(self, model, scope: str = "all"):
        model.eval()
        total, hit, top5_hit = 0, 0, 0
        latency = 0

        # Optional: measure dot-only time for dense Linear layers (CUDA only).
        # This is comparable to BSI's dot_ms, which aggregates across all BSIQuantizedLinear modules.
        dense_dot_ms_total = 0.0
        dense_dot_q_total = 0
        dense_dot_elem_total = 0

        dense_dot_timer = None
        if self.device == 'cuda' and torch.cuda.is_available():
            scope = (scope or "all").lower().strip()

            def is_attention_linear(name: str) -> bool:
                tokens = ['attn', 'self_attn', 'q_proj', 'k_proj', 'v_proj', 'out_proj']
                return any(t in name for t in tokens)

            def is_mlp_linear(name: str) -> bool:
                tokens = ['mlp', 'ff', 'ffn', 'fc1', 'fc2']
                return any(t in name for t in tokens)

            name_by_module = {m: n for n, m in model.named_modules()}

            class _DenseLinearCudaDotTimer:
                def __init__(self):
                    self._handles = []
                    self._start_stack = {}
                    self._records = []
                    self._batch_q = 0
                    self._batch_elem = 0

                def reset_batch(self):
                    self._records.clear()
                    self._batch_q = 0
                    self._batch_elem = 0

                def install(self):
                    for mod, name in name_by_module.items():
                        if not isinstance(mod, torch.nn.Linear):
                            continue
                        if 'lm_head' in name:
                            continue
                        if scope == 'attention' and not is_attention_linear(name):
                            continue
                        if scope == 'mlp' and not is_mlp_linear(name):
                            continue

                        def _pre_hook(m, inputs):
                            x = inputs[0] if inputs else None
                            if (
                                x is None
                                or (not torch.is_tensor(x))
                                or (not x.is_cuda)
                                or x.numel() == 0
                            ):
                                q_rows = 0
                            else:
                                last = int(x.shape[-1]) if x.dim() > 0 else 0
                                q_rows = int(x.numel() // last) if last > 0 else 0
                            out_features = int(getattr(m, "out_features", 0) or 0)
                            out_elems = q_rows * out_features
                            if q_rows <= 0 or out_features <= 0:
                                return
                            evt = torch.cuda.Event(enable_timing=True)
                            evt.record()
                            self._start_stack.setdefault(m, []).append((evt, q_rows, out_elems))

                        def _post_hook(m, inputs, outputs):
                            evt = torch.cuda.Event(enable_timing=True)
                            evt.record()
                            stack = self._start_stack.get(m)
                            if not stack:
                                return
                            start_evt, q_rows, out_elems = stack.pop()
                            self._records.append((start_evt, evt))
                            self._batch_q += int(q_rows)
                            self._batch_elem += int(out_elems)

                        self._handles.append(mod.register_forward_pre_hook(_pre_hook))
                        self._handles.append(mod.register_forward_hook(_post_hook))

                def remove(self):
                    for h in self._handles:
                        try:
                            h.remove()
                        except Exception:
                            pass
                    self._handles.clear()

                def consume_batch_ms(self) -> float:
                    # Assumes caller has synchronized the stream for this forward.
                    total_ms = 0.0
                    for s, e in self._records:
                        total_ms += float(s.elapsed_time(e))
                    return total_ms

            dense_dot_timer = _DenseLinearCudaDotTimer()
            dense_dot_timer.install()
        
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        for batch in self.dataset:
            if dense_dot_timer is not None:
                dense_dot_timer.reset_batch()
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

                if dense_dot_timer is not None:
                    dense_dot_ms_total += dense_dot_timer.consume_batch_ms()
                    dense_dot_q_total += int(dense_dot_timer._batch_q)
                    dense_dot_elem_total += int(dense_dot_timer._batch_elem)
            else:
                start_time = time.perf_counter()
                outputs = model(input_ids)
                end_time = time.perf_counter()
                latency += (end_time - start_time) * 1000 
            
            last_token_logits = outputs.logits[:, -2 - pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            
            # top-5 accuracy
            topk = torch.topk(last_token_logits, k=5, dim=-1)
            top_ids = [int(x) for x in topk.indices[0].tolist()]
            if int(label.item()) in top_ids:
                top5_hit += 1
        
        accuracy = hit / total if total else 0.0
        top5_accuracy = top5_hit / total if total else 0.0
        avg_latency = latency / max(1, self.num_samples)
        peak_memory = torch.cuda.max_memory_allocated() / (1024**2) if self.device == 'cuda' and torch.cuda.is_available() else 0

        if dense_dot_timer is not None:
            dense_dot_timer.remove()
        avg_dense_dot_ms = dense_dot_ms_total / max(1, self.num_samples)
        # Dot-only averages derived from (sum of Linear CUDA time) and denominators.
        avg_dense_dot_us_per_query = (dense_dot_ms_total * 1e3 / dense_dot_q_total) if dense_dot_q_total > 0 else 0.0
        avg_dense_dot_ns_per_scalar = (dense_dot_ms_total * 1e6 / dense_dot_elem_total) if dense_dot_elem_total > 0 else 0.0

        return (
            accuracy,
            top5_accuracy,
            avg_latency,
            peak_memory,
            avg_dense_dot_ms,
            avg_dense_dot_us_per_query,
            avg_dense_dot_ns_per_scalar,
        )

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

def get_model_static_bytes(model) -> int:
    """Return total static bytes (parameters + buffers) for a torch.nn.Module."""
    p = sum(p.nelement() * p.element_size() for p in model.parameters())
    b = sum(buf.nelement() * buf.element_size() for buf in model.buffers())
    return int(p + b)

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

def print_metric_banner(dataset_name: str, split: str, num_samples: int, max_seq_len: int):
    print("\n[Metric]")
    print(f"  Type       : NEXT-TOKEN ACCURACY (last-token prediction)")
    print(f"  Dataset    : {dataset_name}  [{split} split]")
    print(f"  Samples    : {num_samples}")
    print(f"  Definition : Right-pad/truncate to max_seq_len={max_seq_len};")
    print("               label is the last token; logits are taken at the last non-pad position.")

def warmup_forward(model, tokenizer, device, max_seq_len, pad_id):
    model.eval()
    # cheap synthetic warmup; avoids 1st-run overhead in timed loops
    sample = torch.randint(low=100, high=1000, size=(1, max_seq_len), device=device)
    with torch.no_grad():
        _ = model(sample)

def write_report(report_dir, run_cfg, fp16_static_mb, results):
    os.makedirs(report_dir, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

    # 1) run-level JSON
    with open(os.path.join(report_dir, f"run_{stamp}.json"), "w") as f:
        json.dump({
            "config": run_cfg,
            "fp16_total_static_mb": fp16_static_mb,
            "results": results
        }, f, indent=2)

    # 2) per-layer CSV (from the last result)
    if not results:
        return
    last = results[-1]
    summary = last["summary"]
    layers = summary.get("layers", [])
    with open(os.path.join(report_dir, f"layers_{stamp}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        # Report sizes in MB and include FP16 per-layer weight size
        w.writerow(["name","in","out","decimalPlaces","compress_threshold",
                    "bsi_weight_mb","bsi_disk_mb","dense_weight_mb","fp16_weight_mb",
                    "bias_mb","total_slices","compressed_slices","verbatim_slices",
                    "compressed_pct","verbatim_pct"])
        for L in layers:
            def _mb(x):
                try:
                    return (float(x) / (1024**2))
                except Exception:
                    return 0.0
            bsi_mb = _mb(L.get("weight_bsi_bytes", 0))
            bsi_disk_mb = _mb(L.get("weight_bsi_disk_bytes", 0))
            dense_mb = _mb(L.get("weight_dense_bytes", 0))
            fp16_mb = _mb(L.get("weight_fp16_bytes", 0) or (L.get("weight_dense_bytes", 0) / 2))
            bias_mb = _mb(L.get("bias_bytes", 0))
            w.writerow([
                L["name"], L["in_features"], L["out_features"],
                L["decimalPlaces"], L["compress_threshold"],
                f"{bsi_mb:.4f}", f"{bsi_disk_mb:.4f}", f"{dense_mb:.4f}", f"{fp16_mb:.4f}",
                f"{bias_mb:.4f}",
                L.get("weight_total_slices",0),
                L.get("weight_compressed_slices",0),
                L.get("weight_verbatim_slices",0),
                L.get("weight_compressed_pct",0.0),
                L.get("weight_verbatim_pct",0.0)
            ])

def normalize_model_name(name: str) -> str:
    name = name.strip()
    # Accept "opt-125m", "opt-1.3b", "opt-30b" and auto-prefix to HF namespace
    if "/" not in name:
        return f"facebook/{name}"
    return name

def main():
    parser = argparse.ArgumentParser(description='Benchmark FP16 vs BSI (accuracy, latency, and/or static memory)')
    parser.add_argument('--model_name', type=str, default='facebook/opt-125m',
                    help='HuggingFace model name')
    parser.add_argument('--decimal_places', type=int, nargs='+', default=[2],
                    help='Decimal-place settings to test for BSI')
    parser.add_argument('--skip_baseline', action='store_true',
                    help='Skip evaluating FP baseline (useful on CPU)')
    parser.add_argument('--decimal_attention', type=int, default=None,
                    help='[DEPRECATED; ignored] Optional decimal-place override for attention layers')
    parser.add_argument('--decimal_mlp', type=int, default=None,
                    help='[DEPRECATED; ignored] Optional decimal-place override for MLP/FFN layers')
    parser.add_argument('--compress_threshold', type=float, default=0.2,
                    help='Base compression threshold for BSI slices (0 disables compression)')
    parser.add_argument('--threshold_attention', type=float, default=None,
                    help='[DEPRECATED; ignored] Compression threshold override for attention layers')
    parser.add_argument('--threshold_mlp', type=float, default=None,
                    help='[DEPRECATED; ignored] Compression threshold override for MLP/FFN layers')
    parser.add_argument('--layer_stats_batches', type=int, default=0,
                    help='Collect per-layer error metrics for the first N batches (0 disables)')
    parser.add_argument('--report_dir', type=str, default=None, help='Write per-run JSON + per-layer CSV here')
    parser.add_argument('--simple_report_txt', type=str, default=None,
        help='If set, append a simple line per configuration to this text file (memory-only and/or eval runs)')
    parser.add_argument('--memory_only', action='store_true', help='Only compute and print static BSI weights memory (no evaluation)')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of validation samples from LAMBADA')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Max sequence length (padding/truncation length)')
    parser.add_argument('--scope', type=str, default='all', choices=['all', 'attention', 'mlp'], help='Layers to quantize with BSI')
    parser.add_argument('--models', type=str, default='', help='Comma-separated list of models (e.g., "opt-125m,opt-1.3b,opt-30b"). '
         'If set, overrides --model_name. Names without a "/" are prefixed with "facebook/".')

    parser.add_argument('--datasets', type=str, default='lambada',
        help='Comma-separated list of HF datasets to evaluate (default: "lambada"). '
            'Each will be loaded as `load_dataset(name, split=f"{--split}[:{--num_samples}]")`.')
    parser.add_argument('--split', type=str, default='validation',
        help='Dataset split to use (default: validation).')
    parser.add_argument('--save_bsi_dir', type=str, default=None,
        help='If set, save quantized BSI keysets for each config to this directory')
    parser.add_argument('--base_dtype', type=str, choices=['fp32','fp16'], default='fp32',
        help='dtype to load the base (pre-quantization) model for reference size (default: fp32)')
    parser.add_argument('--show_compression_summary', action='store_true',
        help='Print per-layer slice compression summary (default: off)')
    parser.add_argument('--bsi_device', type=str, choices=['auto','cpu','cuda'], default='auto',
        help='Preferred device for BSI dot kernels (auto=CUDA if available)')
    
    args = parser.parse_args()

    fp16_model_name = args.model_name
    device = get_device()
    print(f"Auto-detected device: {device}")

    # Build the list of models to test
    if args.models.strip():
        model_names = [normalize_model_name(m) for m in args.models.split(",") if m.strip()]
    else:
        model_names = [normalize_model_name(args.model_name)]

    # Build the list of datasets to test
    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]

    for dataset_name in dataset_names:
        split_expr = f"{args.split}[:{args.num_samples}]" if args.num_samples and args.num_samples > 0 else args.split

        for fp16_model_name in model_names:
            print("\n" + "=" * 90)
            print(f"Dataset: {dataset_name} | Split: {split_expr} | Model: {fp16_model_name}")
            print("=" * 90)

            run_results = []
            tokenizer = None
            evaluator = None

            if not args.memory_only:
                print("Initializing tokenizer and dataset...")
                tokenizer = AutoTokenizer.from_pretrained(fp16_model_name)
                dataset = load_dataset(dataset_name, split=split_expr)
                evaluator = Evaluator(
                    dataset,
                    tokenizer,
                    device=device,
                    max_seq_len=args.max_seq_len,
                    layer_stats_batches=args.layer_stats_batches
                )
                print_metric_banner(dataset_name, args.split, args.num_samples, args.max_seq_len)

            print(f"\n--- Benchmarking FP baseline: {fp16_model_name} ---")
            fp16_total_static_mb = 0.0
            fp16_linear_weight_mb = 0.0
            if args.memory_only and args.skip_baseline:
                print("\n--- Skipping separate baseline model load for memory-only (--skip_baseline) ---")
            else:
                baseline_kwargs = {
                    'torch_dtype': torch.float16 if device == 'cuda' else torch.float32,
                    'device_map': 'auto' if device == 'cuda' else 'cpu'
                }
                model_fp16 = OPTForCausalLM.from_pretrained(fp16_model_name, **baseline_kwargs)
                # Print dtype of the FP baseline model
                try:
                    print(f"Baseline model dtype: {next(model_fp16.parameters()).dtype}")
                except StopIteration:
                    pass
                param_size, buffer_size = print_model_size(model_fp16)
                fp16_total_static_mb = (param_size + buffer_size) / (1024 ** 2)
                fp16_linear_weight_bytes = 0
                for name, module in model_fp16.named_modules():
                    if isinstance(module, torch.nn.Linear) and 'lm_head' not in name:
                        w = module.weight
                        fp16_linear_weight_bytes += w.nelement() * w.element_size()
                fp16_linear_weight_mb = fp16_linear_weight_bytes / (1024 ** 2)
                print(f"FP16 linear weights size: {fp16_linear_weight_mb:.2f}MB")

                baseline_acc = None
                baseline_top5 = None
                baseline_latency = None
                baseline_peak_mem = 0.0
                if not args.memory_only:
                    if args.skip_baseline:
                        print("Skipping baseline evaluation (--skip_baseline)")
                    else:
                        acc_fp16, top5_fp16, latency_fp16, mem_fp16, dot_fp16_ms, dot_fp16_q_us, dot_fp16_s_ns = evaluator.evaluate(
                            model_fp16, scope=args.scope
                        )
                        print(
                            f"-> [NEXT-TOKEN ACC] FP baseline top1={acc_fp16:.4f}, top5={top5_fp16:.4f}, "
                            f"avg_fwd={latency_fp16:.3f}ms, dot={dot_fp16_ms:.3f}ms, "
                            f"dot_q={dot_fp16_q_us:.3f}us, dot_s={dot_fp16_s_ns:.3f}ns, peak_mem={mem_fp16:.2f}MB"
                        )
                        baseline_acc = acc_fp16
                        baseline_top5 = top5_fp16
                        baseline_latency = latency_fp16
                        baseline_peak_mem = mem_fp16

                run_results.append({
                    "name": "FP16 baseline",
                    "decimal": None,
                    "scope": "dense",
                    "accuracy_top1": baseline_acc,
                    "accuracy_top5": baseline_top5,
                    "avg_forward_ms": baseline_latency,
                    "avg_build_ms": None,
                    "avg_dot_ms": dot_fp16_ms if baseline_latency is not None else None,
                    "avg_dot_us_per_query": dot_fp16_q_us if baseline_latency is not None else None,
                    "avg_dot_ns_per_scalar": dot_fp16_s_ns if baseline_latency is not None else None,
                    "peak_mem_mb": baseline_peak_mem,
                    "summary": {
                        "total_dense_bytes": fp16_linear_weight_bytes,
                        "total_static_bytes": param_size + buffer_size,
                        "linear_weight_mb": fp16_linear_weight_mb
                    }
                })

                del model_fp16
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            bsi_configs = [(f"BSI (dec={dec})", dec) for dec in args.decimal_places]
            threshold_global_cfg = float(args.compress_threshold)

            for name, dec in bsi_configs:
                print(f"\n--- Benchmarking {name} ---")
                base_torch_dtype = torch.float16 if args.base_dtype.lower() == 'fp16' else torch.float32
                model_bsi = OPTForCausalLM.from_pretrained(
                    fp16_model_name,
                    torch_dtype=base_torch_dtype,
                    device_map='cpu'
                )
                # Print dtype of the base (pre-quantization) model used for reference size
                try:
                    print(f"Reference Full Model dtype (pre-quantization): {next(model_bsi.parameters()).dtype}")
                except StopIteration:
                    pass
                decimal_cfg = dec

                if args.memory_only:
                    # Compute base (pre-quantization) full static size on the same model instance
                    base_full_static_mb = get_model_static_bytes(model_bsi) / (1024 ** 2)
                    model_bsi = quantize_model_bsi(
                        model_bsi,
                        decimalPlaces=decimal_cfg,
                        scope=args.scope,
                        compress_threshold=threshold_global_cfg
                    )
                    summary = summarize_bsi_model(model_bsi)
                    # Compute full-model static size (MB): params+buffers+BSI weight bytes
                    bsi_full_total_mb = bsi_full_model_static_bytes(model_bsi, summary) / (1024 ** 2)
                    # Optionally persist quantized keysets
                    if args.save_bsi_dir:
                        safe_dataset = dataset_name.replace('/', '_')
                        safe_model = fp16_model_name.replace('/', '_').replace(':', '_')
                        cfg_folder = f"{safe_dataset}__{safe_model}__{args.scope}__dec{dec}"
                        out_dir = os.path.join(args.save_bsi_dir, cfg_folder)
                        save_bsi_model(model_bsi, out_dir)
                    bsi_weight_mb = summary["total_bsi_bytes"] / (1024 ** 2)
                    bsi_bias_mb = summary["total_bias_bytes"] / (1024 ** 2)
                    bsi_total_mb = summary["total_static_bytes"] / (1024 ** 2)
                    dense_linear_mb = summary["total_dense_bytes"] / (1024 ** 2)
                    compression_vs_fp16_linear = (
                        fp16_linear_weight_mb / bsi_total_mb if bsi_total_mb > 0 else 0
                    )
                    compression_vs_dense = (
                        dense_linear_mb / bsi_total_mb if bsi_total_mb > 0 else 0
                    )
                    print(f"Baseline model static size: {base_full_static_mb:.3f}MB")
                    print(f"BSI model static size: {bsi_full_total_mb:.3f}MB")
                    compression_ratio = base_full_static_mb / bsi_full_total_mb if bsi_full_total_mb > 0 else 0
                    print(f"Compression: {compression_ratio:.2f}x")

                    if args.show_compression_summary:
                        print_compression_summary(summary, heading=f"{name} Compression Summary")
                    result_entry = {
                        "name": name,
                        "decimal": dec,
                        "scope": args.scope,
                        "accuracy_top1": None,
                        "accuracy_top5": None,
                        "avg_forward_ms": None,
                        "avg_dot_ms": None,
                        "summary": summary,
                        "bsi_full_static_mb": bsi_full_total_mb,
                        "base_full_static_mb": base_full_static_mb,
                        "compression_vs_full": (base_full_static_mb / bsi_full_total_mb if bsi_full_total_mb > 0 else 0)
                    }
                    # Append to simple text report if requested
                    if args.simple_report_txt:
                        try:
                            os.makedirs(os.path.dirname(args.simple_report_txt) or '.', exist_ok=True)
                        except Exception:
                            pass
                        with open(args.simple_report_txt, 'a') as tf:
                            stamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                            tf.write(
                                f"[{stamp}] dataset={dataset_name} model={fp16_model_name} scope={args.scope} dec={dec} thr={args.compress_threshold} base_dtype={args.base_dtype} "
                                f"base_full_mb={base_full_static_mb:.4f} bsi_full_mb={bsi_full_total_mb:.4f} compression_full={base_full_static_mb / bsi_full_total_mb if bsi_full_total_mb > 0 else 0:.4f}\n"
                            )
                else:
                    # Compute base (pre-quantization) full static size on the same model instance
                    base_full_static_mb = get_model_static_bytes(model_bsi) / (1024 ** 2)
                    acc_bsi, fwd_ms, build_ms, dot_ms, dot_q_us, dot_s_ns, top5_acc, summary, layer_stats = evaluator.evaluate_with_bsi(
                        model_bsi,
                        decimal_cfg,
                        scope=args.scope,
                        threshold_cfg=threshold_global_cfg,
                        bsi_device=args.bsi_device
                    )
                    # Model is quantized in-place; optionally persist keysets
                    if args.save_bsi_dir:
                        safe_dataset = dataset_name.replace('/', '_')
                        safe_model = fp16_model_name.replace('/', '_').replace(':', '_')
                        cfg_folder = f"{safe_dataset}__{safe_model}__{args.scope}__dec{dec}"
                        out_dir = os.path.join(args.save_bsi_dir, cfg_folder)
                        save_bsi_model(model_bsi, out_dir)
                    # Compute full-model static size (MB): params+buffers+BSI weight bytes
                    bsi_full_total_mb = bsi_full_model_static_bytes(model_bsi, summary) / (1024 ** 2)
                    bsi_weight_mb = summary["total_bsi_bytes"] / (1024 ** 2)
                    bsi_bias_mb = summary["total_bias_bytes"] / (1024 ** 2)
                    bsi_total_mb = summary["total_static_bytes"] / (1024 ** 2)
                    dense_linear_mb = summary["total_dense_bytes"] / (1024 ** 2)
                    compression_vs_fp16_linear = (
                        fp16_linear_weight_mb / bsi_total_mb if bsi_total_mb > 0 else 0
                    )
                    compression_vs_dense = (
                        dense_linear_mb / bsi_total_mb if bsi_total_mb > 0 else 0
                    )
                    print(
                        f"-> [NEXT-TOKEN ACC] {name} top1={acc_bsi:.4f}, top5={top5_acc:.4f}, "
                        f"avg_fwd={fwd_ms:.3f}ms, build={build_ms:.3f}ms, dot={dot_ms:.3f}ms (per-sample), "
                        f"dot_q={dot_q_us:.3f}us, dot_s={dot_s_ns:.3f}ns"
                    )
                    print(f"Baseline model static size: {base_full_static_mb:.3f}MB")
                    print(f"BSI model static size: {bsi_full_total_mb:.3f}MB")
                    compression_ratio = base_full_static_mb / bsi_full_total_mb if bsi_full_total_mb > 0 else 0
                    print(f"Compression: {compression_ratio:.2f}x")
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
                    if args.show_compression_summary:
                        print_compression_summary(summary, heading=f"{name} Compression Summary")
                    result_entry = {
                        "name": name,
                        "decimal": dec,
                        "scope": args.scope,
                        "accuracy_top1": acc_bsi,
                        "accuracy_top5": top5_acc,
                        "avg_forward_ms": fwd_ms,
                        "avg_build_ms": build_ms,
                        "avg_dot_ms": dot_ms,
                        "avg_dot_us_per_query": dot_q_us,
                        "avg_dot_ns_per_scalar": dot_s_ns,
                        "summary": summary,
                        "bsi_full_static_mb": bsi_full_total_mb,
                        "base_full_static_mb": base_full_static_mb,
                        "compression_vs_full": (base_full_static_mb / bsi_full_total_mb if bsi_full_total_mb > 0 else 0),
                        "layer_stats": layer_stats
                    }
                    # Append to simple text report if requested
                    if args.simple_report_txt:
                        try:
                            os.makedirs(os.path.dirname(args.simple_report_txt) or '.', exist_ok=True)
                        except Exception:
                            pass
                        with open(args.simple_report_txt, 'a') as tf:
                            stamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
                            tf.write(
                                f"[{stamp}] dataset={dataset_name} model={fp16_model_name} scope={args.scope} dec={dec} thr={args.compress_threshold} base_dtype={args.base_dtype} "
                                f"base_full_mb={base_full_static_mb:.4f} bsi_full_mb={bsi_full_total_mb:.4f} compression_full={base_full_static_mb / bsi_full_total_mb if bsi_full_total_mb > 0 else 0:.4f} "
                                f"top1={acc_bsi:.4f} top5={top5_acc:.4f} fwd_ms={fwd_ms:.3f} build_ms={build_ms:.3f} dot_ms={dot_ms:.3f} "
                                f"dot_q_us={dot_q_us:.3f} dot_s_ns={dot_s_ns:.3f}\n"
                            )

                run_results.append(result_entry)

                del model_bsi
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if args.report_dir:
                report_dir = args.report_dir
                if len(dataset_names) > 1 or len(model_names) > 1:
                    safe_dataset = dataset_name.replace('/', '_')
                    safe_model = fp16_model_name.replace('/', '_').replace(':', '_')
                    report_dir = os.path.join(report_dir, f"{safe_dataset}__{safe_model}")
                run_cfg = {
                    "model": fp16_model_name,
                    "dataset": dataset_name,
                    "split": split_expr,
                    "num_samples": args.num_samples,
                    "max_seq_len": args.max_seq_len,
                    "scope": args.scope,
                    "decimal_places": args.decimal_places,
                    "compress_threshold": args.compress_threshold,
                    "layer_stats_batches": args.layer_stats_batches,
                    "memory_only": args.memory_only,
                    "base_dtype": args.base_dtype
                }
                write_report(report_dir, run_cfg, fp16_total_static_mb, run_results)

if __name__ == '__main__':
    main()
