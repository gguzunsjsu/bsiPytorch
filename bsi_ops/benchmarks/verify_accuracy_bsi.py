import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, OPTForCausalLM
from torch.nn.functional import pad
from datasets import load_dataset
import bsi_ops
import gc
import numpy as np
from typing import Dict, Any, List, Union, Tuple
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

_cpu_count = os.cpu_count() or 1
_query_builder_workers = max(1, _cpu_count // 2)
_QUERY_BUILD_EXECUTOR = ThreadPoolExecutor(max_workers=_query_builder_workers)

def _build_query_cpu(tensor_cpu: torch.Tensor, decimal_places: int, threshold: float):
    start = time.perf_counter_ns()
    capsule, mem_bytes, slices = bsi_ops.build_bsi_query(tensor_cpu, decimal_places, threshold)
    elapsed = time.perf_counter_ns() - start
    return capsule, mem_bytes, slices, elapsed

def _build_query_cuda(tensor_gpu: torch.Tensor, decimal_places: int, threshold: float, device_index: int):
    prev_device = torch.cuda.current_device()
    if prev_device != device_index:
        torch.cuda.set_device(device_index)
    start = time.perf_counter_ns()
    capsule, mem_bytes, slices, words = bsi_ops.build_bsi_query_cuda(tensor_gpu, decimal_places, threshold)
    elapsed = time.perf_counter_ns() - start
    if prev_device != device_index:
        torch.cuda.set_device(prev_device)
    return capsule, mem_bytes, (slices, words), elapsed

def get_device():
    """Auto-detect best available device"""
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

class BSIQuantizedLinear(torch.nn.Module):
    """BSI quantized linear layer using prebuilt BSI keys for efficiency and diagnostics."""

    def __init__(self, original_linear, decimalPlaces=2, compress_threshold=0.2, query_threshold=None, prefer_cuda: bool=True):
        super().__init__()
        self.decimalPlaces = int(decimalPlaces)
        self.compress_threshold = float(compress_threshold)
        self.query_compress_threshold = float(query_threshold) if query_threshold is not None else float(compress_threshold)

        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features

        # tracking counters (dot-only)
        self.dot_ns_total = 0
        self.dot_calls = 0

        dense_weight = original_linear.weight.detach().to(torch.float32)
        self.weight_fp32 = dense_weight.cpu()
        self.weight_dense_memory_bytes = dense_weight.numel() * dense_weight.element_size()

        if original_linear.bias is not None:
            bias = original_linear.bias.detach().to(torch.float32)
            self.register_buffer("bias", bias)
            self.bias_fp32 = bias.cpu()
            self.bias_memory_bytes = bias.numel() * bias.element_size()
        else:
            self.register_buffer("bias", None)
            self.bias_fp32 = None
            self.bias_memory_bytes = 0

        self._bsi_keys_cuda = None
        with torch.no_grad():
            assert prefer_cuda and hasattr(bsi_ops, 'build_bsi_keys_cuda') and torch.cuda.is_available(), \
                "CUDA path required: build_bsi_keys_cuda not available or CUDA not present"
            # Build keys directly on CUDA (verbatim layout)
            self._bsi_keys_cuda, total_mem_bytes, num_keys, d, W = bsi_ops.build_bsi_keys_cuda(
                dense_weight, self.decimalPlaces, float(self.compress_threshold)
            )
            assert num_keys == self.out_features and d == self.in_features
            self.weight_bsi_memory_bytes = int(total_mem_bytes)
        self.total_bsi_static_bytes = self.weight_bsi_memory_bytes + self.bias_memory_bytes
        # Debug/verbosity flag: when True, prints one-time diagnostics on first forward
        self.verbose = False

        # Tracking: only dot time (GPU), no CPU build comparisons
        self.dot_ns_total = 0
        self.dot_calls = 0
        self.build_ns_total = 0  # optional: track Python-side build time if desired
        self.build_calls = 0
        self._query_cache: OrderedDict = OrderedDict()
        self.max_query_cache = 512

    def reset_error_stats(self):
        self.mse_sum = 0.0
        self.mae_sum = 0.0
        self.cosine_sum = 0.0
        self.max_abs_error = 0.0
        self.samples_tracked = 0

    def clear_query_cache(self):
        self._query_cache.clear()

    def _query_cache_key(self, tensor: torch.Tensor):
        storage = tensor.untyped_storage()
        return (
            int(storage.data_ptr()),
            tensor.storage_offset(),
            tensor.numel(),
            str(tensor.dtype),
            tensor.device.type,
            int(tensor._version)
        )

    def _query_cache_get(self, key):
        entry = self._query_cache.get(key)
        if entry is not None:
            self._query_cache.move_to_end(key)
        return entry

    def _query_cache_set(self, key, value):
        cache = self._query_cache
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > self.max_query_cache:
            cache.popitem(last=False)

    def forward(self, x):
        original_device = x.device
        original_dtype = x.dtype
        original_shape = x.shape
        if len(x.shape) > 2:
            x = x.view(-1, x.shape[-1])
        x = x.to(torch.float32)

        output_list = []
        dense_outputs = None
        dot_ns_this_forward = 0
        
        # Debug printing gated by `self.verbose` to avoid overhead in benchmarks
        debug_first = bool(self.verbose) and not hasattr(self, '_debug_printed')
        
        use_cuda = True
        device_index = torch.cuda.current_device()

        batch_size = x.shape[0]
        batch_inputs: List[torch.Tensor] = []
        cache_keys: List[Any] = []
        cpu_inputs: List[torch.Tensor] = [None] * batch_size
        future_builds: Dict[Any, Any] = {}
        cached_queries: Dict[Any, Tuple[Any, int]] = {}

        for i in range(batch_size):
            input_vec = x[i].detach().contiguous()
            batch_inputs.append(input_vec)
            cache_key = self._query_cache_key(input_vec)
            cache_keys.append(cache_key)

            cached_entry = self._query_cache_get(cache_key)
            if cached_entry is not None:
                cached_queries[cache_key] = cached_entry
                continue

            future_builds[cache_key] = _QUERY_BUILD_EXECUTOR.submit(
                _build_query_cuda,
                input_vec,
                self.decimalPlaces,
                float(self.query_compress_threshold),
                device_index,
            )

        for i in range(batch_size):
            input_vec = batch_inputs[i]
            input_vec_cpu = None

            if debug_first and i == 0:
                print(f"\n[DEBUG {self.__class__.__name__}] First forward pass diagnostics:")
                print(f"  Input vec stats: min={input_vec.min():.4f}, max={input_vec.max():.4f}, mean={input_vec.mean():.4f}, std={input_vec.std():.4f}")
                print(f"  Decimal Places used: {self.decimalPlaces}")

            cache_key = cache_keys[i]
            cached_entry = cached_queries.get(cache_key)
            if cached_entry is not None:
                query_capsule, query_mem = cached_entry
            else:
                future = future_builds.pop(cache_key)
                query_capsule, query_mem, _meta, python_build_ns = future.result()
                # Only track python build time separately; exclude from dot latency
                self.build_ns_total += python_build_ns
                self.build_calls += 1
                self._query_cache_set(cache_key, (query_capsule, query_mem))

            build_ns_cpp = 0
            scores, time_taken_ns, build_ns_cpp, dot_ns, _query_bsi_size = bsi_ops.batch_dot_product_prebuilt_cuda_caps(
                query_capsule,
                self._bsi_keys_cuda
            )
            if build_ns_cpp:
                self.build_ns_total += int(build_ns_cpp)
                self.build_calls += 1
            
            # Debug: Check raw scores from C++
            if debug_first and i == 0:
                print(f"  Raw BSI scores from CUDA: shape={scores.shape}, dtype={scores.dtype}")
                print(f"    min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}, std={scores.std():.4f}")
                print(f"    First 5 values: {scores[:5].tolist()}")
                    
            
            # Add bias if present
            scores = scores.to(device=original_device, dtype=original_dtype, non_blocking=scores.is_cuda and original_device.type == 'cuda')
            if self.bias is not None:
                scores = scores + self.bias.to(device=original_device, dtype=original_dtype)
            if debug_first and i == 0:
                print(f"  After adding bias: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

            output_list.append(scores)
            dot_ns_this_forward += int(dot_ns)

            # No CPU accuracy comparisons in GPU-only mode
        
        # Mark that we've printed debug info
        if debug_first:
            self._debug_printed = True

        # accumulate dot-only timing
        self.dot_ns_total += dot_ns_this_forward
        self.dot_calls += 1

        output = torch.stack(output_list)

        # No CPU accuracy summary in GPU-only mode

        if len(original_shape) == 3:
            output = output.view(original_shape[0], original_shape[1], -1)
        # Bias has already been added above, don't add it again
        return output

def reset_bsi_dot_counters(model: nn.Module):
    for m in model.modules():
        if isinstance(m, BSIQuantizedLinear):
            m.dot_ns_total = 0
            m.dot_calls = 0
            m.build_ns_total = 0
            m.build_calls = 0
            m.clear_query_cache()

def sum_bsi_dot_counters(model: nn.Module) -> int:
    total = 0
    for m in model.modules():
        if isinstance(m, BSIQuantizedLinear):
            total += int(m.dot_ns_total)
    return total

def sum_bsi_build_counters(model: nn.Module) -> int:
    total = 0
    for m in model.modules():
        if isinstance(m, BSIQuantizedLinear):
            total += int(m.build_ns_total)
    return total


def enable_bsi_error_stats(model: nn.Module, enabled: bool) -> None:
    """Toggle per-layer error tracking for BSIQuantizedLinear modules."""
    for module in model.modules():
        if isinstance(module, BSIQuantizedLinear):
            if enabled and not module.collect_stats:
                module.collect_stats = True
                module.reset_error_stats()
            elif not enabled:
                module.collect_stats = False


def collect_bsi_error_stats(model: nn.Module) -> List[Dict[str, float]]:
    """Gather accumulated per-layer error metrics for debugging."""
    layer_stats: List[Dict[str, float]] = []
    for name, module in model.named_modules():
        if isinstance(module, BSIQuantizedLinear) and module.samples_tracked > 0:
            samples = float(module.samples_tracked)
            layer_stats.append({
                "name": name,
                "decimalPlaces": module.decimalPlaces,
                "compress_threshold": module.compress_threshold,
                "samples_tracked": samples,
                "mse": module.mse_sum / samples,
                "mae": module.mae_sum / samples,
                "cosine": module.cosine_sum / samples,
                "max_abs": module.max_abs_error,
            })
    layer_stats.sort(key=lambda item: item["mse"], reverse=True)
    return layer_stats

def summarize_bsi_model(model: nn.Module) -> Dict[str, Any]:
    summary = {
        "total_bsi_bytes": 0,
        "total_dense_bytes": 0,
        "total_bias_bytes": 0,
        "total_fp16_linear_bytes": 0,
        "total_static_bytes": 0,
        "total_dense_with_bias_bytes": 0,
        "num_quantized_layers": 0,
        "total_slices": 0,
        "compressed_slices": 0,
        "verbatim_slices": 0,
        "layers": [],
    }
    for name, module in model.named_modules():
        if isinstance(module, BSIQuantizedLinear):
            layer_stats = {
                "name": name,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "decimalPlaces": module.decimalPlaces,
                "compress_threshold": module.compress_threshold,
                "weight_bsi_bytes": module.weight_bsi_memory_bytes,
                "weight_dense_bytes": module.weight_dense_memory_bytes,
                "bias_bytes": module.bias_memory_bytes,
            }
            # Add equivalent FP16 weight bytes for apples-to-apples reporting
            try:
                fp16_bytes = int(module.in_features * module.out_features * 2)
            except Exception:
                fp16_bytes = 0
            layer_stats["weight_fp16_bytes"] = fp16_bytes
            layer_stats["weight_total_slices"] = getattr(module, "weight_total_slices", 0)
            layer_stats["weight_verbatim_slices"] = getattr(module, "weight_verbatim_slices", 0)
            layer_stats["weight_compressed_slices"] = getattr(module, "weight_compressed_slices", 0)
            layer_stats["weight_compressed_pct"] = getattr(module, "weight_compressed_pct", 0.0)
            layer_stats["weight_verbatim_pct"] = getattr(module, "weight_verbatim_pct", 0.0)
            layer_stats["weight_bsi_disk_bytes"] = getattr(module, "weight_bsi_disk_bytes", 0)
            summary["layers"].append(layer_stats)
            summary["total_bsi_bytes"] += module.weight_bsi_memory_bytes
            summary["total_dense_bytes"] += module.weight_dense_memory_bytes
            summary["total_bias_bytes"] += module.bias_memory_bytes
            summary["total_fp16_linear_bytes"] += fp16_bytes
            summary["total_bsi_disk_bytes"] = summary.get("total_bsi_disk_bytes", 0) + layer_stats["weight_bsi_disk_bytes"]
            summary["num_quantized_layers"] += 1
            summary["total_slices"] += layer_stats["weight_total_slices"]
            summary["compressed_slices"] += layer_stats["weight_compressed_slices"]
            summary["verbatim_slices"] += layer_stats["weight_verbatim_slices"]
    summary["total_static_bytes"] = summary["total_bsi_bytes"] + summary["total_bias_bytes"]
    summary["total_dense_with_bias_bytes"] = summary["total_dense_bytes"] + summary["total_bias_bytes"]
    if summary["total_slices"]:
        summary["compressed_pct"] = (summary["compressed_slices"] * 100.0) / summary["total_slices"]
        summary["verbatim_pct"] = (summary["verbatim_slices"] * 100.0) / summary["total_slices"]
    else:
        summary["compressed_pct"] = 0.0
        summary["verbatim_pct"] = 0.0
    return summary

def save_bsi_model(model: nn.Module, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    meta = []
    for name, module in model.named_modules():
        if isinstance(module, BSIQuantizedLinear):
            layer_dir = os.path.join(out_dir, name.replace('.', '_'))
            os.makedirs(layer_dir, exist_ok=True)
            bsi_ops.save_keyset(module._bsi_keys, layer_dir)
            meta.append({
                "name": name,
                "in": module.in_features,
                "out": module.out_features,
                "decimalPlaces": module.decimalPlaces,
                "compress_threshold": module.compress_threshold
            })
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        import json; json.dump(meta, f, indent=2)


def bsi_full_model_static_bytes(model: nn.Module, summary: Dict[str, Any]) -> int:
    """Return full model static size (bytes) for a BSI-quantized model.

    This sums all registered parameters and buffers (which include non-BSI
    modules and BSI biases) and adds the total BSI weight bytes which are not
    stored as parameters.
    """
    param_bytes = 0
    for p in model.parameters():
        param_bytes += p.nelement() * p.element_size()
    buffer_bytes = 0
    for b in model.buffers():
        buffer_bytes += b.nelement() * b.element_size()
    return int(param_bytes + buffer_bytes + int(summary.get("total_bsi_bytes", 0)))


def compression_summary_lines(summary: Dict[str, Any]) -> Tuple[List[str], Tuple[int, int, float]]:
    layers = summary.get("layers", [])
    lines: List[str] = []
    for idx, layer in enumerate(layers, start=1):
        total = int(layer.get("weight_total_slices", 0) or 0)
        compressed = int(layer.get("weight_compressed_slices", 0) or 0)
        pct = (compressed * 100.0 / total) if total else 0.0
        layer_name = layer.get("name", f"Layer {idx}")
        lines.append(f"Layer {idx} ({layer_name}): {compressed}/{total} ({pct:.2f}%)")

    total_slices = int(summary.get("total_slices", 0) or 0)
    compressed_slices = int(summary.get("compressed_slices", 0) or 0)
    overall_pct = (compressed_slices * 100.0 / total_slices) if total_slices else 0.0
    return lines, (compressed_slices, total_slices, overall_pct)


def print_compression_summary(summary: Dict[str, Any], heading: str = "Compression Summary") -> None:
    lines, overall = compression_summary_lines(summary)
    compressed, total, pct = overall
    if not lines and total == 0:
        print(f"\n=== {heading} ===")
        print("No compression statistics available.")
        return

    print(f"\n=== {heading} ===")
    for line in lines:
        print(line)
    print(f"Overall compressed slices: {compressed}/{total} ({pct:.2f}%)")

def _resolve_layer_value(config: Union[float, Dict[str, float]], layer_kind: str, fallback: float) -> float:
    if isinstance(config, dict):
        if layer_kind == 'attention':
            return float(config.get('attention', config.get('default', fallback)))
        if layer_kind == 'mlp':
            return float(config.get('mlp', config.get('default', fallback)))
        return float(config.get('default', fallback))
    return float(config)


def quantize_model_bsi(model, decimalPlaces=2, skip_lm_head=True, scope='all', compress_threshold=0.2, prefer_cuda: bool=False):
    """Replace linear layers with BSI quantized versions.

    scope options:
      - 'all': quantize all Linear layers except lm_head (default)
      - 'attention': quantize only attention projection layers (q_proj, k_proj, v_proj, out_proj, self_attn)
      - 'mlp': quantize only MLP/FFN linears (fc1, fc2, mlp, ffn)
    """
    def is_attention_linear(name: str) -> bool:
        tokens = ['attn', 'self_attn', 'q_proj', 'k_proj', 'v_proj', 'out_proj']
        return any(t in name for t in tokens)

    def is_mlp_linear(name: str) -> bool:
        tokens = ['mlp', 'ff', 'ffn', 'fc1', 'fc2']
        return any(t in name for t in tokens)

    decimal_config = decimalPlaces
    threshold_config = compress_threshold
    default_decimal = float(decimal_config if isinstance(decimal_config, (int, float)) else decimal_config.get('default', 2))
    default_threshold = float(threshold_config if isinstance(threshold_config, (int, float)) else threshold_config.get('default', 0.2))

    layer_count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if skip_lm_head and 'lm_head' in name:
                continue
            if scope == 'attention' and not is_attention_linear(name):
                continue
            if scope == 'mlp' and not is_mlp_linear(name):
                continue
            parent_name = '.'.join(name.split('.')[:-1])
            module_name = name.split('.')[-1]
            parent = model
            if parent_name:
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
            layer_kind = 'attention' if is_attention_linear(name) else 'mlp' if is_mlp_linear(name) else 'other'
            decimal_layer = _resolve_layer_value(decimal_config, layer_kind, default_decimal)
            threshold_layer = _resolve_layer_value(threshold_config, layer_kind, default_threshold)
            setattr(parent, module_name, BSIQuantizedLinear(module, decimalPlaces=decimal_layer, compress_threshold=threshold_layer, prefer_cuda=prefer_cuda))
            layer_count += 1
    print(f"Quantized {layer_count} linear layers to BSI with decimalPlaces={decimal_config} (scope={scope})")
    return model

class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.dataset = self.dataset.map(self.tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids"])

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        total, hit = 0, 0
        for batch in self.dataset:
            input_ids = batch["input_ids"].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc

def main():
    print("Quick BSI Verification with Small Model")
    print("=" * 50)
    
    model_name = "facebook/opt-125m"
    device = get_device()
    print(f"Auto-detected device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("lambada", split="validation[:10]") 
    evaluator = Evaluator(dataset, tokenizer, device)
    
    prefer_cuda = device == 'cuda' and torch.cuda.is_available()

    print("Loading small model for quick test...")
    if prefer_cuda:
        model_fp16 = OPTForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto", use_safetensors=True
        )
        model_bsi = OPTForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="auto", use_safetensors=True
        )
    else:
        model_fp16 = OPTForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="cpu", use_safetensors=True
        )
        model_bsi = OPTForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, device_map="cpu", use_safetensors=True
        )
    
    print("Testing FP16 baseline...")
    acc_fp16 = evaluator.evaluate(model_fp16)
    print(f"Original model accuracy: {acc_fp16:.4f}")
    
    print("Applying BSI quantization to small subset...")
    layers_quantized = 0
    for name, module in model_bsi.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name:
            if layers_quantized < 5:
                parent_name = '.'.join(name.split('.')[:-1])
                layer_name = name.split('.')[-1]
                parent = model_bsi.get_submodule(parent_name)
                setattr(parent, layer_name, BSIQuantizedLinear(
                    module,
                    decimalPlaces=2,
                    compress_threshold=0.2,
                    prefer_cuda=prefer_cuda,
                ))
                layers_quantized += 1
            else:
                break

    print(f"Quantized {layers_quantized} layers")

    if prefer_cuda:
        model_bsi.to('cuda')
    
    stats = summarize_bsi_model(model_bsi)
    bsi_weight_mb = stats["total_bsi_bytes"] / (1024**2)
    bsi_bias_mb = stats["total_bias_bytes"] / (1024**2)
    bsi_total_mb = stats["total_static_bytes"] / (1024**2)
    dense_linear_mb = stats["total_dense_bytes"] / (1024**2)
    fp16_linear_mb = sum(
        mod.in_features * mod.out_features * 2 for _, mod in model_bsi.named_modules()
        if isinstance(mod, BSIQuantizedLinear)
    ) / (1024**2)
    compression = (fp16_linear_mb / bsi_total_mb) if bsi_total_mb > 0 else 0.0

    print("\nBSI Static Model Size (quantized layers only):")
    print(f"  BSI Quantized Linear Weights: {bsi_weight_mb:.2f} MB")
    print(f"  Bias Storage (FP32): {bsi_bias_mb:.2f} MB")
    print(f"  Total BSI Linear Storage: {bsi_total_mb:.2f} MB")
    print(f"  Dense Linear Weights (reference dtype): {dense_linear_mb:.2f} MB")
    print(f"  FP16 Linear Weights (same dims): {fp16_linear_mb:.2f} MB")
    print(f"  Compression vs FP16 Linear Weights: {compression:.2f}x")
    print_compression_summary(stats)

if __name__ == "__main__":
    main()
