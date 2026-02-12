import torch
from transformers import AutoTokenizer, OPTForCausalLM
from smoothquant.opt import Int8OPTForCausalLM
import gc
from torch.nn.functional import pad
from datasets import load_dataset

class Evaluator:
    def __init__(self, dataset, tokenizer, device='cuda'):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.dataset = self.dataset.map(self.tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    def tokenize_function(self, examples):
        return self.tokenizer(examples['text'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        total, hit = 0, 0
        latency = 0

        torch.cuda.reset_peak_memory_stats()

        for batch in self.dataset:
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            pad_len = 512 - input_ids.shape[1]
            input_ids = pad(input_ids, (0, pad_len), value=1)

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()

            start.record()
            outputs = model(input_ids)
            end.record()

            torch.cuda.synchronize()
            latency += start.elapsed_time(end)

            last_token_logits = outputs.logits[:, -2-pad_len, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()

        accuracy = hit / total
        avg_latency = latency / len(self.dataset)
        peak_memory = torch.cuda.max_memory_allocated() / (1024**2)

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

def main():
    # --- Models changed to 30b ---
    fp16_model_name = 'facebook/opt-30b'
    int8_model_name = 'mit-han-lab/opt-30b-smoothquant'

    print("Initializing tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(fp16_model_name)
    dataset = load_dataset('lambada', split='validation[:1000]')
    evaluator = Evaluator(dataset, tokenizer)

    # --- FP16 Benchmark ---
    print(f"\n--- Benchmarking FP16 model: {fp16_model_name} ---")
    model_fp16 = OPTForCausalLM.from_pretrained(fp16_model_name, torch_dtype=torch.float16, device_map='auto')
    print_model_size(model_fp16)
    acc_fp16, latency_fp16, mem_fp16 = evaluator.evaluate(model_fp16)
    print(f'-> FP16 Accuracy: {acc_fp16:.4f}, Avg Latency: {latency_fp16:.3f}ms, Peak Memory: {mem_fp16:.2f}MB')

    del model_fp16
    gc.collect()
    torch.cuda.empty_cache()

    # --- SmoothQuant INT8 Benchmark ---
    print(f"\n--- Benchmarking SmoothQuant INT8 model: {int8_model_name} ---")
    model_smoothquant = Int8OPTForCausalLM.from_pretrained(int8_model_name, device_map='auto')
    print_model_size(model_smoothquant)
    acc_smoothquant, latency_smoothquant, mem_smoothquant = evaluator.evaluate(model_smoothquant)
    print(f'-> SmoothQuant INT8 Accuracy: {acc_smoothquant:.4f}, Avg Latency: {latency_smoothquant:.3f}ms, Peak Memory: {mem_smoothquant:.2f}MB')

if __name__ == '__main__':
    main()