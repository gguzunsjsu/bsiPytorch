import torch
from transformers import AutoTokenizer, OPTForCausalLM
from datasets import load_dataset
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import quantize_model

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
    model_name = "facebook/opt-1.3b"
    device = "cuda"

    print("Initializing tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset("lambada", split="validation[:1000]")
    evaluator = Evaluator(dataset, tokenizer, device)

    print(f"Loading FP16 model: {model_name}")
    model_fp16 = OPTForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    acc_fp16 = evaluator.evaluate(model_fp16)
    print(f"Original model (FP16) accuracy: {acc_fp16:.4f}\n")

    print("Applying naive W8A8 quantization...")
    model_w8a8 = OPTForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    quantize_model(model_w8a8)
    acc_w8a8 = evaluator.evaluate(model_w8a8)
    print(f"Naive W8A8 quantized model accuracy: {acc_w8a8:.4f}\n")
    del model_w8a8

    print("Applying SmoothQuant W8A8 quantization...")
    model_smooth = OPTForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )
    act_scales = torch.load("act_scales/opt-1.3b.pt") 
    smooth_lm(model_smooth, act_scales, 0.5)
    model_smoothquant_w8a8 = quantize_model(model_smooth)

    acc_smoothquant_w8a8 = evaluator.evaluate(model_smoothquant_w8a8)
    print(f"SmoothQuant W8A8 quantized model accuracy: {acc_smoothquant_w8a8:.4f}\n")

    print("Script finished successfully!")

if __name__ == "__main__":
    main()