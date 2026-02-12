"""
Debug BSI quantized model to understand why accuracy is still 0.0
"""

import torch
from transformers import AutoTokenizer, OPTForCausalLM
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bsi_ops
from verify_accuracy_bsi import BSIQuantizedLinear

def test_single_layer():
    """Test a single BSI quantized layer in isolation"""
    print("="*80)
    print("Testing Single BSI Layer")
    print("="*80)
    
    torch.manual_seed(42)
    linear = torch.nn.Linear(768, 768)
    
    x = torch.randn(1, 10, 768)  # [batch_size=1, seq_len=10, hidden_dim=768]
    
    with torch.no_grad():
        original_output = linear(x)
    
    print(f"\nOriginal Linear Layer:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {original_output.shape}")
    print(f"  Output stats: min={original_output.min():.6f}, max={original_output.max():.6f}, mean={original_output.mean():.6f}")
    print(f"  First 5 values: {original_output[0, 0, :5].tolist()}")
    
    for pf in [127, 255]:
        print(f"\n{'='*40}")
        print(f"Testing BSIQuantizedLinear with pf={pf}")
        print(f"{'='*40}")
        
        bsi_linear = BSIQuantizedLinear(linear, precision_factor=pf)
        
        with torch.no_grad():
            bsi_output = bsi_linear(x)
        
        print(f"  Output shape: {bsi_output.shape}")
        print(f"  Output stats: min={bsi_output.min():.6f}, max={bsi_output.max():.6f}, mean={bsi_output.mean():.6f}")
        print(f"  First 5 values: {bsi_output[0, 0, :5].tolist()}")
        
        error = torch.abs(bsi_output - original_output).mean()
        print(f"  Mean Absolute Error: {error:.6f}")
        
        if torch.all(bsi_output == 0):
            print("  WARNING: All outputs are zero!")
        if torch.any(torch.isnan(bsi_output)):
            print("  WARNING: Output contains NaN!")
        if torch.any(torch.isinf(bsi_output)):
            print("  WARNING: Output contains Inf!")

def test_in_model():
    """Test BSI quantization in actual model context"""
    print("\n" + "="*80)
    print("Testing BSI in OPT Model")
    print("="*80)
    
    model_name = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = OPTForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map="cpu"
    )
    
    text = "The capital of France is"
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        original_outputs = model(**inputs)
        original_logits = original_outputs.logits
    
    print(f"\nOriginal Model:")
    print(f"  Input shape: {inputs.input_ids.shape}")
    print(f"  Logits shape: {original_logits.shape}")
    print(f"  Logits stats: min={original_logits.min():.6f}, max={original_logits.max():.6f}, mean={original_logits.mean():.6f}")
    print(f"  Top 5 predictions for last token:")
    top5 = torch.topk(original_logits[0, -1], 5)
    for i, (val, idx) in enumerate(zip(top5.values, top5.indices)):
        token = tokenizer.decode([idx.item()])
        print(f"    {i+1}. {token}: {val.item():.4f}")
    
    print(f"\n{'='*40}")
    print("Quantizing only first linear layer with BSI")
    print(f"{'='*40}")
    
    first_linear_name = None
    first_linear = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and 'lm_head' not in name:
            first_linear_name = name
            first_linear = module
            break
    
    print(f"  Replacing layer: {first_linear_name}")
    print(f"  Original layer: in_features={first_linear.in_features}, out_features={first_linear.out_features}")
    
    parent_name = '.'.join(first_linear_name.split('.')[:-1])
    module_name = first_linear_name.split('.')[-1]
    parent = model
    if parent_name:
        for part in parent_name.split('.'):
            parent = getattr(parent, part)
    
    bsi_linear = BSIQuantizedLinear(first_linear, precision_factor=127)
    setattr(parent, module_name, bsi_linear)
    
    with torch.no_grad():
        bsi_outputs = model(**inputs)
        bsi_logits = bsi_outputs.logits
    
    print(f"\nBSI Model (1 layer quantized):")
    print(f"  Logits shape: {bsi_logits.shape}")
    print(f"  Logits stats: min={bsi_logits.min():.6f}, max={bsi_logits.max():.6f}, mean={bsi_logits.mean():.6f}")
    
    if torch.all(bsi_logits == 0):
        print("  ERROR: All logits are zero!")
    if torch.any(torch.isnan(bsi_logits)):
        print("  ERROR: Logits contain NaN!")
    if torch.any(torch.isinf(bsi_logits)):
        print("  ERROR: Logits contain Inf!")
    
    logit_diff = torch.abs(bsi_logits - original_logits).mean()
    print(f"  Mean Absolute Error in logits: {logit_diff:.6f}")
    
    print(f"  Top 5 predictions for last token:")
    top5_bsi = torch.topk(bsi_logits[0, -1], 5)
    for i, (val, idx) in enumerate(zip(top5_bsi.values, top5_bsi.indices)):
        token = tokenizer.decode([idx.item()])
        print(f"    {i+1}. {token}: {val.item():.4f}")

def debug_bsi_internals():
    """Debug BSI internal computation step by step"""
    print("\n" + "="*80)
    print("Debugging BSI Internals Step by Step")
    print("="*80)
    
    torch.manual_seed(42)
    linear = torch.nn.Linear(10, 5)
    x = torch.randn(2, 10)  # batch_size=2
    
    print(f"\nTest Setup:")
    print(f"  Weight shape: {linear.weight.shape}")
    print(f"  Input shape: {x.shape}")
    
    with torch.no_grad():
        expected = linear(x)
    print(f"  Expected output shape: {expected.shape}")
    print(f"  Expected values: {expected.tolist()}")
    
    pf = 127
    weight = linear.weight.data  # [5, 10]
    bias = linear.bias.data
    
    print(f"\nBSI Computation (pf={pf}):")
    
    for i in range(x.shape[0]):
        print(f"\n  Sample {i}:")
        input_vec = x[i].detach().cpu().contiguous()
        print(f"    Input vec: {input_vec[:5].tolist()}")
        
        scores, _, _, _ = bsi_ops.batch_dot_product(
            input_vec,
            weight,
            float(pf)
        )
        print(f"    Raw BSI scores: {scores.tolist()}")
        
        scores_scaled = scores / (pf * pf)
        print(f"    Scaled scores (÷{pf*pf}): {scores_scaled.tolist()}")
        
        final = scores_scaled + bias
        print(f"    With bias: {final.tolist()}")
        print(f"    Expected: {expected[i].tolist()}")
        
        error = torch.abs(final - expected[i]).mean()
        print(f"    Error: {error:.6f}")

if __name__ == "__main__":
    test_single_layer()
    test_in_model()
    debug_bsi_internals()
    
    print("\n" + "="*80)
    print("DEBUGGING SUMMARY:")
    print("Check the outputs above to identify:")
    print("1. Are BSI outputs all zeros, NaN, or Inf?")
    print("2. Is the scaling working correctly?")
    print("3. Is the issue with specific layer sizes?")
    print("4. Is bias handling correct?")
    print("="*80)
