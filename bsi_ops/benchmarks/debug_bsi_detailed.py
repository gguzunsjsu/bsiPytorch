"""
Detailed debugging of BSI quantization issue with dtype tracking
"""

import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bsi_ops

def test_batch_dot_product_directly():
    """Test batch_dot_product return types and values"""
    print("="*80)
    print("Testing batch_dot_product directly")
    print("="*80)
    
    torch.manual_seed(42)
    q = torch.randn(10).float()
    K = torch.randn(5, 10).float()
    
    print(f"Input dtypes: q={q.dtype}, K={K.dtype}")
    
    pf = 127
    scores, time_taken, query_bsi_size, keys_bsi_size = bsi_ops.batch_dot_product(
        q, K, float(pf)
    )
    
    print(f"\nRaw batch_dot_product output:")
    print(f"  scores type: {type(scores)}")
    print(f"  scores dtype: {scores.dtype}")
    print(f"  scores shape: {scores.shape}")
    print(f"  scores values: {scores.tolist()}")
    
    scores_scaled = scores / (pf * pf)
    print(f"\nAfter scaling by pf² ({pf*pf}):")
    print(f"  dtype: {scores_scaled.dtype}")
    print(f"  values: {scores_scaled.tolist()}")
    
    scores_float32 = scores_scaled.to(torch.float32)
    scores_float16 = scores_scaled.to(torch.float16)
    
    print(f"\nAfter dtype conversion:")
    print(f"  float32: {scores_float32.tolist()}")
    print(f"  float16: {scores_float16.tolist()}")
    
    if torch.all(scores_float16 == 0):
        print("  WARNING: float16 conversion resulted in all zeros!")

def test_bsi_linear_step_by_step():
    """Test BSIQuantizedLinear computation step by step"""
    print("\n" + "="*80)
    print("Testing BSIQuantizedLinear step by step")
    print("="*80)
    
    from verify_accuracy_bsi import BSIQuantizedLinear
    
    torch.manual_seed(42)
    linear = torch.nn.Linear(10, 5)
    x = torch.randn(2, 10)
    
    print(f"Setup:")
    print(f"  Linear weight shape: {linear.weight.shape}")
    print(f"  Input shape: {x.shape}, dtype: {x.dtype}")
    
    with torch.no_grad():
        expected = linear(x)
    print(f"  Expected output: {expected.tolist()}")
    
    bsi_linear = BSIQuantizedLinear(linear, precision_factor=127)
    
    print(f"\nManual BSI forward pass:")
    original_device = x.device
    original_dtype = x.dtype
    original_shape = x.shape
    
    print(f"  Original dtype: {original_dtype}")
    print(f"  BSI weight dtype: {bsi_linear.weight.dtype}")
    
    output_list = []
    for i in range(x.shape[0]):
        print(f"\n  Processing sample {i}:")
        input_vec = x[i].detach().cpu().contiguous()
        print(f"    input_vec dtype: {input_vec.dtype}")
        
        scores, _, _, _ = bsi_ops.batch_dot_product(
            input_vec,
            bsi_linear.weight,
            float(bsi_linear.precision_factor)
        )
        print(f"    BSI scores dtype: {scores.dtype}")
        print(f"    BSI scores (C++ already scaled): {scores.tolist()}")
        
        output_list.append(scores)
    
    output = torch.stack(output_list)
    print(f"\n  After torch.stack:")
    print(f"    dtype: {output.dtype}")
    print(f"    values: {output.tolist()}")
    
    output = output.to(original_device).to(original_dtype)
    print(f"\n  After .to(original_dtype={original_dtype}):")
    print(f"    dtype: {output.dtype}")
    print(f"    values: {output.tolist()}")
    
    if bsi_linear.bias is not None:
        bias_on_device = bsi_linear.bias.to(original_device)
        print(f"\n  Bias dtype: {bias_on_device.dtype}")
        print(f"  Bias values: {bias_on_device.tolist()}")
        output = output + bias_on_device
        print(f"  After adding bias:")
        print(f"    values: {output.tolist()}")
    
    print(f"\nComparison:")
    print(f"  Expected: {expected.tolist()}")
    print(f"  Got: {output.tolist()}")
    error = torch.abs(output - expected).mean()
    print(f"  Mean Absolute Error: {error:.6f}")

def test_with_float16():
    """Test specifically with float16 which models often use"""
    print("\n" + "="*80)
    print("Testing with float16 (common in models)")
    print("="*80)
    
    from verify_accuracy_bsi import BSIQuantizedLinear
    
    torch.manual_seed(42)
    
    linear = torch.nn.Linear(768, 768)
    linear = linear.to(torch.float16)
    
    x = torch.randn(1, 10, 768, dtype=torch.float16)
    
    print(f"Setup:")
    print(f"  Weight dtype: {linear.weight.dtype}")
    print(f"  Input dtype: {x.dtype}")
    
    with torch.no_grad():
        expected = linear(x)
    print(f"  Expected output stats: min={expected.min():.6f}, max={expected.max():.6f}, mean={expected.mean():.6f}")
    
    bsi_linear = BSIQuantizedLinear(linear, precision_factor=127)
    
    with torch.no_grad():
        output = bsi_linear(x)
    
    print(f"\nBSI output:")
    print(f"  Output dtype: {output.dtype}")
    print(f"  Output stats: min={output.min():.6f}, max={output.max():.6f}, mean={output.mean():.6f}")
    
    if torch.all(output == 0):
        print("  ERROR: All outputs are zero!")
        
        print("\n  Debugging zero output:")
        input_vec = x[0, 0].detach().cpu().contiguous().to(torch.float32)
        weight_cpu = bsi_linear.weight.to(torch.float32)
        
        scores, _, _, _ = bsi_ops.batch_dot_product(
            input_vec,
            weight_cpu,
            float(127)
        )
        scores = scores / (127 * 127)
        
        print(f"    Manual computation (float32):")
        print(f"      scores min={scores.min():.6f}, max={scores.max():.6f}")
        print(f"      After converting to float16: {scores.to(torch.float16)[:5].tolist()}")

if __name__ == "__main__":
    test_batch_dot_product_directly()
    test_bsi_linear_step_by_step()
    test_with_float16()
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("1. Check if batch_dot_product returns float64 (double)")
    print("2. Check if conversion to float16 causes underflow")
    print("3. Check if the weight needs to be float32 for BSI")
    print("="*80)
