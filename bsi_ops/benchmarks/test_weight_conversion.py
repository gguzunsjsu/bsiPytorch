"""Test weight conversion in BSIQuantizedLinear to find the scaling issue."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import bsi_ops
from benchmarks.verify_accuracy_bsi import BSIQuantizedLinear

def test_weight_conversion():
    print("=" * 60)
    print("Testing Weight Conversion in BSIQuantizedLinear")
    print("=" * 60)
    
    # Create a simple linear layer with known weights
    in_features = 4
    out_features = 3
    
    # Create a linear layer
    linear = nn.Linear(in_features, out_features, bias=True)
    
    # Set specific weights for testing
    with torch.no_grad():
        linear.weight = nn.Parameter(torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [2.0, 2.0, 2.0, 2.0]
        ], dtype=torch.float32))
        linear.bias = nn.Parameter(torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32))
    
    print(f"Original weight shape: {linear.weight.shape}")
    print(f"Original weight:\n{linear.weight}")
    print(f"Original bias: {linear.bias}")
    
    # Convert to BSIQuantizedLinear
    decimal_places = 2
    bsi_linear = BSIQuantizedLinear(linear, decimalPlaces=decimal_places, compress_threshold=0.2)
    
    print(f"\nBSIQuantizedLinear created with decimalPlaces={decimal_places}")
    print(f"Stored weight_fp32 shape: {bsi_linear.weight_fp32.shape}")
    print(f"Stored weight_fp32:\n{bsi_linear.weight_fp32}")
    
    # Test with a simple input
    test_input = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    print(f"\nTest input: {test_input}")
    
    # Expected output (using original linear layer)
    with torch.no_grad():
        expected = linear(test_input.unsqueeze(0)).squeeze(0)
    print(f"Expected output (dense): {expected}")
    
    # BSI output
    with torch.no_grad():
        bsi_output = bsi_linear(test_input.unsqueeze(0)).squeeze(0)
    print(f"BSI output: {bsi_output}")
    print(f"Difference: {(bsi_output - expected).abs()}")
    
    # Now test with the same weights but calling BSI directly
    print("\n" + "=" * 60)
    print("Direct BSI call with same weights:")
    
    # Build BSI keys directly
    keys_capsule, total_mem, num_keys, d = bsi_ops.build_bsi_keys(
        linear.weight.detach().cpu(), decimal_places, 0.2
    )
    print(f"Built {num_keys} BSI keys, dimension {d}")
    
    # Do batch dot product directly
    scores, _, _ = bsi_ops.batch_dot_product_prebuilt(test_input, keys_capsule, 0.2)
    # Add bias manually
    scores = scores + linear.bias.detach().cpu()
    print(f"Direct BSI scores (with bias): {scores}")
    print(f"Expected (with bias): {expected}")
    print(f"Difference: {(scores - expected.cpu()).abs()}")
    
    # Check if there's a discrepancy in how weights are being passed
    print("\n" + "=" * 60)
    print("Checking weight scaling:")
    scale = 10 ** decimal_places
    scaled_input = test_input * scale
    print(f"Input scaled by 10^{decimal_places}: {scaled_input}")
    raw_dot = torch.matmul(linear.weight, test_input)
    print(f"Raw dot product (no BSI): {raw_dot}")
    both_scaled = torch.matmul(linear.weight * scale, scaled_input)
    print(f"Both scaled by 10^{decimal_places}: {both_scaled}")
    print(f"Both scaled / 10^{2 * decimal_places}: {both_scaled / (scale * scale)}")

def test_opt_weight_range():
    """Check the actual weight ranges in OPT-125m model."""
    print("\n" + "=" * 60)
    print("Checking OPT-125m weight ranges")
    print("=" * 60)
    
    from transformers import OPTForCausalLM
    
    decimal_places = 2
    scale = 10 ** decimal_places

    model = OPTForCausalLM.from_pretrained(
        "facebook/opt-125m",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    # Check a few attention layers
    for i in [0, 5, 11]:
        layer = model.model.decoder.layers[i]
        
        # Check q_proj weights
        w = layer.self_attn.q_proj.weight
        print(f"\nLayer {i} q_proj weight stats:")
        print(f"  Shape: {w.shape}")
        print(f"  Min: {w.min().item():.6f}, Max: {w.max().item():.6f}")
        print(f"  Mean: {w.mean().item():.6f}, Std: {w.std().item():.6f}")
        print(f"  First 5 values: {w.flatten()[:5].tolist()}")
        
        # Check what happens when we scale by 10^decimal_places
        w_scaled = w * scale
        print(f"  Scaled by 10^{decimal_places} - Min: {w_scaled.min().item():.2f}, Max: {w_scaled.max().item():.2f}")

if __name__ == "__main__":
    test_weight_conversion()
    test_opt_weight_range()
