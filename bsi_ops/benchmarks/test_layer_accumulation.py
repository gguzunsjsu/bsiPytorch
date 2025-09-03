#!/usr/bin/env python3
"""Test to check if BSI outputs are accumulating incorrectly through layers."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import bsi_ops
from benchmarks.verify_accuracy_bsi import BSIQuantizedLinear

def test_sequential_layers():
    print("=" * 60)
    print("Testing Sequential BSI Layers")
    print("=" * 60)
    
    # Create a sequence of linear layers
    in_features = 768
    hidden_features = 768
    num_layers = 3
    pf = 127
    
    # Create sequential model with normal linear layers
    normal_model = nn.Sequential()
    for i in range(num_layers):
        layer = nn.Linear(in_features if i == 0 else hidden_features, hidden_features)
        # Initialize with small weights to avoid explosion
        with torch.no_grad():
            layer.weight.data = torch.randn_like(layer.weight) * 0.02
            if layer.bias is not None:
                layer.bias.data = torch.zeros_like(layer.bias)
        normal_model.add_module(f"layer_{i}", layer)
    
    # Create BSI version
    bsi_model = nn.Sequential()
    for i in range(num_layers):
        orig_layer = normal_model[i]
        bsi_layer = BSIQuantizedLinear(orig_layer, precision_factor=pf, compress_threshold=0.0)
        bsi_model.add_module(f"layer_{i}", bsi_layer)
    
    # Test input
    test_input = torch.randn(1, in_features) * 0.1
    
    print(f"Input stats: mean={test_input.mean():.4f}, std={test_input.std():.4f}")
    print(f"Input range: [{test_input.min():.4f}, {test_input.max():.4f}]")
    
    # Forward through normal model
    x_normal = test_input
    print("\nNormal model forward pass:")
    for i, layer in enumerate(normal_model):
        x_normal = layer(x_normal)
        print(f"  After layer {i}: mean={x_normal.mean():.4f}, std={x_normal.std():.4f}, "
              f"range=[{x_normal.min():.4f}, {x_normal.max():.4f}]")
    
    # Forward through BSI model
    x_bsi = test_input
    print("\nBSI model forward pass:")
    for i, layer in enumerate(bsi_model):
        x_bsi = layer(x_bsi)
        print(f"  After layer {i}: mean={x_bsi.mean():.4f}, std={x_bsi.std():.4f}, "
              f"range=[{x_bsi.min():.4f}, {x_bsi.max():.4f}]")
        
        # Check if values are exploding
        if x_bsi.abs().max() > 1000:
            print(f"  WARNING: Values exploding at layer {i}!")
            print(f"  First 5 values: {x_bsi.flatten()[:5].tolist()}")
            break
    
    # Compare final outputs
    print("\n" + "=" * 60)
    print("Final comparison:")
    print(f"Normal output range: [{x_normal.min():.4f}, {x_normal.max():.4f}]")
    print(f"BSI output range: [{x_bsi.min():.4f}, {x_bsi.max():.4f}]")
    print(f"Ratio of means: {x_bsi.mean() / x_normal.mean() if x_normal.mean() != 0 else 0:.2f}")

def test_residual_connections():
    """Test if the issue is related to residual connections in transformers."""
    print("\n" + "=" * 60)
    print("Testing with Residual Connections (like in Transformers)")
    print("=" * 60)
    
    in_features = 768
    pf = 127
    
    # Create a simple attention-like block
    class SimpleAttentionBlock(nn.Module):
        def __init__(self, use_bsi=False):
            super().__init__()
            self.q_proj = nn.Linear(in_features, in_features)
            self.k_proj = nn.Linear(in_features, in_features)
            self.v_proj = nn.Linear(in_features, in_features)
            self.out_proj = nn.Linear(in_features, in_features)
            
            # Initialize with small weights
            for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
                with torch.no_grad():
                    module.weight.data = torch.randn_like(module.weight) * 0.02
                    if module.bias is not None:
                        module.bias.data = torch.zeros_like(module.bias)
            
            if use_bsi:
                self.q_proj = BSIQuantizedLinear(self.q_proj, pf, 0.0)
                self.k_proj = BSIQuantizedLinear(self.k_proj, pf, 0.0)
                self.v_proj = BSIQuantizedLinear(self.v_proj, pf, 0.0)
                self.out_proj = BSIQuantizedLinear(self.out_proj, pf, 0.0)
        
        def forward(self, x):
            # Simple attention without actual attention computation
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            
            # Simplified: just use v directly (skipping attention)
            attn_output = v
            
            # Output projection
            output = self.out_proj(attn_output)
            
            # Residual connection
            return x + output
    
    # Test input
    test_input = torch.randn(1, 32, in_features) * 0.1  # batch=1, seq_len=32
    
    # Normal block
    normal_block = SimpleAttentionBlock(use_bsi=False)
    normal_output = normal_block(test_input)
    
    print(f"Normal block:")
    print(f"  Input range: [{test_input.min():.4f}, {test_input.max():.4f}]")
    print(f"  Output range: [{normal_output.min():.4f}, {normal_output.max():.4f}]")
    
    # BSI block
    bsi_block = SimpleAttentionBlock(use_bsi=True)
    bsi_output = bsi_block(test_input)
    
    print(f"\nBSI block:")
    print(f"  Input range: [{test_input.min():.4f}, {test_input.max():.4f}]")
    print(f"  Output range: [{bsi_output.min():.4f}, {bsi_output.max():.4f}]")
    
    # Check if residual is causing issues
    print(f"\nDifference in outputs:")
    print(f"  Mean absolute difference: {(bsi_output - normal_output).abs().mean():.4f}")
    print(f"  Max absolute difference: {(bsi_output - normal_output).abs().max():.4f}")
    
    # Stack multiple blocks to see accumulation
    print("\n" + "=" * 60)
    print("Stacking 3 blocks:")
    
    x_normal = test_input
    x_bsi = test_input
    
    for i in range(3):
        # Create new blocks for each layer
        normal_block = SimpleAttentionBlock(use_bsi=False)
        bsi_block = SimpleAttentionBlock(use_bsi=True)
        
        x_normal = normal_block(x_normal)
        x_bsi = bsi_block(x_bsi)
        
        print(f"\nAfter block {i+1}:")
        print(f"  Normal range: [{x_normal.min():.4f}, {x_normal.max():.4f}]")
        print(f"  BSI range: [{x_bsi.min():.4f}, {x_bsi.max():.4f}]")
        
        if x_bsi.abs().max() > 1000:
            print(f"  WARNING: BSI values exploding!")
            break

if __name__ == "__main__":
    test_sequential_layers()
    test_residual_connections()
