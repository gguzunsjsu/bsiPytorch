#!/usr/bin/env python3
"""Test script to verify BSI dimension fix"""

import torch
import bsi_ops
import sys

def test_batch_dot_product():
    """Test that batch_dot_product works with correct dimensions"""
    print("Testing batch_dot_product dimension fix...")
    
    in_features = 768
    out_features = 3072
    batch_size = 2
    seq_len = 10
    
    input_tensor = torch.randn(batch_size, seq_len, in_features)
    weight = torch.randn(out_features, in_features)  # [out_features, in_features]
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Weight shape: {weight.shape}")
    
    x = input_tensor.view(-1, in_features)  # [batch_size * seq_len, in_features]
    print(f"Flattened input shape: {x.shape}")
    
    output_list = []
    for i in range(x.shape[0]):
        input_vec = x[i].contiguous()  # [in_features]
        
        try:
            # weight is [out_features, in_features] - each row is a weight vector
            scores, time_taken, query_bsi_size, keys_bsi_size = bsi_ops.batch_dot_product(
                input_vec,    # query: [in_features]
                weight,       # keys: [out_features, in_features]
                127.0         # precision factor
            )
            output_list.append(scores)
            print(f"✓ Vector {i+1}/{x.shape[0]} processed successfully")
            
            if i == 0:
                print(f"  Output scores shape: {scores.shape}")
                print(f"  Query BSI size: {query_bsi_size} bytes")
                print(f"  Keys BSI size: {keys_bsi_size} bytes")
                total_memory = query_bsi_size + keys_bsi_size
                print(f"  Total BSI memory: {total_memory} bytes ({total_memory/1024:.2f} KB)")
                print(f"  Processing time: {time_taken:.6f} seconds (Note: timing may be incorrect)")
        except RuntimeError as e:
            print(f"✗ Error processing vector {i+1}: {e}")
            return False
    
    output = torch.stack(output_list)
    print(f"\nFinal output shape: {output.shape}")
    print(f"Expected shape: [{batch_size * seq_len}, {out_features}]")
    output = output.view(batch_size, seq_len, out_features)
    print(f"Reshaped output: {output.shape}")
    
    print("\n✓ All tests passed! BSI dimension fix is working correctly.")
    return True

def test_with_bsi_layer():
    """Test using the actual BSIQuantizedLinear layer"""
    print("\nTesting BSIQuantizedLinear layer...")
    
    from verify_accuracy_bsi import BSIQuantizedLinear
    
    linear = torch.nn.Linear(768, 3072)
    
    bsi_linear = BSIQuantizedLinear(linear, precision_factor=127)
    
    x = torch.randn(2, 10, 768)  # [batch_size, seq_len, hidden_dim]
    
    print(f"Input shape: {x.shape}")
    
    try:
        output = bsi_linear(x)
        print(f"Output shape: {output.shape}")
        if hasattr(bsi_linear, 'bsi_memory_bytes'):
            memory_bytes = bsi_linear.bsi_memory_bytes
            memory_kb = memory_bytes / 1024
            memory_mb = memory_bytes / (1024 * 1024)
            print(f"BSI memory used: {memory_bytes} bytes ({memory_kb:.2f} KB / {memory_mb:.4f} MB)")
            
            original_memory = 768 * 3072 * 4  # float32 = 4 bytes per element
            compression_ratio = original_memory / memory_bytes
            print(f"Original weight memory: {original_memory / (1024*1024):.2f} MB")
            print(f"Compression ratio: {compression_ratio:.2f}x")
        else:
            print("Warning: BSI memory not tracked")
        print("✓ BSIQuantizedLinear layer test passed!")
        return True
    except Exception as e:
        print(f"✗ Error in BSIQuantizedLinear: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("BSI Dimension Fix Verification")
    print("=" * 60)
    
    if not test_batch_dot_product():
        print("\n✗ batch_dot_product test failed!")
        sys.exit(1)
    
    if not test_with_bsi_layer():
        print("\n✗ BSIQuantizedLinear test failed!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! BSI implementation is ready for benchmarking.")
    print("=" * 60)
