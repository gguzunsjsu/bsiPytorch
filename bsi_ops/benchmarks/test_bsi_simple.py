#!/usr/bin/env python3
"""Simple test to verify BSI operations are working correctly."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import bsi_ops

def test_bsi_operations():
    print("=" * 60)
    print("Testing BSI Operations")
    print("=" * 60)
    
    # Test 1: Simple dot product with known values
    print("\nTest 1: Simple dot product")
    print("-" * 40)
    
    # Create simple test vectors
    vec1 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    vec2 = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float32)
    
    # Expected result: 1*4 + 2*3 + 3*2 + 4*1 = 4 + 6 + 6 + 4 = 20
    expected = torch.dot(vec1, vec2).item()
    print(f"Input vec1: {vec1.tolist()}")
    print(f"Input vec2: {vec2.tolist()}")
    print(f"Expected dot product: {expected}")
    
    # Test with different decimal-place settings
    for decimal_places in [0, 1, 2]:
        result, time_ns, bsi1_size, bsi2_size = bsi_ops.dot_product_decimal(vec1, vec2, decimal_places)
        print(f"\nDecimal places {decimal_places}:")
        print(f"  BSI result: {result:.6f}")
        print(f"  Error: {abs(result - expected):.6f}")
        print(f"  BSI sizes: {bsi1_size} bytes, {bsi2_size} bytes")
    
    # Test 2: Batch dot product with prebuilt keys
    print("\n\nTest 2: Batch dot product with prebuilt keys")
    print("-" * 40)
    
    # Create a simple weight matrix (3 keys, 4 dims)
    K = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0],
        [2.0, 2.0, 2.0, 2.0]
    ], dtype=torch.float32)
    
    # Query vector
    q = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    
    print(f"Weight matrix K shape: {K.shape}")
    print(f"Query vector q: {q.tolist()}")
    
    # Expected results using torch
    expected_scores = torch.matmul(K, q)
    print(f"Expected scores (torch.matmul): {expected_scores.tolist()}")
    
    for decimal_places in [0, 1, 2]:
        print(f"\nTesting with decimal places {decimal_places}:")
        
        keys_capsule, total_mem, num_keys, d = bsi_ops.build_bsi_keys(K, decimal_places, 0.2)
        print(f"  Built {num_keys} BSI keys, dimension {d}, total memory: {total_mem} bytes")
        
        scores, time_ns, query_mem = bsi_ops.batch_dot_product_prebuilt(q, keys_capsule, 0.2)
        print(f"  BSI scores: {scores.tolist()}")
        print(f"  Expected: {expected_scores.tolist()}")
        print(f"  Errors: {[abs(a-b) for a,b in zip(scores.tolist(), expected_scores.tolist())]}")
        print(f"  Max error: {(scores - expected_scores).abs().max().item():.6f}")
    
    # Test 3: Check if C++ is scaling correctly
    print("\n\nTest 3: Verify C++ scaling behavior")
    print("-" * 40)
    
    # Use larger values to see scaling effects
    vec1 = torch.tensor([100.0, 200.0, 300.0], dtype=torch.float32)
    vec2 = torch.tensor([3.0, 2.0, 1.0], dtype=torch.float32)
    
    expected = torch.dot(vec1, vec2).item()
    print(f"Large vec1: {vec1.tolist()}")
    print(f"Small vec2: {vec2.tolist()}")
    print(f"Expected dot product: {expected}")
    
    decimal_places = 2
    result, _, _, _ = bsi_ops.dot_product_decimal(vec1, vec2, decimal_places)
    print(f"\nWith decimal places={decimal_places}:")
    print(f"  BSI result: {result:.2f}")
    print(f"  Expected: {expected:.2f}")
    print(f"  Error: {abs(result - expected):.2f}")
    print(f"  Error as % of expected: {100 * abs(result - expected) / abs(expected):.2f}%")
    
    scale = 10 ** decimal_places
    unscaled_result = expected * (scale * scale)
    print(f"\nIf C++ wasn't dividing by 10^{2 * decimal_places}, result would be: {unscaled_result:.2f}")
    print(f"  Ratio of actual to unscaled: {result / unscaled_result:.6f}")

if __name__ == "__main__":
    test_bsi_operations()
