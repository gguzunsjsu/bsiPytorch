"""Quick test to verify the BSI fix handles negative values correctly."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import bsi_ops

def test_decimal_dots():
    print("=" * 60)
    print("Testing with decimal dot product")
    print("=" * 60)

def test_dot_positive_withoutDecimal():
    print("*" * 60)
    print("Testing dot with positive numbers")
    print("*" * 60)

    vec1 = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    vec2 = torch.tensor([4, 3, 2, 1], dtype=torch.float32)
    expected = torch.dot(vec1, vec2).item()
    
    #printng torch dot results
    print(f"Vec1 (with positive numbers):  {vec1.tolist()}")
    print(f"Vec2 (with positive numbers):  {vec2.tolist()}")
    print(f"Expected dot product: {expected}")

    #testing with BSI
    # pf = 127
    result, _, _, _ = bsi_ops.dot_product_decimal(vec1, vec2, 2)
    print(f"\nBSI result: {result:.6f}")
    print(f"Error: {abs(result - expected):.6f}")
    print(f"Success: {'✓' if abs(result - expected) < 0.01 else '✗'}")

def test_dot_positive_withDecimal():
    print("\n" + "*" * 60)
    print("Testing dot with positive numbers and decimal values")
    print("*" * 60)

    vec1 = torch.tensor([-1.5, 2.5, -3.5, -4.5], dtype=torch.float32)
    vec2 = torch.tensor([4.5, -3.5, 2.5, 1.5], dtype=torch.float32)
    expected = torch.dot(vec1, vec2).item()
    
    #printing torch dot results
    print(f"Vec1 (with positive decimal numbers):  {vec1.tolist()}")
    print(f"Vec2 (with positive decimal numbers):  {vec2.tolist()}")
    print(f"Expected dot product: {expected}")

    #testing with BSI
    # pf = 31
    result, _, _, _ = bsi_ops.dot_product_decimal(vec1, vec2, 2)
    print(f"\nBSI result: {result:.6f}")
    print(f"Error: {abs(result - expected):.6f}")
    print(f"Success: {'✓' if abs(result - expected) < 0.01 else '✗'}")

def test_positiveFP_bsiBuildWithDecimal():
    print("*" * 60)
    print("Testing BSI with positive float values and using buldBSI(<vec1>, <vec2>, decimalpoints)")
    print("*" * 60)

    vec1 = torch.tensor([0.02, 0.04, 0.08, 0.16], dtype=torch.float32)
    vec2 = torch.tensor([0.505, 0.123, 0.216, 0.512], dtype=torch.float32)
    expected = torch.dot(vec1, vec2).item()
    print(f"Vec1 (with positive float numbers):  {vec1.tolist()}")
    print(f"Vec2 (with positive float numbers):  {vec2.tolist()}")
    print(f"Expected dot product: {expected}")

def test_negative_values():
    print("=" * 60)
    print("Testing BSI with Negative Values")
    print("=" * 60)
    
    # Test vectors with negative values
    vec1 = torch.tensor([-1.0, 2.0, -3.0, 4.0], dtype=torch.float32)
    vec2 = torch.tensor([4.0, -3.0, 2.0, -1.0], dtype=torch.float32)
    
    # Expected result: (-1)*4 + 2*(-3) + (-3)*2 + 4*(-1) = -4 - 6 - 6 - 4 = -20
    expected = torch.dot(vec1, vec2).item()
    print(f"Vec1 (with negatives): {vec1.tolist()}")
    print(f"Vec2 (with negatives): {vec2.tolist()}")
    print(f"Expected dot product: {expected}")
    
    # Test with BSI
    decimal_places = 2
    result, _, _, _ = bsi_ops.dot_product_decimal(vec1, vec2, decimal_places)
    print(f"\nBSI result (decimal_places={decimal_places}): {result:.6f}")
    print(f"Error: {abs(result - expected):.6f}")
    print(f"Success: {'✓' if abs(result - expected) < 0.01 else '✗'}")
    
    # Test batch dot product with negatives
    print("\n" + "-" * 40)
    print("Testing batch dot product with negatives:")
    
    # Weight matrix with negative values
    K = torch.tensor([
        [-1.0, 2.0, -3.0, 4.0],
        [4.0, -3.0, 2.0, -1.0],
        [-2.0, -2.0, 2.0, 2.0]
    ], dtype=torch.float32)
    
    q = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float32)
    
    # Expected using torch
    expected_scores = torch.matmul(K, q)
    print(f"Query: {q.tolist()}")
    print(f"Expected scores: {expected_scores.tolist()}")
    
    # BSI prebuilt
    keys_capsule, _, _, _ = bsi_ops.build_bsi_keys(K, decimal_places, 0.2)
    scores, _, _ = bsi_ops.batch_dot_product_prebuilt(q, keys_capsule, 0.2)
    print(f"BSI scores: {scores.tolist()}")
    print(f"Errors: {[abs(a-b) for a,b in zip(scores.tolist(), expected_scores.tolist())]}")
    max_error = (scores - expected_scores).abs().max().item()
    print(f"Max error: {max_error:.6f}")
    print(f"Success: {'✓' if max_error < 0.01 else '✗'}")

def test_large_negative_range():
    print("\n" + "=" * 60)
    print("Testing Large Negative Range")
    print("=" * 60)
    
    # Test with large negative values (like real model weights)
    vec1 = torch.randn(768) * 0.1 - 0.05  # Mean around -0.05
    vec2 = torch.randn(768) * 0.1
    
    print(f"Vec1 stats: min={vec1.min():.4f}, max={vec1.max():.4f}, mean={vec1.mean():.4f}")
    print(f"Vec2 stats: min={vec2.min():.4f}, max={vec2.max():.4f}, mean={vec2.mean():.4f}")
    
    expected = torch.dot(vec1, vec2).item()
    print(f"Expected dot product: {expected:.4f}")
    
    decimal_places = 2
    result, _, _, _ = bsi_ops.dot_product_decimal(vec1, vec2, decimal_places)
    print(f"BSI result: {result:.4f}")
    print(f"Error: {abs(result - expected):.4f}")
    print(f"Relative error: {100 * abs(result - expected) / abs(expected):.2f}%")
    print(f"Success: {'✓' if abs(result - expected) / abs(expected) < 0.1 else '✗'}")



if __name__ == "__main__":
    test_negative_values()
    test_large_negative_range()
    test_dot_positive_withDecimal()
    test_dot_positive_withoutDecimal()
    test_positiveFP_bsiBuildWithDecimal()
    
    print("\n" + "=" * 60)
    print("If all tests show ✓, the fix is working correctly!")
    print("If any test shows ✗, the BsiSigned fix may not be applied.")
    print("=" * 60)
