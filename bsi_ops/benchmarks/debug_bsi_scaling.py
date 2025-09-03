"""
Debug script to identify and fix BSI quantization scaling issues
"""

import torch
import bsi_ops

def test_bsi_scaling():
    """Test BSI dot product scaling to understand the issue"""
    
    torch.manual_seed(42)
    q = torch.randn(10).float()
    K = torch.randn(5, 10).float()  # 5 keys, 10 dimensions
    
    precision_factors = [15, 31, 63, 127, 255]
    
    print("="*80)
    print("Testing BSI Scaling Issues")
    print("="*80)
    
    ground_truth = torch.matmul(K, q)
    print(f"\nGround truth (PyTorch matmul):")
    print(f"  Result shape: {ground_truth.shape}")
    print(f"  Values: {ground_truth[:3].tolist()}")
    print(f"  Min: {ground_truth.min():.6f}, Max: {ground_truth.max():.6f}")
    
    for pf in precision_factors:
        print(f"\n{'='*40}")
        print(f"Precision Factor: {pf}")
        print(f"{'='*40}")
        
        # Test batch_dot_product
        scores, time_taken, query_bsi_size, keys_bsi_size = bsi_ops.batch_dot_product(
            q,  # query
            K,  # keys matrix
            float(pf)
        )
        
        print(f"BSI batch_dot_product raw output:")
        print(f"  Result shape: {scores.shape}")
        print(f"  Raw values: {scores[:3].tolist()}")
        print(f"  Min: {scores.min():.6f}, Max: {scores.max():.6f}")
        
        # Try different scaling methods
        scaled_by_pf = scores / pf
        print(f"\nScaled by pf (÷{pf}):")
        print(f"  Values: {scaled_by_pf[:3].tolist()}")
        print(f"  Min: {scaled_by_pf.min():.6f}, Max: {scaled_by_pf.max():.6f}")
        
        scaled_by_pf2 = scores / (pf * pf)
        print(f"\nScaled by pf² (÷{pf*pf}):")
        print(f"  Values: {scaled_by_pf2[:3].tolist()}")
        print(f"  Min: {scaled_by_pf2.min():.6f}, Max: {scaled_by_pf2.max():.6f}")
        
        # Compute error compared to ground truth
        error_raw = torch.abs(scores - ground_truth).mean()
        error_pf = torch.abs(scaled_by_pf - ground_truth).mean()
        error_pf2 = torch.abs(scaled_by_pf2 - ground_truth).mean()
        
        print(f"\nMean Absolute Errors:")
        print(f"  Raw BSI output: {error_raw:.6f}")
        print(f"  Scaled by pf: {error_pf:.6f}")
        print(f"  Scaled by pf²: {error_pf2:.6f}")
        
        # Find best scaling factor empirically
        best_scale = ground_truth[0].item() / scores[0].item() if scores[0].item() != 0 else 1.0
        print(f"\nEmpirical best scale factor: {best_scale:.6f}")
        print(f"  Ratio to pf²: {(pf*pf)/best_scale:.6f}")

def test_linear_layer():
    """Test BSI quantized linear layer behavior"""
    print("\n" + "="*80)
    print("Testing BSI Quantized Linear Layer")
    print("="*80)
    
    torch.manual_seed(42)
    linear = torch.nn.Linear(10, 5)
    
    x = torch.randn(2, 10)
    
    with torch.no_grad():
        ground_truth = linear(x)
    
    print(f"\nGround truth output:")
    print(f"  Shape: {ground_truth.shape}")
    print(f"  First sample: {ground_truth[0].tolist()}")
    print(f"  Min: {ground_truth.min():.6f}, Max: {ground_truth.max():.6f}")
    
    for pf in [127, 255]:
        print(f"\n{'='*40}")
        print(f"Testing with pf={pf}")
        print(f"{'='*40}")
        
        weight = linear.weight.data  # [out_features, in_features]
        bias = linear.bias.data if linear.bias is not None else None
        
        output_list = []
        for i in range(x.shape[0]):
            input_vec = x[i].detach().cpu().contiguous()
            
            scores, _, _, _ = bsi_ops.batch_dot_product(
                input_vec,
                weight,
                float(pf)
            )
            
            scores_raw = scores
            scores_scaled_pf = scores / pf
            scores_scaled_pf2 = scores / (pf * pf)
            
            print(f"\nSample {i}:")
            print(f"  Raw BSI: {scores_raw[:3].tolist()}")
            print(f"  Scaled by pf: {scores_scaled_pf[:3].tolist()}")
            print(f"  Scaled by pf²: {scores_scaled_pf2[:3].tolist()}")
            print(f"  Ground truth: {ground_truth[i][:3].tolist()}")
            
            error_raw = torch.abs(scores_raw - ground_truth[i]).mean()
            error_pf = torch.abs(scores_scaled_pf - ground_truth[i]).mean()
            error_pf2 = torch.abs(scores_scaled_pf2 - ground_truth[i]).mean()
            
            print(f"  Errors: raw={error_raw:.6f}, pf={error_pf:.6f}, pf²={error_pf2:.6f}")

if __name__ == "__main__":
    test_bsi_scaling()
    test_linear_layer()
    
    print("\n" + "="*80)
    print("CONCLUSION:")
    print("Based on the tests above, identify which scaling factor works best.")
    print("The BSI dot product likely needs scaling by 1/pf² since both")
    print("query and keys are scaled by pf before the dot product.")
    print("="*80)
