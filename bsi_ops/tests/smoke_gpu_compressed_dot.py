import torch
import bsi_ops

def run_smoke(R=8, d=2048, decimals=2, threshold=0.2):
    assert torch.cuda.is_available(), 'CUDA required'
    K = torch.randn(R, d, device='cuda', dtype=torch.float32)
    q = torch.randn(d, device='cuda', dtype=torch.float32)

    # Build keys (CPU builder -> CUDA packed). Also builds compressed grouped buffers internally.
    key_cap, _, num_keys, d_built, W = bsi_ops.build_bsi_keys_cuda(K, int(decimals), float(threshold))
    assert num_keys == R and d_built == d

    # Baseline (verbatim path): CPU-built query + CUDA verbatim dot
    out_verbatim, *_ = bsi_ops.batch_dot_product_prebuilt_cuda_keys(q, key_cap)

    # Compressed-aware path: GPU query capsule (verbatim words) + compressed dot
    q_cap, _, S, Wq = bsi_ops.build_bsi_query_cuda(q, int(decimals), float(threshold))
    out_compressed, *_ = bsi_ops.batch_dot_product_prebuilt_cuda_caps(q_cap, key_cap)

    dv = out_verbatim.to('cpu')
    dc = out_compressed.to('cpu')
    max_diff = (dv - dc).abs().max().item()
    print(f"Compressed dot smoke: R={R}, d={d}, S={S}, W={W}, max_diff={max_diff:.3e}")
    assert max_diff < 1e-6, 'Compressed-aware kernel must match verbatim within tolerance'

if __name__ == '__main__':
    run_smoke()

