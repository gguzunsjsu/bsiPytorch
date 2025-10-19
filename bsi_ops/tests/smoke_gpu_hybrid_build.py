import torch
import bsi_ops

def run_smoke(length=3072, decimals=2, threshold=0.2):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    assert device == 'cuda', 'CUDA required for hybrid builder smoke test'
    x = torch.randn(length, device=device, dtype=torch.float32)

    # CPU reference: build BSI and get verbatim words
    cap_cpu, _, _ = bsi_ops.build_bsi_query(x.cpu(), decimals, float(threshold))
    words_cpu, rows_cpu, off_cpu, dec_cpu, twos_cpu = bsi_ops.debug_bsi_query_words(cap_cpu)

    # GPU hybrid builder: compressed-only
    cap_cuda, mem_bytes, S, W = bsi_ops.build_bsi_query_cuda_hybrid(x, decimals, float(threshold))
    words_gpu = bsi_ops.debug_bsi_query_hybrid_decompress(cap_cuda)

    # Basic metadata parity
    assert words_cpu.shape == words_gpu.shape == (S, W), f"shape mismatch: CPU {words_cpu.shape}, GPU {words_gpu.shape}"
    # Bitwise equality of decompressed words
    diff = (words_cpu != words_gpu).sum().item()
    print(f"Hybrid GPU builder smoke: S={S}, W={W}, comp_bytes={mem_bytes}, word_mismatches={diff}")
    assert diff == 0, "Decompressed GPU words must match CPU verbatim words"

if __name__ == '__main__':
    run_smoke()

