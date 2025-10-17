import pytest
import torch

bsi_ops = pytest.importorskip("bsi_ops")


@pytest.mark.cuda
def test_gpu_ewah_compression_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU compression test")

    torch.manual_seed(0)

    lengths = [1, 7, 64, 127, 1024]
    decimal_places_list = [0, 2, 4]

    for length in lengths:
        base = torch.randn(length, device="cuda", dtype=torch.float32)
        variants = [
            base,
            torch.zeros_like(base),
            base.abs(),
            -base.abs(),
        ]

        for values in variants:
            cpu_tensor = values.cpu()
            for decimals in decimal_places_list:
                # CPU verbatim words + stats
                cpu_cap, _, _ = bsi_ops.build_bsi_query(cpu_tensor, decimals)
                cpu_words, cpu_rows, cpu_offset, cpu_decimals, cpu_twos = bsi_ops.debug_bsi_query_words(cpu_cap)
                cpu_runs, cpu_lits, cpu_rlws = bsi_ops.cpu_hybrid_stats(cpu_cap)

                # GPU build (verbatim), then compress + decompress
                gpu_cap, _, _, _ = bsi_ops.build_bsi_query_cuda(values, decimals)
                # compress
                cwords, offsets, lengths, stats, S, W = bsi_ops.build_bsi_query_cuda_compressed(gpu_cap)
                # decompress on GPU and fetch to CPU
                dec_gpu_words = bsi_ops.debug_bsi_query_cuda_decompress(gpu_cap)

                # Parity: decompressed GPU must match CPU verbatim words
                assert dec_gpu_words.shape == cpu_words.shape
                torch.testing.assert_close(dec_gpu_words, cpu_words)

                # Stats / footprint: totals should match CPU EWAH counts
                gpu_runs = stats[:, 0]
                gpu_lits = stats[:, 1]
                gpu_rlws = lengths - gpu_lits
                assert torch.equal(gpu_runs.sum(), cpu_runs.sum())
                assert torch.equal(gpu_lits.sum(), cpu_lits.sum())
                # total compressed words equals rlw_count + literal_words
                assert int(lengths.sum().item()) == int((gpu_rlws + gpu_lits).sum().item())
                assert int(lengths.sum().item()) == int((cpu_rlws + cpu_lits).sum().item())

