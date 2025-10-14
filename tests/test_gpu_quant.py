import pytest
import torch

bsi_ops = pytest.importorskip("bsi_ops")


@pytest.mark.cuda
def test_gpu_quant_matches_cpu():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU quantisation parity test")

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
                cpu_cap, _, _ = bsi_ops.build_bsi_query(cpu_tensor, decimals)
                cpu_words, cpu_rows, cpu_offset, cpu_decimals, cpu_twos = bsi_ops.debug_bsi_query_words(cpu_cap)

                gpu_cap, _, _, _ = bsi_ops.build_bsi_query_cuda(values, decimals)
                gpu_words, gpu_rows, gpu_offset, gpu_decimals, gpu_twos = bsi_ops.debug_bsi_query_cuda(gpu_cap)

                assert gpu_rows == cpu_rows == cpu_tensor.numel()
                assert gpu_offset == cpu_offset
                assert gpu_decimals == cpu_decimals == decimals
                assert bool(gpu_twos) == bool(cpu_twos)
                assert gpu_words.shape == cpu_words.shape
                torch.testing.assert_close(gpu_words, cpu_words)
