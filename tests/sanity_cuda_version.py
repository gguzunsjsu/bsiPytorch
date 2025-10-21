import sys
import torch

def main():
    try:
        import bsi_ops as mod
    except Exception as e:
        print("FAIL: cannot import bsi_ops:", repr(e))
        sys.exit(1)

    print("Python:", sys.version.split()[0])
    print("Torch:", getattr(torch, "__version__", "unknown"))
    print("bsi_ops module:", getattr(mod, "__file__", "<no __file__>"))

    ver = getattr(mod, "cuda_builder_version", None)
    if callable(ver):
        print("CUDA_BUILDER_VERSION:", ver())
    else:
        print("CUDA_BUILDER_VERSION: <missing> (old .so?)")

    has_dbg_int = hasattr(mod, "debug_quantize_int64_cuda")
    has_dbg_det = hasattr(mod, "debug_quantize_details_cuda")
    print("has debug_quantize_int64_cuda:", has_dbg_int)
    print("has debug_quantize_details_cuda:", has_dbg_det)

    if not (has_dbg_int and has_dbg_det):
        print("NOTE: Debug helpers not found. Reinstall bsi_ops to pick up latest .so.")
        return

    # Small probe to verify rounding parity (half-away-from-zero)
    # Values chosen around 0.5 thresholds
    x_cpu = torch.tensor([-0.49, -0.5, -0.51, -0.1, 0.0, 0.1, 0.49, 0.5, 0.51], dtype=torch.float32)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    x = x_cpu.to(device)
    dp = 0

    # GPU path via extension
    ints = mod.debug_quantize_int64_cuda(x, dp).to("cpu")
    sx, rx, ix = mod.debug_quantize_details_cuda(x, dp, k=x.numel())

    # CPU expected: std::round semantics (half away from zero)
    # Implemented here using torch so it is easy to compare
    def cpu_round(v):
        vp = torch.floor(v + 0.5)
        vn = -torch.floor(-v + 0.5)
        return torch.where(v.ge(0), vp, vn)

    expected = cpu_round(x_cpu.double()).to(torch.int64)

    print("values     :", x_cpu.tolist())
    print("scaled_fp  :", sx.tolist())
    print("rounded_fp :", rx.tolist())
    print("gpu_ints   :", ix.tolist())
    print("gpu_ints(v2):", ints.tolist())
    print("cpu_expect :", expected.tolist())

    if not torch.equal(ints.cpu(), expected):
        print("MISMATCH: staged ints differ from CPU expectation -> stale .so or rounding bug")
        sys.exit(2)

    print("OK: staged ints match CPU rounding (half-away-from-zero)")


if __name__ == "__main__":
    main()

