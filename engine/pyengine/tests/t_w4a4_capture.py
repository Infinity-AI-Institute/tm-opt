"""exp-0015b probe: is the W4A4 grouped MoE path CUDA-GRAPH CAPTURABLE?

The decode step is captured (exp-0012 graphrun); routing the graphed step's
routed experts through kernels/moe_gemm_w4a4.py is only possible if every op
in that path is capture-legal (no device->host sync, no host-visible
data-dependent control flow) and if replay reproduces the eager result
bitwise. This probe answers that on small synthetic PackedExperts, before any
engine wiring, on GPU 4.

Arms:
  a  eager reference (already-validated prefill path) on a fixed routing
  b  the same call warmed up + CAPTURED into a CUDAGraph on static input
     buffers, then replayed -> bitwise vs arm a
  c  second replay after re-writing the SAME input bytes -> bitwise vs b
     (same-schedule determinism, the t_b3-adapted worker arm)
  d  replay with a DIFFERENT routing written into the static buffers ->
     bitwise vs an eager call on that routing (the graph must be a pure
     function of buffer CONTENTS, not of the routing captured with)
"""
import sys

import torch

from engine.pyengine.loader import PackedExperts
from engine.pyengine.kernels.moe_gemm_w4a4 import moe_experts_w4a4

DEV = torch.device("cuda:4")
E, HID, FFN, T, TOPK = 8, 256, 128, 16, 4


def _pack(n, k, deint, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    packed = torch.randint(0, 256, (E, n, k // 2), generator=g, device=DEV,
                           dtype=torch.uint8)
    scale = (torch.randint(60, 70, (E, n, k // 16), generator=g, device=DEV,
                           dtype=torch.uint8).view(torch.float8_e4m3fn))
    scale2 = torch.full((E,), 0.02, device=DEV, dtype=torch.float32)
    p = PackedExperts(packed, scale, scale2, 16, deint, input_amax=4.375)
    return p


def _routing(seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    idx = torch.stack([torch.randperm(E, generator=g, device=DEV)[:TOPK]
                       for _ in range(T)]).to(torch.int64)
    wt = torch.rand(T, TOPK, generator=g, device=DEV).to(torch.bfloat16)
    return idx, wt


def main():
    torch.cuda.set_device(DEV)
    p13, p2 = _pack(2 * FFN, HID, True, 1), _pack(HID, FFN, False, 2)
    x0 = torch.randn(T, HID, device=DEV, dtype=torch.bfloat16)
    i0, w0 = _routing(10)
    x1 = torch.randn(T, HID, device=DEV, dtype=torch.bfloat16)
    i1, w1 = _routing(11)

    # arm a: eager reference on routing 0 and routing 1
    ref0 = moe_experts_w4a4(x0, p13, p2, i0, w0).clone()
    ref1 = moe_experts_w4a4(x1, p13, p2, i1, w1).clone()
    print(f"a  eager ok: ref0 {tuple(ref0.shape)} "
          f"absmax {ref0.abs().max().item():.4f} finite "
          f"{bool(torch.isfinite(ref0).all())}")

    # static input buffers, as the graphed step would hold them
    sx = x0.clone()
    si = i0.clone()
    sw = w0.clone()

    # warm up on a side stream exactly like graphrun._capture does (this is
    # where the DSL JIT compile must happen -- never inside the capture)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            out = moe_experts_w4a4(sx, p13, p2, si, sw)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # arm b: capture + replay
    pool = torch.cuda.graph_pool_handle()
    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g, pool=pool):
            gout = moe_experts_w4a4(sx, p13, p2, si, sw)
    except Exception as exc:                       # capture-illegal op
        print(f"b  CAPTURE FAILED: {type(exc).__name__}: {exc}")
        return 1
    g.replay()
    torch.cuda.synchronize()
    ok_b = torch.equal(gout, ref0)
    print(f"b  capture+replay vs eager: bitwise={ok_b} "
          f"maxabsdiff={(gout.float() - ref0.float()).abs().max().item():.3e}")

    # arm c: same bytes again -> same bits (determinism)
    sx.copy_(x0); si.copy_(i0); sw.copy_(w0)
    g.replay()
    torch.cuda.synchronize()
    first = gout.clone()
    g.replay()
    torch.cuda.synchronize()
    ok_c = torch.equal(gout, first) and torch.equal(first, ref0)
    print(f"b  replay determinism (same schedule, 2 runs): bitwise={ok_c}")

    # arm d: different routing through the SAME graph
    sx.copy_(x1); si.copy_(i1); sw.copy_(w1)
    g.replay()
    torch.cuda.synchronize()
    ok_d = torch.equal(gout, ref1)
    print(f"d  replay on a DIFFERENT routing vs eager: bitwise={ok_d} "
          f"maxabsdiff={(gout.float() - ref1.float()).abs().max().item():.3e}")

    all_ok = ok_b and ok_c and ok_d
    print(f"RESULT: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
