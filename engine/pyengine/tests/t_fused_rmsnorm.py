"""exp-0026 evidence, ISOLATED: B2.3's Triton rmsnorm at the prefill sites.

The kernel is not new — kernels/rmsnorm.py has been in the tree and gated by
t_b2's B2.3 arm since bring-up. What is new is that model.rmsnorm_prefill
CALLS it, so these arms measure the two things that decides:

  arm a  EQUIVALENCE — fused vs the torch-reference chain (model.rmsnorm) at
         the four REAL prefill site shapes, bf16 and fp32. The claim is NOT
         bitwise: the fp32 variance is a register tree instead of torch's
         reduction, so this arm prints max|delta|, max relative error and
         the max bf16-ULP distance (the units B2.3 gates in), plus the
         exact-match fraction.
  arm b  DETERMINISM — two identical fused calls bitwise equal (the
         t_b3-adapted same-schedule arm; fixed grid, no atomics, no
         autotune, so the kernel sequence is a pure function of the shape).
  arm c  SPEED — CUDA-event timed, interleaved arms after discarded
         warm-ups, per site and summed to a per-layer / per-traversal
         figure.
  arm d  LAYOUT — the same two arms on a CHANNEL-MAJOR input, because that
         is what the eager sconv chain hands the k-norm site when
         PYENGINE_FUSED_SCONV=0 (exp-0025 arm d). The kernel flattens to
         rows itself and always stores contiguous.

Run: CUDA_VISIBLE_DEVICES=4 python -m engine.pyengine.tests.t_fused_rmsnorm
"""
import sys

import torch

from engine.pyengine import model as pmodel

# canonical cohort shape: 6 rows right-padded to the cohort max (the
# decode_heavy prompt draw is randint(512, 1024) words -> T ~ 700-1400)
B, T = 6, 1408
HIDDEN, HEAD_DIM = 6144, 128
# the four norm sites of one prefill layer: (name, trailing shape, per-layer
# count on an SWA layer)
SITES = [("attn_norm/mlp_norm (hidden)", (HIDDEN,), 2),
         ("q norm (64 heads)", (64, HEAD_DIM), 1),
         ("k norm (16 heads, SWA)", (16, HEAD_DIM), 1)]
GLOBAL_SITES = [("k norm (8 heads, global)", (8, HEAD_DIM), 1)]


def _eager(x, w):
    """model.rmsnorm_prefill with the switch forced off = the torch chain."""
    saved = pmodel._FUSED_RMSNORM
    pmodel._FUSED_RMSNORM = False
    try:
        return pmodel.rmsnorm_prefill(x, w)
    finally:
        pmodel._FUSED_RMSNORM = saved


def _fused(x, w):
    saved = pmodel._FUSED_RMSNORM
    pmodel._FUSED_RMSNORM = True
    try:
        return pmodel.rmsnorm_prefill(x, w)
    finally:
        pmodel._FUSED_RMSNORM = saved


def _ulps(a, b):
    """Max ULP distance between two same-dtype float tensors, via the
    monotone integer key of a sign-magnitude float (so a 1-ULP step is a
    key step of 1 across zero as well)."""
    if a.dtype is torch.bfloat16:
        ia = a.view(torch.int16).to(torch.int64) & 0xFFFF
        ib = b.view(torch.int16).to(torch.int64) & 0xFFFF
        sign, mag = 0x8000, 0x7FFF
    else:
        ia = a.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        ib = b.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        sign, mag = 0x80000000, 0x7FFFFFFF
    ka = torch.where(ia & sign != 0, sign - (ia & mag), sign + (ia & mag))
    kb = torch.where(ib & sign != 0, sign - (ib & mag), sign + (ib & mag))
    return int((ka - kb).abs().max())


def _time(fn, reps=20):
    """CUDA-event timed, warm-ups discarded."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(reps):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / reps


def main():
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(26)
    print(f"[t0026] shapes B={B} T={T}", flush=True)
    ok = True
    #1. arm a — equivalence at every site shape, both dtypes
    for name, tail, _ in SITES + GLOBAL_SITES:
        for dt in (torch.bfloat16, torch.float32):
            x = torch.randn((B, T) + tail, generator=g, device=dev,
                            dtype=torch.float32).to(dt)
            w = (torch.randn(tail[-1:], generator=g, device=dev,
                             dtype=torch.float32) * 0.5 + 1.0).to(dt)
            want, got = _eager(x, w), _fused(x, w)
            d = (got.float() - want.float()).abs()
            scale = want.float().abs().clamp_min(1e-6)
            exact = int((got == want).sum())
            n = want.numel()
            u = _ulps(got, want)
            print(f"[t0026] arm a  {name:28s} {str(dt).split('.')[-1]:9s} "
                  f"max|d| {d.max():.3e} rel {(d / scale).max():.3e} "
                  f"ULP<= {u} exact {exact}/{n} = {exact / n:.4f}",
                  flush=True)
            #   B2.3's bar: fp32 reduction-order noise only, i.e. a couple
            #   of bf16 ULP. Anything larger is a real defect.
            if u > (2 if dt is torch.bfloat16 else 64):
                print(f"[t0026] arm a  FAIL {name}: {u} ULP", flush=True)
                ok = False
            if not torch.isfinite(got).all():
                print(f"[t0026] arm a  FAIL {name}: non-finite", flush=True)
                ok = False
    #2. arm b — determinism of the fused path
    x = torch.randn((B, T, HIDDEN), generator=g, device=dev,
                    dtype=torch.float32).to(torch.bfloat16)
    w = torch.randn((HIDDEN,), generator=g, device=dev,
                    dtype=torch.float32).to(torch.bfloat16)
    det = bool((_fused(x, w) == _fused(x, w)).all())
    print(f"[t0026] arm b  fused call1 == call2 (bitwise): {det}",
          flush=True)
    ok = ok and det
    #3. arm c — speed per site, interleaved, summed per layer
    per_layer_off = per_layer_on = 0.0
    for name, tail, count in SITES:
        x = torch.randn((B, T) + tail, generator=g, device=dev,
                        dtype=torch.float32).to(torch.bfloat16)
        w = torch.randn(tail[-1:], generator=g, device=dev,
                        dtype=torch.float32).to(torch.bfloat16)
        off1 = _time(lambda: _eager(x, w))
        on1 = _time(lambda: _fused(x, w))
        off2 = _time(lambda: _eager(x, w))
        on2 = _time(lambda: _fused(x, w))
        off, on = (off1 + off2) / 2, (on1 + on2) / 2
        per_layer_off += off * count
        per_layer_on += on * count
        print(f"[t0026] arm c  {name:28s} x{count}: {off:.3f} -> {on:.3f} "
              f"ms = {off / on:.2f}x  (repeats {off1:.3f}/{off2:.3f} vs "
              f"{on1:.3f}/{on2:.3f})", flush=True)
    print(f"[t0026] arm c  SWA layer (4 sites): {per_layer_off:.3f} -> "
          f"{per_layer_on:.3f} ms = {per_layer_off / per_layer_on:.2f}x "
          f"(-{per_layer_off - per_layer_on:.3f} ms/layer, "
          f"-{(per_layer_off - per_layer_on) * 66 / 1000:.3f} s over 66 "
          f"layers at this shape)", flush=True)
    #4. arm d — channel-major input (the eager-sconv layout, exp-0025 arm d)
    xc = torch.randn((B, HIDDEN, T), generator=g, device=dev,
                     dtype=torch.float32).to(torch.bfloat16).transpose(1, 2)
    w = torch.randn((HIDDEN,), generator=g, device=dev,
                    dtype=torch.float32).to(torch.bfloat16)
    want, got = _eager(xc, w), _fused(xc, w)
    u = _ulps(got, want)
    print(f"[t0026] arm d  channel-major input: ULP<= {u} "
          f"max|d| {(got.float() - want.float()).abs().max():.3e} "
          f"out contiguous {got.is_contiguous()}", flush=True)
    ok = ok and u <= 2
    print(f"[t0026] RESULT {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
