"""exp-0025 evidence, ISOLATED: the fused prefill sconv kernel.

  arm a  EQUIVALENCE — fused vs the eager fp32 conv1d chain at the four
         real prefill site shapes, bf16 and fp32, contiguous and
         channel-major inputs. The claim is BITWISE (a 4-tap dot has one
         sensible order and the kernel uses the reference's), so this arm
         prints exact mismatch counts, not a tolerance.
  arm b  DETERMINISM — two identical fused calls bitwise equal (the
         t_b3-adapted same-schedule arm).
  arm c  SPEED — CUDA-event timed, interleaved arms after discarded
         warm-ups, per site and summed to a per-layer / per-traversal
         figure.
  arm d  LAYOUT — the eager chain's output is CHANNEL-MAJOR (it inherits
         `conv.transpose(1,2) + xf`'s transposed operand strides); the
         fused one is contiguous. Also times the two consumers that
         layout reaches first: the per-head K rmsnorm and the residual
         add, on each layout.

Run: CUDA_VISIBLE_DEVICES=4 python -m engine.pyengine.tests.t_fused_sconv
"""
import torch

from engine.pyengine import model as pmodel
from engine.pyengine.kernels.sconv_prefill import sconv_prefill_fused

# canonical cohort shape: 6 rows right-padded to the cohort max (the
# decode_heavy prompt draw is randint(512, 1024) words -> T ~ 700-1400)
B, T = 6, 1408
# the four sconv sites of one layer: (name, channels)
SITES = [("attn K (SWA)", 2048), ("attn V (SWA)", 2048),
         ("attn out", 6144), ("MoE out", 6144)]
GLOBAL_SITES = [("attn K (global)", 1024), ("attn V (global)", 1024)]


def _eager(x, w):
    """model.sconv_prefill's chain, verbatim, with the switch forced off."""
    saved = pmodel._FUSED_SCONV
    pmodel._FUSED_SCONV = False
    try:
        return pmodel.sconv_prefill(x, w)
    finally:
        pmodel._FUSED_SCONV = saved


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
    g = torch.Generator(device=dev).manual_seed(25)
    print(f"[t0025] shapes B={B} T={T}", flush=True)
    #1. arm a — equivalence at every site shape, both dtypes
    bad = 0
    for name, C in SITES + GLOBAL_SITES:
        for dt in (torch.bfloat16, torch.float32):
            x = torch.randn((B, T, C), generator=g, device=dev,
                            dtype=torch.float32).to(dt)
            w = (torch.randn((C, 1, 4), generator=g, device=dev,
                             dtype=torch.float32) * 0.3).to(dt)
            ref, got = _eager(x, w), pmodel.sconv_prefill(x, w)
            eq = torch.equal(ref, got)
            d = (ref.float() - got.float()).abs()
            nmis = int((ref.float() != got.float()).sum())
            bad += 0 if eq else 1
            print(f"[t0025] arm a  {name:16s} {str(dt):14s} C={C:5d} "
                  f"bitwise={eq} mismatches={nmis}/{ref.numel()} "
                  f"max|d|={d.max().item():.3e}", flush=True)
    #1b. a channel-major input (what the eager chain itself produces, so
    #    the kernel must serve it: the engine feeds sconv outputs forward)
    C = 6144
    x = torch.randn((B, T, C), generator=g, device=dev,
                    dtype=torch.float32).to(torch.bfloat16)
    xt = x.transpose(1, 2).contiguous().transpose(1, 2)   # channel-major
    w = (torch.randn((C, 1, 4), generator=g, device=dev,
                     dtype=torch.float32) * 0.3).to(torch.bfloat16)
    eq = torch.equal(_eager(xt, w), pmodel.sconv_prefill(xt, w))
    bad += 0 if eq else 1
    print(f"[t0025] arm a  channel-major input: bitwise={eq} "
          f"(strides {tuple(xt.stride())})", flush=True)
    #2. arm b — same-schedule determinism
    r1, r2 = sconv_prefill_fused(x, w), sconv_prefill_fused(x, w)
    det = torch.equal(r1, r2)
    bad += 0 if det else 1
    print(f"[t0025] arm b  two identical runs bitwise equal: {det}",
          flush=True)
    #3. arm c — speed per site, interleaved arms
    tot_e = tot_f = 0.0
    for name, C in SITES + GLOBAL_SITES:
        x = torch.randn((B, T, C), generator=g, device=dev,
                        dtype=torch.float32).to(torch.bfloat16)
        w = (torch.randn((C, 1, 4), generator=g, device=dev,
                         dtype=torch.float32) * 0.3).to(torch.bfloat16)
        te1 = _time(lambda: _eager(x, w))
        tf1 = _time(lambda: sconv_prefill_fused(x, w))
        te2 = _time(lambda: _eager(x, w))
        tf2 = _time(lambda: sconv_prefill_fused(x, w))
        te, tf = min(te1, te2), min(tf1, tf2)
        if name in [s[0] for s in SITES]:
            tot_e += te
            tot_f += tf
        print(f"[t0025] arm c  {name:16s} C={C:5d}: {te:.3f} -> {tf:.3f} ms "
              f"= {te / tf:.2f}x  (reps {te1:.3f}/{te2:.3f} vs "
              f"{tf1:.3f}/{tf2:.3f})", flush=True)
    print(f"[t0025] arm c  SWA layer (4 sites): {tot_e:.3f} -> {tot_f:.3f} "
          f"ms = {tot_e / tot_f:.2f}x, -{tot_e - tot_f:.3f} ms/layer; "
          f"x66 layers = -{66 * (tot_e - tot_f) / 1000:.3f} s/traversal",
          flush=True)
    #4. arm d — layout, and the two consumers it reaches first
    C = 6144
    x = torch.randn((B, T, C), generator=g, device=dev,
                    dtype=torch.float32).to(torch.bfloat16)
    w = (torch.randn((C, 1, 4), generator=g, device=dev,
                     dtype=torch.float32) * 0.3).to(torch.bfloat16)
    oe, of = _eager(x, w), sconv_prefill_fused(x, w)
    print(f"[t0025] arm d  eager out contiguous={oe.is_contiguous()} "
          f"strides={tuple(oe.stride())} | fused contiguous="
          f"{of.is_contiguous()} strides={tuple(of.stride())}", flush=True)
    nw = torch.randn((128,), generator=g, device=dev,
                     dtype=torch.float32).to(torch.bfloat16)
    res = torch.randn((B, T, C), generator=g, device=dev,
                      dtype=torch.float32).to(torch.bfloat16)
    for tag, o in (("channel-major", oe), ("contiguous  ", of)):
        tn = _time(lambda: pmodel.rmsnorm(o.view(B, T, C // 128, 128), nw))
        ta = _time(lambda: res + o)
        print(f"[t0025] arm d  consumer on {tag}: per-head rmsnorm "
              f"{tn:.3f} ms, residual add {ta:.3f} ms", flush=True)
    #5. arm f — WHERE the engine's numerics move. The kernel is bitwise
    #   (arm a), so any downstream token change must come from the LAYOUT:
    #   the eager chain hands its consumers a channel-major tensor, and a
    #   GEMM on a channel-major operand is a different cuBLAS kernel with a
    #   different accumulation order than the same VALUES contiguous. This
    #   arm isolates that with identical values in both layouts — it is the
    #   B3.1/B3.3 drift class the D13 envelope gates, not an arithmetic
    #   change (t_fused_sconv_insitu arm b sees it end to end).
    cm = of.transpose(1, 2).contiguous().transpose(1, 2)
    cm.copy_(of)
    same_vals = torch.equal(cm.float(), of.float())
    wq = (torch.randn((C, C), generator=g, device=dev,
                      dtype=torch.float32) * 0.02).to(torch.bfloat16)
    lin_cm = torch.nn.functional.linear(cm, wq)
    lin_ct = torch.nn.functional.linear(of, wq)
    n_ct = pmodel.rmsnorm(cm.view(B, T, C // 128, 128), nw)
    n_cm = pmodel.rmsnorm(of.view(B, T, C // 128, 128), nw)
    print(f"[t0025] arm f  same values in both layouts: {same_vals} | "
          f"GEMM bitwise across layouts: "
          f"{torch.equal(lin_cm, lin_ct)} (max|d| "
          f"{(lin_cm.float() - lin_ct.float()).abs().max().item():.3e}) | "
          f"rmsnorm bitwise across layouts: {torch.equal(n_cm, n_ct)}",
          flush=True)
    print(f"[t0025] {'ALL OK' if bad == 0 else f'{bad} ARM(S) FAILED'}",
          flush=True)
    return bad


if __name__ == "__main__":
    raise SystemExit(main())
