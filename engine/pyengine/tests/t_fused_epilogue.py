"""exp-0022 local validation: the routed-expert inter-GEMM epilogue fused
into the NVFP4 activation quant (kernels/fp4_quant.nvfp4_quant_silu) vs the
eager chain it replaces (moe_gemm_w4a4 step 2/3 before this experiment).

Arms, all on ONE GPU at the REAL padded row shapes the two phases use
(decode m_cap = 32896 from the graphed step, prefill m_cap from a
canonical G=6 cohort), no model load needed:
  a  EQUIVALENCE: fused (packed, sf) vs eager-chain + nvfp4_quant —
     scales bitwise, packed bytes bitwise or tie-class, and the max
     dequantized-value delta.
  b  DETERMINISM: two identical fused runs -> bitwise identical (the
     t_b3-adapted same-schedule arm).
  c  SPEED: eager chain + quant vs fused, CUDA-event timed, interleaved
     arms after a discarded warm-up. This is the per-layer number the
     spec's cycle model is built on.
  d  CAPTURE: the fused kernel replays inside a CUDA graph and gives
     bitwise the same bytes as the eager launch (the graphed decode step
     is a captured region, exp-0012).

Run from the worktree root:
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python \
      engine/pyengine/tests/t_fused_epilogue.py
"""
import sys
import time

import torch

from engine.pyengine.kernels.fp4_quant import nvfp4_quant, nvfp4_quant_silu

DEV = "cuda:0"
FFN = 3072                 # expert intermediate (config.json)
M_DECODE = 32896           # graphed decode step m_cap (exp-0015b)
M_PREFILL = 59392          # G=6 cohort m_cap (P=26820 pairs, E=256)
GS2 = 0.00372              # a representative w2 input_scale (amax/(448*6))
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def eager(c1, ffn, gs):
    """The chain this experiment deletes: 9 elementwise passes over the
    padded row space, then the quant kernel's own read of `act`."""
    m = c1.shape[0]
    gu = c1.view(m, ffn, 2)
    g = gu[..., 0].float()
    u = gu[..., 1].float()
    act = ((g * torch.sigmoid(g)).to(torch.bfloat16).float()
           * u).to(torch.bfloat16).contiguous()
    return nvfp4_quant(act, gs)


def deq(packed, sf_flat, m, ffn):
    """Dequantize packed+swizzled scales back to fp32 values (no global
    scale) — used only to size any packed-byte disagreement."""
    lut = E2M1.to(packed.device)
    lo, hi = (packed & 0xF).long(), (packed >> 4).long()
    w = torch.stack([lut[lo], lut[hi]], dim=-1).view(m, ffn)
    #1. un-swizzle the 32_4_4 blocked scales back to [m, ffn/16]
    kg, nc = ffn // 16, ffn // 16 // 4
    s = sf_flat.view(m // 128, nc, 32, 4, 4).permute(0, 3, 2, 1, 4)
    s = s.reshape(m, kg).float()
    return w * s.repeat_interleave(16, dim=1)


def timeit(fn, n=20):
    """CUDA-event mean over n reps after 3 discarded warm-ups."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n


def main():
    free = torch.cuda.mem_get_info(0)[0] / (1 << 30)
    print(f"[t0022] free on this GPU: {free:.0f} GiB", flush=True)
    if free < 12:
        print("[t0022] GPU busy — NOT running", flush=True)
        return 2
    torch.manual_seed(0)
    ok = True

    for tag, M in (("decode", M_DECODE), ("prefill", M_PREFILL)):
        #0. a GEMM-output-like buffer: mixed magnitudes, some exact zeros
        #   (pad rows) and a few extremes, in the interleaved gate/up layout
        c1 = (torch.randn(M, 2 * FFN, device=DEV, dtype=torch.bfloat16)
              * 3.0)
        c1[: M // 8] = 0.0
        c1[M // 2, :16] = torch.tensor(
            [40.0, -40.0, 1e-3, -1e-3, 0.0, -0.0, 6.0, -6.0,
             0.5, -0.5, 12.0, -12.0, 1e-5, -1e-5, 3.0, -3.0],
            device=DEV, dtype=torch.bfloat16)

        #1. arm a — equivalence vs the eager chain
        pk_e, sf_e = eager(c1, FFN, GS2)
        pk_f, sf_f = nvfp4_quant_silu(c1, FFN, GS2)
        sf_same = torch.equal(sf_e.view(torch.uint8), sf_f.view(torch.uint8))
        nbytes = (pk_e != pk_f).sum().item()
        d_e = deq(pk_e, sf_e, M, FFN)
        d_f = deq(pk_f, sf_f, M, FFN)
        dmax = (d_e - d_f).abs().max().item()
        rel = ((d_e - d_f).abs().sum() / d_e.abs().sum().clamp_min(1e-30)
               ).item()
        print(f"[t0022] arm a {tag} M={M}: scales bitwise {sf_same}; "
              f"packed bytes differing {nbytes}/{pk_e.numel()} "
              f"({100.0 * nbytes / pk_e.numel():.4f}%); dequant max|d| "
              f"{dmax:.3e} rel_mean {rel:.3e}", flush=True)
        ok = ok and sf_same and nbytes * 1e4 <= pk_e.numel()

        #2. arm b — determinism
        pk_f2, sf_f2 = nvfp4_quant_silu(c1, FFN, GS2)
        det = (torch.equal(pk_f, pk_f2)
               and torch.equal(sf_f.view(torch.uint8),
                               sf_f2.view(torch.uint8)))
        print(f"[t0022] arm b {tag}: two identical runs bitwise equal {det}",
              flush=True)
        ok = ok and det

        #3. arm c — speed, interleaved
        t_e = t_f = 0.0
        for _ in range(2):
            t_e += timeit(lambda: eager(c1, FFN, GS2)) / 2
            t_f += timeit(lambda: nvfp4_quant_silu(c1, FFN, GS2)) / 2
        print(f"[t0022] arm c {tag} M={M}: eager {t_e:.3f} ms/layer -> "
              f"fused {t_f:.3f} ms/layer = {t_e / max(t_f, 1e-9):.2f}x "
              f"(-{t_e - t_f:.3f} ms/layer)", flush=True)

        #4. arm d — capture safety (decode shape only; the captured region)
        if tag == "decode":
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                nvfp4_quant_silu(c1, FFN, GS2)
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            static_in = c1.clone()
            with torch.cuda.graph(g):
                pk_g, sf_g = nvfp4_quant_silu(static_in, FFN, GS2)
            g.replay()
            torch.cuda.synchronize()
            cap = (torch.equal(pk_g, pk_f)
                   and torch.equal(sf_g.view(torch.uint8),
                                   sf_f.view(torch.uint8)))
            g.replay()
            torch.cuda.synchronize()
            cap2 = (torch.equal(pk_g, pk_f)
                    and torch.equal(sf_g.view(torch.uint8),
                                    sf_f.view(torch.uint8)))
            print(f"[t0022] arm d decode: capture+replay == eager launch "
                  f"{cap}; second replay identical {cap2}", flush=True)
            ok = ok and cap and cap2
            del g, static_in, pk_g, sf_g
        del c1, pk_e, sf_e, pk_f, sf_f, pk_f2, sf_f2, d_e, d_f
        torch.cuda.empty_cache()

    print(f"[t0022] {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"[t0022] done in {time.time() - t0:.0f} s", flush=True)
    sys.exit(rc)
