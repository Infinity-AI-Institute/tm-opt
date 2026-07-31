"""exp-0019 arms: the grouped bf16 twin for the un-exported MoE layer 2.

Arms (all on ONE GPU, real checkpoint SHAPES with synthetic weights — the
kernel is shape-specialized, not value-specialized, and the real-weight
numeric check is the D13 teacher-forced gate against the live server):
  a  equivalence vs the reference per-hit-expert loop at decode (T=65) and
     prefill (T=745) shapes. NOT bitwise — B3.1/B3.3: fp32 GEMM
     accumulation order and the slot-sum differ exactly as they already do
     on the 63 packed layers. Reported as rel error + cosine so the drift
     can be compared against the packed path's own accepted drift.
  b  determinism: two identical grouped runs bitwise equal (the t_b3-adapted
     worker arm).
  c  capture safety: the grouped call inside torch.cuda.graph, replayed,
     bitwise equal to the eager grouped result on the same inputs. This is
     the property the reference loop lacks (unique().tolist() syncs) and the
     whole reason exp-0012 carved layer 2 out as an eager island.
  d  cost: reference loop vs grouped, decode and prefill shapes.

Usage: CUDA_VISIBLE_DEVICES=<one free gpu> python -m engine.pyengine.tests.t_moe_bf16
"""
import time

import torch

from engine.pyengine import model as pmodel
from engine.pyengine.kernels.moe_gemm_bf16 import moe_experts_bf16

DEV = "cuda:0"
E, FFN, HID, TOPK = 256, 3072, 6144, 6                 # checkpoint shapes
SHAPES = {"decode": 65, "prefill": 745}                # pooled step / one seq


def _route(T, gen):
    """top-6 of 256 with distinct slots per token (topk positions, B2.8) and
    normalized positive weights — the router's output contract."""
    scores = torch.rand(T, E, generator=gen, device=DEV)
    w, idx = torch.topk(scores, TOPK, dim=-1)
    w = (w / w.sum(-1, keepdim=True)).to(torch.bfloat16)
    return idx.contiguous(), w.contiguous()


def _ref(x, w13, w2, idx, wt):
    """model.moe_experts with the grouped twin forced OFF — the reference
    loop verbatim, which is what this experiment replaces."""
    saved = pmodel._BF16_GROUPED
    pmodel._BF16_GROUPED = False
    try:
        return pmodel.moe_experts(x, w13, w2, idx, wt)
    finally:
        pmodel._BF16_GROUPED = saved


def _err(a, b):
    d, r = (a.float() - b.float()), b.float()
    return (d.abs().max().item(),
            (d.abs().mean() / r.abs().mean()).item(),
            torch.nn.functional.cosine_similarity(
                a.float().flatten(), r.flatten(), dim=0).item())


def main():
    torch.manual_seed(0)
    gen = torch.Generator(device=DEV).manual_seed(1234)
    print(f"[t0019] device {torch.cuda.get_device_name(0)}  "
          f"E={E} ffn={FFN} hidden={HID} top_k={TOPK}")
    #1. real-shape synthetic weights: w13 de-interleaved [gates; ups] halves.
    #   Generated per-expert directly in bf16 — a fp32 staging tensor for the
    #   whole stack would be 38 GiB before the cast
    w13 = torch.empty(E, 2 * FFN, HID, device=DEV, dtype=torch.bfloat16)
    w2 = torch.empty(E, HID, FFN, device=DEV, dtype=torch.bfloat16)
    for e in range(E):
        w13[e].normal_(0, 0.02, generator=gen)
        w2[e].normal_(0, 0.02, generator=gen)
    print(f"[t0019] weights resident: "
          f"{(w13.numel() + w2.numel()) * 2 / 2**30:.1f} GiB")
    #2. the layer-2 dispatch predicate must actually select the twin here
    assert pmodel.experts_capturable(w13), "dispatch predicate rejects w13"
    assert not pmodel.experts_capturable(
        torch.zeros(4, 192, 512, device=DEV, dtype=torch.bfloat16)), \
        "predicate must reject non-128-multiple toy shapes"

    ok = True
    for name, T in SHAPES.items():
        x = (torch.randn(T, HID, generator=gen, device=DEV,
                         dtype=torch.float32) * 0.5).to(torch.bfloat16)
        idx, wt = _route(T, gen)
        pairs = int((torch.bincount(idx.reshape(-1), minlength=E) > 0).sum())
        #a. equivalence vs the reference loop
        ref = _ref(x, w13, w2, idx, wt)
        got = moe_experts_bf16(x, w13, w2, idx, wt)
        amax, rel, cos = _err(got, ref)
        print(f"[t0019][a] {name:7s} T={T:4d} hit_experts={pairs:3d}  "
              f"abs_max {amax:.4e}  rel_mean {rel:.3e}  cos {cos:.8f}")
        ok &= cos > 0.999 and rel < 5e-3
        #b. determinism: same schedule, twice, bitwise
        again = moe_experts_bf16(x, w13, w2, idx, wt)
        bit = torch.equal(got, again)
        print(f"[t0019][b] {name:7s} determinism (2 runs bitwise): {bit}")
        ok &= bit
        #c. capture safety — static inputs, capture, replay, compare
        sx, sidx, swt = x.clone(), idx.clone(), wt.clone()
        eager = moe_experts_bf16(sx, w13, w2, sidx, swt)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                moe_experts_bf16(sx, w13, w2, sidx, swt)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g):
                out_g = moe_experts_bf16(sx, w13, w2, sidx, swt)
            g.replay()
            torch.cuda.synchronize()
            cap = torch.equal(out_g, eager)
            print(f"[t0019][c] {name:7s} CAPTURED + replayed, "
                  f"bitwise vs eager: {cap}")
            ok &= cap
        except Exception as exc:                      # noqa: BLE001
            print(f"[t0019][c] {name:7s} CAPTURE FAILED: "
                  f"{type(exc).__name__}: {exc}")
            ok = False
        del g
        #d. cost
        for label, fn in (("loop   ", lambda: _ref(x, w13, w2, idx, wt)),
                          ("grouped", lambda: moe_experts_bf16(
                              x, w13, w2, idx, wt))):
            for _ in range(2):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000 / 5
            print(f"[t0019][d] {name:7s} {label} {ms:8.3f} ms/call")
    print(f"[t0019] ALL ARMS {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
