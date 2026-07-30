"""exp-0015b local validation: routing the DECODE MoE through the native
W4A4 grouped GEMM, on REAL layer-3 routed experts. Run under
CUDA_VISIBLE_DEVICES=4 (GPUs 4-7 rule; ~12 GB peak):

  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python engine/pyengine/tests/t_w4a4_decode.py

Arms:
  a  PREFILL NUMERICS UNCHANGED: the searchsorted group-boundary scan that
     replaced torch.bincount (bincount's CUDA path syncs on max().cpu(),
     illegal under graph capture) yields bitwise-identical offs/dst/src at
     both prefill and decode shapes. offs/dst/src are the only outputs of
     the changed block, so exp-0016's green D13 TF gate carries over
     unchanged to this tree.
  b  decode-shape numerics vs the W4A16 grouped path it replaces (arm-J
     class: W4A4 is the REFERENCE's own recipe, so this distance is the
     one the D11 goldens were generated with).
  c  decode-shape per-layer wall, W4A4 vs W4A16 — the quantitative
     grounding for the predicted delta (exp-0012 kernel table: routed
     experts are 266.4 ms of the 322.7 ms graphed step).
  d  CUDA-graph capture at the real decode shape: replay bitwise == eager,
     same-schedule determinism (t_b3 arm), replay under a DIFFERENT routing
     bitwise == eager on that routing (the graph must be a pure function of
     buffer contents, since routing changes every step), + graph memory.
"""
import time

import torch

from engine.pyengine.kernels.moe_gemm import moe_experts_packed
from engine.pyengine.kernels.moe_gemm_w4a4 import moe_experts_w4a4
from engine.pyengine.tests.t_moe_w4a4 import DEV, load_pack, make_routing

E, K, TOPK = 256, 6144, 6
MB = 64          # the graphed step's padded decode batch (engine _pool_batch)


def _derived(flat, E):
    """offs/dst/src both ways: the shipped searchsorted scan vs the
    bincount+cumsum it replaced."""
    dev = flat.device
    P = flat.shape[0]
    sorted_e, perm = torch.sort(flat, stable=True)

    def finish(cnt, starts_real):
        pad_cnt = ((cnt + 127) // 128) * 128
        ends = torch.cumsum(pad_cnt, 0)
        starts_pad = ends - pad_cnt
        rank = torch.arange(P, device=dev) - starts_real[sorted_e]
        return (ends.to(torch.int32), (starts_pad[sorted_e] + rank).to(torch.int32))

    bounds = torch.searchsorted(
        sorted_e, torch.arange(E + 1, device=dev, dtype=sorted_e.dtype))
    new = finish(bounds[1:] - bounds[:E], bounds[:E])
    cnt = torch.bincount(sorted_e, minlength=E)
    old = finish(cnt, torch.cumsum(cnt, 0) - cnt)
    return new, old


def main():
    torch.manual_seed(0)
    p13 = load_pack(3, "w13_weight", True)
    p2 = load_pack(3, "w2_weight", False)
    ffn = p13.packed.shape[1] // 2
    print(f"real layer-3 experts: E={E} hidden={K} ffn={ffn} "
          f"amax13={p13.input_amax} amax2={p2.input_amax}")

    # ---- arm a: prefill numerics unchanged --------------------------
    ok_a = True
    for T in (736, MB, 1, 3000):
        tki, _ = make_routing(T, E, TOPK)
        new, old = _derived(tki.reshape(-1), E)
        same = all(torch.equal(n, o) for n, o in zip(new, old))
        ok_a &= same
        print(f"a  T={T:5d}: searchsorted offs/dst bitwise == bincount: {same}")
    print(f"a  {'PASS' if ok_a else 'FAIL'} — prefill path bit-identical to "
          f"exp-0016's gated tree")

    # ---- arm b: numerics vs W4A16, decode shape AND the prefill-shape
    #      CONTROL. The control is the whole argument: prefill W4A4 is
    #      already accepted (exp-0015a, 93.0) and its TF gate moved TOWARD
    #      the goldens, so decode W4A4 is safe iff its distance to W4A16 is
    #      the same class as prefill's, not larger.
    def _delta(T):
        xx = (torch.randn(T, K, device=DEV) * 0.4).to(torch.bfloat16)
        ti, tw = make_routing(T, E, TOPK)
        o4 = moe_experts_w4a4(xx, p13, p2, ti, tw)
        o16 = moe_experts_packed(xx, p13, p2, ti, tw)
        d = (o4.float() - o16.float()).abs()
        rel = (d / o16.float().abs().clamp(min=1e-3)).mean().item()
        cos = torch.nn.functional.cosine_similarity(
            o4.float().flatten(), o16.float().flatten(), dim=0).item()
        nz = int((torch.bincount(ti.reshape(-1), minlength=E) > 0).sum())
        fin = bool(torch.isfinite(o4).all())
        print(f"b  T={T:5d} ({nz:3d}/{E} experts hit): W4A4 vs W4A16 "
              f"rel_mean {rel:.3e} cos {cos:.5f} finite {fin}")
        return xx, ti, tw, cos

    _, _, _, cos_pf = _delta(736)          # PREFILL shape = the control
    x, tki, tkw, cos_dec = _delta(MB)      # decode shape
    print(f"b  decode-vs-prefill cos ratio {cos_dec / cos_pf:.4f} "
          f"(>=1 means decode is no further from W4A16 than accepted prefill)")

    # ---- arm c: decode-shape per-layer wall -------------------------
    walls = {}
    for fn, nm in ((moe_experts_w4a4, "w4a4"), (moe_experts_packed, "w4a16")):
        for _ in range(3):
            fn(x, p13, p2, tki, tkw)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            fn(x, p13, p2, tki, tkw)
        torch.cuda.synchronize()
        walls[nm] = (time.perf_counter() - t0) / 20 * 1e3
        print(f"c  {nm} moe_experts layer wall at decode T={MB}: "
              f"{walls[nm]:.3f} ms")
    spd = walls["w4a16"] / walls["w4a4"]
    print(f"c  decode layer speedup {spd:.2f}x -> 63 MoE layers: "
          f"{walls['w4a16'] * 63:.0f} ms -> {walls['w4a4'] * 63:.0f} ms/step")

    # ---- arm d: capture at the real decode shape --------------------
    sx, si, sw = x.clone(), tki.clone(), tkw.clone()
    x2 = (torch.randn(MB, K, device=DEV) * 0.4).to(torch.bfloat16)
    tki2, tkw2 = make_routing(MB, E, TOPK)
    ref1 = moe_experts_w4a4(x, p13, p2, tki, tkw).clone()
    ref2 = moe_experts_w4a4(x2, p13, p2, tki2, tkw2).clone()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            moe_experts_w4a4(sx, p13, p2, si, sw)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    free0 = torch.cuda.mem_get_info(DEV)[0]
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=torch.cuda.graph_pool_handle()):
        gout = moe_experts_w4a4(sx, p13, p2, si, sw)
    free1 = torch.cuda.mem_get_info(DEV)[0]
    g.replay()
    torch.cuda.synchronize()
    ok_b1 = torch.equal(gout, ref1)
    first = gout.clone()
    g.replay()
    torch.cuda.synchronize()
    ok_det = torch.equal(gout, first)
    sx.copy_(x2); si.copy_(tki2); sw.copy_(tkw2)
    g.replay()
    torch.cuda.synchronize()
    ok_b2 = torch.equal(gout, ref2)
    print(f"d  capture+replay vs eager bitwise={ok_b1}; same-schedule "
          f"determinism={ok_det}; different-routing replay bitwise={ok_b2}")
    print(f"d  capture reserved {(free0 - free1) / 2**20:+.0f} MiB "
          f"(free-mem delta; caching allocator makes this a loose bound)")

    # ---- arm e: the honest per-layer number — CAPTURED REPLAY wall.
    #      The eager walls of arm c still carry host dispatch (the very
    #      cost exp-0012's graph deleted); in the serving engine both paths
    #      run inside a graph, so replay-vs-replay is the comparison that
    #      predicts the step.
    def _replay_ms(fn, label):
        sx2, si2, sw2 = x.clone(), tki.clone(), tkw.clone()
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            for _ in range(2):
                fn(sx2, p13, p2, si2, sw2)
        torch.cuda.current_stream().wait_stream(st)
        torch.cuda.synchronize()
        gg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gg, pool=torch.cuda.graph_pool_handle()):
            fn(sx2, p13, p2, si2, sw2)
        for _ in range(3):
            gg.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            gg.replay()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 20 * 1e3
        print(f"e  {label} CAPTURED decode MoE layer replay: {ms:.3f} ms")
        return ms

    r4 = _replay_ms(moe_experts_w4a4, "w4a4 ")
    r16 = _replay_ms(moe_experts_packed, "w4a16")
    #      NB: the ABSOLUTE walls here are on UNIFORM-RANDOM routing, which
    #      hits ~201 of 256 experts at T=64; both kernels are dominated by
    #      per-hit-expert weight traffic, so the trained router's
    #      concentration makes both sides smaller in situ (w4a16 measures
    #      {r16} ms/layer here vs the exp-0012 in-situ table's 266.4/63 =
    #      4.23 ms). The RATIO is what transfers: it is measured on matched
    #      routing, both sides captured, same weights.
    frac = r4 / r16
    print(f"e  graphed layer speedup {r16 / r4:.2f}x on matched routing "
          f"(absolute walls are uniform-random-routing upper bounds, not "
          f"in-situ times)")
    print(f"e  ratio applied to the exp-0012 in-situ table: routed-expert "
          f"term 266.4 -> {266.4 * frac:.0f} ms, graphed step 322.7 -> "
          f"{322.7 - 266.4 * (1 - frac):.0f} ms")

    ok = ok_a and ok_b1 and ok_det and ok_b2
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
