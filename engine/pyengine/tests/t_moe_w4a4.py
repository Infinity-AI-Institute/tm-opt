"""exp-0015a local validation: the W4A4 prefill MoE path (moe_gemm_w4a4)
on REAL layer-3 routed experts. Arms:
  a  vectorized _sfb_blocked bitwise == the vendored per-expert to_blocked
  b  nvfp4_quant_scatter bitwise == torch reference (scales exact; packed
     bytes tie-class tolerant, groundwork arm L semantics)
  c  full moe_experts_w4a4 vs an fp32 emulation of the exact same quantized
     math (per-expert dequant GEMM chain) — catches any layout/scale bug
  d  determinism: same schedule twice -> bitwise identical output
  e  delta vs the W4A16 grouped path (arm-J expectation: rel_mean ~0.1)
  f  perf per layer at the T=736 prefill shape vs the W4A16 kernel
Includes a forced >128-row expert group (2 M-tiles + 256-row padding) and
naturally-empty groups. Run from the worktree root:
  CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python engine/pyengine/tests/t_moe_w4a4.py
"""
import time

import torch
from safetensors import safe_open

from engine.pyengine import loader
from engine.pyengine.kernels.fp4_quant import nvfp4_quant, nvfp4_quant_scatter
from engine.pyengine.kernels.moe_gemm import moe_experts_packed
from engine.pyengine.kernels.moe_gemm_w4a4 import (
    _sfb_blocked, moe_experts_w4a4)
from engine.pyengine.vendor.cutlass_moe.torch_scaled_grouped_mm import (
    to_blocked)

MODEL = "/workspace/models/inkling-nvfp4"
DEV = "cuda:0"
torch.manual_seed(0)

E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], device=DEV)


def deq_ref(packed, scale):
    """[R, K/2] uint8 + [R, K/16] e4m3 -> [R, K] fp32, NO global scale."""
    lo = (packed & 0xF).long()
    hi = (packed >> 4).long()
    w = torch.stack([E2M1_LUT[lo], E2M1_LUT[hi]],
                    dim=-1).view(packed.shape[0], -1)
    return w * scale.float().repeat_interleave(16, dim=1)


def quant_ref(x, gs):
    """Torch reference NVFP4 quant mirroring the KERNEL's fp32 association
    exactly (sf = e4m3(amax * (fp32(1/gs)/6)), zero-group guard via
    where(scf==0)): bf16 [M, K] -> (packed u8 [M, K/2], e4m3 scales
    [M, K/16], dequant fp32 incl. gs). vLLM's own quant kernel uses the
    same reciprocal-multiply association (nvfp4 SFScaleVal), so 1-ulp
    scale deltas vs any divide-form reference are envelope-class."""
    M, K = x.shape
    xf = x.float().view(M, K // 16, 16)
    amax = xf.abs().amax(dim=-1)
    inv6 = torch.tensor(1.0 / gs, dtype=torch.float32, device=x.device) / 6.0
    sc = (amax * inv6).to(torch.float8_e4m3fn)
    scf = sc.float() * gs
    div = torch.where(scf == 0.0, torch.ones_like(scf), scf)
    y = (xf / div[:, :, None]).view(M, K)
    mags = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=x.device)
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device)
    code = torch.bucketize(y.abs(), mids)
    q = mags[code] * torch.sign(y)
    codes = (code + torch.where(y < 0, 8, 0)).to(torch.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed, sc, q * scf.repeat_interleave(16, dim=1).view(M, K), y


def load_pack(layer, wname, deint):
    idx = loader.build_shard_index(MODEL)
    base = f"model.llm.layers.{layer}.mlp.experts.{wname}"

    def rd(name):
        with safe_open(str(idx.shard_path(name)), framework="pt") as f:
            return f.get_tensor(name)

    pk = rd(base).to(DEV)
    sc = rd(base + ".scale")
    if sc.dtype == torch.uint8:
        sc = sc.view(torch.float8_e4m3fn)
    return loader.PackedExperts(
        pk, sc.to(DEV), rd(base + ".scale2").float().to(DEV), 16, deint,
        input_amax=rd(base + ".input_amax").float().max())


def make_routing(T, E, top_k, hot=None, hot_rows=0):
    """Distinct experts per token (B2.8); optionally force expert `hot`
    into slot 0 of the first hot_rows tokens lacking it (a >128-row group)."""
    tki = torch.stack([torch.randperm(E, device=DEV)[:top_k]
                       for _ in range(T)]).to(torch.int64)
    if hot is not None:
        done = 0
        for t in range(T):
            if done >= hot_rows:
                break
            if (tki[t] == hot).sum() == 0:
                tki[t, 0] = hot
                done += 1
    tkw = torch.rand(T, top_k, device=DEV, dtype=torch.float32)
    return tki, tkw


def emulate(x, p13, p2, tki, tkw):
    """fp32 emulation of the kernel path's exact quantized math, per hit
    expert: a_deq(gs13) @ w13_hat.T * s2_13[e] -> bf16 -> eager silu chain
    -> quant(gs2) -> a2_deq @ w2_hat.T * s2_2[e] -> bf16 -> *wt -> bf16 ->
    fp32 slot sum."""
    T, K = x.shape
    gs13 = p13.input_amax / (448.0 * 6.0)
    gs2 = p2.input_amax / (448.0 * 6.0)
    top_k = tki.shape[1]
    outs = torch.zeros(T, top_k, K, device=DEV, dtype=torch.bfloat16)
    for e in torch.unique(tki).tolist():
        tok, slot = torch.where(tki == e)
        _, _, a_deq, _ = quant_ref(x[tok], gs13)
        w13 = deq_ref(p13.packed[e], p13.scale[e])   # interleaved rows
        c1 = (a_deq @ w13.t() * p13.scale2[e]).to(torch.bfloat16)
        gu = c1.view(-1, c1.shape[1] // 2, 2)
        g = gu[..., 0].float()
        u = gu[..., 1].float()
        act = ((g * torch.sigmoid(g)).to(torch.bfloat16).float()
               * u).to(torch.bfloat16)
        _, _, a2_deq, _ = quant_ref(act, gs2)
        w2 = deq_ref(p2.packed[e], p2.scale[e])
        c2 = (a2_deq @ w2.t() * p2.scale2[e]).to(torch.bfloat16)
        outs[tok, slot] = (c2.float() * tkw[tok, slot, None]).to(torch.bfloat16)
    return outs.float().sum(dim=1).to(torch.bfloat16)


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = False
    p13 = load_pack(3, "w13_weight", True)
    p2 = load_pack(3, "w2_weight", False)
    E = p13.packed.shape[0]
    K = p13.packed.shape[2] * 2
    ffn = p13.packed.shape[1] // 2
    print(f"layer 3 loaded: E={E} K={K} ffn={ffn} "
          f"amax13={p13.input_amax:.4g} amax2={p2.input_amax:.4g}")

    # a. vectorized SFB swizzle == per-expert to_blocked (bitwise)
    for pack, nm in ((p13, "w13"), (p2, "w2")):
        blk = _sfb_blocked(pack.scale)
        for e in (0, 1, 137, E - 1):
            ref = to_blocked(pack.scale[e].view(torch.float8_e4m3fn)
                             if pack.scale[e].dtype == torch.uint8
                             else pack.scale[e])
            assert torch.equal(blk[e].view(torch.uint8),
                               ref.view(torch.uint8)), (nm, e)
    print("arm a PASS: _sfb_blocked bitwise == to_blocked")

    # b. scatter-quant vs torch reference on a 128-padded group layout
    gs = p13.input_amax / (448.0 * 6.0)
    Tq, Pq = 300, 512
    xq = (torch.randn(Tq, K, device=DEV) * 0.4).to(torch.bfloat16)
    src = torch.randint(0, Tq, (Pq,), device=DEV, dtype=torch.int32)
    grp = torch.sort(torch.randint(0, 6, (Pq,), device=DEV)).values
    cnt = torch.bincount(grp, minlength=6)
    pad = ((cnt + 127) // 128) * 128
    starts = torch.cumsum(pad, 0) - pad
    rank = torch.arange(Pq, device=DEV) - (torch.cumsum(cnt, 0) - cnt)[grp]
    dstq = (starts[grp] + rank).to(torch.int32)
    m_pad = int(pad.sum())
    pk_k, sf_k = nvfp4_quant_scatter(xq, src, dstq, m_pad, gs)
    r_pk, r_sc, _, r_y = quant_ref(xq[src.long()], gs)
    ref_raw = torch.zeros(m_pad, K // 16, dtype=torch.float8_e4m3fn,
                          device=DEV)
    ref_raw[dstq.long()] = r_sc
    assert torch.equal(sf_k.view(torch.uint8),
                       to_blocked(ref_raw).view(torch.uint8)), "sfa mismatch"
    #   every packed-byte diff must be RTNE-vs-ties-up at an exact e2m1
    #   midpoint (or the sign of a rounded-to-zero value) — dequant-value
    #   drift of at most the tie step, the groundwork arm-L class
    kr = pk_k[dstq.long()]
    ck = torch.stack([kr & 0xF, kr >> 4], -1).view(Pq, K)
    cr = torch.stack([r_pk & 0xF, r_pk >> 4], -1).view(Pq, K)
    mism = ck != cr
    rate = mism.float().mean().item()
    mk, mr = (ck & 7).long(), (cr & 7).long()
    mids_f = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                          device=DEV)
    lo_i = torch.minimum(mk, mr).clamp(max=6)
    tie = (((mk - mr).abs() == 1) & ((ck >> 3) == (cr >> 3))
           & (r_y.abs() == mids_f[lo_i]))
    zero_sign = (mk == 0) & (mr == 0)
    bad = mism & ~(tie | zero_sign)
    assert not bad.any(), f"non-tie packed diffs: {int(bad.sum())}"
    assert torch.equal(pk_k[dstq.long()], kr)  # scatter rows == real rows
    print(f"arm b PASS: scales bitwise, packed diffs all tie-class "
          f"(rate {rate:.2e})")

    # c+d. per-stage arbiter + determinism, with a >128-row group.
    #   Arbiter = fp32-out torch._scaled_mm (groundwork arm A's proven
    #   native vehicle) on IDENTICAL quantized operands with the identical
    #   alpha association (acc * (gs*scale2) in fp32, ONE bf16 round) —
    #   the DSL grouped kernel must match bitwise up to fp32-accumulation-
    #   order ties (observed: single elements at 1 bf16 ulp). An
    #   independent end-to-end emulation can only agree to ~cos 0.997:
    #   pervasive 1-ulp bf16 GEMM rounding flips get amplified by the
    #   second quant's e2m1 midpoints — the D13-envelope drift class, so
    #   it is reported, not gated tightly.
    from engine.pyengine.kernels import moe_gemm_w4a4 as W
    T = 200
    x = (torch.randn(T, K, device=DEV) * 0.4).to(torch.bfloat16)
    tki, tkw = make_routing(T, E, 6, hot=7, hot_rows=150)
    t0 = time.time()
    out_k = moe_experts_w4a4(x, p13, p2, tki, tkw)
    print(f"first call (2 JIT compiles): {time.time() - t0:.1f} s")
    out_k2 = moe_experts_w4a4(x, p13, p2, tki, tkw)
    assert torch.equal(out_k, out_k2), "nondeterministic output"
    print("arm d PASS (bitwise-deterministic)")

    #   internals, mirroring moe_experts_w4a4 step for step
    gs13 = p13.input_amax / (448.0 * 6.0)
    gs2 = p2.input_amax / (448.0 * 6.0)
    prep = W._prep(p13, p2)
    flat = tki.reshape(-1)
    P = flat.shape[0]
    se, perm = torch.sort(flat, stable=True)
    cnt = torch.bincount(se, minlength=E)
    pad = ((cnt + 127) // 128) * 128
    ends = torch.cumsum(pad, 0)
    offs = ends.to(torch.int32)
    sp = ends - pad
    rank = torch.arange(P, device=DEV) - (torch.cumsum(cnt, 0) - cnt)[se]
    dst = (sp[se] + rank).to(torch.int32)
    src = (perm // 6).to(torch.int32)
    m_cap = int(ends[-1])
    a1, sfa1 = nvfp4_quant_scatter(x, src, dst, m_cap, gs13)
    c1 = torch.empty(m_cap, 2 * ffn, device=DEV, dtype=torch.bfloat16)
    W._run_gemm(K, 2 * ffn, a1, sfa1.view(m_cap, K // 16), prep.b13,
                prep.sfb13_c, c1, offs, prep.gsa13_c, prep.gsb13_c)
    gu = c1.view(m_cap, ffn, 2)
    g = gu[..., 0].float()
    u = gu[..., 1].float()
    act = ((g * torch.sigmoid(g)).to(torch.bfloat16).float()
           * u).to(torch.bfloat16).contiguous()
    a2, sfa2 = nvfp4_quant(act, gs2)
    c2 = torch.empty(m_cap, K, device=DEV, dtype=torch.bfloat16)
    W._run_gemm(ffn, K, a2, sfa2.view(m_cap, ffn // 16), prep.b2,
                prep.sfb2_c, c2, offs, prep.gsa2_c, prep.gsb2_c)
    #   consistency: the real function's output reconstructs from these
    wt_s = tkw.reshape(-1)[perm]
    vals = (c2.index_select(0, dst.long()).float()
            * wt_s.float()[:, None]).to(torch.bfloat16)
    op = torch.empty(P, K, device=DEV, dtype=torch.bfloat16)
    op[perm] = vals
    assert torch.equal(op.view(T, 6, K).sum(dim=1), out_k), "internals drift"

    fp4 = lambda t: t.view(torch.float4_e2m1fn_x2)
    worst = (0, 0.0)
    for e in torch.unique(se).tolist():
        s0, s1 = int(sp[e]), int(sp[e] + cnt[e])
        xr = x[(perm[(se == e).nonzero()[:, 0]] // 6).long()]
        _, rsc, _, _ = quant_ref(xr, gs13)
        o1 = torch._scaled_mm(
            fp4(a1[s0:s1]), fp4(p13.packed[e]).t(), to_blocked(rsc),
            to_blocked(p13.scale[e]), out_dtype=torch.float32)
        o1 = (o1 * (gs13 * float(p13.scale2[e]))).to(torch.bfloat16)
        _, rsc2, _, _ = quant_ref(act[s0:s1], gs2)
        o2 = torch._scaled_mm(
            fp4(a2[s0:s1]), fp4(p2.packed[e]).t(), to_blocked(rsc2),
            to_blocked(p2.scale[e]), out_dtype=torch.float32)
        o2 = (o2 * (gs2 * float(p2.scale2[e]))).to(torch.bfloat16)
        for o_ref, o_dsl in ((o1, c1[s0:s1]), (o2, c2[s0:s1])):
            d = (o_ref.float() - o_dsl.float()).abs()
            n = int((d > 0).sum())
            if n:
                ulp = 2.0 ** torch.floor(
                    torch.log2(o_ref.float().abs().clamp(min=1e-30))) * 2 ** -7
                mu = (d / ulp).max().item()
                worst = max(worst, (n, mu))
                assert n <= max(4, d.numel() * 5e-5) and mu <= 1.01, \
                    (e, n, mu)
    print(f"arm c PASS: all hit experts (incl. the >128-row 2-tile group, "
          f"{int(cnt[7])} rows) match the _scaled_mm arbiter bitwise up to "
          f"accumulation ties (worst expert: {worst[0]} elems, "
          f"{worst[1]:.2f} bf16 ulp)")
    out_em = emulate(x, p13, p2, tki, tkw)
    cos = torch.nn.functional.cosine_similarity(
        out_k.float().flatten(), out_em.float().flatten(), dim=0).item()
    print(f"arm c report: end-to-end vs independent emulation cos {cos:.5f} "
          f"(quant-amplified accumulation noise; gate is the TF envelope)")
    assert cos > 0.99

    # e. distance to the W4A16 grouped path (arm-J class, sanity only)
    out_16 = moe_experts_packed(x, p13, p2, tki, tkw)
    d16 = (out_k.float() - out_16.float()).abs()
    rel16 = (d16 / out_16.float().abs().clamp(min=1e-3)).mean().item()
    cos16 = torch.nn.functional.cosine_similarity(
        out_k.float().flatten(), out_16.float().flatten(), dim=0).item()
    print(f"arm e: W4A4 vs W4A16 rel_mean {rel16:.3e} cos {cos16:.4f} "
          f"(arm-J expectation ~1e-1 / ~0.995)")

    # f. perf at the T=736 prefill shape
    T = 736
    x = (torch.randn(T, K, device=DEV) * 0.4).to(torch.bfloat16)
    tki, tkw = make_routing(T, E, 6)
    for fn, nm in ((moe_experts_w4a4, "w4a4"), (moe_experts_packed, "w4a16")):
        for _ in range(3):
            fn(x, p13, p2, tki, tkw)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            fn(x, p13, p2, tki, tkw)
        torch.cuda.synchronize()
        print(f"arm f: {nm} moe_experts layer wall at T=736: "
              f"{(time.perf_counter() - t0) / 20 * 1e3:.2f} ms")
    print("t_moe_w4a4 done.")
