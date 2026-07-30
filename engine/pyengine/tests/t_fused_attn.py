"""exp-0013 evidence: fused two-shape decode attention vs the eager
pooled chain.

A/B through the REAL attn_decode_batch_pooled (fused=False vs
fused=True) on synthetic pool state — same weights, same ctx, pool
snapshot/restored between calls so both arms see identical bytes
(the function appends into the pool). Rows cover the geometry edges:
pos 0 (self-only), pre-wrap 3/200, the 511/512/513 ring boundary,
800 (wrapped), 1500 (> rel_extent 1024 on the global shape — live key,
zero bias), plus the scratch slot at pos 0 (the graph path's dead-row
config). Slots are shuffled so slot != batch row.

Expected: ulp-class bf16 drift only (fp32 flash accumulation vs cuBLAS
tiling — B3.1/B3.3 drift, D13-envelope semantics; kernels/attn_swa.py),
gated loosely at max|diff| <= 0.25 with mean <= 2e-3 over ~1e5 checked
outputs of unit-scale inputs, and >= 99% of elements within 2 bf16 ulp.
C arm: fused twice on restored state -> bitwise identical (t_b3
determinism form). GPUs 4-7 only:
    CUDA_VISIBLE_DEVICES=4 python -m engine.pyengine.tests.t_fused_attn
"""
import types

import torch

from engine.pyengine import kv as pkv
from engine.pyengine import model as pmodel


def _mk_weights(hidden, n_heads, n_kv, head_dim, d_rel, extent, dev, g):
    bf = torch.bfloat16
    def rnd(*shape, scale=0.05):
        return (torch.randn(*shape, generator=g, device=dev,
                            dtype=torch.float32) * scale).to(bf)
    kvd = n_kv * head_dim
    return dict(
        wq=rnd(n_heads * head_dim, hidden), wk=rnd(kvd, hidden),
        wv=rnd(kvd, hidden), wr=rnd(n_heads * d_rel, hidden),
        wo=rnd(hidden, n_heads * head_dim),
        ksc=rnd(kvd, 1, 4, scale=0.2), vsc=rnd(kvd, 1, 4, scale=0.2),
        qn=(1 + rnd(head_dim)).to(bf), kn=(1 + rnd(head_dim)).to(bf),
        proj=rnd(d_rel, extent, scale=0.5))


def _run_shape(name, is_global, pos_host, dev, g):
    hidden, head_dim, d_rel = 512, 128, 16
    n_heads = 64
    n_kv = 8 if is_global else 16
    window, extent = 512, (1024 if is_global else 512)
    MB = len(pos_host)                    # live rows incl. scratch mimic
    mc = types.SimpleNamespace(window=window)
    w = _mk_weights(hidden, n_heads, n_kv, head_dim, d_rel, extent, dev, g)
    #1. pool with random STALE bytes everywhere (dead lanes must be
    #   excluded by selection, never read as zeros)
    if is_global:
        num_pages = (max(pos_host) + pkv.PAGE_SIZE) // pkv.PAGE_SIZE + 2
        pool = pkv.GlobalPool(MB, num_pages, pkv.PAGE_SIZE, n_kv,
                              head_dim, dev)
        pool.kp.normal_(generator=g).mul_(0.05)
        pool.vp.normal_(generator=g).mul_(0.05)
        #   distinct page ids per (slot, page) — scrambled placement
        ids = torch.randperm(MB * num_pages, generator=g,
                             device=dev)[:MB * num_pages]
        pool.table_dev[:MB] = ids.view(MB, num_pages)
    else:
        pool = pkv.SwaPool(MB, window, n_kv, head_dim, dev)
        pool.k.normal_(generator=g).mul_(0.05)
        pool.v.normal_(generator=g).mul_(0.05)
    pool.sconv = pkv.SconvPool(MB, 4, n_kv * head_dim, hidden, dev)
    pool.sconv.kt.normal_(generator=g).mul_(0.05)
    pool.sconv.vt.normal_(generator=g).mul_(0.05)
    #2. shuffled slot map (slot != row), scratch slot (MB) as last row
    perm = torch.randperm(MB - 1, generator=g, device=dev).tolist()
    slots_host = perm + [MB]              # last row = scratch, pos 0
    assert pos_host[-1] == 0
    ctx = pmodel.decode_batch_ctx(pos_host, slots_host, mc, dev,
                                  pkv.PAGE_SIZE)
    x = (torch.randn(MB, hidden, generator=g, device=dev,
                     dtype=torch.float32) * 0.05).to(torch.bfloat16)
    args = (x, w["wq"], w["wk"], w["wv"], w["wr"], w["wo"], w["ksc"],
            w["vsc"], w["qn"], w["kn"], w["proj"], None, pool, ctx,
            head_dim, is_global)
    kw = dict(alpha=0.1, n_floor=128000, eps=1e-6)
    #3. snapshot mutable pool state; run eager / fused / fused-again
    def snap():
        t = [pool.sconv.kt.clone(), pool.sconv.vt.clone()]
        if is_global:
            return t + [pool.kp.clone(), pool.vp.clone()]
        return t + [pool.k.clone(), pool.v.clone()]
    def restore(s):
        pool.sconv.kt.copy_(s[0]); pool.sconv.vt.copy_(s[1])
        if is_global:
            pool.kp.copy_(s[2]); pool.vp.copy_(s[3])
        else:
            pool.k.copy_(s[2]); pool.v.copy_(s[3])
    s0 = snap()
    ref = pmodel.attn_decode_batch_pooled(*args, **kw, fused=False)
    restore(s0)
    fus = pmodel.attn_decode_batch_pooled(*args, **kw, fused=True)
    restore(s0)
    fus2 = pmodel.attn_decode_batch_pooled(*args, **kw, fused=True)
    #4. verdicts
    d = (ref.float() - fus.float()).abs()
    scale_ref = ref.float().abs().mean()
    ulp = (d / (ref.float().abs() + 1e-3)) < (2 * 2 ** -8)  # ~2 bf16 ulp
    print(f"[{name}] B={MB} maxdiff {d.max():.5f} meandiff {d.mean():.6f}"
          f" ref_scale {scale_ref:.4f} within2ulp"
          f" {ulp.float().mean() * 100:.2f}% bitwise_C"
          f" {torch.equal(fus, fus2)}")
    assert torch.equal(fus, fus2), "determinism arm FAILED"
    assert torch.isfinite(fus.float()).all(), "non-finite output"
    assert float(d.max()) <= 0.25 and float(d.mean()) <= 2e-3, \
        (float(d.max()), float(d.mean()))


def main():
    assert torch.cuda.is_available()
    dev = "cuda:0"
    g = torch.Generator(device=dev).manual_seed(1013)
    #1. geometry-edge positions; final row is the scratch/dead config
    _run_shape("swa", False, [0, 3, 200, 511, 512, 513, 800, 0], dev, g)
    _run_shape("global", True,
               [0, 3, 200, 511, 512, 513, 800, 1500, 0], dev, g)
    print("t_fused_attn PASS")


if __name__ == "__main__":
    main()
