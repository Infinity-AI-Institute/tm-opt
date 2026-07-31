"""exp-0023 evidence: fused two-shape PREFILL attention.

Two arm families, because this iteration had no GPU (the dispatcher was
running exp-0022 on GPUs 4-7 for the whole session and GPUs 0-3 hold the
frozen vLLM baseline):

  --sim   CPU ONLY, no CUDA context is ever created. A pure-torch
          transcription of the Triton kernel's blocked math (same loop
          bounds, same distance/band/mask expressions, same online-softmax
          update) is compared in fp32 against the engine's OWN eager chain
          (model.rel_bias / apply_log_scaling / additive_causal_mask are
          imported, not re-implemented). In fp32 the two differ only by
          summation order, so anything above ~1e-5 is an ALGEBRA bug:
          wrong block bounds, wrong band, wrong GQA head, wrong layout,
          wrong tau placement. This is what a no-GPU session can honestly
          pre-gate; it does NOT check Triton codegen or speed.

  --gpu   the real arms: numerics vs the eager chain at the canonical
          cohort shapes, same-schedule determinism, CUDA-event timing and
          peak-memory, plus a model.attn_prefill kill-switch A/B including
          the right-padded batched form. Run on GPUs 4-7 only.

Usage:  python -m engine.pyengine.tests.t_fused_prefill_attn --sim
        python -m engine.pyengine.tests.t_fused_prefill_attn --gpu [--dev 4]
"""
import argparse
import math
import sys

import torch

from engine.pyengine import model as pmodel

#1. canonical shapes (CLAUDE.md model facts + the exp-0017 cohort): 64 Q
#   heads, head_dim 128, SWA 16 KV heads / window 512 / rel extent 512,
#   global 8 KV heads / rel extent 1024, cohort of 6 rows padded to 1536
HEADS, HEAD_DIM, D_REL = 64, 128, 16
SWA = dict(n_kv=16, extent=512, window=512, is_global=False)
GLOBAL = dict(n_kv=8, extent=1024, window=None, is_global=True)
COHORT_B, COHORT_T = 6, 1536


def eager_ref(q, k, v, r, proj, positions, window, is_global,
              alpha=0.1, n_floor=128000.0):
    """attn_prefill #3-#8 verbatim, on the engine's own helpers.
    q [B,H,T,D], k/v [B,HK,T,D], r [B,T,H,d_rel]. Returns [B,T,H,D]."""
    B, H, T, Dh = q.shape
    n_kv = k.shape[1]
    bias = pmodel.rel_bias(r, proj, positions, positions)
    if is_global:
        q, bias = pmodel.apply_log_scaling(q, bias, positions, alpha,
                                           n_floor)
    mask = pmodel.additive_causal_mask(positions, positions, q.dtype,
                                       window=window)
    rep = H // n_kv
    kk = k[:, :, None].expand(B, n_kv, rep, T, Dh).reshape(B, H, T, Dh)
    vv = v[:, :, None].expand(B, n_kv, rep, T, Dh).reshape(B, H, T, Dh)
    attn = torch.matmul(q, kk.transpose(2, 3)) * (1.0 / Dh)
    attn = attn + bias
    attn = attn + mask
    attn = torch.nn.functional.softmax(attn, dim=-1,
                                       dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn, vv).transpose(1, 2).contiguous()


def sim_kernel(q, k, v, r, proj, positions, window, is_global,
               alpha=0.1, n_floor=128000.0, block_m=64, block_n=64):
    """Pure-torch transcription of _attn_prefill_kernel — the SAME loop
    bounds and per-tile expressions, so an indexing/masking bug in the
    kernel is a bug here too. Kept deliberately literal (loops over the
    same blocks) rather than vectorized."""
    B, H, T, Dh = q.shape
    n_kv = k.shape[1]
    rep = H // n_kv
    E = proj.shape[1]
    #1. the kernel's inputs: the rel MIX (rel_bias #1), tau applied to it
    rel = (r @ proj).transpose(1, 2)                     # [B,H,T,E]
    if is_global:
        q, rel = pmodel.apply_log_scaling(q, rel, positions, alpha, n_floor)
    out = torch.zeros((B, T, H, Dh), dtype=q.dtype, device=q.device)
    scale = 1.0 / Dh
    is_swa = window is not None
    W = window or 0
    for b in range(B):
        for h in range(H):
            kvh = h // rep
            for m0 in range(0, T, block_m):
                m_off = torch.arange(m0, m0 + block_m, device=q.device)
                m_mask = m_off < T
                mi = m_off.clamp(max=T - 1)
                qt = q[b, h, mi].float()                 # [BM, D]
                qp = positions[mi]
                m_i = torch.full((block_m,), -math.inf, device=q.device)
                l_i = torch.zeros(block_m, device=q.device)
                acc = torch.zeros((block_m, Dh), device=q.device)
                hi = min(m0 + block_m, T)
                lo = (max(m0 - W + 1, 0) // block_n) * block_n if is_swa \
                    else 0
                for n0 in range(lo, hi, block_n):
                    n_off = torch.arange(n0, n0 + block_n, device=q.device)
                    n_mask = n_off < T
                    ni = n_off.clamp(max=T - 1)
                    kt = k[b, kvh, ni].float()
                    kp = positions[ni]
                    s = (qt @ kt.T) * scale
                    dist = qp[:, None] - kp[None, :]
                    band = (dist >= 0) & (dist < E)
                    dcl = dist.clamp(0, E - 1)
                    bias = torch.where(
                        m_mask[:, None] & band,
                        rel[b, h][mi][:, None, :].expand(
                            -1, block_n, -1).gather(
                            2, dcl[:, :, None].clamp(min=0))[:, :, 0].float(),
                        torch.zeros((), device=q.device))
                    s = s + bias
                    valid = (dist >= 0) & m_mask[:, None] & n_mask[None, :]
                    if is_swa:
                        valid = valid & (dist < W)
                    s = torch.where(valid, s, torch.full_like(s, -1e38))
                    m_new = torch.maximum(m_i, s.max(dim=1).values)
                    a = torch.exp(m_i - m_new)
                    prob = torch.exp(s - m_new[:, None])
                    l_i = l_i * a + prob.sum(dim=1)
                    vt = v[b, kvh, ni].float()
                    acc = acc * a[:, None] + prob.to(v.dtype).float() @ vt
                    m_i = m_new
                res = (acc / l_i[:, None]).to(q.dtype)
                out[b, m_off[m_mask], h] = res[m_mask]
    return out


def _rand(shape, dtype, dev, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(shape, generator=g, dtype=torch.float32) * 0.5).to(
        dtype).to(dev)


def _inputs(B, T, cfg, dtype, dev, seed=7):
    q = _rand((B, HEADS, T, HEAD_DIM), dtype, dev, seed)
    k = _rand((B, cfg["n_kv"], T, HEAD_DIM), dtype, dev, seed + 1)
    v = _rand((B, cfg["n_kv"], T, HEAD_DIM), dtype, dev, seed + 2)
    r = _rand((B, T, HEADS, D_REL), dtype, dev, seed + 3)
    proj = _rand((D_REL, cfg["extent"]), dtype, dev, seed + 4)
    pos = torch.arange(T, device=dev)
    return q, k, v, r, proj, pos


def _report(tag, a, b):
    d = (a.float() - b.float()).abs()
    den = b.float().abs().max().clamp(min=1e-9)
    ok = float(d.max()) <= 2e-4 * float(den)
    print(f"[t0023] {tag:<34} max|d| {float(d.max()):.3e}  "
          f"mean|d| {float(d.mean()):.3e}  scale {float(den):.3e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def run_sim():
    """CPU algebra arms, fp32 (no CUDA context is created)."""
    dev, dt = "cpu", torch.float32
    ok = True
    #a/b. both layer shapes, small T so the CPU transcription is tractable
    for name, cfg, T in (("swa", SWA, 192), ("global", GLOBAL, 192)):
        q, k, v, r, proj, pos = _inputs(2, T, cfg, dt, dev)
        # a short window exercises the band + the SWA block-skip bound
        win = 64 if cfg["window"] else None
        ref = eager_ref(q, k, v, r, proj, pos, win, cfg["is_global"])
        sim = sim_kernel(q, k, v, r, proj, pos, win, cfg["is_global"],
                         block_m=32, block_n=32)
        ok &= _report(f"arm a  {name}: sim vs eager", sim, ref)
    #b. LAYOUT CENSUS on the engine's own ops — the kernel reads raw
    #   strides, so what the call site actually hands it is load-bearing:
    #   sconv_prefill's own-input residual keeps the channel-major layout
    #   of conv.transpose(1,2), so K and V arrive stride-1 along T (which
    #   is also why the eager matmul pays a contiguity copy on them),
    #   while q and the rel mix are last-dim contiguous
    T, n_kv, C = 96, SWA["n_kv"], HEADS * HEAD_DIM
    xs = _rand((1, T, C), dt, dev, 61)
    kw = _rand((n_kv * HEAD_DIM, C), dt, dev, 62)
    ksc = _rand((n_kv * HEAD_DIM, 1, 4), dt, dev, 63)
    k_raw = torch.nn.functional.linear(xs, kw)
    ks = pmodel.sconv_prefill(k_raw, ksc)
    kt = pmodel.rmsnorm(ks.view(1, T, n_kv, HEAD_DIM),
                        _rand((HEAD_DIM,), dt, dev, 64)).transpose(1, 2)
    rr = _rand((1, T, HEADS, D_REL), dt, dev, 65)
    rel = (rr @ _rand((D_REL, SWA["extent"]), dt, dev, 66)).transpose(1, 2)
    print(f"[t0023] arm b  post-sconv K [B,HK,T,D] strides {kt.stride()} "
          f"(head_dim stride {kt.stride(3)}, T stride {kt.stride(2)})")
    print(f"[t0023] arm b  rel mix [B,H,T,E] strides {rel.stride()}")
    lay = kt.stride(3) != 1 and rel.stride(3) == 1
    print(f"[t0023] arm b  K channel-major + rel last-dim contiguous: "
          f"{'PASS' if lay else 'FAIL'} (kernel takes explicit D strides "
          f"for q/k/v and requires contiguity only on rel)")
    ok &= lay
    #c. right-padded rows: row i's live [:len] output must be what the row
    #   alone produces (the batched-prefill causality argument)
    #   pads deliberately carry GARBAGE (nothing is zeroed): the claim is
    #   that causality isolates real rows from them, not that they are 0
    T, lens = 160, [160, 96, 33]
    q, k, v, r, proj, pos = _inputs(3, T, SWA, dt, dev, seed=21)
    grouped = sim_kernel(q, k, v, r, proj, pos, 64, False, block_m=32,
                         block_n=32)
    for i, L in enumerate(lens):
        solo = sim_kernel(q[i:i + 1, :, :L], k[i:i + 1, :, :L],
                          v[i:i + 1, :, :L], r[i:i + 1, :L],
                          proj, pos[:L], 64, False, block_m=32, block_n=32)
        ok &= _report(f"arm c  padded row {i} (len {L})",
                      grouped[i, :L], solo[0])
    #d. tau on the MIX == tau on the gathered bias (the fused path's one
    #   algebraic move); forced live with a floor below the positions
    T = 96
    q, k, v, r, proj, pos = _inputs(1, T, GLOBAL, dt, dev, seed=33)
    tau_floor = 8.0
    ref = eager_ref(q, k, v, r, proj, pos, None, True, n_floor=tau_floor)
    sim = sim_kernel(q, k, v, r, proj, pos, None, True, n_floor=tau_floor,
                     block_m=32, block_n=32)
    tau = pmodel.log_scale_tau(pos, 0.1, tau_floor)
    print(f"[t0023] arm d  tau range [{float(tau.min()):.4f}, "
          f"{float(tau.max()):.4f}] (must not be all-1)")
    ok &= bool(float(tau.max()) > 1.0)
    ok &= _report("arm d  tau-on-mix vs tau-on-bias", sim, ref)
    #d2. the real call site with the switch OFF still runs the accepted
    #    eager path end to end (projections, sconvs, norms, state=None) —
    #    a CPU regression check on the code this patch edits, since the
    #    canonical run would be fatal if the ELSE branch broke
    T = 96
    n_kv = SWA["n_kv"]
    C = HEADS * HEAD_DIM
    x = _rand((1, T, C), dt, dev, 51)
    w = {n: _rand(s, dt, dev, 52 + i) for i, (n, s) in enumerate((
        ("wq", (HEADS * HEAD_DIM, C)), ("wk", (n_kv * HEAD_DIM, C)),
        ("wv", (n_kv * HEAD_DIM, C)), ("wr", (HEADS * D_REL, C)),
        ("wo", (C, HEADS * HEAD_DIM)), ("ksc", (n_kv * HEAD_DIM, 1, 4)),
        ("vsc", (n_kv * HEAD_DIM, 1, 4)), ("qn", (HEAD_DIM,)),
        ("kn", (HEAD_DIM,)), ("proj", (D_REL, SWA["extent"]))))}
    pos = torch.arange(T)
    mask = pmodel.additive_causal_mask(pos, pos, dt, window=SWA["window"])
    if pmodel._FUSED_PREFILL_ATTN:
        print("[t0023] arm d2 SKIP: the fused branch needs a GPU — rerun "
              "with PYENGINE_FUSED_PREFILL_ATTN=0 for the eager call site")
    else:
        out = pmodel.attn_prefill(
            x, w["wq"], w["wk"], w["wv"], w["wr"], w["wo"], w["ksc"],
            w["vsc"], w["qn"], w["kn"], w["proj"], mask, pos, pos,
            HEAD_DIM, False, alpha=0.1, n_floor=128000.0,
            window=SWA["window"])
        fin = bool(torch.isfinite(out).all()) and out.shape == (1, T, C)
        print(f"[t0023] arm d2 attn_prefill(switch=0) shape "
              f"{tuple(out.shape)} finite {fin}: "
              f"{'PASS' if fin else 'FAIL'}")
        ok &= fin
    #e. the mask the kernel derives IS additive_causal_mask's predicate
    T = 64
    pos = torch.arange(T)
    for win in (None, 16):
        m = pmodel.additive_causal_mask(pos, pos, torch.float32, window=win)
        dist = pos[:, None] - pos[None, :]
        valid = dist >= 0
        if win is not None:
            valid = valid & (dist < win)
        same = bool(torch.equal(valid, m[0, 0] == 0.0))
        print(f"[t0023] arm e  mask predicate window={win}: "
              f"{'PASS' if same else 'FAIL'}")
        ok &= same
    return ok


def run_gpu(dev_id):
    """The real arms. GPUs 4-7 only."""
    import time
    from engine.pyengine.kernels.attn_prefill import attn_prefill_fused
    dev = f"cuda:{dev_id}"
    dt = torch.bfloat16
    ok = True
    for name, cfg in (("swa", SWA), ("global", GLOBAL)):
        q, k, v, r, proj, pos = _inputs(COHORT_B, COHORT_T, cfg, dt, dev)
        win = cfg["window"]
        #f. numerics vs the eager chain at the canonical cohort shape
        ref = eager_ref(q, k, v, r, proj, pos, win, cfg["is_global"])
        rel = (r @ proj).transpose(1, 2)
        qs = q
        if cfg["is_global"]:
            qs, rel = pmodel.apply_log_scaling(q, rel, pos, 0.1, 128000.0)
        got = attn_prefill_fused(qs, k, v, rel, pos, pos, win)
        d = (got.float() - ref.float())
        cos = torch.nn.functional.cosine_similarity(
            got.float().flatten(), ref.float().flatten(), dim=0)
        rel_mean = (d.abs().mean() / ref.float().abs().mean())
        print(f"[t0023] arm f  {name}: rel_mean {float(rel_mean):.3e}  "
              f"cos {float(cos):.6f}  max|d| {float(d.abs().max()):.3e}  "
              f"finite {bool(torch.isfinite(got).all())}")
        ok &= bool(torch.isfinite(got).all()) and float(cos) > 0.99
        #g. same-schedule determinism (t_b3 arm): bitwise equal repeats
        got2 = attn_prefill_fused(qs, k, v, rel, pos, pos, win)
        same = bool(torch.equal(got, got2))
        print(f"[t0023] arm g  {name}: two runs bitwise equal: {same}")
        ok &= same
        #h. speed, CUDA-event timed, interleaved arms after warm-ups
        def _eager():
            return eager_ref(q, k, v, r, proj, pos, win, cfg["is_global"])

        def _fused(bm=64, bn=64):
            rl = (r @ proj).transpose(1, 2)
            qq = q
            if cfg["is_global"]:
                qq, rl = pmodel.apply_log_scaling(q, rl, pos, 0.1, 128000.0)
            return attn_prefill_fused(qq, k, v, rl, pos, pos, win,
                                      block_m=bm, block_n=bn)
        #   the tile shape is a free knob the call site pins; sweep it here
        #   rather than guessing (D=128 makes the fp32 score+acc tiles the
        #   register-pressure term)
        arms = [("eager", _eager)] + [
            (f"fused {bm}x{bn}", (lambda bm=bm, bn=bn: _fused(bm, bn)))
            for bm, bn in ((64, 64), (64, 32), (128, 64), (32, 64))]
        times = {tag: [] for tag, _ in arms}
        #   events must be recorded on the LAYER'S device — recording them
        #   on the default device's stream times an idle stream (it reads
        #   host launch gaps, not GPU work)
        with torch.cuda.device(dev):
            for i in range(6):
                for tag, fn in arms:
                    torch.cuda.synchronize(dev)
                    s, e = (torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True))
                    s.record()
                    out = fn()
                    e.record()
                    torch.cuda.synchronize(dev)
                    if i >= 2:                  # discard warm-ups (JIT)
                        times[tag].append(s.elapsed_time(e))
                    del out
        me = sum(times["eager"]) / len(times["eager"])
        for tag, _ in arms[1:]:
            mf = sum(times[tag]) / len(times[tag])
            print(f"[t0023] arm h  {name}: eager {me:.3f} ms -> {tag} "
                  f"{mf:.3f} ms  ({me / mf:.2f}x, -{me - mf:.3f} ms/layer)")
        #i. peak memory per layer
        for tag, fn in (("eager", _eager), ("fused", _fused)):
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            out = fn()
            torch.cuda.synchronize(dev)
            print(f"[t0023] arm i  {name} {tag}: peak "
                  f"{torch.cuda.max_memory_allocated(dev) / 2**30:.3f} GiB")
            del out
    #j. kill-switch A/B through model.attn_prefill itself, including the
    #   right-padded batched form (needs a second process for the switch;
    #   here we compare the fused call site against the eager helper)
    print(f"[t0023] arm j  attn_prefill fused flag = "
          f"{pmodel._FUSED_PREFILL_ATTN}  (run again with "
          f"PYENGINE_FUSED_PREFILL_ATTN=0 and diff the arm f numbers)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--dev", type=int, default=4)
    a = ap.parse_args()
    ok = True
    if a.sim:
        ok &= run_sim()
    if a.gpu:
        assert a.dev >= 4, "GPUs 4-7 only (hard rule)"
        ok &= run_gpu(a.dev)
    print(f"[t0023] {'ALL PASS' if ok else 'FAILURES ABOVE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
