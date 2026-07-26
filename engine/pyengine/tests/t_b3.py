"""B3 tests: KV + decode + scheduler + server (PROGRESS.md B3.x).

Run as:  python -m engine.pyengine.tests.t_b3 <kv|decode|batch|server|soak>
Env: CUDA_VISIBLE_DEVICES=4,5,6,7 (GPU budget rule) — DEV cuda:0 is GPU 4.

B3.1 `kv` — the three recurrent stores (kv.py: paged global KV page 16,
ring-512 SWA KV, window-4 sconv rings) + the decode-form model functions
(model.sconv_decode/attn_decode/layer_decode), gated by the item's oracle:
decode of token N+1 equals recompute-from-scratch on a 600-token prompt,
which crosses the 512 SWA window (ring fully wrapped, prefill T > capacity).

Gate design — what "equals" can honestly mean here. Bit-exactness between
a T=1 decode and a T=602 recompute is not attainable in principle:
cuBLAS/reduction kernels choose shape-dependent accumulation orders
(GEMV-class vs GEMM rows) — same math, different bf16 rounding, the B2.9
grouped_mm-vs-eager situation one level down. Two measured consequences
shape the gates: (i) the drift COMPOUNDS when streams propagate their own
outputs (600-row and 602-row streams diverge to 7e-04 by layer 1 with NO
cache involved), so every layer is TEACHER-FORCED with the recompute
stream's own layer input and compared over exactly one layer — end-to-end
composition is B3.2's job against the external transformers oracle;
(ii) MoE routing WEIGHTS pass through an all-bf16 chain (sigmoid ->
logsigmoid -> softmax on a ~2^-8-granular grid, B2.8 semantics), so a
1-ulp logit difference legitimately moves weights ~1e-2 relative and the
layer output ~5e-3 (measured; expert CHOICE is unaffected) — that is
expert-side numerics, not KV state, so the attention half is gated 10x
tighter than the MoE composition. The arms:
  (a) synthetic struct pins, bit-exact by construction: ring wraparound
      order/positions, oversized-append drop arm, paged gather through a
      SHUFFLED page table (real indirection), partial pages, sconv-ring
      windows incl. the implicit-zero short-prefill arm;
  (b) real-data pins, bit-exact REQUIRED: an independent same-shape replay
      of the whole layer on the T=600 prefill input must reproduce the
      prefill output bitwise (certifies the replay) and then pin the cache
      content (post-q/k-norm K, post-sconv V) and ALL FOUR sconv-ring
      tails (raw k/v, attn-out, mlp-out) bitwise — append/gather mechanics
      (paging, ring wrap, positions, seeding sites) add exactly nothing;
  (c) integration positional pins, exact: after 600 prefill + 2 decode
      tokens the SWA ring holds exactly positions 90..601 (the window-512
      allowed set for query 601), the paged cache exactly 0..601 over
      ceil(602/16)=38 non-contiguously allocated pages;
  (d) the item's oracle, per layer and per decode step (tokens 601/602 vs
      the T=602 from-scratch rows 600/601, fp32 L2 rel err):
      d1 attention-half residual (KV-fed part: paged/ring cache + k/v/attn
         sconv rings + rel-bias at decode distances + GQA gather) < 1e-3,
         measured ~1e-5;
      d2 MoE expert CHOICE identical (top-6 set per token);
      d3 full layer output < 1e-3 on dense layers; < 1e-2 (the B2-family
         expert-numerics budget, B2.9/B2.11) on MoE layers — dominated by
         (ii), measured ~5e-3 worst, with any KV-side bug still caught by
         d1/b/c at their tighter/bitwise levels;
      plus the same-shape prefill prefix drift < 1e-3 (diagnostic floor).
Layers exercised: 0, 1 (dense MLP + SWA), 2 (bf16-MoE + SWA), 3
(NVFP4-MoE + SWA), 5 (NVFP4-MoE + global) — every layer archetype; weights
stream one layer at a time (B2.11 pattern); the single propagating stream
is the T=602 recompute (oracle) stream."""
import sys
import time

import torch

from engine.pyengine import config as pcfg
from engine.pyengine import kv as pkv
from engine.pyengine import loader
from engine.pyengine import model as pmodel
from engine.pyengine.tests.t_b2 import (DEV, MODEL_DIR, _disk, _rel_err,
                                        load_layer_weights)

LAYERS = (0, 1, 2, 3, 5)  # dense/SWA x2, bf16-MoE/SWA, nvfp4-MoE/SWA, nvfp4-MoE/global
T_PROMPT = 600            # the item's 600-token prompt (crosses window 512)
N_DECODE = 2              # decode 601 AND 602: decode-form append + gather twice
GATE_ATTN = 1e-3          # arm d1 (and d3 on dense layers; prefix floor)
GATE_MOE = 1e-2           # arm d3 on MoE layers — B2-family expert budget


def _pin_structs():
    """Arm (a): synthetic bit-exact pins of the three stores' mechanics."""
    g = torch.Generator().manual_seed(1)
    bf = torch.bfloat16

    #1. SconvRing: seeded tail + steps == sliding windows of the raw stream
    x = torch.randn(10, 5, generator=g).to(DEV).to(bf)
    ring = pkv.SconvRing(5, 4, DEV)
    ring.seed(x[:7])
    for t in range(7, 10):
        if not torch.equal(ring.step(x[t:t + 1]), x[t - 3:t + 1]):
            raise SystemExit(f"[t_b3 kv] sconv-ring window wrong at t={t}")
    #2. short prefill (T < kernel-1) keeps the implicit conv zeros
    ring2 = pkv.SconvRing(5, 4, DEV)
    ring2.seed(x[:2])
    want = torch.cat([torch.zeros(1, 5, device=DEV, dtype=bf), x[:3]], 0)
    if not torch.equal(ring2.step(x[2:3]), want):
        raise SystemExit("[t_b3 kv] sconv-ring short-prefill zeros wrong")

    #3. RingKV capacity 8, 13 tokens: batched appends, wraparound order,
    #   positions; the oversized-append drop arm both mid-stream and fresh
    ks = torch.randn(13, 2, 3, generator=g).to(DEV).to(bf)
    vs = torch.randn(13, 2, 3, generator=g).to(DEV).to(bf)
    rk = pkv.RingKV(8, 2, 3, DEV)
    rk.append(ks[:3], vs[:3])
    k, v, pos = rk.gather()
    if not (torch.equal(k, ks[:3]) and torch.equal(v, vs[:3])
            and pos.tolist() == [0, 1, 2]):
        raise SystemExit("[t_b3 kv] ring pre-wrap gather wrong")
    rk.append(ks[3:], vs[3:])  # T=10 > cap=8 mid-stream
    rk2 = pkv.RingKV(8, 2, 3, DEV)
    rk2.append(ks, vs)         # T=13 > cap=8 in one shot
    for r in (rk, rk2):
        k, v, pos = r.gather()
        if not (torch.equal(k, ks[5:]) and torch.equal(v, vs[5:])
                and pos.tolist() == list(range(5, 13)) and r.n == 13):
            raise SystemExit("[t_b3 kv] ring post-wrap gather wrong")

    #4. PagedKV page_size 4, SHUFFLED free order: two appends spanning a
    #   partial page; gather through the table must reproduce logical order
    order = torch.randperm(8, generator=g).tolist()
    pk = pkv.PagedKV(8, 4, 2, 3, DEV, free_order=order)
    pk.append(ks[:5], vs[:5])
    pk.append(ks[5:], vs[5:])
    k, v, pos = pk.gather()
    if not (torch.equal(k, ks) and torch.equal(v, vs)
            and pos.tolist() == list(range(13))):
        raise SystemExit("[t_b3 kv] paged gather wrong")
    if pk.table != order[:4]:  # ceil(13/4)=4 pages, in free-list order
        raise SystemExit(f"[t_b3 kv] page table {pk.table} != {order[:4]}")


def _replay_attn_half(x, w, mask, pos, mc, L):
    """The layer's attention half through the same public functions in
    layer_prefill's exact order, exposing the intermediates layer_prefill
    keeps internal: returns (k_raw, v_raw [1,T,kvd], post-norm K, post-
    sconv V [T,n_kv,D], attn out [1,T,hidden], x1 = x + attn_sconv(attn),
    mlpin = mlp_norm(x1))."""
    F = torch.nn.functional
    n_kv = w["wk"].shape[0] // mc.head_dim
    T = x.shape[1]
    #1. the K/V chain exactly as attn_prefill computes it (B2.6/B2.7)
    hn = pmodel.rmsnorm(x, w["attn_norm"], mc.rms_eps)
    k_raw = F.linear(hn, w["wk"])
    v_raw = F.linear(hn, w["wv"])
    kc = pmodel.sconv_prefill(k_raw, w["ksc"])
    vc = pmodel.sconv_prefill(v_raw, w["vsc"])
    kn = pmodel.rmsnorm(kc.view(1, T, n_kv, mc.head_dim), w["kn"],
                        mc.rms_eps)
    #2. the attention half exactly as layer_prefill glues it
    att = pmodel.attn_prefill(
        hn, w["wq"], w["wk"], w["wv"], w["wr"], w["wo"], w["ksc"], w["vsc"],
        w["qn"], w["kn"], w["proj"], mask, pos, pos, mc.head_dim,
        L in mc.global_layers, alpha=mc.log_alpha, n_floor=mc.log_floor,
        eps=mc.rms_eps)
    x1 = x + pmodel.sconv_prefill(att, w["attn_sconv"])
    mlpin = pmodel.rmsnorm(x1, w["mlp_norm"], mc.rms_eps)
    return k_raw, v_raw, kn[0], vc.view(T, n_kv, mc.head_dim), att, x1, mlpin


def _mlp(h, w, mc, L):
    """The layer's MLP block (dense or MoE), same dispatch as
    layer_prefill/layer_decode."""
    if L < mc.dense_idx:
        return pmodel.dense_mlp(h, w["mlp_gate"], w["mlp_up"],
                                w["mlp_down"], w["mlp_gs"])
    return pmodel.moe(h, w["gate_w"], w["gate_b"], w["gate_gs"], w["gu"],
                      w["w2"], w["shg"], w["shu"], w["shd"], mc.topk,
                      mc.n_shared, mc.route_scale)


def _topk_set(h, w, mc):
    """Top-6 expert CHOICE (set) for each token of h, via moe_gate (B2.8)."""
    _, _, ti, _ = pmodel.moe_gate(h, w["gate_w"], w["gate_b"], w["gate_gs"],
                                  mc.topk, mc.n_shared, mc.route_scale)
    return [set(row.tolist()) for row in ti]


def t_kv():
    """B3.1: see module docstring."""
    t0 = time.time()
    torch.manual_seed(0)
    _pin_structs()
    mc = pcfg.load_verified(MODEL_DIR)
    idx = loader.build_shard_index(MODEL_DIR)
    hdr = loader.read_headers(idx)
    dm = loader.build_dtype_map(idx, hdr)
    total = T_PROMPT + N_DECODE
    assert T_PROMPT > mc.window, "item requires crossing the 512 window"

    #1. deterministic token stream: 600 prompt + 2 decode ids, seeded CPU
    #   generator (numerics need real embedding rows, not real text)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, mc.vocab_unpadded, (total,), generator=g).to(DEV)

    #2. the oracle stream starts at embed+embed_norm (B2.2)
    w_emb = _disk(idx, "model.llm.embed.weight").to(DEV)
    w_en = _disk(idx, "model.llm.embed_norm.weight").to(DEV)
    _, h_ref = pmodel.embed(ids[None, :], w_emb, w_en, eps=mc.rms_eps)
    del w_emb, w_en
    torch.cuda.empty_cache()

    #3. masks + positions per stream ((full, window) per B2.11 convention)
    pos_pre = torch.arange(T_PROMPT, device=DEV)
    pos_ref = torch.arange(total, device=DEV)
    dt = h_ref.dtype
    mask_pre = (pmodel.additive_causal_mask(pos_pre, pos_pre, dt),
                pmodel.additive_causal_mask(pos_pre, pos_pre, dt,
                                            window=mc.window))
    mask_ref = (pmodel.additive_causal_mask(pos_ref, pos_ref, dt),
                pmodel.additive_causal_mask(pos_ref, pos_ref, dt,
                                            window=mc.window))

    #4. per-layer recurrent state; the global layer's page pool free order
    #   is a seeded shuffle so page indirection is physically real
    n_pg = (total + pkv.PAGE_SIZE - 1) // pkv.PAGE_SIZE
    states, page_orders = {}, {}
    for L in LAYERS:
        if L in mc.global_layers:
            page_orders[L] = torch.randperm(64, generator=g).tolist()
            states[L] = pkv.LayerKV(mc, L, DEV, num_pages=64,
                                    free_order=page_orders[L])
        else:
            states[L] = pkv.LayerKV(mc, L, DEV)

    #5. streamed layers, teacher-forced on the oracle stream's layer input
    #   x: prefill the first 600 rows into the state, decode rows 600/601
    #   through the cache, recompute all 602 rows from scratch, compare
    att_err, dec_err, pre_err = {}, {}, {}
    with torch.no_grad():
        for L in LAYERS:
            w = load_layer_weights(idx, hdr, dm, mc, L)
            is_moe = L >= mc.dense_idx
            mi = 0 if L in mc.global_layers else 1
            st = states[L]
            x = h_ref  # this layer's input, [1, 602, hidden]
            #5a. cached path: prefill(600) populates the state
            y600 = pmodel.layer_prefill(x[:, :T_PROMPT], w, mask_pre[mi],
                                        pos_pre, mc, L, state=st)
            #5b. arm (b): independent same-shape replay — must reproduce
            #    the prefill output bitwise, then pin cache content + all
            #    four sconv-ring tails bitwise
            kr6, vr6, kn6, v6, att6, x16, mlpin6 = _replay_attn_half(
                x[:, :T_PROMPT], w, mask_pre[mi], pos_pre, mc, L)
            mlpout6 = _mlp(mlpin6, w, mc, L)
            y6r = x16 + pmodel.sconv_prefill(mlpout6, w["mlp_sconv"])
            if not torch.equal(y6r, y600):
                raise SystemExit(f"[t_b3 kv] layer {L} replay != "
                                 f"layer_prefill (replay drift)")
            ck, cv, cpos = st.cache.gather()
            lo = 0 if L in mc.global_layers else T_PROMPT - mc.window
            if cpos.tolist() != list(range(lo, T_PROMPT)):
                raise SystemExit(f"[t_b3 kv] layer {L} post-prefill cache "
                                 f"positions {cpos[0]}..{cpos[-1]} wrong")
            if not (torch.equal(ck, kn6[lo:]) and torch.equal(cv, v6[lo:])):
                raise SystemExit(f"[t_b3 kv] layer {L} cached K/V not "
                                 f"bit-equal to same-shape replay")
            tails = ((st.k_ring, kr6), (st.v_ring, vr6),
                     (st.attn_ring, att6), (st.mlp_ring, mlpout6))
            for nm, (ring, src) in zip(("k", "v", "attn", "mlp"), tails):
                if not torch.equal(ring.tail, src[0, -(mc.sconv_k - 1):]):
                    raise SystemExit(f"[t_b3 kv] layer {L} {nm}_ring tail "
                                     f"not bit-equal to replay")
            del kr6, vr6, kn6, v6, att6, mlpin6, mlpout6, y6r, ck, cv
            #5c. cached path: two decode steps (the second attends to the
            #    first's cache entries), tracing the attention half
            y_dec, tr = [], []
            for s in range(N_DECODE):
                tr.append({})
                y_dec.append(pmodel.layer_decode(
                    x[:, T_PROMPT + s], w, st, T_PROMPT + s, mc, L,
                    trace=tr[s]))
            #5d. oracle: the same 602 rows from scratch (the propagating
            #    stream), plus its attention half for arm d1/d2
            h_ref = pmodel.layer_prefill(x, w, mask_ref[mi], pos_ref, mc, L)
            _, _, _, _, _, x1o, mlpino = _replay_attn_half(
                x, w, mask_ref[mi], pos_ref, mc, L)
            #5e. gates d1/d2/d3 per decode step + prefix floor
            a_errs = [_rel_err(tr[s]["x1"][0], x1o[0, T_PROMPT + s])
                      for s in range(N_DECODE)]
            d_errs = [_rel_err(y_dec[s][0], h_ref[0, T_PROMPT + s])
                      for s in range(N_DECODE)]
            att_err[L], dec_err[L] = max(a_errs), max(d_errs)
            pre_err[L] = _rel_err(y600, h_ref[:, :T_PROMPT])
            if is_moe:
                want_sets = _topk_set(mlpino[0, T_PROMPT:], w, mc)
                for s in range(N_DECODE):
                    got = _topk_set(tr[s]["mlpin"], w, mc)[0]
                    if got != want_sets[s]:
                        raise SystemExit(
                            f"[t_b3 kv] layer {L} step {s} expert choice "
                            f"{sorted(got)} != oracle {sorted(want_sets[s])}")
            for h in (h_ref, *y_dec):
                if not torch.isfinite(h.float()).all():
                    raise SystemExit(f"[t_b3 kv] non-finite after layer {L}")
            if att_err[L] >= GATE_ATTN:
                raise SystemExit(f"[t_b3 kv] layer {L} attention half "
                                 f"decode vs recompute {att_err[L]:.3e} >= "
                                 f"{GATE_ATTN} (steps {a_errs})")
            d_gate = GATE_MOE if is_moe else GATE_ATTN
            if dec_err[L] >= d_gate:
                raise SystemExit(f"[t_b3 kv] layer {L} decode vs recompute "
                                 f"{dec_err[L]:.3e} >= {d_gate} "
                                 f"(steps {d_errs})")
            if pre_err[L] >= GATE_ATTN:
                raise SystemExit(f"[t_b3 kv] layer {L} prefill prefix vs "
                                 f"recompute {pre_err[L]:.3e} >= {GATE_ATTN}")
            del w, y600, y_dec, tr, x16, x1o, mlpino
            torch.cuda.empty_cache()
            print(f"[t_b3 kv] layer {L} done: attn-half {att_err[L]:.1e} "
                  f"out {dec_err[L]:.1e} prefix {pre_err[L]:.1e}, "
                  f"{time.time() - t0:.0f} s", file=sys.stderr, flush=True)

    #6. arm (c): final positional/structural pins over the real run
    for L in LAYERS:
        st = states[L]
        if st.cache.n != total:
            raise SystemExit(f"[t_b3 kv] layer {L} cache count {st.cache.n}")
        if L in mc.global_layers:
            if len(st.cache.table) != n_pg:
                raise SystemExit(f"[t_b3 kv] layer {L}: "
                                 f"{len(st.cache.table)} pages != {n_pg}")
            if st.cache.table != page_orders[L][:n_pg]:
                raise SystemExit(f"[t_b3 kv] layer {L} page table not in "
                                 f"free-list order")
            if st.cache.table == sorted(st.cache.table):
                raise SystemExit(f"[t_b3 kv] layer {L} pages accidentally "
                                 f"contiguous — indirection not exercised")
            _, _, pos = st.cache.gather()
            if pos.tolist() != list(range(total)):
                raise SystemExit(f"[t_b3 kv] layer {L} paged positions wrong")
        else:
            k, _, pos = st.cache.gather()
            if k.shape != (mc.window, mc.s_kv_heads, mc.head_dim):
                raise SystemExit(f"[t_b3 kv] layer {L} ring shape {k.shape}")
            if pos.tolist() != list(range(total - mc.window, total)):
                raise SystemExit(f"[t_b3 kv] layer {L} ring positions wrong")

    #7. summary = the green evidence
    ae = "/".join(f"{att_err[L]:.1e}" for L in LAYERS)
    de = "/".join(f"{dec_err[L]:.1e}" for L in LAYERS)
    pe = max(pre_err.values())
    print(f"kv ok: layers {list(LAYERS)} (dense/bf16-MoE/nvfp4-MoE x "
          f"SWA/global), {T_PROMPT}-token prompt + {N_DECODE} decode steps "
          f"(crosses window {mc.window}), teacher-forced per layer (see "
          f"docstring) — decode vs recompute-from-scratch: attention half "
          f"(KV-fed) {ae} < {GATE_ATTN}, full layer out {de} < "
          f"{GATE_ATTN}(dense)/{GATE_MOE}(moe, bf16 routing-weight "
          f"granularity, expert CHOICE pinned identical 3/3 moe layers x "
          f"{N_DECODE} steps); prefill-written cache K/V + all 4 sconv-ring "
          f"tails BIT-equal same-shape replay 5/5 layers (replay itself "
          f"bit-equal layer_prefill); SWA ring holds exactly positions "
          f"{total - mc.window}..{total - 1}, paged global 0..{total - 1} "
          f"in {n_pg} shuffled pages (page {pkv.PAGE_SIZE}); prefix drift "
          f"<= {pe:.1e}; struct pins (ring wrap order+positions, "
          f"oversized-append drop, shuffled-table paged gather + partial "
          f"page, sconv-ring windows + short-prefill zeros) all bit-exact; "
          f"wall {time.time() - t0:.0f} s")


def _todo(item):
    def f():
        raise SystemExit(f"[t_b3] not implemented — that is item {item}'s job")
    return f


def main():
    #1. dispatch on subcommand; unimplemented ones fail loud with their item
    cmds = {"kv": t_kv,
            "decode": _todo("B3.2"), "batch": _todo("B3.3"),
            "server": _todo("B3.4"), "soak": _todo("B3.6")}
    usage = f"usage: python -m engine.pyengine.tests.t_b3 {{{'|'.join(cmds)}}}"
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        raise SystemExit(usage)
    cmds[sys.argv[1]]()


if __name__ == "__main__":
    main()
