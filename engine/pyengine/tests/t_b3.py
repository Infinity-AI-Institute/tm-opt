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
composition is B3.2's job (self-consistency vs full-prefill-of-prefix +
bitwise determinism; the binding external referee is B3.5 vs vLLM goldens);
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
import threading
import time

import torch

from engine.pyengine import config as pcfg
from engine.pyengine import kv as pkv
from engine.pyengine import loader
from engine.pyengine import model as pmodel
from engine.pyengine import scheduler as psched
from engine.pyengine import server as pserver
from engine.pyengine.tests.t_b2 import (DEV, MODEL_DIR, PROMPTS5, _deint3,
                                        _disk, _load_refs, _rel_err,
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


N_NEW = 32     # B3.2: the item's "32 tokens"
NUM_PAGES = 8  # per-global-layer page budget: 8*16=128 >= max T+31 (=44)
GATE_TIE = 1e-2  # fp32 logit gap below which a greedy flip FLAGs, not fails


def _materialize(pex):
    """Full bf16 expert stack from a loader.PackedExperts — the SAME
    32-expert-chunk dequant + de-interleave as load_layer_weights' dense
    path (B2.9/B2.11). PackedExperts.__getitem__ is pinned bit-equal to
    this chunked form (t_decode step #4), so swapping the packed form for
    this tensor changes no bit anywhere downstream; it exists so the
    160-stream recompute arm dequants each NVFP4 layer ONCE instead of
    once per hit expert per stream forward."""
    #1. chunked dequant, exactly load_layer_weights' resident loop
    E, rows, half = pex.packed.shape
    out = torch.empty(E, rows, 2 * half, dtype=torch.bfloat16,
                      device=pex.packed.device)
    for e0 in range(0, E, 32):
        e1 = e0 + 32
        deq = loader.dequant_nvfp4(pex.packed[e0:e1], pex.scale[e0:e1],
                                   pex.scale2[e0:e1], pex.group_size)
        out[e0:e1] = _deint3(deq) if pex.deinterleave else deq
    del deq
    return out


def t_decode():
    """B3.2: greedy decode loop (batch 1) — SELF-CONSISTENCY gates, not
    transformers (the item as restructured: accumulated eager-vs-
    grouped_mm drift is non-probative at 9.2e-02, B2.11; the binding
    external referee is B3.5 vs vLLM goldens).

    Engine: all 66 layers RESIDENT at once — contiguous layer groups over
    the 4 visible GPUs balanced by checkpoint header bytes (device-to-
    device hidden-state hops are bitwise copies, so placement never
    touches numerics). bf16-dense routed experts (~1.8 TB) fit nowhere,
    so NVFP4 layers keep their experts checkpoint-PACKED behind
    loader.PackedExperts — model.moe_experts dequants each HIT expert on
    demand, and that path is PINNED bit-equal to the proven chunked
    dequant (B2.9/B2.11) on layer 3, experts {0,7,31}, both w13 and w2.

    Arm (b), engine determinism (NO tolerance): model.generate_greedy
    runs TWICE per prompt from fresh kv.LayerKV state; token_ids must be
    IDENTICAL; bitwise equality of the captured fp32 logits rows and
    margins is reported alongside (expected — same kernels, same shapes).

    Arm (a), self-consistency vs recompute: for every step s (0..31) the
    free run is compared against a FULL PREFILL of its own prefix
    prompt + tokens[:s] (fresh state, from scratch):
      - greedy top-1: argmax(prefill-of-prefix logits) == the free run's
        token at EVERY step; a flip FAILS unless the recompute's fp32 gap
        between its top-1 and our token is < 1e-2 — then it is a reported
        near-tie FLAG (item wording);
      - B3.1's decomposed per-layer gates, TEACHER-FORCED in t_kv's
        EXACT structure: each layer_decode is fed the recompute stream's
        own layer input, with its cache/rings freshly populated by
        prefilling the SAME stream's first T+s-1 rows — one layer of
        shape effects per comparison. (First attempt used the state of
        the NEIGHBORING prompt+tokens[:s-1] stream instead: its rows
        carry L layers of compounded row-count drift — the B3.1
        dead-end — and the dense MLP amplified a sub-gate x1 drift to
        7.7e-03 at layer 1, honestly over the 1e-3 gate. Same-stream
        state is what B3.1's gates mean.) Gates: attention-half residual
        x1 < 1e-3, dense layer out < 1e-3, MoE layer out < 1e-2 with
        expert top-6 CHOICE identical — at every decode step of every
        prompt (66 layers x 31 steps x 5 prompts = 10230 comparisons).
        The item's near-tie provision applies to the choice arm too: a
        SINGLE-pair expert swap is FLAGGED (reported), not failed, iff
        the recompute's own choice scores (sigmoid+bias, B2.8) tie
        within 1e-2 — measured live: one swap per ~10^4 comparisons at
        score gap ~1e-5 (rank-6/7 tie moved by ~1e-5 teacher-forced
        attention noise; the same bf16-granularity fact as B3.1's
        routing-weight budget). The MoE-out < 1e-2 gate stays binding at
        flagged steps, so a swap that actually moves the layer output
        beyond the item's budget still fails.
        The truncated-vs-full prefill prefix drift (NO decode or cache
        in that comparison — pure row-count accumulation noise, and the
        dense MLP amplifies it ~10x) is reported with a coarse 1e-2
        sanity ceiling, NOT B3.1's 600-token-calibrated 1e-3 floor:
        measured 1.7e-03 block-wide at T~30 on dense layer 1, where a
        real state/mask bug would sit orders of magnitude higher.
    The 160 prefix streams advance LAYER-major so each NVFP4 layer's
    experts materialize once (_materialize — bit-equal by the pin above)
    instead of ~10^6 single-expert dequants; stream s=0 (the bare prompt)
    must reproduce the free run's step-0 logits BIT-exactly, certifying
    the materialized-vs-packed + trace-hook recompute path end to end.

    No eos early-exit (canonical workloads run ignore_eos, P1/D6); eos
    presence in the 32 tokens is reported. Prompt-0 ids are pinned
    against the frozen B2.1 input_ids capture."""
    from transformers import AutoTokenizer
    t0 = time.time()
    torch.manual_seed(0)
    tens, _, _ = _load_refs()
    mc = pcfg.load_verified(MODEL_DIR)
    idx = loader.build_shard_index(MODEL_DIR)
    hdr = loader.read_headers(idx)
    dm = loader.build_dtype_map(idx, hdr)
    ndev = torch.cuda.device_count()
    assert ndev == 4, f"expected GPUs 4-7 visible, got {ndev}"

    #1. the B2.11 prompts; prompt-0 ids pinned against the frozen capture
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    ids = [tok(p, return_tensors="pt").input_ids[0].to(DEV) for p in PROMPTS5]
    lens = [t.shape[0] for t in ids]
    if not torch.equal(ids[0][None], tens["input_ids"].to(DEV)):
        raise SystemExit("[t_b3 decode] prompt-0 ids != frozen input_ids")

    #2. layer -> GPU assignment: contiguous groups, balanced by header bytes
    lbytes = []
    for L in range(mc.num_layers):
        pre = f"model.llm.layers.{L}."
        lbytes.append(sum(loader.nbytes(d, s) for n, (d, s) in hdr.items()
                          if n.startswith(pre)))
    total_b = sum(lbytes)
    devs, cum, d = [], 0, 0
    for L in range(mc.num_layers):
        if d < ndev - 1 and cum + lbytes[L] / 2 > total_b * (d + 1) / ndev:
            d += 1
        devs.append(f"cuda:{d}")
        cum += lbytes[L]
    splits = [devs.count(f"cuda:{r}") for r in range(ndev)]
    print(f"[t_b3 decode] layer split {splits} over {ndev} GPUs "
          f"({total_b / (1 << 30):.0f} GiB layers)", file=sys.stderr,
          flush=True)

    #3. resident engine load: 66 layers (NVFP4 experts packed) + embed on
    #   the first device, logits head on the last layer's device
    layers = []
    for L in range(mc.num_layers):
        layers.append(load_layer_weights(idx, hdr, dm, mc, L, dev=devs[L],
                                         pack_moe=True))
        if L % 8 == 7 or L == mc.num_layers - 1:
            print(f"[t_b3 decode] loaded layer {L + 1}/{mc.num_layers}, "
                  f"{time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    w_emb = _disk(idx, "model.llm.embed.weight").to(devs[0])
    w_en = _disk(idx, "model.llm.embed_norm.weight").to(devs[0])
    w_fn = _disk(idx, "model.llm.norm.weight").to(devs[-1])
    w_un = _disk(idx, "model.llm.unembed.weight").to(devs[-1])
    assert isinstance(layers[2]["gu"], torch.Tensor)   # bf16 layer: dense
    assert isinstance(layers[3]["gu"], loader.PackedExperts)

    #4. packed-path pin: on-demand PackedExperts[e] must be bit-equal the
    #   proven chunked dequant + de-interleave (B2.9/B2.11 path) on the
    #   same bytes — layer 3, experts {0,7,31}, both w13 (deint) and w2
    with torch.no_grad():
        for name, pex in (("w13", layers[3]["gu"]), ("w2", layers[3]["w2"])):
            chunk = loader.dequant_nvfp4(pex.packed[:32], pex.scale[:32],
                                         pex.scale2[:32], dm.group_size)
            want = _deint3(chunk) if pex.deinterleave else chunk
            for e in (0, 7, 31):
                if not torch.equal(pex[e], want[e]):
                    raise SystemExit(f"[t_b3 decode] PackedExperts {name} "
                                     f"expert {e} != chunked dequant")
            del chunk, want

    #5. arm (b): the item's loop TWICE per prompt from fresh state —
    #   token_ids must be IDENTICAL (no tolerance); fp32 logits rows are
    #   captured for arm (a); cache token counts pinned per layer per run
    ours, ourm, cap, n_bits = [], [], [], 0
    with torch.no_grad():
        for p in range(5):
            runs = []
            for r in range(2):
                states = [pkv.LayerKV(mc, L, devs[L], num_pages=NUM_PAGES)
                          for L in range(mc.num_layers)]
                c = []
                toks, margs = pmodel.generate_greedy(
                    ids[p], N_NEW, layers, states, w_emb, w_en, w_fn,
                    w_un, mc, capture=c)
                assert len(toks) == N_NEW and all(
                    0 <= t < mc.vocab_unpadded for t in toks)
                fed = lens[p] + N_NEW - 1   # last token is never fed back
                for L in range(mc.num_layers):
                    if states[L].cache.n != fed:
                        raise SystemExit(f"[t_b3 decode] p{p} run{r} layer "
                                         f"{L} cache.n {states[L].cache.n} "
                                         f"!= {fed}")
                del states
                runs.append((toks, margs, c))
            (toks, margs, c), (tok2, mar2, c2) = runs
            if toks != tok2:
                raise SystemExit(f"[t_b3 decode] p{p} NON-DETERMINISTIC: "
                                 f"run1 {toks} != run2 {tok2}")
            n_bits += (margs == mar2
                       and all(torch.equal(a, b) for a, b in zip(c, c2)))
            ours.append(toks)
            ourm.append(margs)
            cap.append(c)
            print(f"[t_b3 decode] p{p} x2 ({time.time() - t0:.0f} s): "
                  f"{tok.decode(toks)!r}", file=sys.stderr, flush=True)
    t_gen = time.time() - t0

    #6. arm (a): 160 recompute streams (5 prompts x steps 0..31), stream
    #   (p, s) = full prefill of prompt + tokens[:s]; LAYER-major advance
    keys = [(p, s) for p in range(5) for s in range(N_NEW)]
    streams, mcache, eflags = {}, {}, []
    worst = {"attn": 0.0, "dense": 0.0, "moe": 0.0, "pfx": 0.0}
    n_choice = 0

    def _pm(dev, n):
        #6a. per-(device, length) masks + positions, built once
        if (dev, n) not in mcache:
            pos = torch.arange(n, device=dev)
            mcache[(dev, n)] = (
                pmodel.additive_causal_mask(pos, pos, torch.bfloat16),
                pmodel.additive_causal_mask(pos, pos, torch.bfloat16,
                                            window=mc.window), pos)
        return mcache[(dev, n)]

    with torch.no_grad():
        for p, s in keys:
            seq = torch.cat([ids[p], torch.tensor(ours[p][:s],
                                                  dtype=torch.long,
                                                  device=DEV)])
            _, streams[(p, s)] = pmodel.embed(seq[None], w_emb, w_en,
                                              eps=mc.rms_eps)
        for L in range(mc.num_layers):
            is_moe = L >= mc.dense_idx
            wm = layers[L]
            if is_moe and isinstance(wm["gu"], loader.PackedExperts):
                wm = dict(wm)
                wm["gu"] = _materialize(layers[L]["gu"])
                wm["w2"] = _materialize(layers[L]["w2"])
            mi = 0 if L in mc.global_layers else 1
            ys, trs = {}, {}
            #6b. oracle: prefill every stream through layer L with the
            #    x1/mlpin trace at the last row (no state here — the
            #    teacher-forced arm rebuilds state from each stream's
            #    OWN rows below)
            for p, s in keys:
                x = streams[(p, s)].to(devs[L])
                streams[(p, s)] = x   # the layer input, kept for 6c
                m = _pm(devs[L], x.shape[1])
                tr = {}
                ys[(p, s)] = pmodel.layer_prefill(x, wm, m[mi], m[2], mc,
                                                  L, trace=tr)
                trs[(p, s)] = (tr["x1"][:, -1], tr["mlpin"][:, -1])
            #6c. teacher-forced decode of every step s>=1, t_kv's EXACT
            #    structure: fresh state from prefilling THIS stream's own
            #    first T+s-1 rows (one layer of shape effects — a cross-
            #    stream cache compounds the B3.1 dead-end drift), then
            #    one decode of the last row; gates per B3.1
            for p, s in keys:
                if s == 0:
                    continue
                x = streams[(p, s)]
                m = _pm(devs[L], x.shape[1] - 1)
                st = pkv.LayerKV(mc, L, devs[L], num_pages=NUM_PAGES)
                y_tf = pmodel.layer_prefill(x[:, :-1], wm, m[mi], m[2],
                                            mc, L, state=st)
                trd = {}
                yd = pmodel.layer_decode(x[:, -1], wm, st,
                                         lens[p] + s - 1, mc, L,
                                         trace=trd)
                a = _rel_err(trd["x1"][0], trs[(p, s)][0][0])
                o = _rel_err(yd[0], ys[(p, s)][0, -1])
                f = _rel_err(y_tf, ys[(p, s)][:, :-1])
                worst["attn"] = max(worst["attn"], a)
                worst["pfx"] = max(worst["pfx"], f)
                kind = "moe" if is_moe else "dense"
                worst[kind] = max(worst[kind], o)
                if a >= GATE_ATTN:
                    raise SystemExit(f"[t_b3 decode] p{p} s{s} L{L} attn "
                                     f"half {a:.3e} >= {GATE_ATTN}")
                if o >= (GATE_MOE if is_moe else GATE_ATTN):
                    raise SystemExit(f"[t_b3 decode] p{p} s{s} L{L} layer "
                                     f"out {o:.3e} >= "
                                     f"{GATE_MOE if is_moe else GATE_ATTN}")
                if f >= GATE_MOE:   # coarse sanity ceiling, see docstring
                    raise SystemExit(f"[t_b3 decode] p{p} s{s} L{L} prefix"
                                     f" drift {f:.3e} >= {GATE_MOE}")
                if is_moe:
                    got = _topk_set(trd["mlpin"], wm, mc)[0]
                    want = _topk_set(trs[(p, s)][1], wm, mc)[0]
                    if got != want:
                        #6c'. a flip is tolerable ONLY as a genuine
                        #     near-tie: one swapped pair whose recompute
                        #     choice scores (sigmoid+bias, B2.8) sit
                        #     within the item's 1e-2 gap; MoE-out gate
                        #     above stays binding either way
                        rl = pmodel.moe_gate(
                            trs[(p, s)][1], wm["gate_w"], wm["gate_b"],
                            wm["gate_gs"], mc.topk, mc.n_shared,
                            mc.route_scale)[0]
                        sc = (rl.sigmoid() + wm["gate_b"])[0].float()
                        gap = max(float(abs(sc[e1] - sc[e2]))
                                  for e1 in got - want
                                  for e2 in want - got)
                        if len(got ^ want) != 2 or gap >= GATE_TIE:
                            raise SystemExit(
                                f"[t_b3 decode] p{p} s{s} L{L} expert "
                                f"choice {sorted(got)} != recompute "
                                f"{sorted(want)}, score gap {gap:.3e} — "
                                f"not a near-tie pair")
                        eflags.append((p, s, L, gap))
                    n_choice += 1
                del st, y_tf
            #6d. advance streams, drop this layer's transients
            for k in keys:
                streams[k] = ys[k]
            del ys, trs
            if wm is not layers[L]:
                del wm
                torch.cuda.empty_cache()
            if L % 8 == 7 or L == mc.num_layers - 1:
                print(f"[t_b3 decode] recompute layer {L + 1}"
                      f"/{mc.num_layers}, worst attn {worst['attn']:.1e} "
                      f"dense {worst['dense']:.1e} moe {worst['moe']:.1e},"
                      f" {time.time() - t0:.0f} s", file=sys.stderr,
                      flush=True)

        #7. the per-step logits gate: argmax(prefill-of-prefix) == the
        #   free run's token (a flip FLAGs below GATE_TIE, else fails);
        #   stream s=0 pinned BIT-equal to the free run's prefill logits
        flags, lg_err, rec_min = [], 0.0, []
        for p in range(5):
            pmarg = []
            for s in range(N_NEW):
                row = pmodel.final_logits(streams[(p, s)][:, -1], w_fn,
                                          w_un, mc.logits_mult,
                                          mc.vocab_unpadded,
                                          eps=mc.rms_eps)[0].float().cpu()
                if s == 0 and not torch.equal(row, cap[p][0]):
                    raise SystemExit(f"[t_b3 decode] p{p}: s=0 recompute "
                                     f"logits not bit-equal the free "
                                     f"run's prefill logits")
                top2 = torch.topk(row, 2).values
                pmarg.append(float(top2[0] - top2[1]))
                lg_err = max(lg_err, _rel_err(row, cap[p][s]))
                want = int(row.argmax())
                if want != ours[p][s]:
                    gap = float(row[want] - row[ours[p][s]])
                    if gap >= GATE_TIE:
                        raise SystemExit(
                            f"[t_b3 decode] p{p} s{s}: recompute greedy "
                            f"{want} ({tok.decode([want])!r}) != ours "
                            f"{ours[p][s]} ({tok.decode([ours[p][s]])!r})"
                            f", fp32 gap {gap:.3e} >= {GATE_TIE} — not a "
                            f"near-tie")
                    flags.append((p, s, gap, ourm[p][s]))
            rec_min.append(min(pmarg))

    #8. summary = the green evidence
    flag_s = "; ".join(f"p{pp}s{ss} gap {g:.1e} our margin {m:.1e}"
                       for pp, ss, g, m in flags) or "none"
    eflag_s = "; ".join(f"p{pp}s{ss}L{ll} gap {g:.1e}"
                        for pp, ss, ll, g in eflags) or "none"
    texts = " | ".join(tok.decode(ours[p][:6]).strip()[:24]
                       for p in range(5))
    n_eos = sum(mc.eos in o for o in ours)
    n_moe = mc.num_layers - mc.dense_idx
    print(f"decode ok: batch-1 greedy, {N_NEW} tokens x 5 prompts (T "
          f"{min(lens)}-{max(lens)}), SELF-CONSISTENCY (restructured "
          f"item) — (a) free-run token == recompute argmax at all "
          f"{5 * N_NEW} steps, near-tie flips {len(flags)} ({flag_s}); "
          f"teacher-forced per-layer decode-vs-prefill at every step "
          f"({mc.num_layers}L x {N_NEW - 1} x 5, same-stream state per "
          f"t_kv): attn half {worst['attn']:.1e} < {GATE_ATTN}, dense "
          f"{worst['dense']:.1e} < {GATE_ATTN}, moe {worst['moe']:.1e} < "
          f"{GATE_MOE}, expert top-6 identical "
          f"{n_choice - len(eflags)}/{(N_NEW - 1) * 5 * n_moe} (near-tie "
          f"single-pair swaps FLAGGED {len(eflags)}: {eflag_s}), prefix "
          f"drift "
          f"{worst['pfx']:.1e} < {GATE_MOE} sanity ceiling (row-count "
          f"noise, no decode in that arm); "
          f"free-vs-recompute logits rel err <= {lg_err:.1e}, min top-2 "
          f"margin recompute {min(rec_min):.2f} ours "
          f"{min(min(m) for m in ourm):.2f}; (b) 2 fresh runs/prompt: "
          f"token_ids IDENTICAL 5/5, logits+margins bitwise {n_bits}/5; "
          f"s=0 recompute BIT-equal free-run prefill logits 5/5 "
          f"(packed==materialized end to end); PackedExperts[e] "
          f"bit-equal chunked dequant (L3 e{{0,7,31}} w13+w2); resident "
          f"split {splits}, cache.n == T+{N_NEW - 1} all layers x 2 runs"
          f" x 5 prompts; starts: {texts}; eos in {n_eos}/5 (no early "
          f"exit, ignore_eos semantics); wall gen {t_gen:.0f} s + "
          f"recompute {time.time() - t0 - t_gen:.0f} s")


PROMPTS8 = PROMPTS5 + (          # B3.3: 3 more in the same tiny style
    "The chemical symbol for iron is",
    "The capital of Japan is",
    "One plus one equals",
)
MAX_NEW8 = (32, 24, 17, 32, 9, 27, 12, 21)  # varied -> retirements interleave
ARRIVE8 = (0, 0, 0, 0, 1, 2, 3, 4)          # staggered -> joins mid-flight


def t_batch():
    """B3.3: continuous batching scheduler — 8 concurrent greedy requests
    must produce IDENTICAL tokens to batch-1 runs (batch invariance; NO
    tolerance anywhere in this test).

    Baseline: model.generate_greedy per request from fresh state — the
    exact loop B3.2 certified bitwise-deterministic (arm b, 5/5 prompts,
    live 5x across sessions). The SAME resident engine then runs the
    requests through scheduler.Scheduler under two genuinely different
    schedules:
      A: submission order 0..7, staggered arrivals (0,0,0,0,1,2,3,4),
         cap 8 — late requests join while earlier ones are mid-decode;
         all 8 run CONCURRENTLY (peak pinned == 8); varied max_new (9-32)
         makes retirements interleave with continued decoding.
      B: submission order REVERSED, all arrivals 0, cap 3 — exercises the
         FCFS waiting queue and admission-on-retirement; every sequence
         decodes next to different co-residents than in A.
    Gate: every request's token list == its batch-1 list, exact int
    equality, 8/8 in BOTH schedules. Batch invariance holds by
    construction — Engine transcribes generate_greedy's step forms op for
    op, each sequence owns its kv.LayerKV, co-scheduled sequences share
    weights and never a reduction; B3.1/B3.2's row-count-drift facts are
    WHY fused-batch kernels could not meet a bitwise gate (they are
    Stage-3 work, D4, refereed by this same test). What is genuinely
    under test is therefore the scheduler's bookkeeping: admission
    timing, state isolation, position accounting, retirement — the gate
    proves it all adds exactly nothing. Retirement pin: cache.n ==
    T + max_new - 1 on all 66 layers inside on_retire (the last token is
    never fed back, B3.2), KV freed right after. Prompt content is
    irrelevant to a self-consistency gate; tiny prompts keep the test
    inside the session budget (~174 generated tokens x 3 passes)."""
    from transformers import AutoTokenizer
    t0 = time.time()
    torch.manual_seed(0)
    mc = pcfg.load_verified(MODEL_DIR)
    idx = loader.build_shard_index(MODEL_DIR)
    hdr = loader.read_headers(idx)
    dm = loader.build_dtype_map(idx, hdr)
    ndev = torch.cuda.device_count()
    assert ndev == 4, f"expected GPUs 4-7 visible, got {ndev}"

    #1. tokenize the 8 prompts on the embed device
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    ids = [tok(p, return_tensors="pt").input_ids[0].to(DEV) for p in PROMPTS8]
    lens = [t.shape[0] for t in ids]

    #2. layer -> GPU assignment + resident load — mirrors t_decode #2-#3
    #   (kept inline verbatim: t_decode is BLOCKED-frozen pending the B3.2
    #   adjudication; dedup into a shared helper once that lands)
    lbytes = []
    for L in range(mc.num_layers):
        pre = f"model.llm.layers.{L}."
        lbytes.append(sum(loader.nbytes(d, s) for n, (d, s) in hdr.items()
                          if n.startswith(pre)))
    total_b = sum(lbytes)
    devs, cum, d = [], 0, 0
    for L in range(mc.num_layers):
        if d < ndev - 1 and cum + lbytes[L] / 2 > total_b * (d + 1) / ndev:
            d += 1
        devs.append(f"cuda:{d}")
        cum += lbytes[L]
    splits = [devs.count(f"cuda:{r}") for r in range(ndev)]
    print(f"[t_b3 batch] layer split {splits} over {ndev} GPUs "
          f"({total_b / (1 << 30):.0f} GiB layers)", file=sys.stderr,
          flush=True)
    layers = []
    for L in range(mc.num_layers):
        layers.append(load_layer_weights(idx, hdr, dm, mc, L, dev=devs[L],
                                         pack_moe=True))
        if L % 16 == 15 or L == mc.num_layers - 1:
            print(f"[t_b3 batch] loaded layer {L + 1}/{mc.num_layers}, "
                  f"{time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    w_emb = _disk(idx, "model.llm.embed.weight").to(devs[0])
    w_en = _disk(idx, "model.llm.embed_norm.weight").to(devs[0])
    w_fn = _disk(idx, "model.llm.norm.weight").to(devs[-1])
    w_un = _disk(idx, "model.llm.unembed.weight").to(devs[-1])
    t_load = time.time() - t0

    #3. baseline: batch-1 generate_greedy per request from fresh state
    base = []
    with torch.no_grad():
        for i in range(8):
            states = [pkv.LayerKV(mc, L, devs[L], num_pages=NUM_PAGES)
                      for L in range(mc.num_layers)]
            toks, _ = pmodel.generate_greedy(ids[i], MAX_NEW8[i], layers,
                                             states, w_emb, w_en, w_fn,
                                             w_un, mc)
            del states
            base.append(toks)
            print(f"[t_b3 batch] baseline r{i} ({time.time() - t0:.0f} s): "
                  f"{tok.decode(toks)!r}", file=sys.stderr, flush=True)
    t_base = time.time() - t0 - t_load

    def _same(tag, got, want, rid):
        #3a. exact int-list equality; report the first differing step
        if got != want:
            j = next((k for k in range(min(len(got), len(want)))
                      if got[k] != want[k]), min(len(got), len(want)))
            raise SystemExit(f"[t_b3 batch] {tag} req {rid} tokens != "
                             f"batch-1: first diff step {j} "
                             f"({got[j:j + 1]} vs {want[j:j + 1]}), lens "
                             f"{len(got)}/{len(want)}")

    def _mk_retire(tag):
        #3b. retirement pin: per-layer cache token count, states attached
        def cb(req):
            want = req.ids.shape[0] + req.max_new - 1
            for L in range(mc.num_layers):
                n = req.states[L].cache.n
                if n != want:
                    raise SystemExit(f"[t_b3 batch] {tag} req {req.id} "
                                     f"layer {L} cache.n {n} != {want}")
            print(f"[t_b3 batch] {tag} req {req.id} retired at step "
                  f"{req.finish_step} ({time.time() - t0:.0f} s)",
                  file=sys.stderr, flush=True)
        return cb

    #4. schedule A: staggered arrivals, cap 8 — all 8 concurrent mid-run
    eng = psched.Engine(layers, w_emb, w_en, w_fn, w_un, mc, NUM_PAGES)
    reqs_a = [psched.Request(i, ids[i], MAX_NEW8[i]) for i in range(8)]
    sched_a = psched.Scheduler(eng, max_batch=8, on_retire=_mk_retire("A"))
    for i, r in enumerate(reqs_a):
        sched_a.submit(r, arrival_step=ARRIVE8[i])
    with torch.no_grad():
        out_a = sched_a.run()
    for i in range(8):
        _same("schedule A", out_a[i], base[i], i)
    if sched_a.peak_running != 8:
        raise SystemExit(f"[t_b3 batch] A peak running "
                         f"{sched_a.peak_running} != 8 — the 8 requests "
                         f"were never concurrent")
    admits_a = [r.admit_step for r in reqs_a]
    if admits_a != list(ARRIVE8):
        raise SystemExit(f"[t_b3 batch] A admit steps {admits_a} != "
                         f"arrivals {list(ARRIVE8)}")
    fins_a = [r.finish_step for r in reqs_a]
    if len(set(fins_a)) < 4:
        raise SystemExit(f"[t_b3 batch] A finish steps {fins_a}: "
                         f"retirements did not interleave")
    t_a = time.time() - t0 - t_load - t_base

    #5. schedule B: reversed submission, all due at 0, cap 3 — FCFS queue
    #   + admission-on-retirement, different co-residents than A
    reqs_b = [psched.Request(i, ids[i], MAX_NEW8[i]) for i in range(8)]
    sched_b = psched.Scheduler(eng, max_batch=3, on_retire=_mk_retire("B"))
    for r in reversed(reqs_b):
        sched_b.submit(r, arrival_step=0)
    with torch.no_grad():
        out_b = sched_b.run()
    for i in range(8):
        _same("schedule B", out_b[i], base[i], i)
    if sched_b.peak_running != 3:
        raise SystemExit(f"[t_b3 batch] B peak running "
                         f"{sched_b.peak_running} != 3 (cap)")
    first3 = sorted(r.id for r in reqs_b if r.admit_step == 0)
    if first3 != [5, 6, 7]:
        raise SystemExit(f"[t_b3 batch] B step-0 admits {first3} != "
                         f"[5, 6, 7] — FCFS from reversed submission broken")
    last_admit_b = max(r.admit_step for r in reqs_b)
    if last_admit_b == 0:
        raise SystemExit("[t_b3 batch] B queue never held a request — "
                         "admission-on-retirement not exercised")
    if any(r.states is not None for r in reqs_a + reqs_b):
        raise SystemExit("[t_b3 batch] KV states not freed at retirement")
    t_b = time.time() - t0 - t_load - t_base - t_a

    #6. summary = the green evidence
    print(f"batch ok: continuous batching scheduler — 8 concurrent greedy "
          f"requests (T {min(lens)}-{max(lens)}, max_new {min(MAX_NEW8)}-"
          f"{max(MAX_NEW8)}, {sum(MAX_NEW8)} tokens) IDENTICAL to batch-1 "
          f"generate_greedy 8/8 requests x BOTH schedules (exact int "
          f"equality, no tolerance) — A: staggered arrivals "
          f"{list(ARRIVE8)} cap 8, peak 8 concurrent, admit==arrival 8/8, "
          f"retirements interleaved ({len(set(fins_a))} distinct finish "
          f"steps {min(fins_a)}..{max(fins_a)}); B: REVERSED submission "
          f"cap 3, FCFS queue (step-0 admits {{5,6,7}}, last admit step "
          f"{last_admit_b}), peak 3; batch-1 math per sequence on its own "
          f"LayerKV by construction (fused-batch kernels = Stage-3/D4); "
          f"cache.n == T+max_new-1 all 66 layers x 8 reqs x 2 schedules "
          f"at retirement, KV freed after; resident split {splits}; wall "
          f"load {t_load:.0f} s + baseline {t_base:.0f} s + A {t_a:.0f} s "
          f"+ B {t_b:.0f} s")


def t_server():
    """B3.4: OpenAI server over the B3.3 scheduler — protocol shape +
    self-consistency (the external semantic referee is B3.5 vs vLLM
    goldens). What this test pins:
      (a) 8 requests POSTed CONCURRENTLY to /v1/completions with the EXACT
          parity-gate payload shape (harness/correctness.py: temperature 0,
          logprobs 1, seed 0, return_token_ids true) plus ignore_eos true:
          HTTP 200 8/8; choice-level token_ids EQUAL (exact ints) to
          batch-1 model.generate_greedy on the same resident weights;
          token_logprobs EQUAL (exact float64 — JSON round-trips float64
          losslessly) to server.chosen_logprobs applied to generate_greedy
          capture= rows: the served path and the reference are the same
          fp32 CPU math on rows produced by the same op sequence (B3.3
          batch invariance), so exactness is the honest gate;
      (b) protocol fields: text == the same detok call on the ids,
          logprobs arrays aligned (tokens/token_logprobs/top_logprobs/
          text_offset), usage exact (benchmark.py reads
          usage.completion_tokens), finish_reason, envelope fields;
          token_ids ABSENT unless requested + logprobs null unless
          requested (correctness.py's fail-loud contract needs the
          present-iff-requested distinction);
      (c) requests were genuinely co-resident (scheduler peak_running
          >= 2; expected 8) — continuous batching through the HTTP path,
          arrivals mapped onto step boundaries;
      (d) ignore_eos semantics: default (absent) arms eos — expected
          tokens = the batch-1 baseline truncated at the first eos
          (dynamic: no assumption eos appears in tiny continuations);
          plus a no-GPU Request.done unit pin (armed eos stops early,
          eos=None runs past eos to max_new — B3.3's canonical form);
      (e) contract violations fail LOUD, never silently mis-serve:
          400 for temperature != 0 / temperature absent / missing prompt /
          logprobs > 1 / stream / echo / max_tokens over KV capacity /
          max_tokens < 1, 404 for unknown paths; /health 200 before,
          DURING the concurrent batch, and after; clean shutdown (engine
          thread joins, queues empty, no engine error)."""
    import requests as rq
    from concurrent.futures import ThreadPoolExecutor
    from transformers import AutoTokenizer
    t0 = time.time()
    torch.manual_seed(0)

    #1. resident engine via the SERVER's own build path (NUM_PAGES=8 ->
    #   128-token capacity, >= max T+max_new = 45 here and small enough
    #   to make the capacity-400 arm cheap)
    eng, splits = pserver.build_resident(MODEL_DIR, num_pages=NUM_PAGES)
    mc = eng.mc
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    ids = [tok(p, return_tensors="pt").input_ids[0].to(eng.w_emb.device)
           for p in PROMPTS8]
    lens = [t.shape[0] for t in ids]
    t_load = time.time() - t0

    #2. batch-1 baselines with fp32 logits capture — tokens + the
    #   reference logprobs through the same helper the server uses
    base, ref_lps = [], []
    with torch.no_grad():
        for i in range(8):
            cap = []
            states = eng.new_states()
            toks, _ = pmodel.generate_greedy(
                ids[i], MAX_NEW8[i], eng.layers, states, eng.w_emb,
                eng.w_en, eng.w_fn, eng.w_un, mc, capture=cap)
            del states
            base.append(toks)
            ref_lps.append(pserver.chosen_logprobs(cap, toks))
            print(f"[t_b3 server] baseline r{i} ({time.time() - t0:.0f} s)"
                  f": {tok.decode(toks)!r}", file=sys.stderr, flush=True)
    t_base = time.time() - t0 - t_load

    #3. serve on an ephemeral port (CLI default is 8200; a stray listener
    #   must not be able to fail this test) — http thread + engine thread
    httpd, loop = pserver.serve(eng, tok, host="127.0.0.1", port=0,
                                max_batch=8)
    port = httpd.server_address[1]
    url = f"http://127.0.0.1:{port}"
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    h0 = rq.get(f"{url}/health", timeout=10)
    if h0.status_code != 200 or h0.json().get("status") != "ok":
        raise SystemExit(f"[t_b3 server] /health before: {h0.status_code} "
                         f"{h0.text}")

    #4. 8 concurrent parity-shaped posts + one /health probe mid-flight
    def _post(i):
        return rq.post(f"{url}/v1/completions",
                       json={"model": MODEL_DIR, "prompt": PROMPTS8[i],
                             "max_tokens": MAX_NEW8[i], "temperature": 0,
                             "logprobs": 1, "seed": 0,
                             "return_token_ids": True, "ignore_eos": True},
                       timeout=600)

    def _health_during():
        time.sleep(3)  # the batch runs ~2 min; 3 s in it is mid-flight
        return rq.get(f"{url}/health", timeout=10)

    with ThreadPoolExecutor(max_workers=9) as pool:
        fut_h = pool.submit(_health_during)
        resps = list(pool.map(_post, range(8)))
        hd = fut_h.result()
    if hd.status_code != 200:
        raise SystemExit(f"[t_b3 server] /health during: {hd.status_code}")
    t_conc = time.time() - t0 - t_load - t_base

    #5. per-response gates: ids exact, logprobs exact, protocol fields
    for i, r in enumerate(resps):
        if r.status_code != 200:
            raise SystemExit(f"[t_b3 server] req {i} HTTP {r.status_code}: "
                             f"{r.text[:400]}")
        j = r.json()
        ch = j["choices"][0]
        got = ch.get("token_ids")
        if got != base[i]:
            k = next((k for k in range(min(len(got or []), len(base[i])))
                      if got[k] != base[i][k]), None)
            raise SystemExit(f"[t_b3 server] req {i} token_ids != batch-1 "
                             f"(first diff at {k}): {got} vs {base[i]}")
        lp = ch["logprobs"]
        if lp["token_logprobs"] != ref_lps[i]:
            dm = max(abs(a - b) for a, b in
                     zip(lp["token_logprobs"], ref_lps[i]))
            raise SystemExit(f"[t_b3 server] req {i} token_logprobs != "
                             f"reference (max |delta| {dm:.3e})")
        want_strs = [tok.decode([t]) for t in base[i]]
        if lp["tokens"] != want_strs:
            raise SystemExit(f"[t_b3 server] req {i} logprobs.tokens "
                             f"mismatch: {lp['tokens'][:4]}...")
        if lp["top_logprobs"] != [{s: l} for s, l in
                                  zip(want_strs, ref_lps[i])]:
            raise SystemExit(f"[t_b3 server] req {i} top_logprobs mismatch")
        if len(lp["text_offset"]) != MAX_NEW8[i]:
            raise SystemExit(f"[t_b3 server] req {i} text_offset len "
                             f"{len(lp['text_offset'])} != {MAX_NEW8[i]}")
        if ch["text"] != tok.decode(base[i], skip_special_tokens=True):
            raise SystemExit(f"[t_b3 server] req {i} text mismatch: "
                             f"{ch['text']!r}")
        if ch["finish_reason"] != "length":
            raise SystemExit(f"[t_b3 server] req {i} finish_reason "
                             f"{ch['finish_reason']!r} != 'length'")
        want_usage = {"prompt_tokens": lens[i],
                      "completion_tokens": MAX_NEW8[i],
                      "total_tokens": lens[i] + MAX_NEW8[i]}
        if j["usage"] != want_usage:
            raise SystemExit(f"[t_b3 server] req {i} usage {j['usage']} != "
                             f"{want_usage}")
        if (j["object"] != "text_completion" or j["model"] != MODEL_DIR
                or not str(j["id"]).startswith("cmpl-")):
            raise SystemExit(f"[t_b3 server] req {i} envelope fields: "
                             f"{j['id']} {j['object']} {j['model']}")
    peak = loop.sched.peak_running
    if peak < 2:
        raise SystemExit(f"[t_b3 server] peak_running {peak} — requests "
                         f"were never co-resident; not continuous "
                         f"batching through HTTP")

    #6. default-eos + minimal-fields arm: no logprobs / return_token_ids /
    #   ignore_eos -> logprobs null, NO token_ids key, eos armed (expected
    #   = baseline truncated at first eos — dynamic, works either way)
    r = rq.post(f"{url}/v1/completions",
                json={"model": MODEL_DIR, "prompt": PROMPTS8[0],
                      "max_tokens": 8, "temperature": 0}, timeout=600)
    if r.status_code != 200:
        raise SystemExit(f"[t_b3 server] eos-arm HTTP {r.status_code}: "
                         f"{r.text[:400]}")
    j = r.json()
    ch = j["choices"][0]
    exp = base[0][:8]
    exp_eos = (exp[:exp.index(mc.eos) + 1] if mc.eos in exp else exp)
    fin_want = "stop" if mc.eos in exp else "length"
    if "token_ids" in ch:
        raise SystemExit("[t_b3 server] token_ids present without "
                         "return_token_ids — breaks the present-iff-"
                         "requested contract")
    if ch["logprobs"] is not None:
        raise SystemExit("[t_b3 server] logprobs not null when unrequested")
    if ch["text"] != tok.decode(exp_eos, skip_special_tokens=True):
        raise SystemExit(f"[t_b3 server] eos-arm text {ch['text']!r} != "
                         f"baseline-derived {exp_eos}")
    if (ch["finish_reason"] != fin_want
            or j["usage"]["completion_tokens"] != len(exp_eos)):
        raise SystemExit(f"[t_b3 server] eos-arm finish/usage: "
                         f"{ch['finish_reason']} {j['usage']}")
    n_eos_base = sum(mc.eos in b for b in base)

    #7. Request-level eos unit pin (no GPU): armed eos stops early;
    #   eos=None (ignore_eos) runs past eos to max_new
    r_eos = psched.Request(99, ids[0].cpu(), 8, eos=7)
    r_eos.tokens.extend([3, 5])
    assert not r_eos.done, "eos pin: done before eos"
    r_eos.tokens.append(7)
    assert r_eos.done, "eos pin: armed eos did not stop the request"
    r_ign = psched.Request(98, ids[0].cpu(), 3, eos=None)
    r_ign.tokens.extend([7, 7])
    assert not r_ign.done, "eos pin: eos=None stopped early"
    r_ign.tokens.append(1)
    assert r_ign.done, "eos pin: length cap did not stop the request"

    #8. loud-failure arms: each of these MUST 400 (silently serving any
    #   of them would corrupt a gate or a benchmark), unknown paths 404
    bad = [({"prompt": "x", "max_tokens": 4, "temperature": 0.7},
            "temperature 0.7"),
           ({"prompt": "x", "max_tokens": 4}, "temperature absent"),
           ({"max_tokens": 4, "temperature": 0}, "prompt missing"),
           ({"prompt": "x", "max_tokens": 4, "temperature": 0,
             "logprobs": 5}, "logprobs 5"),
           ({"prompt": "x", "max_tokens": 4, "temperature": 0,
             "stream": True}, "stream"),
           ({"prompt": "x", "max_tokens": 4, "temperature": 0,
             "echo": True}, "echo"),
           ({"prompt": "x", "max_tokens": NUM_PAGES * pkv.PAGE_SIZE,
             "temperature": 0}, "over capacity"),
           ({"prompt": "x", "max_tokens": 0, "temperature": 0},
            "max_tokens 0")]
    for body, tag in bad:
        rb = rq.post(f"{url}/v1/completions", json=body, timeout=30)
        if rb.status_code != 400 or "error" not in rb.json():
            raise SystemExit(f"[t_b3 server] bad-input arm '{tag}': "
                             f"HTTP {rb.status_code} (want 400+error)")
    r404p = rq.post(f"{url}/v1/chat/completions", json={}, timeout=30)
    r404g = rq.get(f"{url}/nope", timeout=30)
    if r404p.status_code != 404 or r404g.status_code != 404:
        raise SystemExit(f"[t_b3 server] unknown paths: "
                         f"{r404p.status_code}/{r404g.status_code} != 404")

    #9. /health after + clean shutdown; nothing stranded, no engine error
    ha = rq.get(f"{url}/health", timeout=10)
    if ha.status_code != 200:
        raise SystemExit(f"[t_b3 server] /health after: {ha.status_code}")
    httpd.shutdown()
    th.join(10)
    if not loop.stop():
        raise SystemExit("[t_b3 server] engine thread did not join")
    if loop.error is not None:
        raise SystemExit(f"[t_b3 server] engine error:\n{loop.error}")
    if loop.sched.waiting or loop.sched.running:
        raise SystemExit("[t_b3 server] scheduler queues not empty at "
                         "shutdown")
    n_fin = len(loop.sched.finished)
    if n_fin != 9:
        raise SystemExit(f"[t_b3 server] finished {n_fin} != 9 (8 "
                         f"concurrent + 1 eos-arm; 400s must never reach "
                         f"the scheduler)")

    #10. summary = the green evidence
    print(f"server ok: /v1/completions x 8 CONCURRENT parity-shaped "
          f"requests (temperature 0, logprobs 1, seed, return_token_ids, "
          f"ignore_eos; T {min(lens)}-{max(lens)}, max_new {min(MAX_NEW8)}"
          f"-{max(MAX_NEW8)}, {sum(MAX_NEW8)} tokens) — choice-level "
          f"token_ids == batch-1 generate_greedy 8/8 (exact ints), "
          f"token_logprobs == fp32 log-softmax of batch-1 captured logits "
          f"8/8 (exact float64 via JSON), tokens/top_logprobs/text_offset "
          f"aligned, text == detok, finish 'length', usage exact 8/8, "
          f"peak_running {peak}; default-eos arm: logprobs null + NO "
          f"token_ids key + eos-armed truncation (eos in {n_eos_base}/8 "
          f"baselines; Request-level pin: armed eos stops early, eos=None "
          f"runs past); loud failures: 400 x {len(bad)} (bad temperature/"
          f"prompt/logprobs/stream/echo/capacity/max_tokens), 404 x 2; "
          f"/health ok before+during+after; ephemeral port {port} (CLI "
          f"default 8200); shutdown clean (engine thread joined, queues "
          f"empty, {n_fin} retired); wall load {t_load:.0f} s + baseline "
          f"{t_base:.0f} s + concurrent {t_conc:.0f} s + arms "
          f"{time.time() - t0 - t_load - t_base - t_conc:.0f} s")


SOAK_SECS = 1800          # the item's 30 minutes — not configurable
SOAK_SAMPLE_S = 10        # monitor cadence (~180 samples over the soak)
SOAK_W1 = (300, 600)      # steady-state reference window (s from soak start)
SOAK_W2 = (1500, 1800)    # final window; drift = mean(W2) - mean(W1)
SOAK_GATE_RSS_MB = 512    # process RSS drift ceiling
SOAK_GATE_ALLOC_MB = 512  # per-device torch allocated drift ceiling
SOAK_GATE_RESV_MB = 1024  # per-device torch reserved drift ceiling
SOAK_MAX_NEW = (32, 24, 17, 32, 9, 27, 12, 16)  # varied -> constant churn
SOAK_LONG = "The quick brown fox jumps over the lazy dog. " * 70


def _vmrss_mb():
    """Process resident set size in MiB (/proc/self/status VmRSS)."""
    #1. VmRSS is reported in kB
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    raise SystemExit("[t_b3 soak] VmRSS not found in /proc/self/status")


def t_soak():
    """B3.6: 30-minute soak at concurrency 8 — zero crashes/leaks, rss +
    vram stable. Drives the B3.4 server in its SERVING config (num_pages
    1024 = the 16384-token capacity, max_batch 8) with 8 driver threads
    posting parity-shaped requests back-to-back on keep-alive sessions;
    thread 7 soaks a ~700-token prompt so every cycle crosses the 512 ring
    window + sconv boundary (B2.7) and exercises multi-page paged-KV
    alloc/free churn; the rest reuse the B3.3 tiny prompts with varied
    max_new so admissions/retirements interleave continuously.

    Leak gates (drift = mean over W2=[1500,1800)s minus mean over
    W1=[300,600)s, ~30 samples each; the first 300 s are allocator warmup
    — every request shape completes a full cycle within ~200 s):
      - vram allocated <= 512 MiB/device: the smallest plausible real KV
        leak is ONE layer's state per retired request (~67 MiB paged pool,
        ~4 MiB ring); ~40 retirements between window midpoints => >= 2.7
        GiB signal, while sampling-phase noise (+-1 resident seq ~ 260
        MiB instantaneous) mean-smooths far below the gate;
      - vram reserved <= 1 GiB/device: fragmentation creep; fixed request
        shapes recycle identical block sizes, so reserved must plateau;
      - rss <= 512 MiB: a leaked fp32 capture row is ~0.8 MiB/token and
        every request asks logprobs; ~1800 tokens between window midpoints
        => ~1.4 GiB signal; in-flight rows are bounded (~8 x 32 rows).
    Crash gates: HTTP 200 on every completion, /health 200 every 10 s
    (before/throughout/after), engine loop error None, clean drain +
    shutdown (thread joins, queues empty, finished == completions).
    Consistency gates (corruption referee): every completion's token_ids
    AND token_logprobs EXACTLY equal the same thread's cycle-0 response
    (same batch-1 op sequence per B3.3 invariance — exact ints / exact
    float64 via JSON, the B3.4 precedent), usage/finish_reason/lengths
    exact. Concurrency evidence: peak_running == 8, mean co-residency
    >= 6 (back-to-back posts leave at most one-step admission gaps)."""
    import math
    import requests as rq
    from transformers import AutoTokenizer
    t0 = time.time()

    #1. resident engine in the SERVING config (server.py CLI defaults),
    #   tokenize the 8 fixed request shapes; pin ring-crossing coverage
    eng, splits = pserver.build_resident(MODEL_DIR, num_pages=1024)
    mc = eng.mc
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    prompts = list(PROMPTS8[:7]) + [SOAK_LONG]
    lens = [int(tok(p, return_tensors="pt").input_ids[0].shape[0])
            for p in prompts]
    if lens[7] <= mc.window + mc.sconv_k - 1:
        raise SystemExit(f"[t_b3 soak] long prompt T={lens[7]} does not "
                         f"cross the window+sconv boundary "
                         f"{mc.window + mc.sconv_k - 1}")
    ndev = torch.cuda.device_count()
    t_load = time.time() - t0

    #2. serve on an ephemeral port (a stray 8200 listener must not be able
    #   to fail this test); /health must answer before the soak starts
    httpd, loop = pserver.serve(eng, tok, host="127.0.0.1", port=0,
                                max_batch=8)
    port = httpd.server_address[1]
    url = f"http://127.0.0.1:{port}"
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    if rq.get(f"{url}/health", timeout=10).status_code != 200:
        raise SystemExit("[t_b3 soak] /health before soak failed")

    #3. shared soak state; any failure aborts the soak LOUDLY via `abort`
    #   so a broken run fails in seconds, not after 30 idle minutes
    errors, err_lock = [], threading.Lock()
    refs = [None] * 8      # (token_ids, token_logprobs) of cycle 0
    counts = [0] * 8       # completed-and-verified cycles per thread
    samples = []           # (t, rss_mb, alloc_mb[ndev], resv_mb[ndev], n_running)
    abort = threading.Event()
    mon_stop = threading.Event()
    t_start = time.time()
    deadline = t_start + SOAK_SECS

    def _fail(msg):
        with err_lock:
            errors.append(msg)
        abort.set()

    def _drive(i):
        #1. one fixed request shape per thread, posted back-to-back on a
        #   keep-alive session until the 30-minute deadline
        sess = rq.Session()
        body = {"model": MODEL_DIR, "prompt": prompts[i],
                "max_tokens": SOAK_MAX_NEW[i], "temperature": 0,
                "logprobs": 1, "seed": 0, "return_token_ids": True,
                "ignore_eos": True}
        while time.time() < deadline and not abort.is_set():
            try:
                r = sess.post(f"{url}/v1/completions", json=body,
                              timeout=600)
            except Exception as e:
                _fail(f"r{i} cycle {counts[i]}: transport {e!r}")
                break
            if r.status_code != 200:
                _fail(f"r{i} cycle {counts[i]}: HTTP {r.status_code} "
                      f"{r.text[:200]}")
                break
            #2. shape checks: exact usage/lengths/finish, finite logprobs
            j = r.json()
            ch = j["choices"][0]
            tids = ch.get("token_ids")
            lps = (ch.get("logprobs") or {}).get("token_logprobs")
            usage = {"prompt_tokens": lens[i],
                     "completion_tokens": SOAK_MAX_NEW[i],
                     "total_tokens": lens[i] + SOAK_MAX_NEW[i]}
            if (not isinstance(tids, list) or len(tids) != SOAK_MAX_NEW[i]
                    or ch["finish_reason"] != "length"
                    or j["usage"] != usage
                    or not isinstance(lps, list)
                    or len(lps) != SOAK_MAX_NEW[i]
                    or not all(math.isfinite(x) for x in lps)):
                _fail(f"r{i} cycle {counts[i]}: malformed response "
                      f"{str(j)[:300]}")
                break
            #3. consistency vs this thread's cycle 0 — exact (B3.3/B3.4)
            if refs[i] is None:
                refs[i] = (tids, lps)
            elif refs[i] != (tids, lps):
                _fail(f"r{i} cycle {counts[i]}: token_ids/logprobs changed "
                      f"vs cycle 0 — cross-request state corruption")
                break
            counts[i] += 1
        sess.close()

    def _monitor():
        #1. every SOAK_SAMPLE_S: rss + per-device vram gauges +
        #   co-residency + a /health probe; engine death aborts
        sess = rq.Session()
        while not mon_stop.is_set():
            t = time.time() - t_start
            alloc = [torch.cuda.memory_allocated(d) / (1 << 20)
                     for d in range(ndev)]
            resv = [torch.cuda.memory_reserved(d) / (1 << 20)
                    for d in range(ndev)]
            samples.append((t, _vmrss_mb(), alloc, resv,
                            len(loop.sched.running)))
            try:
                if sess.get(f"{url}/health",
                            timeout=10).status_code != 200:
                    _fail(f"/health at {t:.0f} s: non-200")
            except Exception as e:
                _fail(f"/health at {t:.0f} s: {e!r}")
            if loop.error is not None:
                _fail(f"engine died at {t:.0f} s:\n{loop.error}")
            mon_stop.wait(SOAK_SAMPLE_S)
        sess.close()

    #4. run the soak: 8 drivers + 1 monitor; drivers exit at the deadline
    #   (after their last response) or on abort
    drv = [threading.Thread(target=_drive, args=(i,)) for i in range(8)]
    mon = threading.Thread(target=_monitor)
    for d in drv:
        d.start()
    mon.start()
    for d in drv:
        d.join()
    t_soaked = time.time() - t_start

    #5. drain (every issued request already answered, so this is at most
    #   one step) then stop the monitor; fail with the recorded errors
    t_dr = time.time()
    while ((loop.sched.running or loop.sched.waiting)
           and time.time() - t_dr < 300):
        time.sleep(1)
    mon_stop.set()
    mon.join()
    if errors:
        raise SystemExit("[t_b3 soak] FAILED:\n" + "\n".join(errors[:10]))
    if loop.sched.running or loop.sched.waiting:
        raise SystemExit("[t_b3 soak] scheduler did not drain in 300 s")

    #6. stability evaluation over the two steady-state windows
    def _win(lo, hi):
        return [s for s in samples if lo <= s[0] < hi]

    def _mean(xs):
        return sum(xs) / len(xs)

    w1, w2 = _win(*SOAK_W1), _win(*SOAK_W2)
    if len(w1) < 10 or len(w2) < 10:
        raise SystemExit(f"[t_b3 soak] windows too thin ({len(w1)}/"
                         f"{len(w2)} samples) — soak did not run 30 min")
    rss1, rss2 = _mean([s[1] for s in w1]), _mean([s[1] for s in w2])
    d_rss = rss2 - rss1
    d_alloc = [_mean([s[2][d] for s in w2]) - _mean([s[2][d] for s in w1])
               for d in range(ndev)]
    d_resv = [_mean([s[3][d] for s in w2]) - _mean([s[3][d] for s in w1])
              for d in range(ndev)]
    mean_run = _mean([s[4] for s in samples
                      if SOAK_W1[0] <= s[0] < SOAK_SECS])
    peak = loop.sched.peak_running
    bad = []
    if d_rss > SOAK_GATE_RSS_MB:
        bad.append(f"rss drift {d_rss:+.0f} MiB > {SOAK_GATE_RSS_MB}")
    for d in range(ndev):
        if d_alloc[d] > SOAK_GATE_ALLOC_MB:
            bad.append(f"cuda:{d} alloc drift {d_alloc[d]:+.0f} MiB > "
                       f"{SOAK_GATE_ALLOC_MB}")
        if d_resv[d] > SOAK_GATE_RESV_MB:
            bad.append(f"cuda:{d} reserved drift {d_resv[d]:+.0f} MiB > "
                       f"{SOAK_GATE_RESV_MB}")
    if peak != 8:
        bad.append(f"peak_running {peak} != 8")
    if mean_run < 6:
        bad.append(f"mean co-residency {mean_run:.2f} < 6")
    if bad:
        raise SystemExit("[t_b3 soak] STABILITY FAIL: " + "; ".join(bad))

    #7. /health after + clean shutdown, nothing stranded, no engine error
    if rq.get(f"{url}/health", timeout=10).status_code != 200:
        raise SystemExit("[t_b3 soak] /health after soak failed")
    httpd.shutdown()
    th.join(10)
    if not loop.stop():
        raise SystemExit("[t_b3 soak] engine thread did not join")
    if loop.error is not None:
        raise SystemExit(f"[t_b3 soak] engine error:\n{loop.error}")
    n_fin = len(loop.sched.finished)
    if n_fin != sum(counts):
        raise SystemExit(f"[t_b3 soak] finished {n_fin} != completions "
                         f"{sum(counts)}")

    #8. summary = the green evidence
    tok_total = sum(c * m for c, m in zip(counts, SOAK_MAX_NEW))
    print(f"soak ok: {SOAK_SECS // 60} min at conc 8 — {sum(counts)} "
          f"completions ({tok_total} tokens; per-thread {counts}, max_new "
          f"{list(SOAK_MAX_NEW)}, T {min(lens)}-{max(lens)} incl. "
          f"{lens[7]}-tok ring-crossing prompt), {loop.sched.step_count} "
          f"steps, peak_running {peak}, mean co-residency {mean_run:.2f}; "
          f"ZERO crashes (200 x {sum(counts)}, /health 200 x {len(samples)}"
          f" probes, engine error None); token_ids + token_logprobs exact-"
          f"equal cycle 0 on all 8 threads; rss {rss1:.0f} -> {rss2:.0f} "
          f"MiB (drift {d_rss:+.0f} <= {SOAK_GATE_RSS_MB}); vram drift/dev "
          f"MiB alloc {[f'{x:+.0f}' for x in d_alloc]} <= "
          f"{SOAK_GATE_ALLOC_MB}, reserved {[f'{x:+.0f}' for x in d_resv]} "
          f"<= {SOAK_GATE_RESV_MB} (mean W2 [1500,1800)s - W1 [300,600)s); "
          f"drain + shutdown clean ({n_fin} retired); wall load "
          f"{t_load:.0f} s + soak {t_soaked:.0f} s")


def main():
    #1. dispatch on subcommand
    cmds = {"kv": t_kv, "decode": t_decode,
            "batch": t_batch,
            "server": t_server, "soak": t_soak}
    usage = f"usage: python -m engine.pyengine.tests.t_b3 {{{'|'.join(cmds)}}}"
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        raise SystemExit(usage)
    cmds[sys.argv[1]]()


if __name__ == "__main__":
    main()
