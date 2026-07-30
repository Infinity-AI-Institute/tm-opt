"""exp-0012: CUDA-graphed batched decode step.

The exp-0011 in-situ phase profile (insitu_phase_exp0011_2026-07-30.log)
puts host ENQUEUE at 98.8% of the decode step's wall (dec_enq 107.1 s of
dec_wall 108.4 s over 256 steps at ~64 rows; exposed GPU tail 1.2%): the
66-layer batched traversal is ~6k eager Python/torch/Triton dispatches
per step and the GPU finishes each step in the shadow of the host still
enqueueing it. This module deletes that dispatch wholesale: the step is
captured ONCE per (device segment, key-length bucket) into CUDA graphs
operating on static buffers, and every subsequent step is input copies +
graph replays + one batched-argmax readback.

Shape staticization (a graph replays a FIXED kernel sequence):
- Batch is padded to max_batch: rows without a live sequence are DEAD —
  prev token 0, position 0, and the pools' dedicated scratch slot
  (kv.SwaPool/GlobalPool/SconvPool row max_batch, scratch page). Dead
  rows are all configured identically, so their colliding scratch
  writes carry identical bytes and scratch state stays a pure function
  of the schedule; per-row ops (norms, row-wise GEMM outputs, per-row
  attention/softmax, per-pair expert GEMM segments) never mix dead
  content into live rows. Dead rows never NaN: position 0 always has
  its own key valid, so no all-masked softmax exists.
- The global-attention padded key length is a MONOTONE high-water
  512-bucket (L_pad): capture happens at most once per bucket per
  lifetime, buckets only grow, and only the current bucket's graphs are
  ever replayed after a new capture — so per-device graph memory pools
  are shared safely in capture order. Padded columns are dead by the
  same j < n predicate the dynamic ctx used for short rows.
- Layer placement is a contiguous per-device split (server
  build_resident): one graph per device segment, eager 786 KB hidden-
  state copies between segments, so no multi-device capture is needed.

The one un-capturable region — layer dense_idx's bf16-unquantized
routed experts (the reference loop: data-dependent unique().tolist() +
per-hit-expert GEMMs) — runs as an EAGER ISLAND between two device-0
graphs, on the LIVE rows only (_run_island: dead rows in its expert
batches would couple live-token bits to scratch history through the
per-expert GEMM row counts). Its ~1-2k enqueues overlap the GPU's
execution of the captured segments, so its host cost hides in the GPU
shadow.

Correctness semantics (B3.1/B3.3, D13): fixed-B GEMMs and padded-length
softmax reductions retile vs the variable-B eager step, so per-token
bits drift within the D13 envelope exactly like every batched rewrite
since exp-0001; prefill (the teacher-forced gate path) is untouched.
Same-schedule determinism holds: bucket sequence, capture points,
dead-row configs and replayed kernel sequences are pure functions of
the schedule, every captured op is deterministic (stable sort,
searchsorted offsets, no atomics), and warmup/capture touch only
scratch state. Kill switch: PYENGINE_CUDAGRAPH=0 restores the eager
pooled path unchanged."""
import os

import torch

from engine.pyengine import kv as pkv
from engine.pyengine import model as pmodel

BUCKET = 512  # L_pad granularity: recapture at most ceil(16384/512) times


def enabled():
    """Env kill switch; default ON (the experiment's serving mode)."""
    return os.environ.get("PYENGINE_CUDAGRAPH", "1") != "0"


class _Seg:
    """One captured device segment: fn is the capture body (pure device
    work on static buffers), graph the captured CUDAGraph after
    _capture."""

    def __init__(self, dev, fn):
        self.dev, self.fn, self.graph = dev, fn, None


class GraphRunner:
    """Owns the static buffers, the per-bucket capture cache, and the
    replay step. Built lazily by Engine.decode_batch on the engine
    thread (all CUDA work single-threaded there, so default capture
    mode is safe)."""

    def __init__(self, engine):
        e = engine
        mc = e.mc
        self.eng = e
        self.MB = e._pool_batch
        self.scratch = self.MB          # pools' extra row (kv.SwaPool)
        #1. contiguous layer->device ranges (server build_resident split)
        ranges = []
        for L in range(mc.num_layers):
            dev = e.layers[L]["attn_norm"].device
            if ranges and ranges[-1][0] == dev:
                ranges[-1][2] = L + 1
            else:
                ranges.append([dev, L, L + 1])
        assert len({r[0] for r in ranges}) == len(ranges), \
            "layer split must be contiguous per device"
        assert e.w_emb.device == ranges[0][0], "embed off first segment"
        assert e.w_fn.device == ranges[-1][0], "unembed off last segment"
        self.ranges = [tuple(r) for r in ranges]
        self.ndev = len(ranges)
        #2. eager islands: MoE layers without packed experts (the bf16-
        #   unquantized layer dense_idx; data-dependent host loop)
        self.islands = [L for L in range(mc.num_layers)
                        if L >= mc.dense_idx
                        and not hasattr(e.layers[L]["gu"], "packed")]
        for L in self.islands:
            dev, lo, hi = next(r for r in self.ranges if r[1] <= L < r[2])
            assert L + 1 < hi, "island at a segment tail unsupported"
        #3. static buffers: padded row config (per device), segment
        #   hand-offs, island hand-offs, outputs
        MB, hid = self.MB, mc.hidden
        d0, dl = self.ranges[0][0], self.ranges[-1][0]
        self.prev_buf = torch.zeros(MB, dtype=torch.int64, device=d0)
        self.pos_buf = [torch.zeros(MB, dtype=torch.int64, device=r[0])
                        for r in self.ranges]
        self.slots_buf = [torch.full((MB,), self.scratch,
                                     dtype=torch.int64, device=r[0])
                          for r in self.ranges]
        self.hin = [None] + [torch.zeros(MB, hid, dtype=torch.bfloat16,
                                         device=r[0])
                             for r in self.ranges[1:]]
        self.hout = [torch.zeros(MB, hid, dtype=torch.bfloat16,
                                 device=r[0])
                     for r in self.ranges[:-1]] + [None]
        self.isl = {L: tuple(torch.zeros(MB, hid, dtype=torch.bfloat16,
                                         device=e.layers[L]
                                         ["attn_norm"].device)
                             for _ in range(3))       # x2, h2n, m
                    for L in self.islands}
        self.logits_buf = torch.zeros(MB, mc.vocab_unpadded,
                                      dtype=torch.float32, device=dl)
        self.argmax_buf = torch.zeros(MB, dtype=torch.int64, device=dl)
        #4. pinned staging (one per kind; reads complete before the next
        #   step's overwrite — the end-of-step argmax sync drains every
        #   device's chain, input copies included)
        self.pin = {k: torch.empty(MB, dtype=torch.int64,
                                   pin_memory=True)
                    for k in ("prev", "pos", "slots")}
        #5. capture cache: L_pad -> program; shared per-device mem pools
        self.programs = {}
        self.mempool = {}
        self.hiwater = 0

    # ---- capture-time program construction -------------------------

    def _ctx(self, di, L_pad):
        #1. per-device static-shape step context, built IN-graph from
        #   the device's static pos/slots buffers (model.py)
        return pmodel.decode_batch_ctx_static(
            self.pos_buf[di], self.slots_buf[di], L_pad, self.eng.mc,
            pkv.PAGE_SIZE)

    def _make_seg(self, di, entry, Ls, exit_kind, L_pad):
        """One segment closure. entry: ("embed",) | ("hin",) |
        ("resume", L). exit_kind: ("attn_half", L) | ("hout",) |
        ("logits",). All tensors it touches live on ranges[di]'s device
        and are either weights, pools, or this runner's static bufs."""
        e, mc = self.eng, self.eng.mc

        def fn():
            ctx = self._ctx(di, L_pad)
            #1. entry
            if entry[0] == "embed":
                _, h = pmodel.embed(self.prev_buf, e.w_emb, e.w_en,
                                    eps=mc.rms_eps)
            elif entry[0] == "hin":
                h = self.hin[di]
            else:
                L = entry[1]
                w, pool = e.layers[L], e._pools[L]
                hh = pmodel.sconv_decode_batch_pooled(
                    self.isl[L][2], w["mlp_sconv"], pool.sconv.mt,
                    ctx["slots"])
                h = self.isl[L][0] + hh
            #2. full layers (pooled batched form; states=None — page
            #   allocation is hoisted to step #1)
            for L in Ls:
                h = pmodel.layer_decode_batch(h, e.layers[L], None, None,
                                              mc, L, ctx=ctx,
                                              pool=e._pools[L])
            #3. exit
            if exit_kind[0] == "attn_half":
                L = exit_kind[1]
                w, pool = e.layers[L], e._pools[L]
                hh = pmodel.rmsnorm(h, w["attn_norm"], mc.rms_eps)
                hh = pmodel.attn_decode_batch_pooled(
                    hh, w["wq"], w["wk"], w["wv"], w["wr"], w["wo"],
                    w["ksc"], w["vsc"], w["qn"], w["kn"], w["proj"],
                    None, pool, ctx, mc.head_dim,
                    L in mc.global_layers, alpha=mc.log_alpha,
                    n_floor=mc.log_floor, eps=mc.rms_eps)
                hh = pmodel.sconv_decode_batch_pooled(
                    hh, w["attn_sconv"], pool.sconv.at, ctx["slots"])
                x = h + hh
                self.isl[L][0].copy_(x)
                self.isl[L][1].copy_(
                    pmodel.rmsnorm(x, w["mlp_norm"], mc.rms_eps))
            elif exit_kind[0] == "hout":
                self.hout[di].copy_(h)
            else:
                rows = pmodel.final_logits(h, e.w_fn, e.w_un,
                                           mc.logits_mult,
                                           mc.vocab_unpadded,
                                           eps=mc.rms_eps)
                self.logits_buf.copy_(rows)     # exact bf16->fp32 cast
                self.argmax_buf.copy_(self.logits_buf.argmax(dim=-1))

        return _Seg(self.ranges[di][0], fn)

    def _build_program(self, L_pad):
        """Ordered replay plan: _Seg | ("copy", di) | ("island", L)."""
        program = []
        for di, (dev, lo, hi) in enumerate(self.ranges):
            if di > 0:
                program.append(("copy", di))
            entry = ("embed",) if di == 0 else ("hin",)
            start = lo
            for L in (x for x in self.islands if lo <= x < hi):
                program.append(self._make_seg(
                    di, entry, range(start, L), ("attn_half", L), L_pad))
                program.append(("island", L))
                entry, start = ("resume", L), L + 1
            tail = ("logits",) if di == self.ndev - 1 else ("hout",)
            program.append(self._make_seg(
                di, entry, range(start, hi), tail, L_pad))
        return program

    def _run_island(self, L, B):
        #1. the bf16 routed-expert MoE half, eager (reference loop with
        #   its unique().tolist() — host cost hides under the GPU's
        #   execution of the surrounding replays). LIVE ROWS ONLY: dead
        #   rows batched into the per-hit-expert GEMMs would make the
        #   expert row counts — and through bf16 tiling the LIVE tokens'
        #   bits — depend on scratch-state history, which two runs of
        #   the same schedule need not share (a fresh server vs a warm
        #   one). Slicing to [:B] makes the island a pure function of
        #   live routing; dead output rows are a fixed zero
        e, mc = self.eng, self.eng.mc
        w = e.layers[L]
        m = pmodel.moe(self.isl[L][1][:B], w["gate_w"], w["gate_b"],
                       w["gate_gs"], w["gu"], w["w2"], w["shg"],
                       w["shu"], w["shd"], mc.topk, mc.n_shared,
                       mc.route_scale)
        self.isl[L][2][:B].copy_(m)
        self.isl[L][2][B:].zero_()

    def _capture(self, L_pad):
        """Warm up (side stream, twice) and capture every segment of the
        L_pad bucket. Runs with DEAD row config in every input buffer so
        the executed warmups touch only scratch state — live pool state
        is never advanced outside a real replay; capture itself records
        without executing. Deterministic: capture points and the dead
        config are pure functions of the schedule."""
        #1. dead config + zeroed hand-offs
        self.prev_buf.zero_()
        for di in range(self.ndev):
            self.pos_buf[di].zero_()
            self.slots_buf[di].fill_(self.scratch)
        for t in list(self.hin) + list(self.hout):
            if t is not None:
                t.zero_()
        for bufs in self.isl.values():
            for t in bufs:
                t.zero_()
        #2. per-segment warmup + capture, program order
        program = self._build_program(L_pad)
        for item in program:
            if not isinstance(item, _Seg):
                continue
            with torch.cuda.device(item.dev):
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    item.fn()
                    item.fn()
                torch.cuda.current_stream().wait_stream(s)
                torch.cuda.synchronize()
                if item.dev not in self.mempool:
                    self.mempool[item.dev] = torch.cuda.graph_pool_handle()
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.mempool[item.dev]):
                    item.fn()
                item.graph = g
        self.programs[L_pad] = program

    # ---- the per-step replay path ----------------------------------

    def _stage(self, kind, vals, pad):
        #1. fill one pinned buffer: live rows then the dead-row constant
        pin = self.pin[kind]
        pin[:len(vals)] = torch.tensor(vals, dtype=torch.int64)
        pin[len(vals):] = pad
        return pin

    def step(self, reqs, pos, slots):
        """One graphed decode step for the live rows `reqs` (row order =
        the scheduler's admission order). Returns the B greedy tokens.
        Host work: page-boundary allocation, 3 pinned stages + fan-out
        copies, segment replays + islands, ONE argmax readback sync."""
        e, mc = self.eng, self.eng.mc
        B = len(reqs)
        ps = pkv.PAGE_SIZE
        #1. page-boundary allocation, hoisted from the captured layers:
        #   position p needs page p//ps only when p % ps == 0 (prefill
        #   covered (T-1)//ps; each earlier decode covered its own)
        for r, p in zip(reqs, pos):
            if p % ps == 0:
                for st in r.states:
                    if st.is_global:
                        st.cache.ensure_pages(p // ps)
        #2. monotone high-water bucket; capture on first sight
        L_pad = -(-(max(pos) + 1) // BUCKET) * BUCKET
        L_pad = min(max(L_pad, self.hiwater), e.num_pages * ps)
        if L_pad not in self.programs:
            self._capture(L_pad)
        self.hiwater = L_pad
        #3. stage the padded row config; fan out to every device
        prev = self._stage("prev", [r.tokens[-1] for r in reqs], 0)
        posp = self._stage("pos", pos, 0)
        slotsp = self._stage("slots", slots, self.scratch)
        self.prev_buf.copy_(prev, non_blocking=True)
        for di in range(self.ndev):
            self.pos_buf[di].copy_(posp, non_blocking=True)
            self.slots_buf[di].copy_(slotsp, non_blocking=True)
        #4. replay the program
        for item in self.programs[L_pad]:
            if isinstance(item, _Seg):
                with torch.cuda.device(item.dev):
                    item.graph.replay()
            elif item[0] == "copy":
                di = item[1]
                self.hin[di].copy_(self.hout[di - 1], non_blocking=True)
            else:
                self._run_island(item[1], B)
        #5. outputs: one batched-argmax readback (the step's single
        #   sync); fp32 logits rows for capture consumers (D13 gate
        #   form — same values the eager per-row .float() produced)
        toks = self.argmax_buf[:B].cpu().tolist()
        for i, r in enumerate(reqs):
            if r.capture is not None:
                r.capture.append(self.logits_buf[i].cpu())
        #6. pooled appends bypass per-seq .append — keep host counters
        #   truthful (release/gather safety; scheduler.decode_batch #4)
        for r in reqs:
            for st in r.states:
                st.cache.n += 1
        return toks
