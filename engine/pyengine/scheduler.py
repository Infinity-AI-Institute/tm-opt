"""B3.3: continuous batching scheduler — iteration-level scheduling (the
Orca/vLLM shape: requests join the running set at any step boundary, leave
the moment they finish, every running sequence advances one token per step).

DECODE IS BATCHED (Stage-3 exp-0001; supersedes B3.3's batch-1-bitwise
design note): a step advances ALL decodable co-residents through ONE
stack traversal (Engine.decode_batch / model.layer_decode_batch), so each
layer's weights are read once per step, not once per sequence. Prefill is
unchanged — whole-prompt, per-sequence, batch-1 exact (Engine.prefill,
still bitwise generate_greedy's; the D13 gate's max_tokens=1 requests
exercise exactly this path). Tokens from batched decode are NOT bitwise
batch-1: bf16 accumulation order is row-count-dependent (B3.1/B3.3
established this is unattainable in principle), so the binding gates are
(a) the D13 teacher-forced envelope parity check and (b) same-schedule
determinism — two runs of an identical schedule are bitwise identical,
because batch composition per step is a pure function of the schedule
(FCFS admission, admission-order rows, deterministic kernels). Engine
.decode (per-sequence batch-1) is retained as the reference arm for
that A/B (t_batched evidence script). Chunked prefill (an arrival not
stalling running decodes) remains deferred; the seam is Engine.prefill.

Scope: greedy only (canonical workloads are greedy, P1), ignore_eos
semantics (no early exit — a request runs to its max_new, D6/P1), text
only. The B3.4 server maps wall-clock arrivals onto step boundaries via
Scheduler.submit(arrival_step=...) and consumes completions through the
on_retire callback."""
import bisect

import torch

from engine.pyengine import graphrun
from engine.pyengine import kv as pkv
from engine.pyengine import model as pmodel


class Request:
    """One greedy generation request. `ids` [T] int64 prompt tokens on the
    embed weight's device; `max_new` >= 1 tokens to generate. `eos` arms
    early exit after that token id is generated (B3.4 server,
    ignore_eos=false semantics; None = run to max_new, the canonical
    ignore_eos form, D6/P1). `capture`, when a list, receives each step's
    full fp32 logits row on CPU (read-only copies, generate_greedy's
    capture= pattern — the B3.4 logprobs source). The scheduler fills
    `tokens` (python ints) and stamps admit_step / finish_step; `states`
    (per-layer kv.LayerKV) lives only while the request runs."""

    def __init__(self, req_id, ids, max_new, eos=None, capture=None):
        #1. identity + inputs
        assert max_new >= 1, "a request must generate at least one token"
        self.id = req_id
        self.ids = ids
        self.max_new = max_new
        self.eos = eos
        self.capture = capture
        #2. progress, owned by the scheduler
        self.tokens = []
        self.states = None
        self.admit_step = None
        self.finish_step = None

    @property
    def done(self):
        #1. length cap — the only stop condition when eos is None
        #   (ignore_eos semantics, the B3.3 default)
        if len(self.tokens) >= self.max_new:
            return True
        #2. armed eos early-exit (B3.4 server, ignore_eos=false)
        return (self.eos is not None and len(self.tokens) > 0
                and self.tokens[-1] == self.eos)


class Engine:
    """Resident-model handle: the 66 per-layer weight dicts (possibly
    spread over GPUs — hidden-state hops are bitwise copies, B3.2), embed /
    final-norm / unembed weights, the verified config, and the per-layer
    page budget for global-KV allocation. Owns NO per-request state.
    prefill() and decode() transcribe model.generate_greedy's two step
    forms op for op (same functions, order, dtypes, devices) because the
    B3.3 gate is bitwise token identity with that loop; masks are pure
    selection, cached per (device, length) exactly like generate_greedy's
    per-device cache."""

    def __init__(self, layers, w_emb, w_en, w_fn, w_un, mc, num_pages):
        #1. weights + config; mask cache keyed (device, length)
        self.layers = layers
        self.w_emb, self.w_en, self.w_fn, self.w_un = w_emb, w_en, w_fn, w_un
        self.mc = mc
        self.num_pages = num_pages
        self._masks = {}
        self._pools = None       # per-layer batch pools (exp-0007)
        self._pool_batch = None
        self._graphrun = None    # CUDA-graphed decode step (exp-0012)

    def init_pools(self, max_batch):
        """exp-0007: per-layer batch pools for the batched decode
        attention core — kv.SwaPool (rings as rows) on the 55 SWA layers,
        kv.GlobalPool (shared pages + device table mirror) on the 11
        global layers. Total capacity equals max_batch per-seq LayerKVs,
        so the KV footprint at full occupancy is unchanged; idempotent
        per max_batch (the t_batched arms reuse one engine)."""
        if self._pools is not None and self._pool_batch == max_batch:
            return
        mc = self.mc
        self._pools = []
        for L in range(mc.num_layers):
            dev = self.layers[L]["attn_norm"].device
            if L in mc.global_layers:
                kv_heads = mc.g_kv_heads
                pool = pkv.GlobalPool(max_batch, self.num_pages,
                                      pkv.PAGE_SIZE, kv_heads, mc.head_dim,
                                      dev)
            else:
                kv_heads = mc.s_kv_heads
                pool = pkv.SwaPool(max_batch, mc.window, kv_heads,
                                   mc.head_dim, dev)
            #2. exp-0008: the layer's sconv tails ride the same slot ids
            pool.sconv = pkv.SconvPool(max_batch, mc.sconv_k,
                                       kv_heads * mc.head_dim, mc.hidden,
                                       dev)
            self._pools.append(pool)
        self._pool_batch = max_batch

    def new_states(self, slot=None):
        """Fresh per-layer recurrent state on each layer's own device.
        `slot` (scheduler admission, exp-0007) backs the caches on the
        batch pools so the batched decode core can index them — the
        per-seq append/gather code paths are unchanged either way
        (views over pool storage, kv.py)."""
        #1. LayerKV picks ring vs paged from the layer type (B3.1)
        if slot is None:
            return [pkv.LayerKV(self.mc, L,
                                self.layers[L]["attn_norm"].device,
                                num_pages=self.num_pages)
                    for L in range(self.mc.num_layers)]
        return [pkv.LayerKV(self.mc, L, self.layers[L]["attn_norm"].device,
                            num_pages=self.num_pages, pool=self._pools[L],
                            slot=slot)
                for L in range(self.mc.num_layers)]

    def release_states(self, states):
        """Return a retiring sequence's shared-pool pages (table order —
        deterministic given the schedule). Ring/sconv rows need no
        action: the next occupant overwrites them, and the batched core
        masks dead slots (kv.SwaPool docstring)."""
        #1. global layers hand their pages back to the shared free list
        for st in states:
            if st.is_global and st.cache.table_dev_row is not None:
                st.cache.free.extend(st.cache.table)
                st.cache.table = []

    def _mask_pos(self, dev, n):
        #1. (full, window, positions) per (device, length), built once —
        #   pure selection, bitwise = generate_greedy's own build (B2.6/7)
        key = (dev, n)
        if key not in self._masks:
            p = torch.arange(n, device=dev)
            dt = self.w_emb.dtype
            self._masks[key] = (
                pmodel.additive_causal_mask(p, p, dt),
                pmodel.additive_causal_mask(p, p, dt, window=self.mc.window),
                p)
        return self._masks[key]

    @torch.no_grad()
    def prefill(self, req):
        """Whole-prompt prefill populating req.states; returns the first
        greedy token (generate_greedy's prefill form, model.py)."""
        mc = self.mc
        #1. embed -> 66 x layer_prefill(state=...) across the layer devices
        _, h = pmodel.embed(req.ids[None], self.w_emb, self.w_en,
                            eps=mc.rms_eps)
        for L in range(mc.num_layers):
            dev = self.layers[L]["attn_norm"].device
            full, win, p = self._mask_pos(dev, req.ids.shape[0])
            h = pmodel.layer_prefill(
                h.to(dev), self.layers[L],
                full if L in mc.global_layers else win, p, mc, L,
                state=req.states[L])
        #2. last-position logits -> fp32 argmax over the unpadded vocab
        #   (the B2.11 comparison form, generate_greedy's _pick); optional
        #   fp32-row capture (B3.4 logprobs, generate_greedy's capture=
        #   pattern — a read-only CPU copy, token math untouched)
        row = pmodel.final_logits(h[:, -1], self.w_fn, self.w_un,
                                  mc.logits_mult, mc.vocab_unpadded,
                                  eps=mc.rms_eps)[0]
        lg = row.float()
        if req.capture is not None:
            req.capture.append(lg.cpu())
        return int(lg.argmax())

    @torch.no_grad()
    def decode(self, req):
        """One decode step: embed the previous token, run the 66 cached-
        decode layers at absolute position T-1+s (generate_greedy's decode
        form; s = tokens generated so far), return the next greedy token."""
        mc = self.mc
        s = len(req.tokens)
        #1. embed the previous token, 66 x layer_decode through req's state
        prev = torch.tensor([req.tokens[-1]], device=self.w_emb.device)
        _, h = pmodel.embed(prev, self.w_emb, self.w_en, eps=mc.rms_eps)
        for L in range(mc.num_layers):
            h = pmodel.layer_decode(
                h.to(self.layers[L]["attn_norm"].device), self.layers[L],
                req.states[L], req.ids.shape[0] - 1 + s, mc, L)
        #2. logits -> fp32 argmax + optional capture, same form as prefill
        row = pmodel.final_logits(h, self.w_fn, self.w_un, mc.logits_mult,
                                  mc.vocab_unpadded, eps=mc.rms_eps)[0]
        lg = row.float()
        if req.capture is not None:
            req.capture.append(lg.cpu())
        return int(lg.argmax())

    @torch.no_grad()
    def decode_batch(self, reqs):
        """One decode step for B co-resident sequences through ONE stack
        traversal (model.layer_decode_batch): every layer's weights are
        read once for the whole batch instead of once per sequence, and
        the per-step Python/launch cost of the 66-layer walk is paid once.
        Row order = the caller's list order (the scheduler passes running-
        list order = admission order), fixed for a given schedule; per-row
        results depend on the row COUNT through bf16 accumulation order
        (B3.1/B3.3), which is exactly the drift D13's envelope covers.
        Returns the B next greedy tokens in row order."""
        mc = self.mc
        pos = [r.ids.shape[0] - 1 + len(r.tokens) for r in reqs]
        slots = [r.states[0].slot for r in reqs]
        #0. CUDA-graphed step (exp-0012): the whole pooled traversal as
        #   per-device graph replays on static buffers — same pools,
        #   B padded to max_batch, key length padded to a 512 bucket
        #   (graphrun module docstring; PYENGINE_CUDAGRAPH=0 disables)
        if (self._pools is not None and None not in slots
                and len(reqs) <= self._pool_batch and graphrun.enabled()):
            if self._graphrun is None:
                self._graphrun = graphrun.GraphRunner(self)
            out = self._graphrun.step(reqs, pos, slots)
            return out
        #1. embed all previous tokens — one [B] lookup + rowwise norm
        prev = torch.tensor([r.tokens[-1] for r in reqs],
                            device=self.w_emb.device)
        _, h = pmodel.embed(prev, self.w_emb, self.w_en, eps=mc.rms_eps)
        #2. pooled path (exp-0007): per-step per-device shared indexing,
        #   built once and reused by every layer on that device; requests
        #   admitted without a pool slot fall back to the per-row core
        ctxs = {}
        if self._pools is not None and None not in slots:
            for L in range(mc.num_layers):
                dev = self.layers[L]["attn_norm"].device
                if dev not in ctxs:
                    ctxs[dev] = pmodel.decode_batch_ctx(pos, slots, mc, dev,
                                                        pkv.PAGE_SIZE)
        #3. 66 batched layers; the [B, hidden] rows hop devices together
        for L in range(mc.num_layers):
            dev = self.layers[L]["attn_norm"].device
            h = pmodel.layer_decode_batch(
                h.to(dev), self.layers[L],
                [r.states[L] for r in reqs], pos, mc, L,
                ctx=ctxs.get(dev),
                pool=self._pools[L] if ctxs else None)
        #4. pooled appends bypass RingKV/PagedKV.append — keep the host-
        #   side per-seq token counters truthful (release/gather safety)
        if ctxs:
            for r in reqs:
                for st in r.states:
                    st.cache.n += 1
        #5. one final-logits GEMM; per-row fp32 argmax + optional capture
        rows = pmodel.final_logits(h, self.w_fn, self.w_un, mc.logits_mult,
                                   mc.vocab_unpadded, eps=mc.rms_eps)
        out = []
        for i, req in enumerate(reqs):
            lg = rows[i].float()
            if req.capture is not None:
                req.capture.append(lg.cpu())
            out.append(int(lg.argmax()))
        return out


class Scheduler:
    """Iteration-level scheduler over one Engine. submit() queues a request
    with an arrival step (a deterministic stand-in for wall-clock arrival);
    each step() admits due arrivals into free slots FCFS (arrival step,
    then submission order), prefills them (an arrival's first token IS its
    token for that step), advances every previously-admitted running
    sequence one decode token, then retires finished sequences — calling
    on_retire(req) while req.states is still attached (the B3.4 completion
    seam / test pin) before freeing the KV. run() drains everything and
    returns {req_id: tokens}."""

    def __init__(self, engine, max_batch, on_retire=None):
        #1. config + queues; running is kept in admission order
        assert max_batch >= 1
        self.engine = engine
        self.max_batch = max_batch
        self.on_retire = on_retire
        self.waiting = []    # (arrival_step, submit_seq, request)
        self.running = []
        self.finished = []
        self.step_count = 0
        self.peak_running = 0
        self._seq = 0
        #2. batch pools + slot free list (exp-0007): admission takes the
        #   lowest free slot, retire returns it — deterministic; slot ids
        #   only pick pool rows, never batch order (= admission order)
        engine.init_pools(max_batch)
        self._free_slots = list(range(max_batch))

    def submit(self, req, arrival_step=0):
        """Queue a request; it becomes admissible at `arrival_step`."""
        #1. FCFS key = (arrival, submission order) — fully deterministic
        self.waiting.append((arrival_step, self._seq, req))
        self._seq += 1

    def step(self):
        """One scheduling iteration: admit -> one token per sequence ->
        retire. Per-sequence forwards are batch-1 exact (Engine), so the
        schedule changes only WHEN work happens, never its numerics."""
        #1. admit due arrivals FCFS while slots are free; prefill is the
        #   arrival's token for this step
        for entry in sorted(w for w in self.waiting
                            if w[0] <= self.step_count):
            if len(self.running) >= self.max_batch:
                break
            self.waiting.remove(entry)
            req = entry[2]
            req.states = self.engine.new_states(
                slot=self._free_slots.pop(0))
            req.admit_step = self.step_count
            self.running.append(req)
            req.tokens.append(self.engine.prefill(req))
        self.peak_running = max(self.peak_running, len(self.running))
        #2. one decode token for every sequence admitted in a PRIOR step —
        #   all decodable co-residents advance through ONE batched stack
        #   traversal; running-list (= admission) order fixes the row order
        todo = [req for req in self.running
                if req.admit_step < self.step_count and not req.done]
        if todo:
            for req, tok in zip(todo, self.engine.decode_batch(todo)):
                req.tokens.append(tok)
        #3. retire finished sequences; free their KV after the callback
        #   (shared-pool pages + batch slot go back to their free lists —
        #   insertion keeps the slot list sorted, so admission stays
        #   lowest-slot-first deterministic)
        still_running = []
        for req in self.running:
            if req.done:
                req.finish_step = self.step_count
                if self.on_retire is not None:
                    self.on_retire(req)
                if req.states[0].slot is not None:
                    self.engine.release_states(req.states)
                    bisect.insort(self._free_slots, req.states[0].slot)
                req.states = None
                self.finished.append(req)
            else:
                still_running.append(req)
        self.running = still_running
        self.step_count += 1

    def run(self, max_steps=1 << 20):
        """Drain the queues; returns {req_id: token list}."""
        #1. step until idle; a wedged schedule fails loud
        while self.waiting or self.running:
            if self.step_count >= max_steps:
                raise RuntimeError("scheduler exceeded max_steps")
            self.step()
        return {r.id: r.tokens for r in self.finished}
