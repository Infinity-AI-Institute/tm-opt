"""B3.4: OpenAI-compatible HTTP server over the B3.3 continuous-batching
scheduler (PROGRESS.md B3.4). Its two consumers define the contract:
harness/correctness.py (the B3.5 parity gate) and harness/benchmark.py
(B4.x) — both non-streamed /v1/completions.

Contract (docs/BASELINE_NOTES.md, verified live against vLLM):
  POST /v1/completions {model, prompt, max_tokens, temperature: 0,
                        logprobs: 0|1, seed, return_token_ids: bool,
                        ignore_eos: bool}
    prompt: one non-empty string, or one non-empty token-id array
    (list[int] — OpenAI completions spec, vLLM supports it; B3.4b: the
    D13 teacher-forced parity gate posts prompt ids + golden-prefix ids
    to eliminate detokenize/retokenize boundary effects). Id arrays are
    used as input ids directly, tokenization skipped; same length/
    capacity checks as strings.
    -> {id, object: "text_completion", created, model,
        choices: [{index, text,
                   token_ids,     # CHOICE level, present iff
                                  # return_token_ids (correctness.py
                                  # fails loud when absent)
                   logprobs: {tokens, token_logprobs, top_logprobs,
                              text_offset} | null,
                   finish_reason: "length" | "stop"}],
        usage: {prompt_tokens, completion_tokens, total_tokens}}
  GET /health -> 200 {"status": "ok", ...}

Design: ONE EngineLoop thread owns all CUDA work (the B3.3 Scheduler:
FCFS admission at step boundaries, prefill-on-admit, one decode per
running sequence per step, retire via on_retire). HTTP handler threads
only tokenize, enqueue, block on a per-request Event, and format the
response; wall-clock arrivals map onto step boundaries by submitting at
the loop's current step_count (the scheduler.py module note). Greedy only:
temperature must be exactly 0 — the canonical workloads are greedy (P1) —
and every knob this engine cannot serve exactly (sampling, stream, echo,
stop, n>1) fails loud with 400 rather than silently changing semantics.
Logprobs come from the Request capture hook (fp32 logits rows,
generate_greedy's capture= pattern): logprob(tid) = row[tid] -
logsumexp(row), computed OFF the engine thread at response time. Capture
memory is one fp32 vocab row (~0.8 MB) per generated token per
logprobs-requesting request — fine for the 64-token parity gate; the
canonical benchmark requests no logprobs.

Tokenization: transformers AutoTokenizer(trust_remote_code) on the model
dir — the exact call the B2/B3 tests baselined; equivalence with vLLM's
`--tokenizer-mode inkling` is exactly what the B3.5 gate referees.
Resident load mirrors t_b3.t_batch #2 (itself mirroring the
BLOCKED-frozen t_decode #2-#3) — dedup all three into loader.py once
B3.2 is adjudicated."""
import argparse
import json
import queue
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

from engine.pyengine import config as pcfg
from engine.pyengine import kv as pkv
from engine.pyengine import loader
from engine.pyengine import scheduler as psched

MODEL_DIR = "/workspace/models/inkling-nvfp4"


def chosen_logprobs(rows, token_ids):
    """Log-softmax value of each chosen token from its captured fp32
    logits row (generate_greedy / Engine capture= form). Pure fp32 CPU
    math — deterministic, so the t_b3 server gate can require exact
    equality between the served values and the same helper applied to a
    batch-1 generate_greedy capture."""
    #1. logprob = row[tid] - logsumexp(row), one scalar per step
    out = []
    for row, tid in zip(rows, token_ids):
        out.append(float(row[tid] - torch.logsumexp(row, 0)))
    return out


def build_resident(model_dir=MODEL_DIR, num_pages=1024):
    """Stream the checkpoint into a layer-split resident model across all
    visible GPUs and wrap it in a scheduler.Engine (routed experts stay
    checkpoint-packed, B3.2 loader.PackedExperts). Returns (engine,
    splits). num_pages is the PER-SEQUENCE page budget of each global
    layer: capacity = num_pages * kv.PAGE_SIZE tokens (1024 -> the 16384
    serve window). Mirrors t_b3.t_batch #2 — dedup into loader.py once
    B3.2 is adjudicated. Wall ~105 s page-cache warm (B1.6/B3.2)."""
    #1. proven test-side loader helpers (deferred import: heavy module;
    #   see the dedup note above)
    from engine.pyengine.tests.t_b2 import _disk, load_layer_weights
    t0 = time.time()
    mc = pcfg.load_verified(model_dir)
    idx = loader.build_shard_index(model_dir)
    hdr = loader.read_headers(idx)
    dm = loader.build_dtype_map(idx, hdr)
    ndev = torch.cuda.device_count()
    #2. contiguous layer -> GPU split, proportional by header bytes
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
    print(f"[server] layer split {splits} over {ndev} GPUs "
          f"({total_b / (1 << 30):.0f} GiB layers)", file=sys.stderr,
          flush=True)
    #3. stream the 66 layers, then embed/embed_norm (first device) and
    #   final norm/unembed (last device)
    layers = []
    for L in range(mc.num_layers):
        layers.append(load_layer_weights(idx, hdr, dm, mc, L, dev=devs[L],
                                         pack_moe=True))
        if L % 16 == 15 or L == mc.num_layers - 1:
            print(f"[server] loaded layer {L + 1}/{mc.num_layers}, "
                  f"{time.time() - t0:.0f} s", file=sys.stderr, flush=True)
    w_emb = _disk(idx, "model.llm.embed.weight").to(devs[0])
    w_en = _disk(idx, "model.llm.embed_norm.weight").to(devs[0])
    w_fn = _disk(idx, "model.llm.norm.weight").to(devs[-1])
    w_un = _disk(idx, "model.llm.unembed.weight").to(devs[-1])
    #4. the engine handle owns weights + config only, never request state
    return psched.Engine(layers, w_emb, w_en, w_fn, w_un, mc, num_pages), \
        splits


class EngineLoop:
    """The single CUDA thread: drains wall-clock arrivals into the B3.3
    Scheduler at step boundaries and steps while work exists. HTTP threads
    talk to it only through submit() (thread-safe queue) and the
    per-request done_event set by the retire callback.

    COHORT-FORMING ADMISSION (exp-0017). Arrivals land in a pool, not
    straight in the scheduler: the scheduler admits every due arrival the
    moment a slot frees, so under the bench's per-completion arrival
    pattern prompts reach Scheduler._prefill_groups ONE AT A TIME and
    every batched prefill group degenerates to a singleton (the measured
    exp-0014 failure — DEADENDS post-mortem: 'grouping must be triggered
    by the scheduler's own cohort-retire signal, not by a wall-clock
    accumulation window racing HTTP arrivals'). The pool releases when
    (a) nothing is generating — a sequential singleton stream, the D13
    gate and cold start keep the batch-1-exact per-seq path and never
    stall; (b) PREFILL_COHORT prompts have gathered AND that many slots
    are free, i.e. a retiring cohort has actually freed room for them —
    released together they prefill as ONE group, then run and retire in
    lockstep, so their replacements co-arrive and the cohort structure
    is self-sustaining after the first round; or (c) COHORT_HOLD_S has
    passed (safety valve — bootstrap and ragged tails admit what fits).
    Only the number of free slots and the pool size gate the release;
    wall clock enters solely through the valve. Scheduler.step() stays a
    pure function of its input schedule (the t_b3 determinism arm pins
    arrival steps and drives the scheduler directly), so same-schedule
    bitwise repeatability is untouched."""

    #1. cohort target: 6 co-prefilled canonical prompts put ~124
    #   (token, expert) rows on each of the 256 experts (6 x T=736 x
    #   top6 / 256), i.e. ONE 128-row native-MMA M tile per expert —
    #   the tile the W4A4 grouped GEMM (exp-0015a/0016/0015b) already
    #   pays in full for the ~17 rows a single sequence routes there
    PREFILL_COHORT = 6
    COHORT_HOLD_S = 4.0

    def __init__(self, engine, max_batch):
        #1. scheduler + inbox + cohort pool + lifecycle state
        self.sched = psched.Scheduler(engine, max_batch,
                                      on_retire=self._retire)
        self.inbox = queue.Queue()
        self._pend = []
        self._pend_since = 0.0
        self.error = None
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run,
                                       name="tmopt-engine", daemon=True)

    def start(self):
        #1. spawn the engine thread
        self.thread.start()

    def stop(self, timeout=60):
        #1. flag + join; True when the thread exited
        self._stop.set()
        self.thread.join(timeout)
        return not self.thread.is_alive()

    def submit(self, req):
        """HTTP-thread entry. req must carry a threading.Event at
        req.done_event; req.ids may still be on CPU (moved onto the embed
        device by the engine thread, keeping ALL CUDA work there)."""
        #1. hand off; the loop maps it onto the next step boundary
        self.inbox.put(req)

    def _retire(self, req):
        #1. wake the waiting HTTP thread — tokens are complete; the
        #   scheduler frees the KV right after this callback (B3.3)
        req.done_event.set()

    def _admit(self, req):
        #1. wall-clock arrival -> CURRENT step boundary (scheduler.py note)
        req.ids = req.ids.to(self.sched.engine.w_emb.device)
        self.sched.submit(req, arrival_step=self.sched.step_count)

    def _pool(self, req):
        #1. hold an arrival for cohort formation; the wait clock starts on
        #   the pool's first member
        if not self._pend:
            self._pend_since = time.time()
        self._pend.append(req)

    def _release_count(self):
        """How many pooled arrivals to hand the scheduler this iteration —
        the class docstring's (a)/(b)/(c). Never more than the free slots:
        a request the scheduler cannot admit now would sit in its waiting
        list and be admitted alone later, which is the fragmentation this
        pool exists to prevent."""
        #1. (a) nothing is generating -> admit everything, never stall
        if not self.sched.running and not self.sched.waiting:
            return len(self._pend)
        free = (self.sched.max_batch - len(self.sched.running)
                - len(self.sched.waiting))
        if free <= 0:
            return 0
        #2. (b) a cohort has gathered and a retiring cohort's slots are
        #   free for it -> release (the excess over PREFILL_COHORT rides
        #   along; Scheduler._prefill_groups repartitions under its own
        #   padded-scores budget)
        if len(self._pend) >= self.PREFILL_COHORT >= 1 and \
                free >= self.PREFILL_COHORT:
            return min(len(self._pend), free)
        #3. (c) safety valve
        if time.time() - self._pend_since >= self.COHORT_HOLD_S:
            return min(len(self._pend), free)
        return 0

    def _run(self):
        #1. drain arrivals into the cohort pool + step while work exists;
        #   when idle, block briefly on the inbox instead of spinning
        try:
            while not self._stop.is_set():
                while True:
                    try:
                        self._pool(self.inbox.get_nowait())
                    except queue.Empty:
                        break
                n = self._release_count() if self._pend else 0
                if n:
                    for req in self._pend[:n]:
                        self._admit(req)
                    del self._pend[:n]
                    self._pend_since = time.time()
                if self.sched.waiting or self.sched.running:
                    self.sched.step()
                    continue
                try:
                    self._pool(self.inbox.get(timeout=0.05))
                except queue.Empty:
                    pass
        except Exception:
            #2. fail loud: record the traceback, wake every in-flight
            #   waiter (their handlers answer 500 — finish_step is None)
            self.error = traceback.format_exc()
            sys.stderr.write(self.error)
            for r in self._pend:
                r.done_event.set()
            for _, _, r in self.sched.waiting:
                r.done_event.set()
            for r in self.sched.running:
                r.done_event.set()
            while True:
                try:
                    self.inbox.get_nowait().done_event.set()
                except queue.Empty:
                    break


class _Ctx:
    """Shared handler context: tokenizer (+lock — HF fast tokenizers are
    not proven thread-safe under concurrent encode), engine loop, verified
    config, request-id counter, per-sequence KV capacity."""

    def __init__(self, tok, loop, mc, model_name, max_seq):
        #1. shared objects + locks
        self.tok = tok
        self.tok_lock = threading.Lock()
        self.loop = loop
        self.mc = mc
        self.model_name = model_name
        self.max_seq = max_seq
        self._n = 0
        self._id_lock = threading.Lock()

    def next_id(self):
        #1. monotonically increasing request id
        with self._id_lock:
            self._n += 1
            return self._n


def _validate(body):
    """Contract checks for /v1/completions; returns (err, prompt, max_new,
    logprobs, return_token_ids, ignore_eos) with err=None on success;
    prompt comes back as a str (tokenize) or list[int] (token-id array,
    used directly — B3.4b). Anything this engine cannot serve EXACTLY is
    a loud 400 (greedy-only, no stream/echo/stop/n>1; malformed prompt
    lists — empty, mixed, bool, float, nested batch — never guess) —
    never a silent semantic change."""
    #1. prompt: one non-empty string OR one non-empty token-id array
    #   (a singleton [str] is unwrapped; list[int] skips tokenization,
    #   B3.4b — OpenAI spec form the D13 teacher-forced gate sends)
    p = body.get("prompt")
    if isinstance(p, list) and len(p) == 1 and isinstance(p[0], str):
        p = p[0]
    if isinstance(p, list):
        if not p or not all(isinstance(t, int) and not isinstance(t, bool)
                            for t in p):
            return ("prompt list must be a non-empty token-id array "
                    "(every element an int)",) + (None,) * 5
    elif not isinstance(p, str) or not p:
        return ("prompt must be one non-empty string or one token-id "
                "array",) + (None,) * 5
    #2. greedy-only: temperature must be present and exactly 0
    t = body.get("temperature")
    if (isinstance(t, bool) or not isinstance(t, (int, float))
            or float(t) != 0.0):
        return (f"greedy-only engine: temperature must be 0 (got {t!r})",
                ) + (None,) * 5
    #3. max_tokens: int >= 1 (OpenAI default 16)
    m = body.get("max_tokens", 16)
    if isinstance(m, bool) or not isinstance(m, int) or m < 1:
        return (f"max_tokens must be an int >= 1 (got {m!r})",) + (None,) * 5
    #4. logprobs: null, 0, or 1 (top-1 only — greedy top-1 IS the choice)
    lp = body.get("logprobs")
    if lp is not None and (isinstance(lp, bool) or lp not in (0, 1)):
        return (f"logprobs supports 0 or 1 only (got {lp!r})",) + (None,) * 5
    #5. unsupported knobs fail loud
    for k in ("stream", "echo", "suffix", "stop"):
        if body.get(k):
            return (f"{k} unsupported",) + (None,) * 5
    if body.get("n", 1) != 1 or body.get("best_of", 1) != 1:
        return ("n/best_of must be 1",) + (None,) * 5
    #6. flags (seed is accepted and ignored: greedy is deterministic)
    return (None, p, m, lp, bool(body.get("return_token_ids")),
            bool(body.get("ignore_eos")))


def _completion_json(ctx, body, req, want_lp, want_ids):
    """Build the /v1/completions response body from a retired Request
    (BASELINE_NOTES contract: token_ids at CHOICE level iff
    return_token_ids; benchmark.py reads usage.completion_tokens)."""
    toks = req.tokens
    #1. detokenize: full text skips specials (vLLM's default), per-token
    #   strings keep them (so an eos step is visible in logprobs.tokens)
    with ctx.tok_lock:
        text = ctx.tok.decode(toks, skip_special_tokens=True)
        strs = [ctx.tok.decode([t]) for t in toks]
    #2. finish reason: armed eos generated -> "stop", else length cap
    fin = ("stop" if (req.eos is not None and toks and toks[-1] == req.eos)
           else "length")
    choice = {"index": 0, "text": text, "logprobs": None,
              "finish_reason": fin}
    if want_ids:
        choice["token_ids"] = toks
    #3. logprobs from the captured fp32 rows; rows freed after use.
    #   text_offset = cumulative offsets over the per-token strings
    #   (nothing in harness/ consumes it; provided for OpenAI shape)
    if want_lp is not None:
        lps = chosen_logprobs(req.capture, toks)
        offs, o = [], 0
        for s in strs:
            offs.append(o)
            o += len(s)
        choice["logprobs"] = {
            "tokens": strs, "token_logprobs": lps,
            "top_logprobs": ([{s: l} for s, l in zip(strs, lps)]
                             if want_lp >= 1 else [{} for _ in strs]),
            "text_offset": offs}
        req.capture = None
    #4. envelope + usage
    T = int(req.ids.shape[0])
    return {"id": f"cmpl-{req.id}", "object": "text_completion",
            "created": int(time.time()),
            "model": body.get("model") or ctx.model_name,
            "choices": [choice],
            "usage": {"prompt_tokens": T, "completion_tokens": len(toks),
                      "total_tokens": T + len(toks)}}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, obj):
        #1. always JSON with explicit Content-Length (keep-alive safe)
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        #1. health probe; anything else is 404
        if self.path == "/health":
            self._send(200, {"status": "ok",
                             "model": self.server.ctx.model_name})
        else:
            self._send(404, {"error": f"unknown path {self.path}"})

    def do_POST(self):
        ctx = self.server.ctx
        if self.path != "/v1/completions":
            #0. drain the body before answering — on a keep-alive
            #   connection unread bytes would be parsed as the next
            #   request line (observed as a spurious 'Bad request syntax')
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self._send(404, {"error": f"unknown path {self.path}"})
            return
        try:
            #1. parse the JSON body
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                assert isinstance(body, dict)
            except Exception:
                self._send(400, {"error": "body is not a JSON object"})
                return
            #2. contract checks -> 400
            err, prompt, max_new, want_lp, want_ids, ignore_eos = \
                _validate(body)
            if err:
                self._send(400, {"error": err})
                return
            #3. input ids: a token-id-array prompt is used directly
            #   (tokenization skipped, B3.4b) after a range check against
            #   the embed table (mc.vocab rows) — an out-of-range id
            #   would device-assert inside the embed gather and kill the
            #   engine thread, so it must 400 here; string prompts
            #   tokenize on CPU (under the tokenizer lock). Length/
            #   capacity checks below are identical for both forms.
            if isinstance(prompt, list):
                bad = next((t for t in prompt
                            if not 0 <= t < ctx.mc.vocab), None)
                if bad is not None:
                    self._send(400, {"error": f"prompt token id {bad} "
                                              f"outside [0, "
                                              f"{ctx.mc.vocab})"})
                    return
                ids = torch.tensor(prompt, dtype=torch.long)
            else:
                with ctx.tok_lock:
                    ids = ctx.tok(prompt, return_tensors="pt").input_ids[0]
            if int(ids.shape[0]) + max_new > ctx.max_seq:
                self._send(400, {"error": f"prompt {int(ids.shape[0])} + "
                                          f"max_tokens {max_new} exceeds "
                                          f"capacity {ctx.max_seq}"})
                return
            #4. submit onto the engine loop; block until retired (or the
            #   engine died — finish_step stays None -> 500)
            req = psched.Request(ctx.next_id(), ids, max_new,
                                 eos=None if ignore_eos else ctx.mc.eos,
                                 capture=[] if want_lp is not None else None)
            req.done_event = threading.Event()
            ctx.loop.submit(req)
            req.done_event.wait()
            if req.finish_step is None:
                self._send(500, {"error": f"engine died:\n{ctx.loop.error}"})
                return
            #5. format the response off the engine thread
            self._send(200, _completion_json(ctx, body, req, want_lp,
                                             want_ids))
        except BrokenPipeError:
            pass
        except Exception:
            try:
                self._send(500, {"error": traceback.format_exc()})
            except Exception:
                pass


class _BacklogHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer with a deep listen backlog (see serve())."""
    request_queue_size = 256


def serve(eng, tok, host="127.0.0.1", port=8200, max_batch=64,
          model_name=MODEL_DIR):
    """Start the engine-loop thread and build the HTTP server (bound, not
    yet serving). Caller runs httpd.serve_forever() — the CLI blocks on
    it, the t_b3 server test runs it in a thread. Returns (httpd, loop)."""
    #1. engine loop thread — all CUDA work lives there
    loop = EngineLoop(eng, max_batch)
    loop.start()
    #2. threaded HTTP server sharing one context. Listen backlog raised
    #   off socketserver's default 5: the bench opens its whole
    #   concurrency at warmup in one blast, and a 5-deep accept queue
    #   dropped three of exp-0014's 64 bench threads before they wrote a
    #   status line (-4% headline on a run whose engine was fine) — the
    #   exp-0014 post-mortem asks every later engine change to carry
    #   this. Measurement hygiene only: the backlog is the kernel's
    #   pre-accept queue depth and cannot make a served request faster.
    httpd = _BacklogHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    httpd.ctx = _Ctx(tok, loop, eng.mc, model_name,
                     eng.num_pages * pkv.PAGE_SIZE)
    return httpd, loop


def main():
    """CLI: load the resident model, serve forever. B3.5 points
    harness/correctness.py at this (default port 8200); long-running
    instances belong in the tmux `serve` session (CLAUDE.md rule 7)."""
    #1. args
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8200)
    ap.add_argument("--max-batch", type=int, default=64)
    ap.add_argument("--num-pages", type=int, default=1024,
                    help="per-global-layer page budget PER SEQUENCE; "
                         "1024*16 = the 16384-token serve window")
    a = ap.parse_args()
    #2. resident engine + tokenizer
    from transformers import AutoTokenizer
    eng, _ = build_resident(a.model, num_pages=a.num_pages)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    #3. serve forever
    httpd, loop = serve(eng, tok, host=a.host, port=a.port,
                        max_batch=a.max_batch, model_name=a.model)
    print(f"[server] ready on http://{a.host}:{a.port} (max_batch "
          f"{a.max_batch}, capacity {httpd.ctx.max_seq} tok/seq)",
          flush=True)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        loop.stop()


if __name__ == "__main__":
    main()
