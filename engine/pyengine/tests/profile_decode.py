"""exp-0005 (STEP-B ruling #1, 2026-07-30): instrumentation-only component
profile of the batched decode step. The engine hot path is UNTOUCHED — this
script is the experiment's entire diff; it monkey-patches timing wrappers
around the model/scheduler component functions and drives Engine.decode_batch
in-process (GPUs 4-7, CUDA_VISIBLE_DEVICES set by the caller).

Why two batch arms (B=8 and B=64): every timeboxed(conc=64) ledger row to
date — including the 15.3 tok/s best — ran the SERVER at its default
--max-batch 8 (worker serve_cmd `python -m engine.pyengine.server --port
8200`; verified live on the exp-0004 process table 2026-07-30). The bench's
64 client threads only kept 8 engine slots saturated, so state-of-world
finding (a) "batch-amortizable work is nearly gone" was inferred from a run
whose ENGINE batch never left 8. The B=64 arm measures the untested
counterfactual; the B=8 arm profiles the operating point behind every
existing row.

Attribution model: innermost-region self time via a region stack (the k/v
sconvs called inside attn_decode_batch count as `sconv`, not `attn_core`;
rmsnorm anywhere counts as `rmsnorm`). Three passes per arm:
  A (sync-region)  torch.cuda.synchronize(all devs) at region boundaries —
                   honest serialized per-region walls (GPU compute + that
                   region's own dispatch); sum ~= step wall (check printed).
  B (no-sync)      perf_counter only, one all-dev sync per step in the
                   driver — host enqueue time per region. If sum(host) ~=
                   step wall, the step is dispatch/host-bound (the CUDA-
                   graph case, attack #2); the residual is GPU tail.
  C (torch.profiler, 1 step) — cudaLaunchKernel count + total self CUDA
                   time -> GPU-busy fraction; top kernels table to stderr.
Known hot-loop syncs the passes will surface (code-reading, to verify):
kv.PagedKV._flat_slots .tolist() on every global-layer append AND gather
(2 x 11 layers x B per step), layer 2's eager moe_experts
unique().tolist(), per-seq torch.tensor(...) H2D in attn cores.

Prompts follow the canonical decode_heavy generator distribution
(harness/benchmark.py make_prompts: seeded, ISL 512-1024, copied verbatim
here so nothing in harness/ is imported or touched). Run:
  CUDA_VISIBLE_DEVICES=4,5,6,7 python -m engine.pyengine.tests.profile_decode
Output: human tables on stderr, one PROFILE_JSON line on stdout.
"""
import argparse
import json
import random
import sys
import time

import torch

from engine.pyengine import model as pmodel
from engine.pyengine import scheduler as psched
from engine.pyengine import server as pserver


def make_prompts(isl, range_ratio, seed, n):
    """Verbatim copy of harness/benchmark.py make_prompts (read-only
    harness stays unimported); same seeded ISL distribution as the bench."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        L = rng.randint(int(isl * range_ratio), isl)
        out.append(f"[req {i}] " + "word " * max(L - 4, 1))
    return out


def sync_all():
    #1. work hops across all visible devices; sync every one
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


class Prof:
    """Region-stack profiler: self time per region (children subtracted).
    sync=True -> pass A semantics, sync=False -> pass B semantics."""

    def __init__(self):
        self.sync = False
        self.enabled = False
        self.stack = []
        self.acc = {}     # region -> [self_seconds, calls]
        self.last_exit = {}  # region -> perf_counter at last _exit
        self._orig = []

    def _enter(self, region):
        if self.sync:
            sync_all()
        self.stack.append([region, time.perf_counter(), 0.0])

    def _exit(self):
        if self.sync:
            sync_all()
        region, t0, child = self.stack.pop()
        now = time.perf_counter()
        dt = now - t0
        a = self.acc.setdefault(region, [0.0, 0])
        a[0] += dt - child
        a[1] += 1
        self.last_exit[region] = now
        if self.stack:
            self.stack[-1][2] += dt

    def wrap(self, obj, name, region):
        orig = getattr(obj, name)
        prof = self

        def wrapped(*a, **k):
            if not prof.enabled:
                return orig(*a, **k)
            prof._enter(region)
            try:
                return orig(*a, **k)
            finally:
                prof._exit()

        setattr(obj, name, wrapped)
        self._orig.append((obj, name, orig))

    def reset(self):
        self.acc = {}

    def snapshot(self, steps):
        return {r: {"ms_per_step": 1e3 * s / steps, "calls_per_step": c / steps}
                for r, (s, c) in sorted(self.acc.items(),
                                        key=lambda kv: -kv[1][0])}


def run_arm(eng, reqs, prof, label, n_warm, n_steps, do_profiler):
    """Profile decode_batch on `reqs`: warmup, pass B (no-sync), pass A
    (sync), optional pass C (torch.profiler). Returns the arm's report."""
    B = len(reqs)

    def step():
        toks = eng.decode_batch(reqs)
        for r, t in zip(reqs, toks):
            r.tokens.append(t)

    #1. warmup (JIT specializations, allocator steady state)
    prof.enabled = False
    for _ in range(n_warm):
        step()
    sync_all()

    #2. pass B: host-enqueue attribution, one sync per step in the driver.
    #   enq = time from step start until the `logits` region exits (= every
    #   kernel of the step enqueued); wall - enq = argmax loop + GPU drain
    prof.enabled, prof.sync = True, False
    prof.reset()
    walls_b, enq_b = [], []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        step()
        sync_all()
        walls_b.append(time.perf_counter() - t0)
        enq_b.append(prof.last_exit.get("logits", t0) - t0)
    pass_b = prof.snapshot(n_steps)
    wall_b = sum(walls_b) / n_steps
    enq = sum(enq_b) / n_steps
    host_total = sum(v["ms_per_step"] for v in pass_b.values())

    #3. pass A: serialized region walls
    prof.sync = True
    prof.reset()
    walls_a = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        step()
        walls_a.append(time.perf_counter() - t0)
    pass_a = prof.snapshot(n_steps)
    wall_a = sum(walls_a) / n_steps
    prof.enabled = False

    #4. pass C: launch count + GPU busy for one step
    pass_c = None
    if do_profiler:
        try:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CPU,
                                     ProfilerActivity.CUDA]) as tp:
                step()
            ka = tp.key_averages()
            launches = sum(e.count for e in ka
                           if e.key in ("cudaLaunchKernel", "cuLaunchKernel",
                                        "cudaLaunchKernelExC"))
            busy_us = 0.0
            for e in ka:
                v = getattr(e, "self_device_time_total", None)
                if v is None:
                    v = getattr(e, "self_cuda_time_total", 0)
                if getattr(e, "device_type", None) is not None and \
                        "CUDA" in str(e.device_type):
                    busy_us += v
            pass_c = {"launches_per_step": launches,
                      "gpu_busy_ms": busy_us / 1e3}
            print(f"\n[{label}] top kernels (1 profiled step):",
                  file=sys.stderr)
            try:
                print(ka.table(sort_by="self_cuda_time_total", row_limit=12),
                      file=sys.stderr)
            except Exception:
                pass
        except Exception as e:  # profiler failure must not kill the run
            pass_c = {"error": repr(e)}

    #5. human table
    mean_ctx = sum(int(r.ids.shape[0]) + len(r.tokens) for r in reqs) / B
    print(f"\n[{label}] B={B} steps={n_steps} mean_ctx={mean_ctx:.0f} "
          f"| passA step {wall_a * 1e3:.1f} ms | passB step "
          f"{wall_b * 1e3:.1f} ms (enqueue {enq * 1e3:.1f} ms = "
          f"{100 * enq / wall_b:.0f}%, host-attr {host_total:.1f} ms) "
          f"| aggregate {B / wall_b:.1f} tok/s", file=sys.stderr)
    print(f"  {'region':12s} {'A ms(sync)':>11s} {'B ms(host)':>11s} "
          f"{'calls':>6s}", file=sys.stderr)
    for r in sorted(pass_a, key=lambda r: -pass_a[r]["ms_per_step"]):
        b = pass_b.get(r, {"ms_per_step": 0.0})
        print(f"  {r:12s} {pass_a[r]['ms_per_step']:11.1f} "
              f"{b['ms_per_step']:11.1f} "
              f"{pass_a[r]['calls_per_step']:6.0f}", file=sys.stderr)

    return {"B": B, "steps": n_steps, "mean_ctx": mean_ctx,
            "step_wall_ms_sync": wall_a * 1e3,
            "step_wall_ms_nosync": wall_b * 1e3,
            "enqueue_ms_nosync": enq * 1e3,
            "host_ms_nosync": host_total,
            "aggregate_tok_s": B / wall_b,
            "pass_a_sync_ms": pass_a, "pass_b_host_ms": pass_b,
            "pass_c": pass_c}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=pserver.MODEL_DIR)
    ap.add_argument("--conc", type=int, default=64)
    ap.add_argument("--num-pages", type=int, default=1024,
                    help="production per-seq global-layer page budget")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--steps-b8", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=2)
    a = ap.parse_args()

    #1. resident model, exactly the server's build (layer split, packing)
    from transformers import AutoTokenizer
    eng, splits = pserver.build_resident(a.model, num_pages=a.num_pages)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

    #2. canonical-distribution prompts -> Requests; per-seq prefill (timed:
    #   admission serialization is part of the max-batch story)
    cfg = {"isl": 1024, "range_ratio": 0.5, "seed": 1001}  # decode_heavy
    prompts = make_prompts(cfg["isl"], cfg["range_ratio"], cfg["seed"],
                           a.conc)
    reqs, pre_walls = [], []
    for i, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt").input_ids[0].to(eng.w_emb.device)
        r = psched.Request(i, ids, max_new=1 << 20)
        r.states = eng.new_states()
        t0 = time.perf_counter()
        r.tokens.append(eng.prefill(r))
        sync_all()
        pre_walls.append(time.perf_counter() - t0)
        reqs.append(r)
        if i % 16 == 15:
            print(f"[profile] prefilled {i + 1}/{a.conc}, "
                  f"last {pre_walls[-1]:.2f} s", file=sys.stderr, flush=True)
    pre = {"mean_s": sum(pre_walls) / len(pre_walls),
           "min_s": min(pre_walls), "max_s": max(pre_walls),
           "mean_isl": sum(int(r.ids.shape[0]) for r in reqs) / len(reqs)}
    print(f"[profile] prefill mean {pre['mean_s']:.2f} s "
          f"(ISL~{pre['mean_isl']:.0f})", file=sys.stderr, flush=True)

    #3. instrument: innermost-region self-time wrappers
    prof = Prof()
    for name, region in [("sconv_decode_batch", "sconv"),
                         ("rmsnorm", "rmsnorm"),
                         ("attn_decode_batch", "attn_core"),
                         ("moe_gate", "gate"),
                         ("moe_experts", "experts"),
                         ("moe_shared", "shared"),
                         ("dense_mlp", "dense_mlp"),
                         ("embed", "embed"),
                         ("final_logits", "logits"),
                         ("layer_decode_batch", "layer_glue")]:
        prof.wrap(pmodel, name, region)
    prof.wrap(psched.Engine, "decode_batch", "step_glue")

    #4. arms: B=conc (the untested counterfactual), then B=8 (the operating
    #   point behind every timeboxed ledger row to date)
    with torch.no_grad():
        arm64 = run_arm(eng, reqs, prof, f"B={a.conc}", a.warmup, a.steps,
                        do_profiler=True)
        arm8 = run_arm(eng, reqs[:8], prof, "B=8", 2, a.steps_b8,
                       do_profiler=True)

    out = {"experiment": "exp-0005-step-profile", "conc": a.conc,
           "num_pages": a.num_pages, "layer_split": splits,
           "prefill": pre, "arm_b64": arm64, "arm_b8": arm8}
    print("PROFILE_JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
