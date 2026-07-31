"""exp-0018 evidence: prefill/decode traversals pipelined across the
layer-split GPUs.

The engine shards the 66 layers into 4 contiguous per-device segments
(server.build_resident), so ONE traversal keeps exactly one GPU busy and
idles the other three. Every device->host sync inside a traversal pins the
host to that device's progress and stops the next traversal from being
enqueued, which is why the live bench shows a marching 100/0/0/0
utilization wave (/workspace/logs/t_pipeline_bubble_exp0018_util.log).
This script measures the two arms on the real resident model:

  arm a  EQUIVALENCE — same schedule, _PIPELINE off vs on. Per-sequence
         numerics are untouched by the change (identical kernels, shapes
         and op order; only host enqueue interleaving moves), so the token
         streams must be EXACTLY equal, not merely envelope-close.
  arm b  WALL — the step that owns both a grouped prefill and a decode
         traversal, timed in both arms. That step is the whole mechanism:
         serial it costs 4 prefill segments + 4 decode segments, pipelined
         it should cost 4 prefill segments + 1.
  arm c  DETERMINISM — two identical pipelined runs, bitwise (the
         t_b3-adapted worker arm).

Run: CUDA_VISIBLE_DEVICES=4,5,6,7 python -m engine.pyengine.tests\
.t_pipeline_prefill
"""
import sys
import time

import torch

from engine.pyengine import scheduler as psched
from engine.pyengine.server import build_resident

MAX_BATCH = 64
# the graphed decode step always replays at B = max_batch (dead rows
# padded, exp-0012), so its wall is independent of how many rows are
# actually live — 8 residents buy the real decode traversal cheaply
RESIDENT = 8            # rows already decoding when the cohort lands
COHORT = 6              # exp-0017's PREFILL_COHORT
LENS = [745, 700, 800, 745, 690, 760]   # right-padding is exercised
MAX_NEW = 4


def _prompts(n, base_len, seed):
    #1. deterministic pseudo-prompts at canonical-ISL scale; the token
    #   VALUES are irrelevant to a wall measurement, the shapes are not
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1, 190000, (base_len[i % len(base_len)],),
                          generator=g, dtype=torch.long)
            for i in range(n)]


def _run(eng, dev0, pipeline, warm=False):
    """One full schedule under one arm; returns (tokens, step_walls)."""
    #1. the arm switch is the module global Scheduler.step reads
    psched._PIPELINE = pipeline
    sched = psched.Scheduler(eng, MAX_BATCH)
    #   the warm arm MUST use the same length class: the graphed step's
    #   key-length bucket is a monotone high-water mark, so capturing at
    #   1024 during warm-up keeps a recapture out of the timed arms
    n_res, n_new = (2, 2) if warm else (RESIDENT, COHORT)
    lens = LENS[:2] if warm else LENS
    ids = _prompts(n_res + n_new, lens, 1001)
    for i in range(n_res):
        sched.submit(psched.Request(f"r{i}", ids[i].to(dev0), MAX_NEW),
                     arrival_step=0)
    for i in range(n_res, n_res + n_new):
        sched.submit(psched.Request(f"c{i}", ids[i].to(dev0), MAX_NEW),
                     arrival_step=1)
    #2. step until drained, timing each step end to end (a step already
    #   ends on a host sync — the decode argmax readback)
    walls = []
    while sched.waiting or sched.running:
        t0 = time.perf_counter()
        sched.step()
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)
        walls.append(time.perf_counter() - t0)
    return {r.id: list(r.tokens) for r in sched.finished}, walls


def main():
    #1. one resident load for every arm
    t0 = time.time()
    eng, splits = build_resident()
    dev0 = eng.w_emb.device
    print(f"[t0018] resident in {time.time() - t0:.0f} s, layer split "
          f"{splits}", flush=True)

    #2. warm the JIT/autotune caches (first CUTE DSL prefill compiles)
    t0 = time.time()
    _run(eng, dev0, True, warm=True)
    _run(eng, dev0, False, warm=True)
    print(f"[t0018] warm-up {time.time() - t0:.0f} s", flush=True)

    #3. arms
    tok_ser, w_ser = _run(eng, dev0, False)
    tok_pipe, w_pipe = _run(eng, dev0, True)
    tok_pipe2, _ = _run(eng, dev0, True)

    #4. arm a — exact token equality (NOT envelope: nothing numeric moved)
    same = tok_ser == tok_pipe
    print(f"[t0018] arm a  serial == pipelined tokens : {same} "
          f"({len(tok_ser)} seqs, {sum(len(v) for v in tok_ser.values())} "
          f"tokens)", flush=True)
    if not same:
        bad = [k for k in tok_ser if tok_ser[k] != tok_pipe.get(k)]
        print(f"[t0018]        MISMATCH on {bad[:5]}", flush=True)

    #5. arm b — the step that owns a prefill group AND a decode traversal
    #   is step 1 (arrivals at arrival_step=1; step 0 prefills the
    #   residents and has no decode work)
    print(f"[t0018] arm b  step walls (s)", flush=True)
    print(f"[t0018]        serial    : "
          f"{' '.join(f'{w:.3f}' for w in w_ser[:6])} | total "
          f"{sum(w_ser):.3f}", flush=True)
    print(f"[t0018]        pipelined : "
          f"{' '.join(f'{w:.3f}' for w in w_pipe[:6])} | total "
          f"{sum(w_pipe):.3f}", flush=True)
    if len(w_ser) > 1 and len(w_pipe) > 1:
        print(f"[t0018]        MIXED step 1: {w_ser[1]:.3f} -> "
              f"{w_pipe[1]:.3f} = {w_ser[1] / w_pipe[1]:.3f}x", flush=True)
    print(f"[t0018]        schedule total: {sum(w_ser):.3f} -> "
          f"{sum(w_pipe):.3f} = {sum(w_ser) / sum(w_pipe):.3f}x", flush=True)

    #6. arm c — same-schedule determinism of the pipelined arm
    det = tok_pipe == tok_pipe2
    print(f"[t0018] arm c  pipelined run1 == run2 : {det}", flush=True)

    ok = same and det
    print(f"[t0018] RESULT {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
