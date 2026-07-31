"""exp-0021 evidence: the graphed decode step is enqueued BEFORE the
step's prefill Python, not after it.

t_traversal_sync_probe established the fact this rests on: a grouped
prefill traversal has ZERO device->host syncs left in it (exp-0018) and
still costs 696 ms of HOST time against a 0.1 ms device tail — the GPUs
are idle for essentially all of it. The graphed decode step is the
opposite shape (a few ms of host work, ~170 ms of GPU work), so a step
that owns both should hand the decode replays over first and spend its
prefill Python underneath them.

  arm a  EQUIVALENCE — same schedule, _DECODE_FIRST off vs on. Nothing
         numeric moves (identical kernels, shapes and per-row op order;
         only the host's issue order and the page ids the shared free
         list hands out), so the token streams must be EXACTLY equal.
  arm b  WALL — the MIXED step (a 6-prompt cohort landing on residents
         that are already decoding, the canonical steady state) in both
         arms, INTERLEAVED and repeated so drift cancels. Prefill-only
         and decode-only steps must not move.
  arm c  DETERMINISM — two identical decode-first runs, bitwise (the
         t_b3-adapted worker arm).

Run: CUDA_VISIBLE_DEVICES=4,5,6,7 python -m engine.pyengine.tests\
.t_decode_first
"""
import sys
import time

import torch

from engine.pyengine import scheduler as psched
from engine.pyengine.server import build_resident

MAX_BATCH = 64
RESIDENT = 8            # rows already decoding when the cohort lands
COHORT = 6              # exp-0017's PREFILL_COHORT
LENS = [745, 700, 800, 745, 690, 760]
MAX_NEW = 4
NEED_FREE_GIB = 200
WAIT_MAX_S = 900


def _wait_for_gpus():
    #1. never load on top of a running canonical bench (exp-0018 lesson)
    t0 = time.time()
    while time.time() - t0 < WAIT_MAX_S:
        free = [torch.cuda.mem_get_info(d)[0] / (1 << 30)
                for d in range(torch.cuda.device_count())]
        if min(free) >= NEED_FREE_GIB:
            return True
        print(f"[t0021d] waiting, free/GPU = "
              f"{' '.join(f'{f:.0f}' for f in free)} GiB", flush=True)
        time.sleep(20)
    return False


def _prompts(n, base_len, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1, 190000, (base_len[i % len(base_len)],),
                          generator=g, dtype=torch.long)
            for i in range(n)]


def _run(eng, dev0, first, warm=False):
    """One full schedule under one arm; returns (tokens, step_walls)."""
    #1. the arm switch is the module global Scheduler.step reads
    psched._DECODE_FIRST = first
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
    if not _wait_for_gpus():
        print("[t0021d] GPUs busy — NOT loading", flush=True)
        return 2
    t0 = time.time()
    eng, splits = build_resident()
    dev0 = eng.w_emb.device
    print(f"[t0021d] resident in {time.time() - t0:.0f} s, layer split "
          f"{splits}", flush=True)

    #1. warm both arms (discarded): the first schedule in a process pays
    #   whole-traversal JIT + the graph capture
    t0 = time.time()
    _run(eng, dev0, False, warm=True)
    _run(eng, dev0, True, warm=True)
    print(f"[t0021d] warm-up {time.time() - t0:.0f} s", flush=True)

    #2. interleaved arms
    runs = {False: [], True: []}
    toks = {}
    for first in (False, True, False, True):
        tk, w = _run(eng, dev0, first)
        runs[first].append(w)
        toks.setdefault(first, []).append(tk)
        print(f"[t0021d] arm b  decode_first={first} steps "
              f"{' '.join(f'{x:.3f}' for x in w[:6])} | total "
              f"{sum(w):.3f}", flush=True)

    #3. arm a — exact token equality (NOT envelope: nothing numeric moved)
    same = toks[False][0] == toks[True][0]
    print(f"[t0021d] arm a  serial == decode-first tokens : {same} "
          f"({len(toks[False][0])} seqs, "
          f"{sum(len(v) for v in toks[False][0].values())} tokens)",
          flush=True)
    if not same:
        a, b = toks[False][0], toks[True][0]
        bad = [k for k in a if a[k] != b.get(k)]
        print(f"[t0021d]        MISMATCH on {bad[:5]}", flush=True)

    #4. arm b summary — prefill-only step 0 / MIXED step 1 / total
    def _mean(ws, i):
        return sum(w[i] for w in ws) / len(ws)

    off, on = runs[False], runs[True]
    print(f"[t0021d] arm b  prefill-only step0: {_mean(off, 0):.3f} -> "
          f"{_mean(on, 0):.3f}", flush=True)
    print(f"[t0021d] arm b  MIXED step1       : {_mean(off, 1):.3f} -> "
          f"{_mean(on, 1):.3f} = "
          f"{_mean(off, 1) / _mean(on, 1):.3f}x  "
          f"({(_mean(off, 1) - _mean(on, 1)) * 1e3:+.0f} ms)", flush=True)
    print(f"[t0021d] arm b  decode-only step2 : {_mean(off, 2):.3f} -> "
          f"{_mean(on, 2):.3f}", flush=True)
    print(f"[t0021d] arm b  schedule total    : "
          f"{sum(sum(w) for w in off) / len(off):.3f} -> "
          f"{sum(sum(w) for w in on) / len(on):.3f}", flush=True)

    #5. arm c — same-schedule determinism of the shipped arm
    det = toks[True][0] == toks[True][1]
    print(f"[t0021d] arm c  decode-first run1 == run2 : {det}", flush=True)

    ok = same and det
    print(f"[t0021d] RESULT {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
