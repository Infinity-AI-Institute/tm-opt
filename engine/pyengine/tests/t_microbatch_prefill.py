"""exp-0021 evidence: a released cohort prefills as SEVERAL microbatches.

exp-0018 made the traversal sync-free, so the layer-split pipeline is now
fed by how many traversals the host can hand over before the first one
finishes — and a step only ever had two (its prefill group and its
decode). Cutting the cohort's rows into consecutive sub-groups is the one
free source of more. The trade is exp-0017's flat term: every sub-group
pays the 128-row expert M tile again, so this script measures the
makespan directly instead of trusting the model.

  arm a  WALL — the mixed step (a 6-prompt cohort landing on residents
         that are already decoding, i.e. the canonical steady state) timed
         at _MICROBATCH in {0 = one group, 2, 3}, INTERLEAVED and repeated
         so any drift cancels. Also reports the prefill-only step, where
         the same cut applies without a decode traversal to share with.
  arm b  DETERMINISM — two identical runs at the shipped _MICROBATCH,
         bitwise (the t_b3-adapted worker arm).
  arm c  GROUPING — the sub-group structure the scheduler actually built,
         asserted rather than assumed.
  arm d  NUMERIC CLASS — tokens at _MICROBATCH=0 vs the shipped value.
         These are NOT required to be equal: changing which rows share a
         padded traversal is the row-count bf16 accumulation drift class
         of decode_batch/prefill_batch (B3.1/B3.3), whose binding gates
         are the D13 envelope + same-schedule determinism. Reported so the
         size of the class is on the record.

Waits for GPUs 4-7 to be free before loading (the worker owns them during
a canonical bench; exp-0018 died because someone ignored that).

Run: CUDA_VISIBLE_DEVICES=4,5,6,7 python -m engine.pyengine.tests\
.t_microbatch_prefill
"""
import sys
import time

import torch

from engine.pyengine import scheduler as psched
from engine.pyengine.server import build_resident

MAX_BATCH = 64
RESIDENT = 8            # rows already decoding when the cohort lands
COHORT = 6              # exp-0017's PREFILL_COHORT
LENS = [745, 700, 800, 745, 690, 760]   # right-padding is exercised
MAX_NEW = 4
ARMS = [0, 2, 3, 0, 2, 3]               # interleaved so drift cancels
NEED_FREE_GIB = 200     # weights are ~137 GiB/GPU + KV pool
WAIT_MAX_S = 1500


def _wait_for_gpus():
    """Block until every visible device is essentially empty."""
    #1. a canonical bench holds 170-220 GiB/GPU; refuse to load into it
    t0 = time.time()
    while time.time() - t0 < WAIT_MAX_S:
        free = [torch.cuda.mem_get_info(d)[0] / (1 << 30)
                for d in range(torch.cuda.device_count())]
        if min(free) >= NEED_FREE_GIB:
            print(f"[t0021] GPUs free ({min(free):.0f} GiB min) after "
                  f"{time.time() - t0:.0f} s", flush=True)
            return True
        print(f"[t0021] waiting for GPUs, free/GPU = "
              f"{' '.join(f'{f:.0f}' for f in free)} GiB", flush=True)
        time.sleep(20)
    return False


def _prompts(n, base_len, seed):
    #1. deterministic pseudo-prompts at canonical-ISL scale; the token
    #   VALUES are irrelevant to a wall measurement, the shapes are not
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1, 190000, (base_len[i % len(base_len)],),
                          generator=g, dtype=torch.long)
            for i in range(n)]


def _run(eng, dev0, micro, warm=False):
    """One full schedule at one _MICROBATCH; returns (tokens, walls,
    groups) with groups = the sub-group sizes of the cohort step."""
    #1. the arm switch is the module global _prefill_groups reads
    psched._MICROBATCH = micro
    sched = psched.Scheduler(eng, MAX_BATCH)
    seen = []
    inner = sched._prefill_groups

    def _spy(admits):
        #   record the sub-group structure the scheduler really built
        gs = inner(admits)
        seen.append([len(g) for g in gs])
        return gs

    sched._prefill_groups = _spy
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
    return {r.id: list(r.tokens) for r in sched.finished}, walls, seen


def main():
    #1. never load on top of a running canonical bench
    if not _wait_for_gpus():
        print("[t0021] GPUs still busy — NOT loading (exp-0018 lesson)",
              flush=True)
        return 2
    t0 = time.time()
    eng, splits = build_resident()
    dev0 = eng.w_emb.device
    print(f"[t0021] resident in {time.time() - t0:.0f} s, layer split "
          f"{splits}", flush=True)

    #2. warm the JIT/autotune caches (first CUTE DSL prefill compiles);
    #   DISCARDED, and run at both grouping shapes so neither timed arm
    #   pays a first-shape compile
    t0 = time.time()
    _run(eng, dev0, 0, warm=True)
    _run(eng, dev0, 2, warm=True)
    print(f"[t0021] warm-up {time.time() - t0:.0f} s", flush=True)

    #3. arm a — interleaved wall arms
    by_arm, toks, groups = {}, {}, {}
    for micro in ARMS:
        tk, w, g = _run(eng, dev0, micro)
        by_arm.setdefault(micro, []).append(w)
        toks.setdefault(micro, []).append(tk)
        groups[micro] = g
        print(f"[t0021] arm a  M={micro} steps "
              f"{' '.join(f'{x:.3f}' for x in w[:6])} | total "
              f"{sum(w):.3f}", flush=True)
    print("[t0021] arm a  SUMMARY (prefill-only step 0 / mixed step 1 / "
          "schedule total, mean of repeats, s)", flush=True)
    base = None
    for micro in sorted(by_arm):
        runs = by_arm[micro]
        s0 = sum(w[0] for w in runs) / len(runs)
        s1 = sum(w[1] for w in runs) / len(runs)
        tot = sum(sum(w) for w in runs) / len(runs)
        if base is None:
            base = (s0, s1, tot)
        print(f"[t0021]        M={micro}: {s0:.3f} / {s1:.3f} / {tot:.3f}"
              f"  -> mixed {base[1] / s1:.3f}x, total "
              f"{base[2] / tot:.3f}x", flush=True)

    #4. arm b — same-schedule determinism at the shipped value
    det = toks[2][0] == toks[2][1]
    print(f"[t0021] arm b  M=2 run1 == run2 : {det}", flush=True)

    #5. arm c — the grouping actually built (cohort step is index 1)
    for micro in sorted(groups):
        print(f"[t0021] arm c  M={micro} sub-group sizes per step: "
              f"{groups[micro][:3]}", flush=True)
    shape_ok = groups[2][1] == [2, 2, 2] and groups[0][1] == [6]

    #6. arm d — numeric class of the regrouping
    a, b = toks[0][0], toks[2][0]
    n_eq = sum(1 for k in a if a[k] == b.get(k))
    print(f"[t0021] arm d  M=0 vs M=2 identical streams: {n_eq}/{len(a)} "
          f"(regrouping = the B3.1/B3.3 row-count drift class)", flush=True)

    ok = det and shape_ok
    print(f"[t0021] RESULT {'PASS' if ok else 'FAIL'} (grouping "
          f"{shape_ok}, determinism {det})", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
