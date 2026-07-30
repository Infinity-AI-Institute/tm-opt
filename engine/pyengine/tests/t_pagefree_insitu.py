"""exp-0024 in-situ arm: is the free-list memmove ON THE CRITICAL PATH?

tests/t_pagefree prices the change in isolation (278 ms of host-thread
memmove per canonical cohort, 82x). That is NOT the same claim as "the
cohort gets 278 ms shorter" — exp-0018's post-mortem is explicit that
"I removed the thing that blocks X" is not evidence of X, and a host
thread that outruns the CUDA launch queue can hide host work behind
device work for free. This arm runs the one measurement that settles it,
on the real resident model: the SAME grouped prefill traversal, timed with
the free list as a deque and as a list, arms interleaved, pools reset to
the canonical 65,536-entry length before every repetition.

A 6-row traversal pops 6 x 11 global layers x ceil(745/16) = 3,102 pages.
At the 7.7 us/pop that arm c of t_pagefree measures for a full pool, the
predicted separation is ~24 ms per traversal if the cost is on the
critical path, and ~0 if the launch queue absorbs it. Host time (call ->
return of the sync-free enqueue) and the device tail are reported
separately, so the arm also shows WHERE the difference lands.

Run: CUDA_VISIBLE_DEVICES=4,5,6,7 python -m engine.pyengine.tests\
.t_pagefree_insitu
"""
import collections
import sys
import time

import torch

from engine.pyengine import scheduler as psched
from engine.pyengine.server import build_resident

LENS = [745, 700, 800, 745, 690, 760]      # canonical-distribution group
GROUP = 6                                   # exp-0017 admission group size
REPS = 3                                    # per arm, interleaved
NEED_FREE_GIB = 200
WAIT_MAX_S = 900


def _wait_for_gpus():
    t0 = time.time()
    while time.time() - t0 < WAIT_MAX_S:
        free = [torch.cuda.mem_get_info(d)[0] / (1 << 30)
                for d in range(torch.cuda.device_count())]
        if min(free) >= NEED_FREE_GIB:
            return True
        print(f"[t0024] waiting for GPUs, free/GPU = "
              f"{' '.join(f'{f:.0f}' for f in free)} GiB", flush=True)
        time.sleep(20)
    return False


def _prompts(n, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1, 190000, (LENS[i % len(LENS)],),
                          generator=g, dtype=torch.long)
            for i in range(n)]


def _mkreqs(eng, dev0, ids, slot0):
    out = []
    for i, t in enumerate(ids):
        r = psched.Request(f"p{slot0 + i}", t.to(dev0), 2)
        r.states = eng.new_states(slot=slot0 + i)
        out.append(r)
    return out


def _sync_all():
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def _reset_pools(eng, kind):
    """Rebind every GlobalPool's free list to a FRESH full-length structure
    of `kind`. Must run BEFORE new_states(): LayerKV binds pool.free by
    reference at construction (kv.PagedKV backing), so a rebind only
    reaches caches created afterwards. Returns the pool count."""
    n = 0
    for pool in eng._pools:
        if hasattr(pool, "free") and pool.free is not None:
            total = pool.kp.shape[0] - 1        # scratch page is never free
            pool.free = kind(range(total))
            n += 1
    return n


def main():
    if not _wait_for_gpus():
        print("[t0024] GPUs busy — NOT loading", flush=True)
        return 2
    t0 = time.time()
    eng, splits = build_resident()
    dev0 = eng.w_emb.device
    eng.init_pools(64)
    print(f"[t0024] resident in {time.time() - t0:.0f} s, split {splits}",
          flush=True)
    ids = _prompts(GROUP, 2024)
    npool = _reset_pools(eng, collections.deque)
    print(f"[t0024] {npool} global pools, "
          f"{eng._pools[0].kp.shape[0] - 1 if npool else 0} pages each; "
          f"group {GROUP} rows, lens {LENS[:GROUP]}", flush=True)

    #1. warm — JIT/autotune must not land inside a timed arm
    g = _mkreqs(eng, dev0, ids, 0)
    eng.prefill_batch(g)
    for r in g:
        eng.release_states(r.states)
    _sync_all()
    print("[t0024] warm done", flush=True)

    #2. interleaved arms; pools reset to full length before EVERY rep so
    #   both arms see the canonical free-list length, not a drained one
    res = {"deque": [], "list": []}
    for rep in range(REPS):
        for name, kind in (("deque", collections.deque), ("list", list)):
            _reset_pools(eng, kind)
            g = _mkreqs(eng, dev0, ids, 8)
            _sync_all()
            t0 = time.perf_counter()
            eng.prefill_batch_enqueue(g)
            t1 = time.perf_counter()
            _sync_all()
            t2 = time.perf_counter()
            for r in g:
                eng.release_states(r.states)
            res[name].append((t1 - t0, t2 - t1))
            print(f"[t0024] rep {rep} {name:5s}: host "
                  f"{(t1 - t0) * 1e3:7.1f} ms | device tail "
                  f"{(t2 - t1) * 1e3:6.1f} ms | total "
                  f"{(t2 - t0) * 1e3:7.1f} ms", flush=True)

    def med(name, i):
        return sorted(v[i] for v in res[name])[len(res[name]) // 2] * 1e3

    dq, li = med("deque", 0) + med("deque", 1), med("list", 0) + med("list", 1)
    print(f"[t0024] MEDIAN total per {GROUP}-row traversal: "
          f"deque {dq:.1f} ms | list {li:.1f} ms | delta {li - dq:.1f} ms "
          f"({(li - dq) / max(li, 1e-9) * 100:.2f}%)", flush=True)
    print(f"[t0024] host-half medians: deque {med('deque', 0):.1f} ms | "
          f"list {med('list', 0):.1f} ms", flush=True)
    #   cohort projection: 64 seqs / GROUP traversals + the decode-side pops
    print(f"[t0024] projected per 64-seq cohort (prefill traversals only): "
          f"{(li - dq) * (64 / GROUP):.0f} ms", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
