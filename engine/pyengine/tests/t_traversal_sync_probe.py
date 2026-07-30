"""exp-0021 diagnosis: WHY do two prefill traversals not overlap?

t_microbatch_prefill measured that cutting a cohort into N prefill
sub-groups costs N-1 extra flat terms (~180 ms each) and returns ZERO of
the layer-split pipeline gain — the sub-group traversals are fully
serialized, which is also the simplest explanation of exp-0020's null
in-situ result (rejected at 222.6 vs 222.9). Two candidate causes:

  (i)  a device->host sync still inside the traversal, which pins the host
       to one device's progress so the next traversal is never handed
       over (exp-0018 removed the ones it knew about: kv._flat_slots and
       the closing argmax);
  (ii) the traversal is HOST-bound — the Python/launch cost of one
       traversal is comparable to its GPU cost, so there is no host
       run-ahead to give away in the first place.

The probe separates them on the real resident model:

  arm a  SYNC CENSUS — every implicit device->host sync entry point is
         counted during ONE grouped traversal, with a traceback for the
         first hit of each kind. (i) is true iff this is nonzero.
  arm b  HOST vs DEVICE — time from call to RETURN of the sync-free
         enqueue (host cost) vs the following device sync (device cost).
         (ii) is true iff the host half is a large fraction.
  arm c  OVERLAP — two groups of 3 enqueued back to back, then one sync,
         versus the same two groups each followed by its own sync. If the
         traversals can overlap at all, the back-to-back form is faster.

Run: CUDA_VISIBLE_DEVICES=4,5,6,7 python -m engine.pyengine.tests\
.t_traversal_sync_probe
"""
import sys
import time
import traceback

import torch

from engine.pyengine import scheduler as psched
from engine.pyengine.server import build_resident

LENS = [745, 700, 800, 745, 690, 760]
NEED_FREE_GIB = 200
WAIT_MAX_S = 900

COUNTS = {}
FIRST = {}
_ARMED = [False]


def _wrap(owner, name, key):
    """Count calls to one implicit-sync entry point."""
    orig = getattr(owner, name)

    def probe(*a, **kw):
        if _ARMED[0]:
            COUNTS[key] = COUNTS.get(key, 0) + 1
            if key not in FIRST:
                FIRST[key] = "".join(traceback.format_stack()[-4:-1])
        return orig(*a, **kw)

    setattr(owner, name, probe)


def _install():
    #1. the ops that force a device->host sync when fed a CUDA tensor
    for name in ("item", "tolist", "cpu", "numpy", "nonzero"):
        if hasattr(torch.Tensor, name):
            _wrap(torch.Tensor, name, f"Tensor.{name}")
    for name in ("nonzero", "unique", "bincount"):
        _wrap(torch, name, f"torch.{name}")
    _wrap(torch.cuda, "synchronize", "cuda.synchronize")


def _wait_for_gpus():
    t0 = time.time()
    while time.time() - t0 < WAIT_MAX_S:
        free = [torch.cuda.mem_get_info(d)[0] / (1 << 30)
                for d in range(torch.cuda.device_count())]
        if min(free) >= NEED_FREE_GIB:
            return True
        print(f"[t0021p] waiting, free/GPU = "
              f"{' '.join(f'{f:.0f}' for f in free)} GiB", flush=True)
        time.sleep(20)
    return False


def _prompts(n, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1, 190000, (LENS[i % len(LENS)],),
                          generator=g, dtype=torch.long)
            for i in range(n)]


def _mkreqs(eng, dev0, ids, slot0):
    """Requests with fresh KV state, ready for a grouped prefill."""
    out = []
    for i, t in enumerate(ids):
        r = psched.Request(f"p{slot0 + i}", t.to(dev0), 2)
        r.states = eng.new_states(slot=slot0 + i)
        out.append(r)
    return out


def _sync_all():
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)


def main():
    if not _wait_for_gpus():
        print("[t0021p] GPUs busy — NOT loading", flush=True)
        return 2
    t0 = time.time()
    eng, splits = build_resident()
    dev0 = eng.w_emb.device
    eng.init_pools(64)
    print(f"[t0021p] resident in {time.time() - t0:.0f} s, split {splits}",
          flush=True)
    ids = _prompts(12, 1001)

    #1. warm: JIT/autotune must not land inside a timed or counted arm
    g = _mkreqs(eng, dev0, ids[:3], 0)
    eng.prefill_batch(g)
    for r in g:
        eng.release_states(r.states)
    _sync_all()
    print("[t0021p] warm done", flush=True)

    #2. arm a — sync census over ONE grouped traversal
    _install()
    g = _mkreqs(eng, dev0, ids[:3], 8)
    _ARMED[0] = True
    rows = eng.prefill_batch_enqueue(g)
    _ARMED[0] = False
    _sync_all()
    print(f"[t0021p] arm a  sync census over one 3-row traversal: "
          f"{COUNTS if COUNTS else 'NONE'}", flush=True)
    for k, tb in FIRST.items():
        print(f"[t0021p]        first {k}:\n{tb}", flush=True)
    for r in g:
        eng.release_states(r.states)

    #3. arm b — host cost vs device cost of one traversal
    walls = []
    for rep in range(2):
        g = _mkreqs(eng, dev0, ids[:3], 16)
        t0 = time.perf_counter()
        rows = eng.prefill_batch_enqueue(g)
        t1 = time.perf_counter()
        _sync_all()
        t2 = time.perf_counter()
        walls.append((t1 - t0, t2 - t1))
        for r in g:
            eng.release_states(r.states)
    for host, dev in walls:
        print(f"[t0021p] arm b  host enqueue {host * 1e3:.1f} ms | device "
              f"tail {dev * 1e3:.1f} ms | total {(host + dev) * 1e3:.1f} ms",
              flush=True)

    #4. arm c — back-to-back vs individually synced
    def _two(sync_between):
        ga = _mkreqs(eng, dev0, ids[:3], 24)
        gb = _mkreqs(eng, dev0, ids[3:6], 32)
        t0 = time.perf_counter()
        eng.prefill_batch_enqueue(ga)
        if sync_between:
            _sync_all()
        eng.prefill_batch_enqueue(gb)
        _sync_all()
        w = time.perf_counter() - t0
        for r in ga + gb:
            eng.release_states(r.states)
        return w

    for sb in (True, False, True, False):
        w = _two(sb)
        print(f"[t0021p] arm c  two 3-row traversals, sync_between={sb}: "
              f"{w * 1e3:.1f} ms", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
