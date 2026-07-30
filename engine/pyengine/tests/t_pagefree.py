"""exp-0024 local validation: the global-KV page free list as a FIFO deque
instead of a Python list popped from the front.

Why this is a measurement and not a style note. kv.GlobalPool holds ONE
shared free list of `max_batch * num_pages` page ids — at the canonical
operating point that is 64 * 1024 = 65,536 entries (server.build_resident
num_pages=1024, serve max_batch=64) — and kv.PagedKV.ensure_pages hands
pages out with `self.free.pop(0)`. CPython's list.pop(0) is O(len): it
memmoves every remaining element down one slot, so each page costs a
~0.5 MB memmove. The canonical cohort pops a page for every 16 tokens of
every one of the 11 global layers of every one of the 64 co-resident
sequences, and every one of those pops runs on the host thread INSIDE the
prefill traversal / decode step, where it blocks kernel issue.

collections.deque.popleft() is O(1) and returns the SAME element in the
SAME order; `extend` still appends at the tail, so scheduler.release_states
returns pages exactly where it did. The page-id sequence — hence every
page table, hence every KV placement, hence every tensor value in the
engine — is unchanged. Arm (a) proves that; arm (b) prices it.

Arms (pure CPU, no GPU, no model load — safe to run while the dispatcher
owns the GPUs):
  a  ORDER EQUIVALENCE: replay the canonical cohort schedule against both
     structures and compare the full allocated-page-id streams elementwise,
     plus the final free-list contents.
  b  COST: wall time of the free-list operations alone for one canonical
     cohort, list vs deque, after a discarded warm-up cohort.
  c  SCALE: cost as a function of free-list length, showing the O(len) vs
     O(1) shape (and that the win is not an artifact of one size).

Run from the worktree root (pin to the NUMA-0 cores so the measurement
cannot perturb a GPU4-7 bench):
  taskset -c 4 python engine/pyengine/tests/t_pagefree.py
"""
import collections
import time

#canonical operating point (server.build_resident / serve defaults)
MAX_BATCH = 64
NUM_PAGES = 1024          # per-sequence page budget of one global layer
PAGE_SIZE = 16            # kv.PAGE_SIZE
N_GLOBAL = 11             # global layers: {5,11,...,65}
ISL = 745                 # canonical prompt length (exp-0009/0017 T~736-745)
OSL = 128                 # canonical osl_cap


def cohort_schedule():
    """The (sequence, layer) page-allocation events of ONE canonical
    lockstep cohort, in the order the engine performs them: 64 prefill
    traversals (each seeding 11 global layers with ceil(ISL/16) pages),
    then 128 decode steps whose per-row ensure_pages crosses a page
    boundary every 16 tokens, then 64 releases.

    Returns a list of ('alloc', seq, layer, target_pages) / ('free', seq)
    events — `target_pages` is ensure_pages' argument form (a table SIZE,
    not an increment), so replay allocates exactly what the engine does."""
    ev = []
    pre = -(-ISL // PAGE_SIZE)                     # pages after prefill
    for s in range(MAX_BATCH):                     # 64 prefill traversals
        for L in range(N_GLOBAL):
            ev.append(("alloc", s, L, pre))
    #decode: the cohort steps in lockstep, so every row crosses a page
    #boundary on the same step (every PAGE_SIZE tokens)
    have = pre
    for t in range(OSL):
        need = -(-(ISL + t + 1) // PAGE_SIZE)
        if need > have:
            for s in range(MAX_BATCH):
                for L in range(N_GLOBAL):
                    ev.append(("alloc", s, L, need))
            have = need
    for s in range(MAX_BATCH):
        ev.append(("free", s))
    return ev


def replay(free, popfn, events, record=None):
    """Run the schedule against `free` (list or deque). `popfn` takes the
    structure and returns the next page id — the ONE line this experiment
    changes. A 'free' event returns every table of that sequence, in layer
    order, exactly as scheduler.release_states does."""
    tables = collections.defaultdict(list)
    for e in events:
        if e[0] == "alloc":
            _, s, L, n = e
            tab = tables[(s, L)]
            while len(tab) < n:                    # kv.ensure_pages #1
                pid = popfn(free)
                tab.append(pid)
                if record is not None:
                    record.append(pid)
        else:
            s = e[1]
            for L in range(N_GLOBAL):
                free.extend(tables.pop((s, L), []))
    return free


def main():
    events = cohort_schedule()
    total = MAX_BATCH * NUM_PAGES
    print(f"canonical cohort: pool {total} pages, "
          f"{MAX_BATCH} seqs x {N_GLOBAL} global layers, "
          f"ISL {ISL} OSL {OSL}")

    #arm a: order equivalence — the whole point of the change being free
    ra, rb = [], []
    fa = replay(list(range(total)), lambda f: f.pop(0), events, ra)
    fb = replay(collections.deque(range(total)),
                lambda f: f.popleft(), events, rb)
    same = ra == rb
    tail = list(fa) == list(fb)
    print(f"a  ORDER: {len(ra)} allocated page ids, streams identical: "
          f"{same}; final free lists identical: {tail} "
          f"(len {len(fa)} vs {len(fb)})")
    assert same and tail, "page order changed — NOT value-preserving"

    #arm b: cost of one cohort's free-list traffic, warm-up discarded
    def timed(make, popfn):
        replay(make(), popfn, events)               # warm-up cohort
        f = make()
        t0 = time.perf_counter()
        replay(f, popfn, events)
        return time.perf_counter() - t0

    t_list = timed(lambda: list(range(total)), lambda f: f.pop(0))
    t_deq = timed(lambda: collections.deque(range(total)),
                  lambda f: f.popleft())
    print(f"b  COST/cohort: list.pop(0) {t_list*1e3:8.1f} ms   "
          f"deque.popleft() {t_deq*1e3:8.1f} ms   "
          f"saved {(t_list-t_deq)*1e3:8.1f} ms "
          f"({t_list/max(t_deq,1e-9):.0f}x)")

    #arm c: the O(len) shape — cost vs pool size at a fixed pop count
    print("c  SCALE (pop half the pool, no refill):")
    for n in (8192, 16384, 32768, 65536, 131072):
        k = n // 2
        f = list(range(n))
        t0 = time.perf_counter()
        for _ in range(k):
            f.pop(0)
        t1 = time.perf_counter()
        d = collections.deque(range(n))
        t2 = time.perf_counter()
        for _ in range(k):
            d.popleft()
        t3 = time.perf_counter()
        print(f"     pool {n:7d} ({k:6d} pops): list {(t1-t0)*1e3:8.1f} ms"
              f"   deque {(t3-t2)*1e3:6.2f} ms   "
              f"per pop {(t1-t0)/k*1e6:6.1f} us vs "
              f"{(t3-t2)/k*1e6:.2f} us")


if __name__ == "__main__":
    main()
