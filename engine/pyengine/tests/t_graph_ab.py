"""exp-0012 GPU evidence — eager pooled decode vs CUDA-graphed decode.

One resident engine load (server.build_resident), then three identical
schedules (64 requests, deterministic pseudo-random prompts, ISL mix
crossing the 512->1024 L_pad bucket mid-decode so a live re-capture is
exercised), fresh Scheduler per arm on the shared pools:
  A  eager pooled path   (PYENGINE_CUDAGRAPH=0)
  B  graphed path, run 1 (PYENGINE_CUDAGRAPH=1)
  C  graphed path, run 2 — same schedule as B, warm scratch/capture
     state: token streams must be BITWISE identical to B (the worker's
     t_b3-adapted determinism arm; graphrun._run_island note)
Reported: B==C bitwise verdict (hard assert), A-vs-B stream agreement
(envelope drift expected, NOT bitwise — B3.1/B3.3), first-divergence
step histogram, and mean decode-step wall per arm at full B.

Run: CUDA_VISIBLE_DEVICES=4,5,6,7 python -m engine.pyengine.tests.t_graph_ab
Log: /workspace/logs/t_graph_ab_exp0012_2026-07-30.log (tee by caller).
"""
import os
import time

import torch

from engine.pyengine import scheduler as psched
from engine.pyengine import server as pserver

N_REQ = 64
N_NEW = 40          # ISL up to 500 -> pos crosses 512 mid-decode
SEED = 20260730


def prompts(mc):
    #1. deterministic prompt set: lengths cycle 220..500, ids < unpadded
    g = torch.Generator().manual_seed(SEED)
    out = []
    for i in range(N_REQ):
        T = 220 + (i * 7919) % 281          # 220..500
        out.append(torch.randint(0, mc.vocab_unpadded, (T,),
                                 generator=g))
    return out


def run_arm(eng, ids_list, label):
    #1. fresh scheduler on the shared engine/pools; submit everything at
    #   step 0 — one deterministic lockstep schedule
    sched = psched.Scheduler(eng, N_REQ)
    for i, ids in enumerate(ids_list):
        sched.submit(psched.Request(i, ids.to(eng.w_emb.device), N_NEW))
    #2. drive prefills (step 0), then time the pure-decode steps
    t0 = time.time()
    sched.step()
    t_prefill = time.time() - t0
    walls = []
    while sched.running or sched.waiting:
        t1 = time.time()
        sched.step()
        torch.cuda.synchronize()
        walls.append(time.time() - t1)
    toks = {r.id: list(r.tokens) for r in sched.finished}
    full = [w for w in walls[:-1]] or walls     # drop the retire step
    print(f"[{label}] prefill(64) {t_prefill:.1f} s; decode steps "
          f"{len(walls)}, mean {sum(full)/len(full)*1e3:.1f} ms, "
          f"min {min(full)*1e3:.1f} ms", flush=True)
    return toks, sum(full) / len(full)


def main():
    assert torch.cuda.is_available()
    eng, _ = pserver.build_resident()
    mc = eng.mc
    ids_list = prompts(mc)
    #1. arm A: eager pooled
    os.environ["PYENGINE_CUDAGRAPH"] = "0"
    ta, wa = run_arm(eng, ids_list, "A eager")
    #2. arms B/C: graphed, twice (C on warm capture + evolved scratch)
    os.environ["PYENGINE_CUDAGRAPH"] = "1"
    tb, wb = run_arm(eng, ids_list, "B graph")
    tc, _ = run_arm(eng, ids_list, "C graph2")
    #3. determinism: B == C bitwise
    same = all(tb[i] == tc[i] for i in range(N_REQ))
    print(f"[determinism] graph run1 == run2 bitwise: {same}", flush=True)
    #4. A vs B: envelope-drift comparison (greedy chains diverge after a
    #   first low-margin flip; report where)
    div = []
    for i in range(N_REQ):
        d = next((s for s in range(N_NEW) if ta[i][s] != tb[i][s]), None)
        div.append(d)
    exact = sum(1 for d in div if d is None)
    firsts = sorted(d for d in div if d is not None)
    print(f"[a/b] rows bitwise-equal eager-vs-graph: {exact}/{N_REQ}; "
          f"first-divergence steps: {firsts[:10]}"
          f"{'...' if len(firsts) > 10 else ''}", flush=True)
    #5. sanity: token ranges
    for t in (ta, tb, tc):
        assert all(0 <= x < mc.vocab_unpadded for v in t.values()
                   for x in v)
    print(f"[speed] mean step: eager {wa*1e3:.1f} ms -> graph "
          f"{wb*1e3:.1f} ms ({wa/wb:.2f}x)", flush=True)
    assert same, "DETERMINISM FAIL: graph runs differ on same schedule"
    print("ALL OK", flush=True)


if __name__ == "__main__":
    main()
