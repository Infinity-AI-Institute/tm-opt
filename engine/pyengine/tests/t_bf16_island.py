"""exp-0019 in-situ arms — the eager island vs the grouped bf16 twin.

The standalone kernel arms (tests/t_moe_bf16.py) price layer 2's MoE half
in isolation; exp-0011's rejection is the reminder that isolation is not
in-situ, because graphrun's island deliberately runs its host work in the
SHADOW of the surrounding graph replays. This measures the real thing: one
resident engine load (server.build_resident), three identical schedules on
fresh schedulers over the shared pools, with the GraphRunner rebuilt per
arm so each captures under its own predicate:
  A  PYENGINE_BF16_GROUPED=0 — the accepted exp-0017 path: layer 2 on the
     reference loop, carved out as graphrun's eager island, device 0 split
     into two graphs around it
  B  grouped twin on, run 1 — islands must be EMPTY and every device a
     single captured segment
  C  grouped twin on, run 2, same schedule, warm capture + evolved scratch:
     token streams must be BITWISE identical to B (the t_b3-adapted
     determinism arm)
Reported: island/segment structure per arm, mean decode-step wall A vs B,
B==C bitwise verdict (hard assert), and A-vs-B stream agreement (envelope
drift expected, NOT bitwise — B3.1/B3.3, same class as every batched
rewrite since exp-0001).

Run: CUDA_VISIBLE_DEVICES=4,5,6,7 python -m engine.pyengine.tests.t_bf16_island
"""
import time

import torch

from engine.pyengine import graphrun
from engine.pyengine import model as pmodel
from engine.pyengine import scheduler as psched
from engine.pyengine import server as pserver

N_REQ = 64
N_NEW = 40          # ISL up to 500 -> pos crosses the 512 L_pad bucket
SEED = 20260730


def prompts(mc):
    #1. same deterministic prompt set t_graph_ab uses, so the two
    #   experiments' step walls are directly comparable
    g = torch.Generator().manual_seed(SEED)
    out = []
    for i in range(N_REQ):
        T = 220 + (i * 7919) % 281          # 220..500
        out.append(torch.randint(0, mc.vocab_unpadded, (T,), generator=g))
    return out


def run_arm(eng, ids_list, grouped, label):
    #1. force a fresh GraphRunner so capture happens under THIS arm's
    #   predicate (the flag is read at call time by both model.moe_experts
    #   and graphrun's island scan — a stale runner would replay the other
    #   arm's baked-in path)
    pmodel._BF16_GROUPED = grouped
    eng._graphrun = None
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
    gr = eng._graphrun
    segs = 0
    if gr is not None and gr.programs:
        prog = next(iter(gr.programs.values()))
        segs = sum(1 for it in prog if isinstance(it, graphrun._Seg))
        print(f"[{label}] islands {gr.islands} | captured segments {segs} "
              f"| devices {gr.ndev}", flush=True)
    toks = {r.id: list(r.tokens) for r in sched.finished}
    full = walls[:-1] or walls                  # drop the retire step
    mean = sum(full) / len(full)
    print(f"[{label}] prefill(64) {t_prefill:.1f} s; decode steps "
          f"{len(walls)}, mean {mean*1e3:.1f} ms, min {min(full)*1e3:.1f} ms",
          flush=True)
    return toks, mean, (gr.islands if gr else None), segs


def main():
    assert torch.cuda.is_available()
    eng, _ = pserver.build_resident()
    ids_list = prompts(eng.mc)
    #0. DISCARDED warm-up arm. The first arm in a process pays one-time JIT
    #   for the whole traversal (Triton specializations + the CUTLASS DSL
    #   W4A4 compile), which lands in its prefill and first decode steps —
    #   in the first version of this test that alone made arm A's prefill
    #   36.2 s vs 6.5 s warm, an artefact that has nothing to do with the
    #   variable. Then arms INTERLEAVE island/grouped/island/grouped so any
    #   residual warm-up trend cancels and each path is measured twice.
    run_arm(eng, ids_list, False, "W warmup ")
    ta, wa, isl_a, seg_a = run_arm(eng, ids_list, False, "A1 island ")
    tb, wb, isl_b, seg_b = run_arm(eng, ids_list, True, "B1 grouped")
    ta2, wa2, _, _ = run_arm(eng, ids_list, False, "A2 island ")
    tc, wb2, _, _ = run_arm(eng, ids_list, True, "B2 grouped")
    print(f"[repeat] island {wa*1e3:.1f}/{wa2*1e3:.1f} ms  "
          f"grouped {wb*1e3:.1f}/{wb2*1e3:.1f} ms", flush=True)
    assert all(ta[i] == ta2[i] for i in range(N_REQ)), \
        "DETERMINISM FAIL: island runs differ on same schedule"
    wa, wb = (wa + wa2) / 2, (wb + wb2) / 2
    #1. structure: the island must be gone, and with it the extra graphs
    print(f"[structure] islands {isl_a} -> {isl_b}; "
          f"captured segments {seg_a} -> {seg_b}", flush=True)
    #2. determinism: B == C bitwise on the same schedule
    same = all(tb[i] == tc[i] for i in range(N_REQ))
    print(f"[determinism] grouped run1 == run2 bitwise: {same}", flush=True)
    #3. A vs B: envelope-drift comparison (greedy chains diverge after a
    #   first low-margin flip; report where)
    div = [next((s for s in range(N_NEW) if ta[i][s] != tb[i][s]), None)
           for i in range(N_REQ)]
    exact = sum(1 for d in div if d is None)
    firsts = sorted(d for d in div if d is not None)
    print(f"[a/b] rows bitwise-equal island-vs-grouped: {exact}/{N_REQ}; "
          f"first-divergence steps: {firsts[:10]}"
          f"{'...' if len(firsts) > 10 else ''}", flush=True)
    for t in (ta, tb, tc):
        assert all(0 <= x < eng.mc.vocab_unpadded for v in t.values()
                   for x in v)
    print(f"[speed] mean decode step: island {wa*1e3:.1f} ms -> grouped "
          f"{wb*1e3:.1f} ms ({wa/wb:.3f}x, {(wa-wb)*1e3:.1f} ms/step)",
          flush=True)
    assert isl_b == [], f"island not removed: {isl_b}"
    assert same, "DETERMINISM FAIL: grouped runs differ on same schedule"
    print("ALL OK", flush=True)


if __name__ == "__main__":
    main()
