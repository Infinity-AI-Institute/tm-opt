"""exp-0023 evidence: fused prefill attention measured IN SITU.

exp-0011's rejection and exp-0018's null are the standing reminders that an
isolated kernel arm can be honest and still not move the box, so the number
that decides this experiment is the one measured on the REAL resident model
through the real Scheduler: the wall of a step that owns a grouped prefill
traversal, with PYENGINE_FUSED_PREFILL_ATTN toggled between arms.

Shapes are the canonical ones, not round numbers: benchmark.make_prompts
draws L = randint(isl*range_ratio, isl) = randint(512, 1024) words for
decode_heavy, so a cohort's rows are ~512-1024 tokens and its padded Tmax
is ~800-1024 (the same LENS t_pipeline_prefill/t_decode_first used).

  arm a  WALL — cohort-prefill step and mixed step, arms INTERLEAVED
         (off/on/off/on) after a discarded warm-up, so any residual trend
         cancels and no CUTE DSL / Triton JIT lands in a timed arm.
  arm b  TOKENS — the same schedule under both arms. Numerics DO move here
         (fp32 flash accumulation vs a bf16-rounded score tensor), so
         equality is a sanity signal, not the gate; the gate is the D13
         teacher-forced envelope check the worker runs.
  arm c  DETERMINISM — two identical fused runs must be bitwise equal
         (the t_b3-adapted arm; the kernel is fixed-grid and atomics-free).

Run: CUDA_VISIBLE_DEVICES=4,5,6,7 python -m engine.pyengine.tests\
.t_prefill_attn_insitu
"""
import sys
import time

import torch

from engine.pyengine import model as pmodel
from engine.pyengine import scheduler as psched
from engine.pyengine.server import build_resident

MAX_BATCH = 64
RESIDENT = 8            # rows already decoding when the cohort lands
COHORT = 6              # exp-0017's PREFILL_COHORT
LENS = [745, 700, 800, 745, 690, 760]
MAX_NEW = 4


def _prompts(n, base_len, seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(1, 190000, (base_len[i % len(base_len)],),
                          generator=g, dtype=torch.long)
            for i in range(n)]


def _run(eng, dev0, fused, warm=False):
    """One full schedule under one arm; returns (tokens, step_walls)."""
    #1. the arm switch is the module global attn_prefill reads per call
    #   (prefill is not captured, so toggling between runs is safe)
    pmodel._FUSED_PREFILL_ATTN = fused
    sched = psched.Scheduler(eng, MAX_BATCH)
    n_res, n_new = (2, 2) if warm else (RESIDENT, COHORT)
    lens = LENS[:2] if warm else LENS
    ids = _prompts(n_res + n_new, lens, 1001)
    for i in range(n_res):
        sched.submit(psched.Request(f"r{i}", ids[i].to(dev0), MAX_NEW),
                     arrival_step=0)
    for i in range(n_res, n_res + n_new):
        sched.submit(psched.Request(f"c{i}", ids[i].to(dev0), MAX_NEW),
                     arrival_step=1)
    walls = []
    while sched.waiting or sched.running:
        t0 = time.perf_counter()
        sched.step()
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)
        walls.append(time.perf_counter() - t0)
    return {r.id: list(r.tokens) for r in sched.finished}, walls


def main():
    t0 = time.time()
    eng, splits = build_resident()
    dev0 = eng.w_emb.device
    print(f"[t0023i] resident in {time.time() - t0:.0f} s, layer split "
          f"{splits}", flush=True)
    #2. warm BOTH arms' JIT (the fused kernel compiles two specializations,
    #   SWA and global; the eager arm warms cuBLAS/softmax) — discarded
    t0 = time.time()
    _run(eng, dev0, True, warm=True)
    _run(eng, dev0, False, warm=True)
    print(f"[t0023i] warm-up {time.time() - t0:.0f} s", flush=True)
    #3. interleaved arms
    tok_off1, w_off1 = _run(eng, dev0, False)
    tok_on1, w_on1 = _run(eng, dev0, True)
    tok_off2, w_off2 = _run(eng, dev0, False)
    tok_on2, w_on2 = _run(eng, dev0, True)
    for tag, w in (("eager  run1", w_off1), ("fused  run1", w_on1),
                   ("eager  run2", w_off2), ("fused  run2", w_on2)):
        print(f"[t0023i] arm a  {tag}: steps "
              f"{' '.join(f'{x:.3f}' for x in w[:6])} | total "
              f"{sum(w):.3f} s", flush=True)
    off0 = (w_off1[0] + w_off2[0]) / 2
    on0 = (w_on1[0] + w_on2[0]) / 2
    print(f"[t0023i] arm a  PREFILL step 0 ({RESIDENT} rows): {off0:.3f} "
          f"-> {on0:.3f} s = {off0 / on0:.3f}x (-{off0 - on0:.3f} s)",
          flush=True)
    if len(w_off1) > 1 and len(w_on1) > 1:
        off1 = (w_off1[1] + w_off2[1]) / 2
        on1 = (w_on1[1] + w_on2[1]) / 2
        print(f"[t0023i] arm a  MIXED step 1 ({COHORT} prefill + "
              f"{RESIDENT} decode): {off1:.3f} -> {on1:.3f} s = "
              f"{off1 / on1:.3f}x (-{off1 - on1:.3f} s)", flush=True)
    tot_off = (sum(w_off1) + sum(w_off2)) / 2
    tot_on = (sum(w_on1) + sum(w_on2)) / 2
    print(f"[t0023i] arm a  SCHEDULE total: {tot_off:.3f} -> {tot_on:.3f} "
          f"s = {tot_off / tot_on:.3f}x", flush=True)
    #4. arm b — tokens under both arms (sanity, not the gate)
    same = tok_off1 == tok_on1
    ntok = sum(len(v) for v in tok_off1.values())
    print(f"[t0023i] arm b  eager tokens == fused tokens: {same} "
          f"({len(tok_off1)} seqs, {ntok} tokens)", flush=True)
    if not same:
        diff = [k for k in tok_off1 if tok_off1[k] != tok_on1.get(k)]
        print(f"[t0023i]        differing seqs: {diff}", flush=True)
        for k in diff[:2]:
            print(f"[t0023i]        {k}: eager {tok_off1[k]} vs fused "
                  f"{tok_on1[k]}", flush=True)
    #5. arm c — same-schedule determinism of the fused arm (the gate)
    det = tok_on1 == tok_on2
    print(f"[t0023i] arm c  fused run1 == run2 (bitwise tokens): {det}",
          flush=True)
    #   eager repeatability, for reference on the same schedule
    print(f"[t0023i] arm c  eager run1 == run2: {tok_off1 == tok_off2}",
          flush=True)
    ok = det
    print(f"[t0023i] RESULT {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
