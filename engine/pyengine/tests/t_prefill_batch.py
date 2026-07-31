"""exp-0014 GPU evidence — serial per-seq prefill vs grouped batched prefill.

One resident engine load (server.build_resident), then on the SAME engine
and pools, three arms over one deterministic canonical-ISL-class prompt set
(N=64, lens 500..999 — the decode_heavy admission cohort shape):
  S   serial arm: eng.prefill(req) x N — the per-seq path every accepted
      bench row runs today (the 65.3 s/cohort prefill term).
  G   grouped arm: eng.prefill_batch(group) over the admission groups
      Scheduler._prefill_groups forms from the same lens in the same order.
  G2  grouped arm re-run on fresh states, same groups: first tokens AND
      captured fp32 logits rows must be BITWISE identical to G (the
      t_b3-adapted same-schedule determinism arm at the prefill seam).
Reported: wall S vs wall G (the prefill-term ratio the bench cycle model
converts to tok/s), S-vs-G first-token agreement + max |logit| delta
(row-count bf16 drift class, NOT bitwise — B3.1/B3.3; the D13 envelope is
the binding gate), G==G2 bitwise verdict (hard asserts), and per-arm state
spot checks (cache.n == len per row after each arm).

Run: CUDA_VISIBLE_DEVICES=4,5,6,7 python -m engine.pyengine.tests.t_prefill_batch
Log: tee by caller into /workspace/logs/.
"""
import time
import types

import torch

from engine.pyengine import scheduler as psched
from engine.pyengine import server as pserver

N_REQ = 64
SEED = 20260730


def prompts(mc):
    #1. deterministic canonical-ISL-class set: lens cycle 500..999,
    #   ids < unpadded vocab (t_graph_ab's generator pattern)
    g = torch.Generator().manual_seed(SEED)
    out = []
    for i in range(N_REQ):
        T = 500 + (i * 7919) % 500
        out.append(torch.randint(0, mc.vocab_unpadded, (T,), generator=g))
    return out


def fresh(eng, ids_list):
    #1. one Request per prompt, slot = list index (admission order), fp32
    #   logits capture on — the bitwise-compare payload
    reqs = []
    for i, ids in enumerate(ids_list):
        r = psched.Request(i, ids.to(eng.w_emb.device), 1, capture=[])
        r.states = eng.new_states(slot=i)
        reqs.append(r)
    return reqs


def release(eng, reqs):
    #1. hand shared-pool pages back (slot rows are overwritten by the next
    #   occupant; free-list order is deterministic given the call order)
    for r in reqs:
        eng.release_states(r.states)
        r.states = None


def main():
    #1. resident load — the exact server build
    eng, splits = pserver.build_resident()
    mc = eng.mc
    eng.init_pools(N_REQ)
    ids_list = prompts(mc)
    lens = [int(t.shape[0]) for t in ids_list]
    print(f"[t_prefill_batch] splits {splits}, N={N_REQ}, "
          f"lens {min(lens)}..{max(lens)} sum {sum(lens)}", flush=True)

    #2. arm S — serial per-seq prefill (today's path)
    reqs = fresh(eng, ids_list)
    torch.cuda.synchronize()
    t0 = time.time()
    tok_s = [eng.prefill(r) for r in reqs]
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)
    wall_s = time.time() - t0
    for r in reqs:
        assert r.states[0].cache.n == int(r.ids.shape[0]), "S cache.n"
    cap_s = [r.capture[0] for r in reqs]
    release(eng, reqs)
    print(f"[t_prefill_batch] S serial   {wall_s:8.2f} s "
          f"({wall_s / N_REQ * 1e3:6.1f} ms/seq)", flush=True)

    #3. admission groups — the exact partition Scheduler.step would form
    shim = types.SimpleNamespace(engine=eng)
    groups_idx = None

    def run_grouped(label):
        nonlocal groups_idx
        reqs = fresh(eng, ids_list)
        groups = psched.Scheduler._prefill_groups(shim, reqs)
        gi = [[r.id for r in g] for g in groups]
        if groups_idx is None:
            groups_idx = gi
            print(f"[t_prefill_batch] groups {[len(g) for g in groups]} "
                  f"(budget {psched.Engine.PREFILL_SCORES_BUDGET >> 30} GiB)",
                  flush=True)
        assert gi == groups_idx, "group partition must be schedule-pure"
        torch.cuda.synchronize()
        t0 = time.time()
        by_id = {}
        for g in groups:
            for r, tok in zip(g, eng.prefill_batch(g)):
                by_id[r.id] = tok
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)
        wall = time.time() - t0
        toks = [by_id[r.id] for r in reqs]     # request (row) order
        for r in reqs:
            assert r.states[0].cache.n == int(r.ids.shape[0]), \
                f"{label} cache.n"
        caps = [r.capture[0] for r in reqs]
        release(eng, reqs)
        print(f"[t_prefill_batch] {label} grouped {wall:8.2f} s "
              f"({wall / N_REQ * 1e3:6.1f} ms/seq)", flush=True)
        return toks, caps, wall

    tok_g, cap_g, wall_g = run_grouped("G ")
    tok_g2, cap_g2, _ = run_grouped("G2")

    #4. G == G2 bitwise (same schedule, fresh states) — hard gate
    assert tok_g == tok_g2, "grouped prefill tokens not run-repeatable"
    for a, b in zip(cap_g, cap_g2):
        assert torch.equal(a, b), "grouped prefill logits not repeatable"
    print("[t_prefill_batch] G == G2 BITWISE (tokens + fp32 logits rows)",
          flush=True)

    #5. S vs G — drift-class report (argmax agreement + logit delta)
    agree = sum(a == b for a, b in zip(tok_s, tok_g))
    dmax = max(float((a - b).abs().max()) for a, b in zip(cap_s, cap_g))
    for c in cap_g:
        assert torch.isfinite(c).all(), "non-finite grouped logits"
    print(f"[t_prefill_batch] S-vs-G first-token agree {agree}/{N_REQ}, "
          f"max |logit delta| {dmax:.4f} (B3.1/B3.3 drift class)", flush=True)
    print(f"[t_prefill_batch] RATIO serial/grouped = {wall_s / wall_g:.2f}x",
          flush=True)

    #6. arm S2 (exp-0017) — the serial arm RE-RUN on the now-warm engine.
    #   Arm S runs first and pays every first-call cost the steady-state
    #   bench never sees again (the CUTE DSL grouped-GEMM compiles per
    #   device x shape, Triton autotune, cuBLAS heuristics), so S/G
    #   overstates the ratio the cycle model needs. S2/G is the honest
    #   one; S2 must reproduce S's tokens exactly (same path, same
    #   states, deterministic kernels).
    reqs = fresh(eng, ids_list)
    torch.cuda.synchronize()
    t0 = time.time()
    tok_s2 = [eng.prefill(r) for r in reqs]
    for d in range(torch.cuda.device_count()):
        torch.cuda.synchronize(d)
    wall_s2 = time.time() - t0
    assert tok_s2 == tok_s, "warm serial arm changed the per-seq tokens"
    release(eng, reqs)
    print(f"[t_prefill_batch] S2 serial  {wall_s2:8.2f} s "
          f"({wall_s2 / N_REQ * 1e3:6.1f} ms/seq), JIT share of S "
          f"{(wall_s - wall_s2) / wall_s * 100:.0f}%", flush=True)
    print(f"[t_prefill_batch] RATIO warm serial/grouped = "
          f"{wall_s2 / wall_g:.2f}x", flush=True)


if __name__ == "__main__":
    main()
