"""exp-0001 evidence: batched decode across co-resident sequences.

Three arms over ONE fixed 8-request schedule (6 arrivals at step 0, 2 at
step 3, max_new 16-24 so the batch grows AND shrinks mid-run):

  A. reference — per-sequence batch-1 loop (Engine.prefill + Engine.decode),
     numerically the pre-exp-0001 scheduler path (t_b3's oracle form).
  B. batched   — the new Scheduler path (Engine.decode_batch): one stack
     traversal per step for all decodable co-residents.
  C. determinism — arm B rerun with fresh state on the identical schedule;
     tokens AND captured fp32 logits rows must be BITWISE identical (the
     adapted t_b3 determinism gate: same schedule -> same bits, even though
     bf16 row-count drift makes batched != batch-1 in principle, B3.1/B3.3).

Reported: per-request token agreement A vs B (with the fp32 top1-top2
margin at any divergence — drift is only legitimate where the margin is
small), chosen-logprob delta stats over the agreeing prefix, aggregate
decode tok/s for A and B, and the C bitwise verdict. GPUs 4-7 only:
    CUDA_VISIBLE_DEVICES=4,5,6,7 python -m engine.pyengine.tests.t_batched
"""
import time

import torch

from engine.pyengine import scheduler as psched
from engine.pyengine import server as pserver


def _requests(tok, dev):
    """The fixed corpus: 8 (ids, max_new, arrival) triples from the B2
    prompt set (varied lengths; two staggered arrivals)."""
    from engine.pyengine.tests.t_b2 import PROMPTS5
    texts = list(PROMPTS5) + [
        PROMPTS5[0] + " " + PROMPTS5[1],
        PROMPTS5[2] + " Explain step by step.",
        PROMPTS5[3] + " " + PROMPTS5[4],
    ]
    max_new = [24, 20, 16, 24, 18, 22, 16, 20]
    arrival = [0, 0, 0, 0, 0, 0, 3, 3]
    ids = [torch.tensor(tok(t).input_ids, device=dev) for t in texts]
    return list(zip(ids, max_new, arrival))


def _margin(row, k=2):
    #1. fp32 top1-top2 logit gap of one captured row
    top = torch.topk(row, k).values
    return float(top[0] - top[1])


def _chosen_lp(row, tid):
    #1. fp32 log-softmax of the chosen token (server.chosen_logprobs form)
    return float(row[tid] - torch.logsumexp(row, 0))


def _run_batched(eng, corpus):
    """Arm B/C: the new scheduler path on the fixed schedule."""
    sched = psched.Scheduler(eng, max_batch=8)
    reqs = []
    for i, (ids, n, arr) in enumerate(corpus):
        r = psched.Request(f"r{i}", ids, n, capture=[])
        reqs.append(r)
        sched.submit(r, arrival_step=arr)
    torch.cuda.synchronize()
    t0 = time.time()
    sched.run()
    torch.cuda.synchronize()
    return reqs, time.time() - t0


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(pserver.MODEL_DIR,
                                        trust_remote_code=True)
    eng, _ = pserver.build_resident()
    corpus = _requests(tok, eng.w_emb.device)
    n_dec = sum(n - 1 for _, n, _ in corpus)  # decode tokens (prefill excl.)

    #1. arm A: per-sequence batch-1 reference (pre-exp-0001 numerics)
    ref = []
    t_pre = 0.0
    torch.cuda.synchronize()
    t0 = time.time()
    for i, (ids, n, _) in enumerate(corpus):
        r = psched.Request(f"r{i}", ids, n, capture=[])
        r.states = eng.new_states()
        tp = time.time()
        r.tokens.append(eng.prefill(r))
        torch.cuda.synchronize()
        t_pre += time.time() - tp
        while len(r.tokens) < n:
            r.tokens.append(eng.decode(r))
        r.states = None
        ref.append(r)
    torch.cuda.synchronize()
    t_ref = time.time() - t0 - t_pre
    print(f"[A ref    ] {n_dec} decode tok in {t_ref:.1f} s "
          f"({n_dec / t_ref:.2f} tok/s aggregate; prefills {t_pre:.1f} s)",
          flush=True)

    #2. arm B: batched scheduler on the same schedule
    bat, t_bat = _run_batched(eng, corpus)
    print(f"[B batched] {n_dec} decode tok in {t_bat:.1f} s wall incl. "
          f"prefills ({n_dec / max(t_bat - t_pre, 1e-9):.2f} tok/s "
          f"aggregate est. on decode wall {t_bat - t_pre:.1f} s) -> "
          f"decode speedup ~{t_ref / max(t_bat - t_pre, 1e-9):.2f}x",
          flush=True)

    #3. A vs B: token agreement + margins at divergence + chosen-lp deltas
    agree_tot, tot, deltas = 0, 0, []
    for ra, rb in zip(ref, bat):
        n = min(len(ra.tokens), len(rb.tokens))
        div = next((s for s in range(n) if ra.tokens[s] != rb.tokens[s]), None)
        upto = n if div is None else div
        agree_tot += upto
        tot += n
        deltas += [abs(_chosen_lp(ra.capture[s], ra.tokens[s])
                       - _chosen_lp(rb.capture[s], rb.tokens[s]))
                   for s in range(upto)]
        note = ("all agree" if div is None else
                f"DIVERGES at s={div} (ref margin "
                f"{_margin(ra.capture[div]):.4f})")
        print(f"[A/B {ra.id}] {upto}/{n} tokens agree — {note}", flush=True)
    dmax = max(deltas) if deltas else float("nan")
    dmean = sum(deltas) / len(deltas) if deltas else float("nan")
    print(f"[A/B] agreement {agree_tot}/{tot} "
          f"({100.0 * agree_tot / tot:.2f}%), chosen-lp delta over "
          f"agreeing prefix: mean {dmean:.6f} max {dmax:.6f}", flush=True)

    #4. arm C: same schedule again -> bitwise tokens AND logits rows
    det, _ = _run_batched(eng, corpus)
    bits = all(rb.tokens == rc.tokens
               and len(rb.capture) == len(rc.capture)
               and all(torch.equal(x, y)
                       for x, y in zip(rb.capture, rc.capture))
               for rb, rc in zip(bat, det))
    print(f"[C determinism] same-schedule rerun bitwise "
          f"{'PASS' if bits else 'FAIL'}", flush=True)
    if not bits:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
