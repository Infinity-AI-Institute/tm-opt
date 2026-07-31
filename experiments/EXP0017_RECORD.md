# exp-0017-cohort-prefill — record (2026-07-30)

Base: f09315b (exp-0015b, ACCEPTED ledger iter 18 @ 194.5 tok/s — native
W4A4 in both the prefill and the graphed decode routed-expert GEMM) merged
with main ac8e8a9.

## Hypothesis (one mechanism)

The native NVFP4 grouped MMA pads every expert group to a 128-row M tile,
so an expert's GEMM cost is FLAT from 1 to 128 rows. One canonical prompt
(T~745, top-6 over 256 experts) routes only ~17 rows per expert — the tile
runs 87% empty — and prefill pays that walk once PER SEQUENCE. Co-prefilling
a cohort fills the tile that is already paid for.

exp-0014 built the batched walk (right-padded groups, length-sorted,
partitioned under the 4 GiB padded-scores budget, per-row state seeding,
singleton fall-through) and measured 1.30x at engine level — under W4A16,
where GEMM cost still scaled with rows, so grouping only amortized launch
overhead. It was REJECTED in situ because its 150 ms wall-clock accumulation
window raced per-completion HTTP arrivals and admitted mostly singletons.

This experiment keeps that machinery verbatim (cherry-picked d9f4525 +
84e67ca + ceb9a91) and changes ONE thing: how admission forms cohorts.

## The admission fix (server.py EngineLoop)

Arrivals land in a pool instead of going straight to the scheduler, which
otherwise admits each due arrival the instant a slot frees — the exact
fragmentation the post-mortem named. Release when:
  (a) nothing is generating -> admit everything (D13 gate's sequential
      singletons, cold start, drain tails keep the batch-1-exact per-seq
      path and never stall);
  (b) PREFILL_COHORT=6 prompts have gathered AND >= 6 slots are free, i.e.
      a cohort actually retired and made room — never release more than
      the free slots, since a request the scheduler cannot admit now would
      sit in its waiting list and be admitted alone later;
  (c) COHORT_HOLD_S=4.0 valve (bootstrap, ragged tails).
Released together, a cohort prefills as ONE group, decodes in lockstep and
retires together, so its replacements co-arrive: the structure is
self-sustaining after one round. This is the scheduler-state trigger the
exp-0014 post-mortem asked for; wall clock enters only through the valve.
Scheduler.step() stays a pure function of its input schedule, so the t_b3
same-schedule determinism arm (which pins arrival steps and drives the
scheduler directly) is untouched.

Carried alongside as measurement hygiene, at the exp-0014 post-mortem's
explicit request: listen backlog 5 -> 256. That artifact dropped 3 of 64
bench threads before they wrote a status line and took -4% off exp-0014's
headline (89.7 x 61/64 = 85.5 ~ its measured 86.0). A kernel pre-accept
queue depth cannot make a served request faster.

## GPU evidence (GPUs 4-7, logs in /workspace/logs)

t_prefill_batch_exp0017_warm_out.log — one resident load, N=64 canonical-ISL
prompts (lens 500..989, sum 47704), same engine and pools:
  S   serial per-seq (today's path), COLD    54.83 s (856.7 ms/seq)
  G   grouped, groups [45, 19]               12.91 s (201.7 ms/seq)
  G2  grouped re-run, fresh states           12.85 s (200.9 ms/seq)
  S2  serial per-seq, WARM                   24.35 s (380.5 ms/seq)
  S/G = 4.25x but 56% of arm S is first-call JIT (CUTE DSL grouped-GEMM
  compiles per device x shape, Triton autotune, cuBLAS heuristics) — the
  steady-state bench never pays it again, so the honest ratio is
  S2/G = 1.89x. Reported both; the prediction uses 1.89x.
  Cross-check: S2's 380.5 ms/seq matches the ~370 ms/seq the accepted
  194.5 tok/s row implies for its prefill term (911 requests x p +
  ~253 s decode = 600 s box), so the arm is measuring the real path.
  G == G2 BITWISE on tokens AND fp32 logit rows (determinism arm at the
  prefill seam). S-vs-G first-token agree 50/64, max |logit delta| 1.8125
  — the B3.1/B3.3 row-count drift class, the one D13's envelope covers,
  same class batched decode already carries.

Group-size fit (two points, per-seq = A/G + B): A = 183.7 ms flat,
B = 196.8 ms scaling -> 1.67x at G=6, 1.79x at G=12, 1.89x at G=37.
Cohort target trades against decode rows held out of the graphed step
(whose cost is B-independent, so tokens/step is proportional to live rows):
modelled throughput is flat at 247-248 tok/s for G in 6..12 and falls to
236 at G=3, so PREFILL_COHORT=6 sits on the plateau.

## D13 teacher-forced gate (tf_gate_exp0017.log) — GREEN

parity_pass true, agree 379/400 = 0.9475 (envelope floor 0.965 is the vLLM
arm; the engine's accepted bar is the exp-0016/0015b row), delta_mean
0.043856 (bar 0.054281), delta_max 0.396472 — BITWISE the accepted
exp-0016/exp-0015b gate numbers. Expected and required: the gate's
sequential singleton stream leaves the running set empty, release rule (a)
fires, and every prompt takes the unchanged per-seq path. It also proves
the pool cannot stall a sequential client.

## Predicted delta

Cycle model on the accepted 194.5 tok/s row (600 s box, conc 64, osl 128 ->
911 requests; prefill 911 x 0.3805 = 347 s = 58% of the box, decode 253 s):
prefill 347 -> 208 s at G=6, decode 253 -> ~265 s from the held rows ->
+25-37%, i.e. 245-267 tok/s against a merge bar of 195.1. Floor case: if
the bench's arrivals still fragment, groups stay singletons and the number
lands back at ~194.5 (the per-seq path is unchanged), which is itself the
measurement that closes out the batched-prefill line.
