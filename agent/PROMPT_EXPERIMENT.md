# PROMPT_EXPERIMENT.md — one optimization iteration (P6 / Stage 3)

You are one iteration of the tm-opt experiment loop. The repo is your memory.
Precondition (verify, else stop): B3.5 is ticked in PROGRESS.md (parity green
under D13) and `experiments/ledger.jsonl` contains iteration-0 rows for BOTH
engines. If either is missing, write nothing and exit.

## Do, in order
1. Read `CLAUDE.md`, `CONTEXT_AND_PLAN.md` (D4 ranking; D12 merge bars; D13
   parity semantics), `docs/ARCHITECTURE.md` §"Where the wins must come from",
   `docs/ROOFLINE.md`, and the LAST 20 lines of `experiments/ledger.jsonl`
   (what worked, what dead-ended — do not repeat dead ends).
2. Backpressure check: if `experiments/queue/` has ≥2 pending specs, STOP —
   write nothing, exit. The dispatcher is behind; more specs help nobody.
3. Form ONE hypothesis: a single mechanism with a predicted effect on the
   canonical workload, grounded in a ledger observation or a roofline number,
   not vibes. CURRENT STATE OF THE WORLD (2026-07-31): decode_heavy best =
   89.7 tok/s timeboxed(600s, conc=64) — 56x over start. Iters 8-15 landed
   pooled batched attention (iter 8, 4.14x step wall), pooled sconv tails
   (iter 9), prefill GEMM BLOCK_M (iter 10), hardware FP4 cvt dequant
   (iter 11), the CUDA-graphed decode step (iter 13 — dispatch floor
   LOCKED: the step is now GPU-BOUND; its kernel table is the current
   ground truth), and fused two-shape flash-decode attention (iter 14,
   -150 ms/step). Rejected-but-instructive: exp-0011 (in-situ phase
   profile — wall-share profiling cannot see GPU near-parity) and
   exp-0014 (batched group prefill — 1.30x engine-level ceiling is real
   but arrival-race fragmentation ate it in situ; post-mortem in
   DEADENDS). The exp-0012 graphed-step kernel table + exp-0013 residuals
   rank the surface:
     #1 exp-0015 NATIVE FP4 MMA WIRING: the CUTE DSL NVFP4 grouped MoE
        kernel is exact-validated on GPU 4 (DEADENDS addendum 3; 408 us
        vs ~4.2 ms/layer Triton W4A16 decode share) — integrate into
        PackedExperts + the graphed pooled step per the committed split
        plan (exp-0015a activation-quant kernel is bitwise-validated
        groundwork). Worth up to ~220 ms of the ~320 ms decode step AND
        the dominant prefill GEMM term simultaneously. Verify EARLY:
        capture-safety under CUDA graphs, ragged 1-2-row decode segments
        (the source's fake-dimension token_offset handling), and both
        determinism arms.
     #2 batched-prefill RETRY with the arrival race fixed (exp-0014
        post-mortem: length-sorted grouping works at engine level, 1.30x;
        the loss was admission fragmentation + 3 dropped bench threads),
     #3 conc 64->128 re-push AFTER #1 lands (native MMA changes the
        rows-per-expert batch calculus),
     #4 remaining copy elimination per the exp-0012 kernel table,
     #5 chunked prefill, #6 scheduler specialization at conc 512,
     #7 TP comm overlap, #8 topology A/B (2xTP=4 vs 8).
   The cycle is PREFILL-TERM-DOMINATED (~2/3 of cohort wall) — attack #1
   moves both terms at once.
   Determinism rule for CUDA graphs / compiled paths: same-schedule
   bitwise repeatability still binds (D13 envelope + the t_b3 batch
   determinism arm) — capture/compile must not introduce run-to-run
   nondeterminism.
4. Implement it in a worktree: `git worktree add experiments/wt-<id> main`,
   change the ONE thing, keep the diff minimal and inside engine/.
5. Local pre-gate (cheap, GPUs 4-7): run the D13 teacher-forced parity check
   against your local engine build:
   `python harness/correctness.py --endpoint <local> --teacher-forced
    --envelope experiments/tf_envelope.json`
   Red → fix or abandon (document either way); never submit red.
   BUDGET the run first (see hard rules): server load ~2 min + gate ~20 min
   fits one session; anything you estimate >40 min does NOT get started.
6. Submit: write `experiments/queue/<id>.json` following
   `experiments/example_spec.json`, with fields REQUIRED by
   docs/LEDGER_SCHEMA.md: `label` (2-4 words, chart hover title) and
   `mechanism` (one line, chart hover subtitle), plus hypothesis, predicted
   delta, worktree ref, commit. AUTHOR THE SPEC OUTSIDE queue/ AND mv IT IN
   (atomic — the dispatcher polls every 15 s and must never see a
   half-written file). Before writing the spec, verify your commit contains
   current main: `git merge-base --is-ancestor main <your-commit>` must
   succeed — a spec pinning a pre-merge ref measures a stale engine.
7. Commit the worktree + spec:
   `git add <files> && git commit -m "[ralph] exp(<id>): <hypothesis in 5-8 words>"`
   — the [ralph] prefix is MANDATORY on every commit you make, and <id>
   must match the spec filename. Stop. The dispatcher/worker run the
   canonical gates; you never run canonical benchmarks yourself and never
   merge to main.

## Hard rules
- One variable per experiment. If your diff does two things, split it.
- NEVER modify harness/, configs/, goldens/, ledger, prompts, loop scripts.
- Never touch /workspace/maverick. GPUs 4-7 only for local runs.
- Your session ends when you stop responding — there is NO "later".
  Background execution is FORBIDDEN in every form: no run_in_background, no
  watchdogs, no "I'll be re-invoked" — you will NOT be. Long commands run as
  ONE FOREGROUND Bash call with an explicit large timeout (the environment
  permits up to ~55 minutes).
- BUDGET CHECK before starting any test or long command: estimate wall time
  first; if the estimate exceeds ~40 minutes, do NOT start it — mark the
  situation in a committed note with the estimate and a proposed split.
- If you end an iteration without committing FOR ANY REASON, you MUST first
  commit a note (IN-PROGRESS or BLOCKED) explaining why — a silent no-commit
  iteration is a rule violation. For runs near the budget edge, commit a
  hedge note BEFORE starting.
- Every commit message you write starts with the literal prefix `[ralph] ` —
  no exceptions, including post-mortems and notes.
- A rejected experiment is DATA: if your last spec was rejected (see ledger),
  your first duty is a one-paragraph post-mortem appended to
  `experiments/DEADENDS.md` before proposing anything new.
- Numbers that look too good are bugs until proven otherwise: check output
  lengths, check the gate actually ran, check the cache_key matches, check
  GPU exclusivity (nvidia-smi snapshot is embedded in benchmark records).
- Correctness semantics for fused/batched kernels: bitwise identity vs
  batch-1 is UNATTAINABLE in principle (bf16 row-count accumulation drift,
  B3.1/B3.3). The binding gates are (a) the D13 teacher-forced envelope
  parity check and (b) t_b3 batch's determinism arm adapted by the worker:
  two identical runs of the SAME schedule must be bitwise-identical (your
  kernel must stay deterministic even if it is no longer batch-size-
  invariant). Design for both from the start.