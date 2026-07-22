# PROMPT_EXPERIMENT.md — one optimization iteration (P6 / Stage 3)

You are one iteration of the tm-opt experiment loop. The repo is your memory.
Precondition (verify, else stop): parity gate green on main, and
`experiments/baseline_vllm.json` + `configs/canonical.lock.json` exist.

## Do, in order
1. Read `CLAUDE.md`, `CONTEXT_AND_PLAN.md` (D4 win-burden ranking),
   `docs/ARCHITECTURE.md` §"Where the wins must come from",
   `docs/ROOFLINE.md`, and the LAST 20 lines of `experiments/ledger.jsonl`
   (what worked, what dead-ended — do not repeat dead ends).
2. Backpressure check: if `experiments/queue/` has ≥2 pending specs, STOP —
   write nothing, exit. The dispatcher is behind; more specs help nobody.
3. Form ONE hypothesis: a single mechanism with a predicted effect on the
   canonical workload (e.g. "fusing sconv into the attention-K epilogue
   removes 55×3 kernel launches per decode step → predict +2-4% decode tok/s").
   Ground it in a ledger observation or a roofline number, not vibes.
4. Implement it in a worktree: `git worktree add experiments/wt-<id> main`,
   change the ONE thing, keep the diff minimal and inside engine/.
5. Local pre-gate (cheap, GPUs 4-7): run
   `python harness/correctness.py --endpoint <your local engine>` on the
   parity subset. Red → fix or abandon (document either way); never submit red.
6. Submit: write `experiments/queue/<id>.json` following
   `experiments/example_spec.json`, with fields REQUIRED by
   docs/LEDGER_SCHEMA.md: `label` (2-4 words, chart hover title) and
   `mechanism` (one line, chart hover subtitle), plus hypothesis, predicted
   delta, worktree ref, commit.
7. Commit the worktree + spec. Stop. The dispatcher/worker run the canonical
   gates; you never run canonical benchmarks yourself and never merge to main.

## Hard rules
- One variable per experiment. If your diff does two things, split it.
- NEVER modify harness/, configs/, goldens/, ledger, prompts, loop scripts.
- Never touch /workspace/maverick. GPUs 4-7 only for local runs.
- A rejected experiment is DATA: if your last spec was rejected (see ledger),
  your first duty is a one-paragraph post-mortem appended to
  `experiments/DEADENDS.md` before proposing anything new.
- Numbers that look too good are bugs until proven otherwise: check output
  lengths, check the gate actually ran, check the cache_key matches.
