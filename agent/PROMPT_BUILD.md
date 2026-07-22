# PROMPT_BUILD.md — one bring-up iteration (P4)

You are one iteration of the tm-opt build loop. You have no memory of prior
iterations; the repo is your memory. Work inside /workspace/tm-opt.

## Do, in order
1. Read `CLAUDE.md`, then `CONTEXT_AND_PLAN.md` (§1 facts, D-decisions), then
   `PROGRESS.md` top to bottom.
2. Select the FIRST unchecked `[ ]` item in PROGRESS.md that is not marked
   `BLOCKED`. That item — nothing else — is your entire scope this iteration.
3. Implement the smallest change that completes it. Stay inside
   `engine/pyengine/` (plus its tests). Prefer boring, obviously-correct code;
   numbered step comments (`#1.`, `#2.`) in every function.
4. Run the item's `test:` command EXACTLY as written. Paste its real output
   into your PROGRESS.md note — never summarize a test you did not run.
5. If the test passes: tick the box `[x]`, append
   `— green <one-line evidence> (commit <hash>)`, then
   `git add -A && git commit -m "build(<item-id>): <what>"`.
   If it fails after honest effort: DO NOT tick. Change the item's marker to
   `[ ] BLOCKED:` with the exact error and your best hypothesis, commit that
   PROGRESS.md edit alone, and stop.
6. Stop after one item. Do not look ahead, refactor beyond scope, or "fix"
   other items in passing.

## Hard rules (violating any invalidates the iteration)
- NEVER modify `harness/`, `configs/`, `goldens/`, `experiments/ledger.jsonl`,
  `PROMPT_*.md`, or the loop scripts.
- NEVER touch `/workspace/maverick`.
- NEVER weaken a test, tolerance, or PROGRESS item to make it pass. If an
  item's test seems wrong, mark BLOCKED and say why — a human adjudicates.
- Reference facts, don't re-derive: model shapes from
  `engine/include/tmopt/config.h` + verify_config; measured facts from
  `docs/BASELINE_NOTES.md`; API format contract from BASELINE_NOTES.
- GPU budget: GPUs 4–7 only (`CUDA_VISIBLE_DEVICES=4,5,6,7`); the vLLM
  reference server owns 0–3. Weight-loading tests may take minutes — that is
  fine; hangs >30 min are not (kill, mark BLOCKED).
- vLLM source in the venv may be READ for reference (it is public code);
  never copied wholesale — cite file:line in a comment when a design follows it.
