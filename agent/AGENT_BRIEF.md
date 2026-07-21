# Agent brief — tm-opt experiment worker

You are one of up to six agent sessions improving the tmopt inference engine
for Inkling. You produce ONE experiment per task.

## Contract
1. One hypothesis, one variable. State it in a single sentence.
2. Author a patch against `main` (git diff format) into `experiments/patches/`.
3. Write a spec JSON (copy `experiments/example_spec.json`) into
   `experiments/queue/`. The dispatcher runs everything else.
4. Read `experiments/ledger.jsonl` before proposing: never repeat a rejected
   hypothesis without a materially different mechanism.

## Hard rules
- Never edit `harness/`, `configs/*.json`, or the ledger.
- Never loosen `LOGPROB_TOL` or the accept threshold to make a result pass.
- No code from other Infinity repositories — public sources only.
- Determinism first: fixed reduction orders in kernels; the parity gate is
  batch-sensitive by design.

## Prioritized attack surface (from docs/ARCHITECTURE.md)
1. moe_grouped_gemm tiles/stages/work-queue scheduling
2. decode_attention split-KV sizing and vector widths
3. Scheduler token budget + chunked prefill sizes
4. CUDA graphs on the decode path
5. MTP speculative decode (draft len sweep)
