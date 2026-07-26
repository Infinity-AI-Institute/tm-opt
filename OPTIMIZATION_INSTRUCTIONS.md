# OPTIMIZATION_INSTRUCTIONS.md — Inkling on 8×B300 (adapted)

Adapted from Infinity's Qwen3/H100 original for this project's model,
hardware, and frozen contract. This is the fairness constitution: every
experiment, merge, and reported number is bound by it. Where the original
said FP8/H100/conc=88, this document says what OUR ledger enforces.
Companion decisions: D1–D12 in CONTEXT_AND_PLAN.md.

## 1. The comparison

- Model: `thinkingmachines/Inkling` full model, NVFP4 checkpoint at
  `/workspace/models/inkling-nvfp4` — IDENTICAL directory for both engines.
  KV cache dtype bf16 both engines (checkpoint declares kv_cache_quant "none").
- Baseline: pinned vLLM `0.23.1rc1.dev1270+g9243e0124` (wheel archived at
  /workspace/wheels), official recipe flags, TP=4 on GPUs 0–3
  (`scripts/serve_vllm_bench.sh` is the only measurement config).
- Candidate: `engine/pyengine` (Triton-first; CUDA ports land as experiments).
- Hardware: one Runpod 8×B300 node. Multi-NODE is out of scope (D9);
  the multi-GPU challenge is realized as TP=4 NCCL + replica orchestration.

## 2. The frozen measurement contract (cache_key 8451a604a8849296)

| parameter | decode_heavy (headline) | prefill_heavy |
|---|---|---|
| ISL | ~1024 (range_ratio 0.5, seed 1001) | ~8192 (range_ratio 0.5, seed 1002) |
| OSL (ignore_eos) | 8192 | 1024 |
| concurrency | 512 | 512 |
| warmup / measured | 1024 / 3×2560 requests | 1024 / 3×2560 requests |
| prefix caching | OFF | OFF |
| KV | pinned bytes: --kv-cache-memory 111231943107 | same |
| MTP / speculative | OFF both engines (canonical); MTP-ON tracked pair (D7) | same |

- Concurrency 512 is the measured PEAK (collapse past it = KV-capacity edge;
  evidence docs/logs/2026-07-2{2,3}_sweep_*).
- The cache_key is the SHA-16 of these parameters. Every ledger row carries
  it; rows with different keys are never compared. Editing any canonical
  config re-keys and orphans all prior numbers — tampering is self-defeating.
- Prompt generator: `make_prompts()` — byte-identical in freeze_canonical.py
  and harness/benchmark.py. Counting: usage.completion_tokens; timing:
  wall-clock over the measured block; non-streamed.

## 3. Baselines and merge bars (measured 2026-07-26)

| workload | vLLM baseline | noise floor | merge bar (max(2×noise, 0.3%)) |
|---|---|---|---|
| decode_heavy | 13,409.5 tok/s | 0.12% | +0.30% (~+40 tok/s) |
| prefill_heavy | 4,999.3 tok/s | 0.82% | +1.64% (~+82 tok/s) |

A candidate result merges only if parity is green AND it exceeds the current
ledger best by the workload's merge bar.

## 4. Invalid optimizations (automatic disqualification)

4.1 **Benchmark manipulation**
- Changing ISL/OSL/concurrency/seeds/range_ratio/warmup counts from the
  frozen configs; benchmarking at a cherry-picked concurrency.
- Prefix caching on (either engine); reusing identical prompts so caching
  or KV reuse inflates throughput.
- KV sizing games: utilization fractions instead of pinned bytes; unequal
  KV budgets between engines.
- Comparing runs with different cache_keys, or quoting warm-vs-cold
  asymmetrically. Server init (JIT/graph capture) is excluded on BOTH sides.

4.2 **Model manipulation**
- Any checkpoint other than the pinned NVFP4 directory; precision swaps;
  pruning/distilling/quantizing further; skipping layers/experts; KV
  eviction, approximation, or windowing beyond trained semantics.
  (Window-512 SWA on the 55 local layers IS trained semantics; verify_config
  enforces the architecture at startup.)
- Reducing computation below trained semantics in any form. Speculative
  decoding (MTP) is output-identical and allowed ONLY in the MTP-ON tracked
  pair, both engines eligible, never in the canonical headline.

4.3 **Measurement integrity**
- Engines never run simultaneously during ANY measurement; GPUs 4–7 must
  read ~0 during baseline runs and vice versa (the benchmark record embeds
  an nvidia-smi snapshot — it is checked).
- Timing is owned by harness/benchmark.py; engines never self-report.
- experiments/ledger.jsonl is append-only and written ONLY by the
  harness/worker. goldens/, configs/, harness/ are read-only (chmod +
  worker diff-rejection + prompt rules). A patch touching them is invalid
  regardless of its numbers. Tolerances (LOGPROB_TOL, merge bars) loosen
  only with a written ledger justification.
- Parity gate: reference = committed goldens (D11): pinned vLLM under
  VLLM_BATCH_INVARIANT=1 (0/50 re-run diff, sha 74d9776). The goldens flag
  is NEVER set on the benchmark server (vLLM is measured at its fastest).
  Candidates face the gate in their own deterministic mode
  (TMOPT_DETERMINISTIC=1). transformers reference retired: loader defect on
  grouped-fp4 experts (docs/logs/2026-07-24 probe runs); bf16 spot-check
  pending as independence hardening (Path B).

4.4 **Known failure patterns (from the original project — pre-answered here)**
- "210% via identical prompts" → seeded varied prompts are IN the frozen key.
- Reference-cache tampering → goldens committed + integrity-checked by the
  gate (prompt drift, count, sha) on every run.
- Too-good-to-be-true numbers are treated as harness bugs first: check
  output lengths, gate execution, cache_key, GPU exclusivity snapshot.

## 5. Experiment protocol (Stage 3)

- One hypothesis, one variable, one worktree, one spec JSON per iteration
  (agent/PROMPT_EXPERIMENT.md is the binding brief).
- Local parity pre-gate on GPUs 4–7 before submission; dispatcher/worker run
  canonical gates; agents never run canonical benchmarks or merge to main.
- Every accepted row carries: iteration, label, mechanism, workload,
  tok_per_s, pct_vs_baseline, baseline_id, cache_key, commit, noise_floor_pct,
  log_path (docs/LEDGER_SCHEMA.md). Rejections are data → DEADENDS.md.
- Claims for the write-up come from same-hour sequential A/B
  (vLLM → candidate → vLLM), both workloads, MTP-off headline + MTP-on pair.

## 6. Reporting

Graph + iteration log are GENERATED from the ledger + canonical.lock.json
(`scripts/plot_trajectory.py`); never hand-edited. Baselines are the dashed
lines; rejected experiments appear in the log. If the chart needs a datum,
the fix is a ledger field, not a manual annotation.