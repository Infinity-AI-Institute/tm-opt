# CLAUDE.md — tm-opt project context

Read this fully before acting. It encodes the project's goal, hard rules,
current state, and verified technical facts.

## Read first, in order
1. `CONTEXT_AND_PLAN.md` — decision record (D1–D9) + execution plan (P1–P6).
   Supersedes older docs on conflict; if you find a conflict, fix the older
   doc in the same commit.
2. `PROGRESS.md` — (once it exists) the current task list; pick the next
   unchecked item, nothing else.
3. `docs/BASELINE_NOTES.md` — measured facts; never re-derive these.
4. `PLAN.md` holds the stage checklists.
5. `OPTIMIZATION_INSTRUCTIONS.md` (adapted) holds the fairness constitution.
6. `docs/ARCHITECTURE.md` (rev 3) holds the engine design.

## Goal

Replicate Infinity's "Surpassing vLLM with a Generated Inference Stack" for
Thinking Machines **Inkling** (full model; Inkling-Small unreleased) on a
Runpod 8×B300 node. Beat vLLM's throughput on a frozen canonical workload
with a from-scratch engine (Triton-first, CUDA replacing Triton during
optimization — decision D5), improved via an agent-driven experiment loop,
with every claim surviving a fairness audit.

## Hard rules (non-negotiable)

1. **Never open, read, grep, or delete `/workspace/maverick`** — prior
   occupant's work; trial forbids accessing other Infinity code.
2. Public sources only: vLLM docs/code, HF transformers, model card, CUTLASS.
3. `harness/`, `configs/*.json`, `goldens/`, `experiments/ledger.jsonl`, and
   the vLLM reference cache are read-only/append-only. A patch touching them
   is invalid regardless of its numbers. Never loosen parity tolerances or
   accept thresholds to make a result pass.
4. Both engines always run the identical checkpoint and KV dtype, identical
   canonical workload, never simultaneously. No computation reduction beyond
   the model's trained semantics (window-512 SWA on the 55 local layers IS
   the trained semantics; KV eviction/approximation/layer-skip are NOT).
   Canonical configs run MTP OFF on both engines; an MTP-ON config pair is
   tracked separately (D7).
5. One hypothesis per experiment; correctness gate before perf; results below
   best + 2×noise floor (min 0.3%) don't merge.
6. Credentials: never commit tokens; everything credential-like is listed in
   `docs/SECURITY_LEDGER.md` for end-of-trial revocation.
7. Long-running work lives in tmux (server = session `serve`; detach with
   Ctrl-b d, NEVER Ctrl-C inside it), artifacts live in `/workspace` or git;
   everything outside `/workspace` dies on pod restart (re-setup steps
   accumulate in `scripts/pod_init.sh`).

## Environment (verified)

- Pod: Runpod 8×B300 SXM6 (sm_103a), 275,040 MiB (~267.7 GiB usable) each,
  full NV18 NVSwitch mesh, 2 NUMA domains (GPU0–3 ↔ cores 0-85; GPU4–7 ↔
  86-171). NUMA-pin serving/bench processes.
- RAM 3.9 TiB; `/dev/shm` 1.9 TB (remountable larger) — BF16 staging if needed.
- `/workspace`: 1.5 TB volume (persistent). Venv at `/workspace/venv`.
- CUDA toolkits: 12.8 (default symlink) AND **13.0 at /usr/local/cuda-13.0
  (installed; REQUIRED — 12.8 cannot compile sm_103a; FlashInfer JIT fails
  without it)**. Serve script exports CUDA_HOME=/usr/local/cuda-13.0.
  PyTorch is cu130. apt re-install lines live in pod_init.sh (container disk
  is ephemeral).
- vLLM baseline build: wheel `vllm-0.23.1rc1.dev1270+g9243e0124` (nightly;
  version string misleadingly old — commit hash is the identifier). Wheel
  archived at `/workspace/wheels/`. Registry contains
  `InklingForConditionalGeneration`. scipy was a missing dep (--no-deps
  install), now fixed.
- **Baseline SERVES** (2026-07-19): `scripts/serve_vllm_baseline.sh`, TP=4 on
  GPUs 0–3, port 8106, greedy output correct. GPUs 4–7 idle (goldens /
  second replica). Cold init 1748 s; FlashInfer JIT + autotune caches
  persisted at `/workspace/.flashinfer`. Serve script tees logs to
  `/workspace/logs/`; evidence logs are hand-curated into `docs/logs/`.
- Weights: `/workspace/models/inkling-nvfp4` (complete, 551.35 GiB,
  33 shards + `mtp.safetensors`).
- Official vLLM recipe (baseline config source of truth):
  recipes.vllm.ai/thinkingmachines/Inkling — env `VLLM_USE_V2_MODEL_RUNNER=1`,
  `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1`; flags `--tokenizer-mode inkling
  --reasoning-parser inkling --tensor-parallel-size 4 --trust-remote-code`.
  NOTE: prefix caching is ON by default; the benchmark serve variant must add
  `--no-enable-prefix-caching` and pin KV via `--kv-cache-memory`.

## Model facts (from config.json — trust these over memory; runtime-enforced
by `./build/verify_config /workspace/models/inkling-nvfp4`)

- `InklingForConditionalGeneration`, multimodal wrapper; text_config is the
  serving target (text-only v1 scope; vision/audio configs exist, unused;
  multi-NODE also out of scope — one pod, D9).
- 66 layers, hidden 6144, vocab 201024 (unpadded 200058), eos 200006,
  context 1,048,576 (we serve at 16,384). bf16 activations/KV
  (`kv_cache_quant_algo: none`).
- Attention hybrid: **11 global layers = {5,11,17,23,29,35,41,47,53,59,65}**
  (every 6th from 5); other 55 are sliding-window, window **512**.
  Global: 64 Q heads / 8 KV heads / head_dim 128.
  SWA: 64 Q heads / **16 KV heads** / head_dim 128.
  KV cost per seq: SWA fixed ≈230 MB total; global ≈45 KB/token.
- **No RoPE.** Relative attention: learned rel-pos bias (d_rel=16,
  rel_extent=1024) + log length scaling (n_floor=128000, alpha=0.1),
  added pre-softmax.
- **sconv**: window-4 short convolutions (use_sconv, kernel 4) on attention
  K/V/output and MoE output — small per-layer ring state, cached like a
  virtual window-4 KV layer.
- MoE: 256 routed experts, top-6, **sigmoid gate with bias**, norm_after_topk,
  route_scale 8.0, use_global_scale; **2 shared experts, shared_expert_sink**;
  expert intermediate 3072 (skinny GEMMs); **layer 2 is dense MLP**
  (intermediate 24576), not MoE.
- MTP: 8 chained draft layers (`num_nextn_predict_layers: 8`, own local/global
  ids), separate `mtp.safetensors`. vLLM reports ~2.7× from MTP — engine must
  implement it for the MTP-ON config pair; canonical headline is MTP-OFF (D7).
- NVFP4 weights except exclude list: embeddings/embed_norm/final norm/unembed,
  layer-0 attention, multimodal encoders (see hf_quant_config.json). Loader
  honors per-module precision.
- **API contract for the parity gate (verified live):** greedy via
  /v1/completions with `temperature:0, logprobs:1, "return_token_ids": true`
  → token IDs at CHOICE level (`choice["token_ids"]`). Our engine's server
  must implement the same fields.

## Architecture bet (rev 3 — KV lever RETIRED)

Two 4-GPU NVFP4 replicas (551 GiB / 4 ≈ 137 GiB/GPU weights measured, + KV),
pure data parallel between replicas, expert/tensor parallel within — vs
vLLM's best config (test TP=4×2 and TP=8, compare against its best).

**Verified 2026-07-19 (docs/BASELINE_NOTES.md): vLLM's Inkling integration is
window-aware in KV ALLOCATION, not just compute** (SlidingWindowSpec for the
55 local layers + sconv; empirical gauge peak 0.21% for a 12K+4K request).
The startup "10.48× concurrency" line is a worst-case reservation convention.
Therefore: per-layer-type KV (ring 512 + paged global) is PARITY WORK, not an
edge; KV memory is not binding at 16K; canonical concurrency is
compute/scheduler-bound (found empirically).

Wins must come from (priority order): CUDA-graphed decode / launch-overhead
deletion; skinny-expert grouped GEMM (N=3072, fused sigmoid→top6→
norm_after_topk→permute, device work queue); sconv fusion; two-shape
attention with fused rel-bias; scheduler specialization to the frozen
workload; MTP tuning (separate config pair); TP=4 comm/compute overlap.
Precision is parity, not an edge.

Engine implementation strategy (D5): `engine/pyengine/` — Python
orchestration + Triton kernels for bring-up speed; C++/CUDA skeleton under
`engine/` is retained as the CUDA porting roadmap (Triton→CUDA ports are
themselves Stage-3 experiments) and as the config provenance gate
(verify_config).

## Workflow

- Harness: dispatcher (queue → GPU slots) + worker (worktree → build → parity
  gate → bench → ledger). Agents propose patches + spec JSONs; never run
  canonical benchmarks directly; never merge to main (dispatcher
  fast-forwards accepted commits).
- Ralph loops: `scripts/ralph_build.sh` (P4 bring-up grind, one PROGRESS.md
  item per iteration) and `scripts/ralph_experiment.sh` (Stage 3 hypothesis
  generation with queue backpressure).
- Every ledger entry embeds the validation checklist; before/after numbers,
  commit hash, canonical cache-key, dead-ends documented.
- Report artifacts are generated FROM the ledger
  (`scripts/plot_trajectory.py` → trajectory graph + ITERATION_LOG.md).
