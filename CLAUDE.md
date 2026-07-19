# CLAUDE.md — tm-opt project context

Read this fully before acting. It encodes the project's goal, hard rules, current
state, and verified technical facts. `PLAN.md` holds the stage checklists;
`OPTIMIZATION_INSTRUCTIONS.md` (adapted) holds the fairness constitution;
`docs/ARCHITECTURE.md` holds the engine design.

## Goal

Replicate Infinity's "Surpassing vLLM with a Generated Inference Stack" for
Thinking Machines **Inkling** (full model; Inkling-Small unreleased) on a Runpod
8×B300 node. Beat vLLM's throughput on a frozen canonical workload with a
from-scratch engine, improved via an agent-driven experiment loop, with every
claim surviving a fairness audit.

## Hard rules (non-negotiable)

1. **Never open, read, grep, or delete `/workspace/maverick`** — prior occupant's
   work; trial forbids accessing other Infinity code.
2. Public sources only: vLLM docs/code, HF transformers, model card, CUTLASS.
3. `harness/`, `configs/*.json`, `experiments/ledger.jsonl`, and the vLLM
   reference cache are read-only/append-only. A patch touching them is invalid
   regardless of its numbers. Never loosen parity tolerances or accept
   thresholds to make a result pass.
4. Both engines always run the identical checkpoint and KV dtype, identical
   canonical workload, never simultaneously. No computation reduction beyond
   the model's trained semantics (window-512 SWA on the 55 local layers IS the
   trained semantics; KV eviction/approximation/layer-skip are NOT).
5. One hypothesis per experiment; correctness gate before perf; results below
   best + 2×noise floor (min 0.3%) don't merge.
6. Credentials: never commit tokens; everything credential-like is listed in
   `docs/SECURITY_LEDGER.md` for end-of-trial revocation.
7. Long-running work lives in tmux, artifacts live in `/workspace` or git;
   everything outside `/workspace` dies on pod restart.

## Environment (verified)

- Pod: Runpod 8×B300 SXM6, 275,040 MiB (~269 GB) usable HBM each, full NV18
  NVSwitch mesh, 2 NUMA domains (GPU0–3 ↔ cores 0-85; GPU4–7 ↔ 86-171).
  NUMA-pin serving/bench processes.
- RAM 3.9 TiB; `/dev/shm` 1.9 TB (remountable larger) — BF16 staging if needed.
- `/workspace`: 1.5 TB volume (persistent). Venv at `/workspace/venv`.
- CUDA toolkit 12.8 (driver 580 fine); install 12.9+/13.x alongside before
  sm_103 kernel work. PyTorch is cu130.
- vLLM baseline build: wheel `vllm-0.23.1rc1.dev1270+g9243e0124` (nightly,
  post-launch main; version string is misleadingly old — commit hash is the
  identifier). Wheel archived at `/workspace/wheels/`. Registry contains
  `InklingForConditionalGeneration`.
- Weights: `/workspace/models/inkling-nvfp4` (complete, 33 shards +
  `mtp.safetensors`).
- Official vLLM recipe (baseline config source of truth):
  recipes.vllm.ai/thinkingmachines/Inkling — env `VLLM_USE_V2_MODEL_RUNNER=1`,
  `FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1`; flags `--tokenizer-mode inkling
  --reasoning-parser inkling --tensor-parallel-size 4 --trust-remote-code`.

## Model facts (from config.json — trust these over memory)

- `InklingForConditionalGeneration`, multimodal wrapper; text_config is the
  serving target (text-only v1 scope; vision/audio configs exist, unused).
- 66 layers, hidden 6144, vocab 201024 (unpadded 200058), eos 200006,
  context 1,048,576. bf16 activations/KV (`kv_cache_quant_algo: none`).
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
  implement it to be competitive.
- NVFP4 weights except exclude list: embeddings/embed_norm/final norm/unembed,
  layer-0 attention, multimodal encoders (see hf_quant_config.json). Loader
  honors per-module precision.

## Architecture bet

Two 4-GPU NVFP4 replicas (600 GB / 4 ≈ 150 GB/GPU weights + KV headroom),
pure data parallel between replicas, expert/tensor parallel within — vs vLLM's
best config (test TP=4×2 and TP=8, compare against its best). Wins come from:
comm/compute overlap, skinny-expert grouped GEMM with device work queue,
per-layer-type KV (ring 512 + paged global), sconv fusion, CUDA-graphed decode,
MTP. Precision is parity, not an edge.

## Workflow

- Harness: dispatcher (queue → GPU slots) + worker (worktree → build → parity
  gate → bench → ledger). Agents propose patches + spec JSONs; never run
  canonical benchmarks directly; never merge to main (dispatcher fast-forwards
  accepted commits).
- Ralph loops: `scripts/ralph_build.sh` (Stage 2 checklist grind, one item per
  iteration, PROGRESS.md as memory) and `scripts/ralph_experiment.sh`
  (Stage 3 hypothesis generation with queue backpressure).
- Every ledger entry embeds the validation checklist; before/after numbers,
  commit hash, dead-ends documented.
