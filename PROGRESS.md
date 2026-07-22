# PROGRESS.md — pyengine bring-up (P4). The build loop's memory.

Rules: the loop takes the FIRST unchecked non-BLOCKED item, does ONLY that,
runs its `test:` command verbatim, pastes real output into the note, ticks,
commits. Humans may reorder/split/unblock items; the loop may not.
Env for all items: `cd /workspace/tm-opt && source /workspace/venv/bin/activate
&& export CUDA_VISIBLE_DEVICES=4,5,6,7`. Model dir: /workspace/models/inkling-nvfp4.

## B0 — scaffolding sanity
- [ ] B0.1 pyengine imports clean.
      test: `python -c "import engine.pyengine as pe; print(pe.__version__)"`
- [ ] B0.2 triton availability + toy kernel compiles on GPU 4.
      test: `python engine/pyengine/tests/test_triton_smoke.py`

## B1 — loader (safetensors → GPU tensors, NVFP4-aware)
- [ ] B1.1 shard index: enumerate 33 shards + mtp.safetensors, map tensor
      name → shard file.  test: `python -m engine.pyengine.tests.t_b1 index`
- [ ] B1.2 tensor census vs config: counts per layer match config.h shapes
      (66 layers; layer 2 dense; 11 global / 55 swa; MoE 256×ffn3072).
      test: `python -m engine.pyengine.tests.t_b1 census`
- [ ] B1.3 dtype map honors hf_quant exclude list (embeds/norms/unembed/
      layer-0 attn stay bf16; rest NVFP4 + scales).
      test: `python -m engine.pyengine.tests.t_b1 dtypes`
- [ ] B1.4 NVFP4 dequant of ONE expert weight to bf16 matches vLLM's
      dequant of the same tensor (read vLLM modelopt code for the block-scale
      layout; cite file:line).  test: `python -m engine.pyengine.tests.t_b1 dequant`
- [ ] B1.5 TP=4 sharding plan: head-parallel attention, expert-parallel MoE;
      per-GPU byte budget printed and ≤150 GiB.
      test: `python -m engine.pyengine.tests.t_b1 plan`
- [ ] B1.6 full load onto GPUs 4-7 under budget, wall time printed.
      test: `python -m engine.pyengine.tests.t_b1 load`

## B2 — model graph (single token correctness, vs transformers reference slices)
Reference: generate per-layer reference activations ONCE with
transformers(trust_remote_code) on the SAME checkpoint, tiny prompt, layers
{0,1,2,5,6}, saved to engine/pyengine/tests/ref/ (script t_b2 ref).
- [ ] B2.1 reference activations generated + committed (small tensors only).
      test: `python -m engine.pyengine.tests.t_b2 ref`
- [ ] B2.2 embed + embed_norm matches ref (rel err < 1e-2 bf16).
      test: `python -m engine.pyengine.tests.t_b2 embed`
- [ ] B2.3 rmsnorm Triton kernel matches torch reference on random tensors.
      test: `python -m engine.pyengine.tests.t_b2 rmsnorm`
- [ ] B2.4 relative-attention bias table + log scaling matches ref math
      (pure torch first; d_rel 16, extent 1024, α 0.1, floor 128000).
      test: `python -m engine.pyengine.tests.t_b2 relbias`
- [ ] B2.5 sconv (window-4, prefill form) matches ref on layer 0.
      test: `python -m engine.pyengine.tests.t_b2 sconv`
- [ ] B2.6 global-attention layer (idx 5) forward matches ref slice.
      test: `python -m engine.pyengine.tests.t_b2 attn_global`
- [ ] B2.7 SWA layer (idx 0, window 512, 16 KV heads) matches ref slice.
      test: `python -m engine.pyengine.tests.t_b2 attn_swa`
- [ ] B2.8 MoE gate: sigmoid+bias → top-6 → norm_after_topk → route_scale 8
      matches ref router outputs exactly (indices) / 1e-2 (weights).
      test: `python -m engine.pyengine.tests.t_b2 gate`
- [ ] B2.9 MoE expert GEMMs + 2 shared experts (sink) match ref layer out.
      test: `python -m engine.pyengine.tests.t_b2 moe`
- [ ] B2.10 dense layer 2 MLP matches ref.
      test: `python -m engine.pyengine.tests.t_b2 dense`
- [ ] B2.11 full 66-layer forward, next-token logits: top-1 matches
      transformers greedy for 5 tiny prompts.
      test: `python -m engine.pyengine.tests.t_b2 logits`

## B3 — KV + decode + scheduler + server (parity milestone)
- [ ] B3.1 KV structs: paged global (page 16) + ring-512 SWA + sconv ring;
      decode of token N+1 equals recompute-from-scratch on a 600-token prompt
      (crosses the 512 window).  test: `python -m engine.pyengine.tests.t_b3 kv`
- [ ] B3.2 greedy decode loop (batch 1) reproduces B2.11 prompts to 32 tokens
      vs transformers.  test: `python -m engine.pyengine.tests.t_b3 decode`
- [ ] B3.3 continuous batching scheduler: 8 concurrent greedy requests give
      IDENTICAL tokens to batch-1 runs (batch invariance).
      test: `python -m engine.pyengine.tests.t_b3 batch`
- [ ] B3.4 OpenAI server: /v1/completions with temperature 0, logprobs,
      return_token_ids (choice-level), ignore_eos, /health.
      test: `python -m engine.pyengine.tests.t_b3 server`
- [ ] B3.5 PARITY GATE vs goldens — the milestone. HUMAN VERIFIES this tick.
      test: `python harness/correctness.py --endpoint http://localhost:8200`
- [ ] B3.6 30-minute soak at conc 8, zero crashes/leaks (rss + vram stable).
      test: `python -m engine.pyengine.tests.t_b3 soak`

## B4 — first honest number
- [ ] B4.1 benchmark.py runs against pyengine on GPUs 4-7 (vLLM idle),
      decode_heavy canonical config; result written as ledger iteration-0
      row for engine=pyengine with cache_key. LOSING IS EXPECTED
      (case-study iteration 0 was 13.6% of vLLM).
      test: `python harness/benchmark.py --endpoint http://localhost:8200 --config configs/canonical_decode_heavy.json --ledger-iteration 0`
- [ ] B4.2 same for prefill_heavy.
      test: `python harness/benchmark.py --endpoint http://localhost:8200 --config configs/canonical_prefill_heavy.json --ledger-iteration 0`

<!-- Loop notes append below this line -->
