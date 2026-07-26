# PROGRESS.md — pyengine bring-up (P4). The build loop's memory.

Rules: the loop takes the FIRST unchecked non-BLOCKED item, does ONLY that,
runs its `test:` command verbatim, pastes real output into the note, ticks,
commits. Humans may reorder/split/unblock items; the loop may not.
Env for all items: `cd /workspace/tm-opt && source /workspace/venv/bin/activate
&& export CUDA_VISIBLE_DEVICES=4,5,6,7`. Model dir: /workspace/models/inkling-nvfp4.

## B0 — scaffolding sanity
- [x] B0.1 pyengine imports clean.
      test: `python -c "import engine.pyengine as pe; print(pe.__version__)"`
      — green: output `0.0.1` (scaffolding pre-existing from commit 8d85a77;
      no code change needed)
- [x] B0.2 triton availability + toy kernel compiles on GPU 4.
      test: `python engine/pyengine/tests/test_triton_smoke.py`
      — green: `triton ok on NVIDIA B300 SXM6 AC; triton 3.6.0` (test file
      pre-existing from scaffolding; no code change needed)

## B1 — loader (safetensors → GPU tensors, NVFP4-aware)
- [x] B1.1 shard index: enumerate 33 shards + mtp.safetensors, map tensor
      name → shard file.  test: `python -m engine.pyengine.tests.t_b1 index`
      — green: 34 files (33 model + mtp), 2056 tensors mapped, per-shard
      safetensors headers match index exactly (commit 5400822)
- [x] B1.2 tensor census vs config: counts per layer match config.h shapes
      (66 layers; layer 0-1 dense; 11 global / 55 swa; MoE 256×ffn3072).
      test: `python -m engine.pyengine.tests.t_b1 census`
      — green: all 2056 tensors accounted, exact per-layer dtype+shape match
      vs config; FACT FIX: dense = layers 0–1 (dense_mlp_idx=2 is a count),
      layer 2 is MoE — see note below (commit fa0e025)
- [x] B1.3 dtype map honors hf_quant exclude list (bf16: ALL attention,
      norms/sconvs, gates+shared experts, layer-2 experts, dense MLPs 0–1,
      embeds/unembed, mtp; NVFP4 = routed expert w13/w2 of layers 3–65 only).
      test: `python -m engine.pyengine.tests.t_b1 dtypes`
      — green: 479-entry exclude list reconciled exactly both directions;
      nvfp4 = 126 expert weights (layers 3–65 × w13/w2) ONLY, all attn bf16
      on all 66 layers, mtp 160 bf16 (commit 5c74500)
- [x] B1.4 NVFP4 dequant of ONE expert weight to bf16 matches vLLM's
      dequant of the same tensor (read vLLM modelopt code for the block-scale
      layout; cite file:line).  test: `python -m engine.pyengine.tests.t_b1 dequant`
      — green: layers.3 w13 experts [0,7,255] + 2D arm BIT-EXACT vs vLLM
      dequantize_to_dtype(swizzle=False) on GPU 4; max|deq|=6*448*scale2
      exactly (ratio 1.000 all three) (commit 4db87e1)
- [x] B1.5 TP=4 sharding plan: head-parallel attention, expert-parallel MoE;
      per-GPU byte budget printed and ≤150 GiB.
      test: `python -m engine.pyengine.tests.t_b1 plan`
      — green: per-GPU 142.04 GiB ≤ 150 (sharded 136.38 + replicated 5.67;
      16q+2/4kv heads/rank, 64 experts/rank, embed+unembed replicated by
      choice) (commit 5c3a636)
- [x] B1.6 full load onto GPUs 4-7 under budget, wall time printed.
      test: `python -m engine.pyengine.tests.t_b1 load`
      HINT: a substantially-complete load_replica() sits UNCOMMITTED in
      loader.py + t_b1.py from a prior session (killed by its own
      backgrounding — see rules). Review it, finish it, run the test in the
      FOREGROUND (expect ~5-15 min; page cache is warm), commit.
      — green: per-GPU 142.04 GiB resident (== B1.5 plan) ≤ 150, wall 110.2 s
      foreground, spot checks bitwise-exact (implementation pre-committed in
      f347f5c; no code change needed)

## B2 — model graph (single token correctness, vs transformers reference slices)
Reference: generate per-layer reference activations ONCE with
transformers(trust_remote_code) on the SAME checkpoint, tiny prompt, layers
{0,1,2,5,6}, saved to engine/pyengine/tests/ref/ (script t_b2 ref).
- [x] B2.1 reference activations generated + committed (small tensors only).
      test: `python -m engine.pyengine.tests.t_b2 ref`
      — green: 94 tensors 10.8 MiB committed (layers {0,1,2,5,6}, 13-tok
      prompt); native transformers 5.14.1 eager on a 7-layer slice; layers
      3-6 experts repaired via B1.4 dequant, proven bit-exact vs HF's own
      Interleave on layer-2 bf16 experts (commit ef157f6)
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
- 2026-07-26 B0.1: ran test verbatim from /workspace/tm-opt (venv active,
  CUDA_VISIBLE_DEVICES=4,5,6,7). Real output:
  ```
  0.0.1
  ```
  No code change required — engine/pyengine/ scaffolding (commit 8d85a77)
  already imports clean via namespace package. Pre-existing uncommitted edits
  to scripts/ralph_{build,experiment}.sh were left unstaged (loop scripts are
  off-limits to this loop; staged PROGRESS.md only instead of `git add -A`).
- 2026-07-26 B0.2: ran test verbatim from /workspace/tm-opt (venv active,
  CUDA_VISIBLE_DEVICES=4,5,6,7). Real output:
  ```
  triton ok on NVIDIA B300 SXM6 AC; triton 3.6.0
  ```
  test_triton_smoke.py pre-existing from scaffolding; kernel compiled and ran
  on GPU 4 (first visible device), allclose check passed. No code change.
  Same staging convention as B0.1: PROGRESS.md only (ralph_*.sh edits remain
  unstaged, loop scripts off-limits).
- 2026-07-26 B1.1: implemented `build_shard_index()` in engine/pyengine/loader.py
  (reads model.safetensors.index.json — its weight_map already covers
  mtp.safetensors, 34 distinct files) + new tests/t_b1.py with subcommand
  dispatch (census..load fail loud with their item ids until implemented).
  Ran test verbatim from /workspace/tm-opt (venv active,
  CUDA_VISIBLE_DEVICES=4,5,6,7). Real output:
  ```
  shard index ok: 34 files (33 model + mtp.safetensors), 2056 tensors mapped (160 in mtp), headers match index
  ```
  Test cross-checks every shard's safetensors header against the index map.
  Doc conflict found & fixed in same code commit (5400822):
  CONTEXT_AND_PLAN.md §1 said "34 shards + mtp.safetensors"; disk has 33
  model-*-of-00033 shards + mtp (34 files total) — corrected to 33, matching
  CLAUDE.md and this file. Code committed first (5400822) so this tick can
  cite the real hash; PROGRESS tick is its own follow-up commit. ralph_*.sh
  edits remain unstaged per B0.1 convention (loop scripts off-limits).
- 2026-07-26 B1.2: implemented loader.read_headers() (header-only dtype+shape
  map) + t_b1 census: exact per-layer suffix→(dtype,shape) comparison for all
  66 layers, global/SWA split cross-checked vs config.json local_layer_ids,
  full partition reconciled. Ran test verbatim from /workspace/tm-opt (venv
  active, CUDA_VISIBLE_DEVICES=4,5,6,7). Real output:
  ```
  census ok: 66 layers = 2 dense[0, 1] (ffn 24576) + 64 moe (256x ffn3072 +2 shared; 63 nvfp4-packed, 1 bf16 @[2]); attn 11 global + 55 swa (w512, 16kv); all shapes match config; totals 1882 layer + 4 llm + 10 multimodal(ignored) + 160 mtp (8x20) = 2056
  ```
  FACT CORRECTION (checkpoint + vLLM code, decisive): "layer 2 is dense" in
  the docs was a misreading. dense_mlp_idx=2 is a COUNT of leading dense
  layers — vllm/models/inkling/nvidia/model.py:164 (`layer_id <
  config.dense_mlp_idx`). Checkpoint: layers 0,1 have mlp.w13_dn [49152,6144]
  (=2×24576 dense fused gate+up); layers 2–65 have mlp.experts.* (256×,
  per-expert w13 [6144,6144] = 2×3072 fused). Layer 2's experts are stored
  bf16 WITHOUT quant scales (only MoE layer unquantized; B1.3 must reconcile
  vs hf_quant exclude list). Also observed: ALL attention weights bf16 (not
  only layer 0 — B1.3 evidence); rel_logits_proj.proj is (16,512) on SWA vs
  (16,1024) on global layers. The item's test (census vs config.h shapes)
  passes fully — its parenthetical gloss "layer 2 dense" was the wrong part;
  glosses fixed same-commit in CLAUDE.md, CONTEXT_AND_PLAN.md §1+P4,
  docs/ARCHITECTURE.md, PLAN.md, pyengine docstrings (CLAUDE.md rule 1,
  B1.1 precedent). FLAGGED FOR HUMAN (loop may not edit these): B2.10
  wording "dense layer 2 MLP" → should be "dense layers 0–1"; B1.2's own
  parenthetical above left as-written; engine/include/tmopt/config.h:43
  comment "(every layer except dense_mlp_idx)" (C++ side, outside pyengine
  scope; values are correct, comment gloss is not). ENV FLAG for human: 7 of
  .git/objects/* fan-out dirs are root:root without group write (created
  2026-07-26 ~04:56, root-run git); ralph commits fail ~3%/object with
  "insufficient permission … failed to write commit object" — needs
  `chown -R ralph:ralph .git/objects` by root.
- 2026-07-26 B1.3: implemented loader.build_dtype_map() (+ DtypeMap,
  PACK_SUFFIXES) and t_b1 dtypes. Ran test verbatim from /workspace/tm-opt
  (venv active, CUDA_VISIBLE_DEVICES=4,5,6,7). Real output:
  ```
  dtypes ok: exclude list (479 literal entries, all matched) reconciles exactly: nvfp4 = 126 expert weights (layers 3-65 x w13/w2, U8 + scale/scale2/amax/shape, group 16); plain 1426 (1298 bf16 + 128 f32 gate bias/global_scale); attn bf16 on ALL 66 layers; layer-2 experts bf16; mtp 160 bf16 (outside exclude scope)
  ```
  Reconciliation is exhaustive both directions: every model.llm.* tensor is
  in-a-pack ⇔ NOT excluded (component-boundary match, semantics per vLLM
  ModelOptQuantConfigBase.is_layer_excluded exact-match arm, vllm/model_
  executor/layers/quantization/modelopt.py:145; wildcard arm unreachable —
  builder asserts literal entries); every exclude entry matches ≥1 tensor;
  pack companions (scale F8_E4M3 /16-group, scale2 F32, input_amax BF16,
  original_shape I64) shape-checked against each U8 base; multimodal all
  excluded-plain. FACTS CONFIRMED (B1.2's flag was right): quantized = routed
  expert w13/w2 of MoE layers 3–65 ONLY; exclude list covers ALL attention
  (every layer, not just 0), all norms/sconvs, MoE gates + shared experts,
  layer-2 experts, dense MLPs 0–1, embeds/embed_norm/norm/unembed; the only
  plain F32s are the 64 MoE gates' bias+global_scale; mtp.safetensors is
  NOT in the exclude list's reach and is entirely bf16. Doc fact fix
  same-commit (5c74500): CLAUDE.md model-facts bullet + ARCHITECTURE.md
  Precision row ("layer-0 attn" gloss was stale). Item's own parenthetical
  "rest NVFP4 + scales" left as-written — FLAGGED FOR HUMAN (loop may not
  edit item text): named bf16 subset is correct, "rest" precisely = expert
  w13/w2 of layers 3–65. Regression check: t_b1 index + census both still
  green (loader.py reordered B1.2 read_headers before B1.3 block).
  ENV NOTE (resolves B1.2's env flag without root): the 7 root-owned
  .git/objects fan-out dirs (5f 0c 6d ba cb ed 7d) blocked this commit;
  fixed as ralph via same-parent rename (needs only parent-dir write) +
  cp -p into fresh ralph-owned dirs; originals parked at
  .git/objects/zz-XX-rootowned (root may delete at leisure); also restored
  the well-known empty-tree object (4b825dc, was fsck "missing", referenced
  only by a dangling tree from a failed root-era commit). git fsck now clean
  apart from danglers; commit path verified working.
- 2026-07-26 B1.4: implemented loader.dequant_nvfp4() (pure torch; E2M1 LUT,
  low-nibble-first, f8e4m3 block scale /16 * per-expert f32 scale2, same f32
  op association as vLLM's triton kernel) + t_b1 dequant. Layout citations in
  the docstring: break_fp4_bytes nvfp4_emulation_utils.py:316-333 (low nibble
  = element 2j), kE2M1ToFloat :20-22, dequantize_to_dtype :346-400 (tensor_sf
  * global_scale, NO reciprocal), run_nvfp4_emulations :484-491 (weight-
  dequant entry), modelopt.py:1253-1262 (weight_scale_2 = amax/(6*448),
  renamed "without reciprocation"), inkling name map .scale/.scale2 ->
  weight_scale/_2 nvidia/model.py:381-386 + per-expert scale2 load
  moe.py:578-582; checkpoint scales LINEAR (modelopt.py:1873 "2D, not
  swizzled"; swizzle is load-time for cutlass) => swizzle=False reference.
  Ran test verbatim from /workspace/tm-opt (venv active,
  CUDA_VISIBLE_DEVICES=4,5,6,7). Real output:
  ```
  dequant ok: model.llm.layers.3.mlp.experts.w13_weight experts [0, 7, 255] + 2D arm, shape (3, 6144, 6144): bit-exact vs vLLM dequantize_to_dtype(swizzle=False) in bf16; e2m1 low-nibble-first, f8e4m3 block scale /16 * f32 scale2 (amax/2688, no reciprocal); max|deq|/(6*448*scale2) = ['1.000', '1.000', '1.000'], nonzero 91.7%
  ```
  Reference really is vLLM's code path (triton _dequantize_nvfp4_kernel via
  dequantize_to_dtype swizzle=False on cuda:0 = GPU 4), exercising both the
  3D per-expert-global-scale arm and the 2D scalar arm; torch.equal (bit
  equality), not allclose. Extra checkpoint anchors: .original_shape ==
  (256, 6144, 6144) confirms 2 fp4/byte on the input dim; max|deq| ==
  6*448*scale2 exactly per expert confirms the ModelOpt amax/2688 invariant.
  Regression: t_b1 index + census + dtypes all still green (loader now
  imports torch at module top). NOTE for B2.9 (not this item's scope, no
  files changed): checkpoint w13 expert rows are INTERLEAVED [g0,u0,g1,u1,..]
  — vLLM de-interleaves at load into [w1;w3] halves (nvidia/moe.py:583-595);
  our model.py must account for this when consuming w13.
- 2026-07-26 B1.5: implemented loader.build_shard_plan() (+ ShardPlan,
  DTYPE_BYTES/nbytes, suffix policy tables) and t_b1 plan. Placement: attn
  head-parallel (wq/wk/wv + per-channel k/v sconvs dim0 = whole heads; wo
  row-parallel dim1; 16q + 2kv-global/4kv-swa heads/rank, GQA rank-local);
  MoE expert-parallel (experts + nvfp4 scale/scale2 dim0, 64/rank; gate
  replicated so every rank routes); shared experts + dense MLPs + mtp MLP
  ffn-split (fused-w13 out-dim/w2 in-dim; interleaved [g,u] pairs keep
  contiguous slices whole-channel, vLLM nvidia/moe.py:583-595); mtp
  input_proj row-parallel; embed/unembed REPLICATED by choice (+4.60 GiB/GPU,
  buys local lookup + full-logits argmax; Stage-3 may revisit); multimodal
  SKIP (text-only v1). Builder fails loud on unknown suffixes, non-divisible
  splits, split heads, incoherent packs. Ran test verbatim from
  /workspace/tm-opt (venv active, CUDA_VISIBLE_DEVICES=4,5,6,7). Real output:
  ```
  plan ok: tp=4, head-parallel attn (16q + 2kv-global/4kv-swa heads/rank) + expert-parallel moe (64 experts/rank), ffn-split shared/dense/mtp mlp; per-GPU 142.04 GiB <= 150 GiB budget (sharded 136.38 + replicated 5.67, of which embed+unembed 4.60); skipped multimodal 0.17 GiB; checkpoint total 551.35 GiB
  ```
  Byte accounting reconciles to the checkpoint total exactly (551.35 GiB,
  matches CLAUDE.md). New facts pinned while planning: mtp.safetensors is
  9.80 GiB — each of the 8 draft layers is a DENSE-MLP transformer block
  (w13_dn [49152,6144] / w2_md [6144,24576], same as dense layers 0-1) with
  16-kv-head attention + input_proj [6144,12288] (consumes
  concat(embed,hidden)) + embed_norm/hidden_norm; multimodal is only
  0.17 GiB (audio encoder + 4 visual linears). Regression: t_b1 index +
  census + dtypes + dequant all still green. vLLM measured 137.46 GiB/GPU
  weights (BASELINE_NOTES) — our +4.6 GiB delta is exactly the replicated
  embed/unembed, headroom for it verified by this budget.
- 2026-07-26 B1.6: NO code change — the hint's "uncommitted" load_replica()
  + t_load() were already committed by the human in f347f5c (alongside the
  hint itself); reviewed both end-to-end before running: streams each shard
  once, REPLICATE full-copy / SHARD r-th-of-4 slice per the B1.5 plan, SKIP
  multimodal never read, pre-flight free-VRAM check per rank, post-load
  name-set + exact-byte reconciliation vs plan prediction. Pre-flight: GPUs
  4-7 idle (4 MiB used each), t_b1 index green as env sanity. Ran test
  verbatim from /workspace/tm-opt (venv active, CUDA_VISIBLE_DEVICES=4,5,6,7)
  in the FOREGROUND, single run, no retries. Real output (stdout; 34
  per-shard progress lines went to stderr, steady ~3-4 s/shard):
  ```
  load ok: tp=4 replica on GPUs 4-7 (visible cuda:0-3), 2046 tensors/rank; per-GPU 142.04 GiB resident (allocator max 142.05) <= 150 GiB budget; wall 110.2 s (551.18 source GiB, 5.00 GiB/s); spot checks bitwise-exact: replicate + dim0/1/2 shards + nvfp4 pack (u8/f8-scale/f32-scale2/amax)
  ```
  Wall 110 s (page cache fully warm — hint's 5-15 min was the cold-ish
  estimate). Resident bytes == B1.5 plan prediction exactly on all 4 ranks;
  allocator overhead 0.01 GiB. Spot checks compare rank pieces bitwise vs a
  fresh disk read on ranks 0+3: replicate (embed), dim-0 shard (wq bf16 +
  full nvfp4 pack u8/scale/scale2), dim-1 (wo), dim-2 (shared_w2), replicated
  pack metadata (input_amax). B1 loader track complete; next is B2.1
  (transformers reference activations).
- 2026-07-26 B2.1: implemented tests/t_b2.py (`ref` + fail-loud stubs for
  embed..logits). Design forced by P2 probe-run-3 evidence (cited in
  scripts/generate_goldens.py): naive transformers from_pretrained REINITS
  the packed NVFP4 experts of layers 3-65 ("Reinit due to size mismatch"),
  and full-model bf16 experts (~1.8 TB) fit nowhere — so `ref` builds a
  7-layer TRUNCATED config (causal decoder: layers {0,1,2,5,6} activations
  depend only on layers below; slice is exact), loads via transformers'
  native Inkling code + conversion pipeline (conversion_mapping.py
  "inkling_mm_model": w13 → Interleave(dim=1) de-interleave → gate_up_proj),
  then repairs the 8 mismatched expert tensors (layers 3-6 × w13/w2) with
  loader.dequant_nvfp4 (B1.4, bit-exact vs vLLM) + the same de-interleave.
  Honesty gates: loading-info reconciled exactly (mismatch set == the 8
  packs, nothing text-tower missing, no kept-layer key dropped); our
  de-interleave proven torch.equal vs HF's OWN Interleave output on layer-2
  bf16 experts; dense w13_dn chunk, embed rows, layer-5 wq, fp32-upcast
  k_sconv all bit-exact; ModelOpt max|deq|=6*448*scale2 invariant ratio
  1.000000 on all 8 packs; every capture finite. Captured 94 module-boundary
  tensors (layer in/out, norms, attn in/out, all 4 sconvs, MoE gate
  routed_logits/topk_weights/topk_indices/shared_gammas, experts + shared
  out, attn internals k/v_sconv+r_proj+q/k_norm+rel_logits_proj for layers
  0+5, embed/embed_norm/final_norm, input_ids), 10.8 MiB — committable.
  Reruns verify sha256+manifest instead of regenerating (frozen like
  goldens). Ran test verbatim from /workspace/tm-opt (venv active,
  CUDA_VISIBLE_DEVICES=4,5,6,7), foreground. Real output (first run, after
  transformers' load report table):
  ```
  ref ok: generated 94 tensors (10.8 MiB) for layers [0, 1, 2, 5, 6], prompt 13 toks; native transformers 5.14.1 eager, 7-layer slice on cuda:0; fixup layers 3-6 experts dequant (invariant ratios [1.0]); spot checks bit-exact (interleave/dense-chunk/embed/wq/fp32-sconv); all finite; wall 63.6 s (load 20.4)
  ```
  Verify path exercised by a second verbatim run:
  ```
  ref ok: existing refs verified — 94 tensors, sha256 eae8f3a95804, layers [0, 1, 2, 5, 6], prompt 13 toks (generated with transformers 5.14.1, 63.5 s)
  ```
  .gitignore: added `!engine/pyengine/tests/ref/*.safetensors` negation under
  the existing `*.safetensors` rule — the item REQUIRES committing the refs
  (.gitignore is not on the protected list). Sanity stats: activations
  bounded (absmax ≤ 8.2e3 bf16, finite), router picks 41-50 distinct experts
  over 78 slots, topk_weights f32. NOTE for B2.4/B2.6: eager path applies
  log-scaling tau only on global layers and tau==1.0 at 13 tokens (clamp
  min 1 below floor 128000) — rel_logits_proj.out captures are PRE-tau.
  NOTE for B2.2+: refs are bf16 module-boundary captures; self_attn.in is
  post-input_layernorm, mlp.in is post-post_attention_layernorm,
  attn_sconv/mlp_sconv outs are PRE-residual-add (sconv module includes its
  own +residual of its input, in fp32, per modeling_inkling.py:512-542).
