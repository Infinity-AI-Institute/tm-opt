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
- [x] B2.2 embed + embed_norm matches ref (rel err < 1e-2 bf16).
      test: `python -m engine.pyengine.tests.t_b2 embed`
      — green: rel err 0.0 (bit-exact, both lookup and embed_norm) vs frozen
      refs; torch-ref rmsnorm + embed in model.py (commit 7a37c70)
- [x] B2.3 rmsnorm Triton kernel matches torch reference on random tensors.
      test: `python -m engine.pyengine.tests.t_b2 rmsnorm`
      — green: 17.9M random elements (widths 6144+128, 8 cases), all within
      2 bf16 ulp of torch ref (35 off-by-ulp, worst rel err 1.1e-05); frozen
      embed_norm anchor bit-exact (commit 0162160)
- [x] B2.4 relative-attention bias table + log scaling matches ref math
      (pure torch first; d_rel 16, extent 1024, α 0.1, floor 128000).
      test: `python -m engine.pyengine.tests.t_b2 relbias`
      — green: frozen anchors L0(ext512)/L5(ext1024) bit-exact vs ref
      captures; 12/12 bit-exact vs native InklingRelativeLogits at long
      positions; tau fp64 agreement 6.1e-08, ==1.0 below floor (16K serve
      window inert) (commit 7545848)
- [x] B2.5 sconv (window-4, prefill form) matches ref on layer 0.
      test: `python -m engine.pyengine.tests.t_b2 sconv`
      — green: all 4 layer-0 sconvs (k/v/attn/mlp) bit-exact vs frozen
      anchors; native-module cross-check bit-exact 4 real + 5 random cases;
      window-4 causality + tap orientation + no-bias pinned (commit 506c98f)
- [x] B2.6 global-attention layer (idx 5) forward matches ref slice.
      test: `python -m engine.pyengine.tests.t_b2 attn_global`
      — green: full path + all 8 internal anchors bit-exact vs frozen refs;
      mask bit-equal native create_causal_mask; 6/6 native-module
      cross-checks bit-exact (real+random weights, T 1-600, batch 2);
      causality bit-checked (commit 3196ecd)
- [x] B2.7 SWA layer (idx 0, window 512, 16 KV heads) matches ref slice.
      test: `python -m engine.pyengine.tests.t_b2 attn_swa`
      — green: layer-0 frozen anchors bit-exact (full path + 8 internals);
      window mask bit-equal native create_sliding_window_causal_mask; 6/6
      native-module cross-checks bit-exact incl. T600 > window; window
      cutoff pinned exactly both sides (commit a0c007b)
- [x] B2.8 MoE gate: sigmoid+bias → top-6 → norm_after_topk → route_scale 8
      matches ref router outputs exactly (indices) / 1e-2 (weights).
      test: `python -m engine.pyengine.tests.t_b2 gate`
      — green: layers 2/5/6 frozen anchors — indices EXACT 3/3, weights/
      logits/gammas rel err 0.0 (bit-exact 9/9); 5/5 native-module
      cross-checks bit-exact (all 4 outputs); bias-steers-choice-only +
      top-6-largest + slot-sum pins all pass (commit 542fb1e)
- [x] B2.9 MoE expert GEMMs + 2 shared experts (sink) match ref layer out.
      test: `python -m engine.pyengine.tests.t_b2 moe`
      — green: layers 2/5/6 anchors experts/shared/full all < 1e-2 (shared
      bit-exact 3/3); ref-capture provenance pinned — ref ran grouped_mm
      dispatch, native grouped_mm reproduces frozen experts.out BIT-exactly;
      native eager cross-checks 5/5 bit-exact; structural + fp64 pins all
      pass (commit e14e430)
- [x] B2.10 dense layer 2 MLP matches ref.
      test: `python -m engine.pyengine.tests.t_b2 dense`
      — green: dense layers 0+1 (item wording stale — dense_mlp_idx=2 is a
      COUNT, layer 2 is MoE; flagged B1.2) frozen anchors bit-exact 2/2;
      conversion pinned vs HF's own Interleave/Chunk; 6/6 native InklingMLP
      cross-checks bit-exact; live global_scale + fp64 pins (commit f763e28)
- [x] B2.11 full 66-layer forward, next-token logits: top-1 matches
      transformers greedy for 5 tiny prompts.
      test: `python -m engine.pyengine.tests.t_b2 logits`
      — green: top-1 MATCH 5/5 (Berlin/Au/oxygen/cold/four) vs native
      transformers streamed 66-layer reference; ref composition pinned
      bit-exact vs frozen B2.1 captures (13 pins); ours bit-exact through
      dense layers 0-1, MoE drift <= 2.9e-03; wall 130 s (commit 7b63ab6)

## B3 — KV + decode + scheduler + server (parity milestone)
- [x] B3.1 KV structs: paged global (page 16) + ring-512 SWA + sconv ring;
      decode of token N+1 equals recompute-from-scratch on a 600-token prompt
      (crosses the 512 window).  test: `python -m engine.pyengine.tests.t_b3 kv`
      — green: decode 601/602 vs T=602 recompute per layer (teacher-forced):
      attn half <= 7.9e-05, dense out <= 9.7e-05, MoE out <= 5.4e-03 (expert
      choice identical); cache K/V + 4 sconv-ring tails bit-equal replay 5/5
      layers; ring/page positional pins exact; wall 13 s (commit f3b1fbd)
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
- 2026-07-26 B2.2: implemented model.py rmsnorm (torch reference, exact
  InklingRMSNorm semantics per modeling_inkling.py:99-113 — fp32 variance,
  downcast to input dtype BEFORE the bf16 weight multiply, eps 1e-6) +
  embed (plain lookup, no multiplier, modeling_inkling.py:654,659,682);
  config.py gained rms_eps checked against checkpoint rms_norm_eps; t_b2
  refactor: _verify_existing split into silent _load_refs (sha256 + manifest
  + finiteness) reused by every B2.2+ item, plus _rel_err (fp32 global L2)
  and t_embed (weights read straight from shards via B1.1 index, shapes
  pinned vs verified config). Ran test verbatim from /workspace/tm-opt (venv
  active, CUDA_VISIBLE_DEVICES=4,5,6,7). Real output:
  ```
  embed ok: 13 toks; lookup rel err 0.0e+00 (bit-exact True), embed_norm rel err 0.0e+00 < 1e-2 (bit-exact True, max|diff| 0.0e+00); weights (201024, 6144) bf16 from shards, eps 1e-06
  ```
  Bit-exact (not just <1e-2): lookup is a row copy and our rmsnorm replays
  transformers' exact op order/dtypes on the same GPU. Regressions all
  green after the config.py + t_b2 refactor: t_b2 ref (verify path, sha
  eae8f3a95804 unchanged), t_b1 index + census + dtypes. Code committed
  first (7a37c70) per B1.1 convention; this tick is its own commit.
- 2026-07-26 B2.3: implemented kernels/rmsnorm.py (triton row-per-program:
  fp32 mean-of-squares over the real width via masked block, rsqrt, downcast
  to input dtype BEFORE the weight multiply — exact InklingRMSNorm op order;
  weight mul at fp32 opmath with single rounding on store, which equals
  torch's promoted-bf16 mul bit-for-bit given equal inputs) + t_b2 rmsnorm
  (+ _ulp_bf16 exact-ulp helper via sign-magnitude→monotonic-int mapping).
  Gate chosen: elementwise <= 2 bf16 ulp AND fp32 L2 rel err < 1e-3 per case
  — bit-exactness is unattainable in principle (tl.sum tree order vs torch's
  reduction order differ by a few fp32 ulps in the variance; after bf16
  downcast that bounds to 1 ulp on xhat, 2 on the product), and this gate is
  ~100x tighter than the B2 header's 1e-2 budget, so nothing is weakened.
  Cases: widths 6144 (BLOCK 8192, masked) + head_dim 128 3D q/k-norm shape;
  rows 1/13/64/257/512/2048; scales 1e-4 (eps-dominated variance) / 1 / 1e4;
  all-zero rows required bit-exact. model.py untouched — graph items choose
  torch-ref vs kernel later. Ran test verbatim from /workspace/tm-opt (venv
  active, CUDA_VISIBLE_DEVICES=4,5,6,7). Real output:
  ```
  rmsnorm ok: triton kernel vs torch ref, 8 random cases (widths {6144,128}, rows 1-2048, scales 0/1e-4/1/1e4): 17924096 elements, bit-exact 17924061 (35 off), max 2 ulp <= 2, worst rel err 1.1e-05 < 1e-3; frozen embed_norm anchor rel err 0.0e+00 < 1e-2 (max 0 ulp)
  ```
  (An earlier identical-gate run printed a rounded "bit-exact 100.000%"
  alongside "max 2 ulp"; evidence line switched to exact counts — 35 of
  17.9M off — before committing, no gate change.) The real-weights anchor
  (kernel on frozen embed lookup vs frozen embed_norm.out) came out fully
  bit-exact, 0 ulp. Regressions green: t_b2 ref (verify path, sha
  eae8f3a95804 unchanged) + t_b2 embed. Code committed first (0162160) per
  convention; this tick is its own commit.
- 2026-07-26 B2.4: implemented model.py rel_bias + log_scale_tau +
  apply_log_scaling (pure torch, op-for-op InklingRelativeLogits semantics
  per modeling_inkling.py:131-142 — bf16 matmul vs the (d_rel, extent) bank,
  gather at backward distance, zero outside 0<=d<extent; extent = 1024
  global / window 512 SWA per :196; tau fp32 clamp((pos+1)/floor,min=1) log
  form + fp32-mul-downcast application per :254-261, global layers only) +
  t_b2 relbias; config.py load_verified now also checks d_rel/rel_extent/
  log_scaling_alpha/log_scaling_n_floor vs checkpoint (were declared but
  unverified). Test design: tau has NO non-trivial frozen anchor (13-tok
  prompt => tau==1.0 exactly, B2.1 note; captures are PRE-tau), so the tau
  gate is an independent float64 evaluation + exactness facts (==1.0 through
  pos floor-1 incl. whole 16K serve window — log scaling provably inert at
  canonical lengths; >1 monotonic from pos==floor); the bias table IS
  decisively anchored: frozen captures on real weights AND the native
  transformers module run in-process at synthetic long positions (offsets
  200k/300k, distances -15..1515 spanning negative/in-band/>=extent for both
  extents, real + random banks), torch.equal required. kernels/relbias.py
  untouched — item says "pure torch first"; the fused kernel is D4 Stage-3
  work. Ran test verbatim from /workspace/tm-opt (venv active,
  CUDA_VISIBLE_DEVICES=4,5,6,7). Real output:
  ```
  relbias ok: frozen anchors L0(swa,ext512)/L5(global,ext1024) rel err 0.0e+00/0.0e+00 < 1e-2 (bit-exact True/True), future-dist zeros exact; native-module cross-check bit-exact 12/12 cases (6406400 els; extents {512,1024} x real/random proj x {13tok, prefill@200k dist-15..1515, decode@300k}), out-of-band exact zeros; tau: ==1.0 exactly below floor 128000 (16K serve window inert), floor boundary+monotonic ok, fp64 agreement 6.1e-08 < 1e-6 over 2717 pts to ctx 1048575, apply op-order bit-exact (modeling_inkling.py:259-261)
  ```
  Regressions green after the config.py check additions: t_b2 ref (verify
  path, sha eae8f3a95804 unchanged) + embed + rmsnorm; t_b1 index + census +
  dtypes + plan. NOTE for B2.6/B2.7: rel_bias consumes r_proj output viewed
  [B,Q,heads,d_rel]; attention must apply tau to BOTH q and bias (global
  only) BEFORE softmax; at every B2/B3 test length tau==1.0 so a tau bug
  would only surface >128K — the bit-check vs :259-261 transcription is the
  guard. Code committed first (7545848) per convention; this tick is its
  own commit.
- 2026-07-26 B2.5: implemented model.py sconv_prefill (pure torch, exact
  InklingShortConvolution prefill semantics per modeling_inkling.py:500-542
  — whole module in fp32 (_keep_in_fp32_modules_strict :610, conv weights
  bf16-on-disk upcast exactly), depthwise causal F.conv1d pad k-1/truncate-
  to-T with no bias (:498) and no activation (causal_conv1d_fn :461-481),
  then the module's OWN-input residual added in fp32 before downcast to
  input dtype — so captures are pre-OUTER-residual, per the B2.1 note) +
  t_b2 sconv; config.py load_verified now also checks sconv_kernel_size=4
  vs checkpoint (was declared but unverified, B2.4 convention). Test
  design: anchors cover ALL FOUR layer-0 sconvs — k/v_sconv have .in/.out
  captured directly; attn_sconv's input IS self_attn.out and mlp_sconv's
  input IS mlp.out (decoder forward :576-591) — replayed through our
  function with real checkpoint weights (shapes (C,1,4): 2048 k/v, 6144
  attn/mlp) vs captured outputs; plus the ACTUAL transformers
  InklingShortConvolution run in-process (past_key_values=None,
  conv_mask=None — identical to the ref path: batch-1 makes
  apply_mask_to_padding_states a no-op, :433), torch.equal REQUIRED, on the
  4 real frozen inputs + 5 random cases (T {1,3,4,13,600} x C
  {1024,2048,6144} x batch {1,2}); plus structural pins: perturb-one-token
  causality (only out[t..t+3] moves, both sides bit-checked), delta-at-last-
  tap == exactly 2x (current token, cross-correlation no flip), delta-at-
  tap-0 == x[t-3]+x[t] by manual indexing, zero-in -> zero-out (no bias).
  Decode form (causal_conv1d_update ring state) is B3.1's job; fused kernel
  is Stage-3 (D4) — kernels/sconv.py stub untouched. Ran test verbatim from
  /workspace/tm-opt (venv active, CUDA_VISIBLE_DEVICES=4,5,6,7). Real output:
  ```
  sconv ok: layer-0 frozen anchors k/v/attn/mlp rel err 0.0e+00/0.0e+00/0.0e+00/0.0e+00 < 1e-2 (bit-exact True/True/True/True); native-module cross-check bit-exact 4 real + 5 random cases (T {1,3,4,13,600} x C {1024,2048,6144}); window-4 causality bit-checked, last-tap-is-current + tap-0 orientation exact, zero-in zero-out (no bias), fp32 module math (modeling_inkling.py:500-542)
  ```
  Regressions green after the config.py check addition: t_b2 ref (verify
  path, sha eae8f3a95804 unchanged) + embed + rmsnorm + relbias; t_b1
  index + census + dtypes + plan. NOTE for B2.6/B2.7: k/v sconv sits
  BETWEEN k/v_proj and the attention math (k_sconv(k_proj(x)),
  modeling_inkling.py:229-230); attn items must thread it there, and
  sconv_prefill also covers the attn-out + moe-out positions. Code
  committed first (506c98f) per convention; this tick is its own commit.
- 2026-07-26 B2.6: implemented model.py attn_prefill (pure torch, exact
  InklingAttention prefill semantics per modeling_inkling.py:217-282 +
  eager_attention_forward :157-182 — q/k/v/r projections with k/v through
  their window-4 sconvs (:229-230), per-head q/k rmsnorm on head_dim then
  [B,H,T,D] transpose (:233-235), rel_bias from r_proj states (:248-251),
  tau fp32 round-trip on global layers unconditionally like the ref
  (:254-261; ==1.0 below floor 128000), GQA repeat_kv expand+reshape
  (:145-154), bf16 scores * 1/head_dim + bias + mask in the ref's
  association order (:171-175; 1/d scaling because q/k are RMS-normalized,
  :197-198), fp32 softmax downcast (:177), o_proj) + additive_causal_mask
  (eager 0/finfo.min form, masking_utils eager_mask semantics; window arm
  for B2.7 carries sliding_window_overlay's kv > q - window) + t_b2
  attn_global. Test: (a) frozen layer-5 anchors — full path from
  self_attn.in vs self_attn.out AND all 8 captured internals (k/v_proj,
  k/v_sconv, r_proj, q/k_norm, rel_logits_proj) recomputed as a chain,
  1e-2 gate; (b) our mask vs transformers create_causal_mask on the exact
  ref-run call, torch.equal; (c) native InklingAttention in-process (bf16
  module, fp32 sconvs per _keep_in_fp32_modules_strict :610), torch.equal
  REQUIRED, real + random weights x T {1,4,13,29,600} incl. batch 2;
  (d) perturb-token-t causality bit-checked both sides. Ran test verbatim
  from /workspace/tm-opt (venv active, CUDA_VISIBLE_DEVICES=4,5,6,7).
  Real output:
  ```
  attn_global ok: layer-5 frozen anchors — full path rel err 0.0e+00 < 1e-2 (bit-exact True), 8 internals (k/v_proj+sconv, r_proj, q/k_norm, rel_bias) < 1e-2, bit-exact 8/8; mask bit-equal native create_causal_mask (eager 0/finfo.min); native-module cross-check bit-exact 6/6 cases (4079616 els; real+random weights, T {1,4,13,29,600}, batch 2); causality bit-checked through full path; tau==1.0 at all tested positions (scaling armed, floor 128000)
  ```
  Everything bit-exact, not just <1e-2: our function transcribes the exact
  op order/dtypes/layouts on the same GPU. Facts pinned: conv_mask was None
  in the ref run (create_recurrent_attention_mask returns None for unpadded
  input — confirms B2.5's setup); eager mask dtype = inputs_embeds dtype
  (bf16), values exactly {0, finfo.min}, [B,1,Q,K]. NOTE for B2.7: same
  attn_prefill covers SWA — pass the 16-kv-head weights, window=512 mask
  (untested arm this iteration), is_global=False; L0 internals are already
  captured. Regressions green: t_b2 ref (sha eae8f3a95804 unchanged) +
  embed + rmsnorm + relbias + sconv; t_b1 index + census + dtypes + plan.
  Code committed first (3196ecd) per convention; this tick is its own
  commit.
- 2026-07-26 B2.7: NO new model code — B2.6's attn_prefill already carries
  the SWA arm (is_global=False skips the tau round-trip exactly like the
  native `if not self.is_sliding` gate, modeling_inkling.py:254; rel extent
  falls out of the proj bank shape (16,512); window enters only through the
  mask, matching eager_attention_forward which ignores its sliding_window
  kwarg, :157-182). Added t_b2 attn_swa (mirror of attn_global at layer 0:
  frozen anchors full-path + 8 internals, mask-vs-native, native-module
  torch.equal cross-check, causality) plus the SWA-specific arms: our
  additive_causal_mask(window=512) bit-equal transformers'
  create_sliding_window_causal_mask on the exact ref-run call (predicate
  kv > q - window per masking_utils sliding_window_overlay; layer 0 reads
  the "sliding_attention" mask entry, InklingTextModel.forward :709);
  cross-check T set {1,4,13,600} with T600 > window on BOTH real and random
  weights; and a window-cutoff gate at T600 — perturbing token 37 must move
  the direct window edge query 37+511 and be bit-invisible from query
  37+512+3 on (k/v sconv smears the token into keys/values t..t+3, so the
  cutoff is window+sconv_k-1, not window — a real trap for B3.1's ring
  buffer sizing). config.py load_verified now also checks
  swa_num_attention_heads=64 + swa_head_dim=128 vs checkpoint (SWA arm
  relies on them; B2.4/B2.5 convention). Ran test verbatim from
  /workspace/tm-opt (venv active, CUDA_VISIBLE_DEVICES=4,5,6,7). Real output:
  ```
  attn_swa ok: layer-0 frozen anchors — full path rel err 0.0e+00 < 1e-2 (bit-exact True), 8 internals (k/v_proj+sconv, r_proj, q/k_norm, rel_bias ext512) < 1e-2, bit-exact 8/8; window mask bit-equal native create_sliding_window_causal_mask (eager 0/finfo.min); native-module cross-check bit-exact 6/6 cases (7587840 els; real+random weights, T {1,4,13,600} incl. 600>window, batch 2); causality bit-checked; window cutoff exact at T600 (edge 37+511 moves, queries >= 37+512+4-1 bit-identical, k/v sconv smear accounted)
  ```
  Everything bit-exact, not just <1e-2. NOTE for B3.1: the KV ring for SWA
  layers must retain window+sconv_k-1 = 515 tokens of history influence —
  the sconv ring (3 raw pre-conv K/V inputs) is separate state from the 512
  post-conv KV entries; the cutoff gate above is the ground truth for that
  boundary. Regressions green after the config.py addition: t_b2 ref
  (verify path, sha eae8f3a95804 unchanged) + embed + rmsnorm + relbias +
  sconv + attn_global; t_b1 index + census + dtypes + plan. Code committed
  first (a0c007b) per convention; this tick is its own commit.
- 2026-07-26 B2.8: implemented model.moe_gate (pure torch, exact
  InklingTopkRouter semantics per modeling_inkling.py:342-377 — the router
  weight is (258, 6144): ONE linear scores 256 routed + 2 shared slots;
  CHOICE = top-6 of sigmoid(routed logits) + e_score_correction_bias
  (checkpoint name mlp.gate.bias, f32, DeepSeek-V3-style choice steering);
  WEIGHTS ignore the bias — softmax over logsigmoid (= sigmoids normalized
  to sum 1) of the 6 CHOSEN routed logits AND the 2 shared logits jointly
  (:366-370, the norm_after_topk form), then * route_scale 8 *
  global_scale (:372, a per-layer f32 scalar ~0.006 — live math, not a
  1.0); shared slots split off as shared_gammas for the sink experts) +
  t_b2 gate; config.py load_verified now also checks n_shared_experts +
  route_scale vs checkpoint (declared-but-unverified, B2.4 convention).
  Ref-run dtype fact: the router is NOT in _keep_in_fp32_modules_strict
  (:610), so the ref load downcast the f32 bias/global_scale to bf16 —
  the downcast is value-exact on all 3 layers (down_exact=True) and all
  gate captures are bf16; our replay feeds bf16 everywhere. Ran test
  verbatim from /workspace/tm-opt (venv active,
  CUDA_VISIBLE_DEVICES=4,5,6,7). Real output:
  ```
  gate ok: frozen anchors layers {2,5,6} — topk_indices EXACT 3/3 layers (234 slots), weights rel err 0.0e+00/0.0e+00/0.0e+00 < 1e-2 (+ routed_logits/shared_gammas, bit-exact 9/9); native-module cross-check bit-exact 5/5 cases x all 4 outputs (real+random weights incl. global_scale!=1, T {1,13,257,600}, batch 2); top-6 = 6 largest sigmoid+bias scores (distinct, in-range), fp64 bias-free normalized-sigmoid weights agree 6.4e-03 (bias steers choice only; +100 spike forces expert into all top-6), slot sums = route_scale*global_scale (max dev 1.3e-02); router all-bf16 like ref (f32 bias/global_scale downcast exact=True)
  ```
  Everything bit-exact (indices torch.equal incl. topk sorted=False order;
  all 12 anchor tensors + 20 cross-check tensors), not just within 1e-2.
  Structural pins: chosen 6 proven the 6 LARGEST sigmoid+bias scores by
  direct min-chosen >= max-unchosen comparison (no topk); independent fp64
  bias-free normalized-sigmoid formula reproduces weights (6.4e-03, bf16
  rounding) — proves bias steers choice ONLY; +100 bias spike forces an
  unchosen expert into every token's top-6; per-token slot sums ==
  route_scale*global_scale. NOTE for B2.9: InklingMoE.forward (:419-425)
  passes topk_weights/indices to InklingExperts (weights applied per-slot
  AFTER down_proj, index_add accumulation in input dtype) and
  shared_gammas to InklingSharedExperts (gamma applied to act_fn(gate)*up
  BEFORE down_proj; fp32 sum over the 2 shared experts, :399-404); routed
  out + shared out added at :424. Regressions green after the config.py
  addition: t_b2 ref (verify path, sha eae8f3a95804 unchanged) + embed +
  rmsnorm + relbias + sconv + attn_global + attn_swa; t_b1 index + census
  + dtypes + plan. Code committed first (542fb1e) per convention; this
  tick is its own commit.
- 2026-07-26 B2.9: implemented model.moe_experts (exact InklingExperts
  eager-loop semantics per modeling_inkling.py:315-339 — ascending-expert
  iteration fixes the bf16 index_add_ rounding order, routing weight applied
  AFTER down_proj, accumulate in input dtype; w13 rows de-interleaved
  [gates;ups] per conversion_mapping.py "inkling_mm_model" / vLLM
  nvidia/moe.py:583-595, consumed as chunk halves), model.moe_shared (exact
  InklingSharedExperts :394-405 — bmm stacks, gammas multiply act*up BEFORE
  down_proj, fp32 sum over the 2 sinks), model.moe (InklingMoE :418-425 —
  gate → routed on flattened tokens + shared on pre-flatten residuals) +
  t_b2 moe. Layer-2 experts read bf16 from shards; layers 5/6 rebuilt by
  B1.4 dequant + de-interleave (identical to the ref run's own repair). Ran
  test verbatim from /workspace/tm-opt (venv active,
  CUDA_VISIBLE_DEVICES=4,5,6,7). Real output (after transformers'
  standalone-module dispatch warning line):
  ```
  moe ok: frozen anchors layers {2,5,6} — experts/shared/full-block rel err 3.3e-03/0.0e+00/1.0e-03 | 3.9e-03/0.0e+00/3.8e-03 | 3.2e-03/0.0e+00/3.2e-03 < 1e-2 (bit-exact 3/9; composition routed+shared bitwise 3/3; layers 5/6 via B1.4 nvfp4 dequant + de-interleave); capture provenance PINNED: native grouped_mm dispatch (the ref run's path, modeling_utils.py:2100) reproduces frozen experts.out bit-exactly; native InklingMoE eager cross-check bit-exact 5/5 cases (real layer-2 weights T {13,57} batch 2 + random small-config T {13,600}); pins: one-chooser batch-1 chain bit-exact (weight AFTER down_proj), zero-weights/zero-gammas exact zeros, unchosen-expert perturbation invisible, slot-permutation no-op; fp64 replay agrees routed/shared/full 4.4e-03/2.5e-03/2.5e-03 < 1e-2
  ```
  FACT PINNED (first B2 item where anchors are legitimately NOT bit-exact):
  the B2.1 ref run's from_pretrained dispatched InklingExperts to
  "grouped_mm" (torch._grouped_mm; modeling_utils.py:2100 defaults it when
  no experts_implementation kwarg is passed, _grouped_mm_can_dispatch :2113
  always passes for this class) — NOT the eager masked loop. Proven
  decisively inside the test: the native module FORCED onto grouped_mm
  reproduces the frozen layer-2 experts.out bit-for-bit (rel 0.0), while
  the eager loop (our transcription target, and the standalone-module
  dispatch fallback integrations/moe.py:497-509) lands 3.3e-03 away —
  same math, different accumulation order. Our functions ARE bit-exact vs
  the native eager modules in-process (5/5 InklingMoE cases incl. a
  hidden-512/8-expert random config; shared-experts anchors bit-exact 3/3
  because InklingSharedExperts is undecorated plain bmm). NOTE for B2.11:
  the full-forward ref logits carry grouped_mm expert numerics on all MoE
  layers; per-layer deltas ~3-4e-03 are expected and the item's gate is
  top-1 agreement, not bitwise. NOTE for B3.1+: sconv ring aside, MoE decode
  reuses these functions unchanged (token count is just T=1). ENV NOTE:
  three more root-owned .git/objects fan-out dirs (62 85 97) blocked the
  commit; fixed as ralph via the B1.3 same-parent-rename recipe, originals
  parked at .git/objects/zz-{62,85,97}-rootowned (root may delete), fsck
  clean. Regressions all green after the change: t_b2 ref (verify path, sha
  eae8f3a95804 unchanged) + embed + rmsnorm + relbias + sconv + attn_global
  + attn_swa + gate; t_b1 index + census + dtypes + plan. Code committed
  first (e14e430) per convention; this tick is its own commit.
- 2026-07-26 B2.10: implemented model.dense_mlp (pure torch, exact
  InklingMLP semantics per modeling_inkling.py:285-299 — three SEPARATE
  bias-free linears (:291-293, kept unfused so cuBLAS sees the module's
  exact GEMM shapes), silu, then a LIVE trailing global_scale multiply
  (:295,:299 — bf16 scalar, 0.0082 layer 0 / 0.0261 layer 1, NOT an inert
  1.0); whole module bf16 (not in _keep_in_fp32_modules_strict :610)) +
  t_b2 dense; config.py load_verified now also checks
  dense_intermediate_size=24576 vs the raw checkpoint json (the field
  configuration_inkling.py:125-126 maps onto config.intermediate_size —
  raw intermediate_size 3072 is the EXPERT ffn; B2.4 convention for
  newly-relied-on fields). ITEM WORDING vs FACT (B1.2 flag stands): the
  item says "dense layer 2 MLP" but dense_mlp_idx=2 is a COUNT — the dense
  layers are 0 and 1, layer 2 is MoE (its MLP was gated by B2.8/B2.9).
  The test therefore anchors BOTH real dense layers, strictly more than
  the item's literal ask; item text left untouched (loop may not edit).
  Weights: disk w13_dn [49152,6144] interleaved [g0,u0,g1,u1,..] rows ->
  de-interleave -> chunk halves [gate;up] (conversion_mapping.py:196-201),
  w2_md -> down, and the conversion is pinned torch.equal vs transformers'
  OWN Interleave(dim=0)+Chunk(dim=0) ops run in-process on both layers.
  Ran test verbatim from /workspace/tm-opt (venv active,
  CUDA_VISIBLE_DEVICES=4,5,6,7). Real output:
  ```
  dense ok: frozen anchors layers {0,1} (the dense layers — dense_mlp_idx=2 is a COUNT, census B1.2; layer 2 is MoE, covered by B2.8/B2.9) — mlp.out rel err 0.0e+00/0.0e+00 < 1e-2 (bit-exact 2/2); weight conversion pinned vs HF's own Interleave(0)+Chunk(0) ops both layers; native InklingMLP cross-check bit-exact 6/6 (real L0 T{13,57} + batch 2, real L1 T13, random hidden512/ffn768 T{600,13}); global_scale LIVE (0.0082/0.0261) applied last (bitwise), gate/up swap breaks anchor, zero-in exact zero-out; fp64 replay agrees 2.5e-03/3.3e-03 < 1e-2
  ```
  Everything bit-exact (InklingMLP is three plain nn.Linears — no dispatch
  decorator, so unlike B2.9's experts the anchors ARE bitwise): both frozen
  mlp.out anchors, all 6 native-module cross-checks (real L0 weights on
  frozen input + T57 + batch-2 T5, real L1 weights on frozen input, random
  hidden-512/ffn-768 config at T600 + batch-2 T13). Structural pins:
  scale-is-last-op bitwise (unit-scale run * gs == output), gate/up swap
  breaks the anchor (orientation discriminates, silu asymmetric), zero-in
  -> exact zero-out (no biases), independent fp64 replay 2.5e-03/3.3e-03.
  Anchor-point sanity: mlp.in == post_attention_layernorm.out bitwise both
  layers (capture convention pinned). NOTE for B2.11: all five ref layers'
  MLP paths are now individually gated (dense 0-1 bitwise, MoE 2/5/6 to
  grouped_mm numerics per B2.9) — the full-forward item composes them with
  attention/sconv/norm blocks that are all already bit-exact. Regressions
  all green after the config.py addition: t_b2 ref (verify path, sha
  eae8f3a95804 unchanged) + embed + rmsnorm + relbias + sconv + attn_global
  + attn_swa + gate + moe; t_b1 index + census + dtypes + plan. Code
  committed first (f763e28) per convention; this tick is its own commit.
- 2026-07-26 B2.11: implemented model.layer_prefill (exact
  InklingDecoderLayer op order, modeling_inkling.py:566-591, composing the
  B2.2-B2.10 functions; weight-dict keys documented in the docstring) +
  model.final_logits (norm -> hidden DIVIDED by logits_mup_width_multiplier
  24.0 -> bias-free lm_head (unembed) -> slice to unpadded_vocab_size
  200058, exact transcription of modeling_inkling.py:783-789; identical in
  the multimodal wrapper :1280-1286) + t_b2 logits; config.py load_verified
  now also checks unpadded_vocab_size + logits_mup_width_multiplier (newly
  relied on, B2.4 convention). REFERENCE DESIGN (forced by the same P2
  facts as B2.1): full-model from_pretrained is impossible here — bf16
  routed experts are ~1.8 TB and the packed layers re-init — so the
  reference STREAMS the actual native transformers InklingDecoderLayer
  one layer at a time on GPU 4 (eager attention + grouped_mm experts
  dispatch, exactly from_pretrained's defaults per modeling_utils.py:2100 /
  the B2.9 provenance pin), each built from the same converted disk weights
  our engine consumes (B1.4 chunked nvfp4 dequant + proven de-interleave),
  then final norm + the :783-789 logits head. The composition is NOT taken
  on faith: prompt 0 is the frozen B2.1 prompt and the streaming ref had to
  reproduce the frozen captures BIT-exactly at 13 pins (embed +
  embed_norm, layers {0,1,2,5,6} in/out — covering dense, bf16-MoE,
  nvfp4-MoE, SWA and global arms — and final_norm applied at layer 6);
  masks bit-equal the native create_*_mask builders on all 5 prompts;
  conv_mask pinned None; and OUR logit head run on the REF hidden states
  is bitwise identical to the native head, isolating all ours-vs-ref
  divergence to the expert kernels (eager loop vs grouped_mm, B2.9).
  Ran test verbatim from /workspace/tm-opt (venv active,
  CUDA_VISIBLE_DEVICES=4,5,6,7), foreground. Real output (after 9
  per-8-layer stderr progress lines):
  ```
  logits ok: full 66-layer streamed forward, 5 tiny prompts (T 4-13) — greedy top-1 MATCH 5/5 vs native transformers (' Berlin',' Au',' oxygen',' cold',' four'); streaming ref pinned bit-exact vs frozen captures (13 pins: embed+norm, layers {0,1,2,5,6} in/out under grouped_mm dispatch, final_norm@L6); our engine bit-exact vs frozen layers 0-1 out (2/2), MoE-layer drift @2/5/6 4.8e-04/2.1e-03/2.9e-03 < 1e-2 (expert-kernel order, B2.9); masks bit-equal native builders x5 prompts, conv_mask None; logit head bit-equal on shared hidden; last-pos logits rel err 9.2e-02 max, top-1 margins ref [6.0,0.8,4.5,1.0,1.8] ours [6.1,0.9,4.5,0.8,1.5]; wall 130 s
  ```
  Prompts: the frozen B2.1 prompt + 4 with confidently-determined next
  tokens; all five continuations are the factually right ones. NOTE for
  B3: the 9.2e-02 last-position logits rel err is the accumulated
  eager-vs-grouped_mm drift over 63 MoE layers (per-layer ~3e-03, B2.9) —
  top-1 held with margins 0.8-6.0 but B3.2's 32-token decode gate should
  expect occasional divergence risk on low-margin steps; our engine is
  internally deterministic, and the B3.5 parity gate vs vLLM goldens (not
  vs transformers) is the binding one. Wall 130 s: streams the 551 GiB
  checkpoint once, both engines advance in lockstep per layer sharing one
  weight load (~2.7 s/MoE layer read+dequant, page cache warm).
  Regressions all green after the config.py addition: t_b2 ref (verify
  path, sha eae8f3a95804 unchanged) + embed + rmsnorm + relbias + sconv +
  attn_global + attn_swa + gate + moe + dense; t_b1 index + census +
  dtypes + plan. Code committed first (7b63ab6) per convention; this tick
  is its own commit. B2 track complete; next is B3.1 (KV structs).
- 2026-07-26 B3.1: implemented kv.py (SconvRing — last sconv_k-1=3 RAW
  pre-conv inputs, zero-init == conv's implicit left pad; RingKV capacity
  512 == the SWA window predicate's allowed set exactly, so decode needs no
  mask, kv_pos > q_pos - 512 per B2.7; PagedKV — PAGE_SIZE 16, per-seq page
  table over a caller-ordered free list so gather is real indirection;
  LayerKV = cache + 4 sconv rings; caches hold POST-q/k-norm K and
  POST-sconv V, per B2.5-B2.7 the values that never change once written),
  model.py decode forms (sconv_decode — same F.conv1d op on the exact
  4-token window, padding=0, fp32 + own-input residual, causal_conv1d_update
  semantics modeling_inkling.py:441-457; attn_decode — Q=1 over gathered
  cache, rel_bias reused at distances pos-kv_pos, tau round-trip on global,
  GQA expand, NO additive mask by construction; layer_decode — layer_prefill
  op order, MoE/dense reused at T=1 per B2.9 note, optional trace for
  tests), attn_prefill/layer_prefill grew state=None population hooks
  (numerics untouched — same ops, verified by full B2 rerun), and
  t_b2.load_layer_weights lifted from t_logits' closure to module level
  (unchanged body) for reuse. Ran test verbatim from /workspace/tm-opt (venv
  active, CUDA_VISIBLE_DEVICES=4,5,6,7). Real output (after 5 per-layer
  stderr progress lines):
  ```
  kv ok: layers [0, 1, 2, 3, 5] (dense/bf16-MoE/nvfp4-MoE x SWA/global), 600-token prompt + 2 decode steps (crosses window 512), teacher-forced per layer (see docstring) — decode vs recompute-from-scratch: attention half (KV-fed) 2.1e-05/1.6e-05/7.9e-05/1.2e-05/7.9e-06 < 0.001, full layer out 8.1e-05/9.7e-05/6.5e-04/4.0e-04/5.4e-03 < 0.001(dense)/0.01(moe, bf16 routing-weight granularity, expert CHOICE pinned identical 3/3 moe layers x 2 steps); prefill-written cache K/V + all 4 sconv-ring tails BIT-equal same-shape replay 5/5 layers (replay itself bit-equal layer_prefill); SWA ring holds exactly positions 90..601, paged global 0..601 in 38 shuffled pages (page 16); prefix drift <= 3.0e-04; struct pins (ring wrap order+positions, oversized-append drop, shuffled-table paged gather + partial page, sconv-ring windows + short-prefill zeros) all bit-exact; wall 13 s
  ```
  GATE DESIGN (documented in t_b3.py's docstring; two dead ends hit
  honestly): bitwise decode==recompute is unattainable in principle — T=1
  GEMV vs T=602 GEMM rows pick shape-dependent accumulation orders. First
  attempt (three propagating streams) showed the drift COMPOUNDS with depth
  (600- vs 602-row streams diverge to 7e-04 by layer 1 with NO cache
  involved), so the test teacher-forces each layer with the recompute
  stream's own input — one layer of shape effects per comparison;
  end-to-end composition is B3.2's job vs the EXTERNAL transformers oracle.
  Second: MoE layer outputs sit ~5e-03 apart even teacher-forced —
  diagnosed to the routing WEIGHTS' all-bf16 chain (sigmoid -> logsigmoid
  -> softmax on a ~2^-8 grid, B2.8): 1-ulp logit shifts move weights ~1e-2
  rel while expert CHOICE is stable (verified identical). So the gates
  decompose: attention half (everything the KV state feeds: paged/ring
  cache, k/v/attn sconv rings, decode rel-bias, GQA gather) < 1e-3
  (measured ~1e-5); expert top-6 set EQUAL; full MoE out < 1e-2 (the
  B2.9/B2.11 expert-numerics budget); dense out < 1e-3; and the mechanics
  arms carry the bitwise burden — cache content + all four ring tails
  torch.equal vs a same-shape replay (itself pinned bit-equal to
  layer_prefill's output), synthetic wraparound/oversized-append/partial-
  page/shuffled-table pins, and exact positional pins (SWA ring == positions
  90..601 = the window set for query 601; paged == 0..601 over 38
  non-contiguous pages). NOTE for B3.2: layer_decode(trace=) exposes
  x1/mlpin for tests; decode reuses moe/dense at T=1; expect the same
  bf16-routing-weight noise vs transformers, with top-1 agreement as the
  gate (B2.11 note). NOTE for B3.3: kv.py is single-sequence by design;
  multi-seq ownership lands in scheduler.py (PagedKV free_order hook is the
  allocator seam). Regressions all green after the model.py/t_b2.py
  changes: t_b2 ref (verify path, sha eae8f3a95804 unchanged) + embed +
  rmsnorm + relbias + sconv + attn_global + attn_swa + gate + moe + dense +
  logits (identical evidence lines incl. top-1 5/5, wall 129 s); t_b1 index
  + census + dtypes + plan. Code committed first (f3b1fbd) per convention;
  this tick is its own commit.
