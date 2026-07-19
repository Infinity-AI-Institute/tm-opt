# tmopt engine architecture — Inkling on 8×B300 (rev 2, from config.json)

Single-model engine for `thinkingmachines/Inkling-NVFP4`, specialized for B300,
deployed as **two 4-GPU replicas** (leading design; 8-GPU×1 is the standing A/B).
Text-only v1: vision/audio configs exist in the checkpoint but are out of scope,
declared in README.

## Verified model shape (config.json is the source of truth)

| Component | Value |
|---|---|
| Layers | 66 (layer 2 = dense MLP, others MoE) |
| Hidden / vocab / ctx | 6144 / 201024 (200058 unpadded) / 1M |
| Global attn layers | 11: {5,11,17,23,29,35,41,47,53,59,65}; 64Q/8KV/128d |
| SWA layers | 55; window 512; 64Q/**16KV**/128d |
| Position encoding | none (no RoPE): relative bias d_rel=16, extent 1024 + log scaling (floor 128000, α 0.1), pre-softmax |
| sconv | window-4 conv on attn K, V, output, and MoE output, every layer |
| MoE | 256 experts, top-6, sigmoid gate + bias, norm_after_topk, route_scale 8, 2 shared experts (sink), expert FFN 3072 |
| Dense layer 2 | FFN 24576 |
| MTP | 8 chained draft layers, separate mtp.safetensors |
| Precision | NVFP4 weights (exclude: embeds, norms, unembed, layer-0 attn); bf16 activations & KV |

## Memory plan (per 4-GPU replica, 269 GB/GPU real)

- Weights: ~600 GB NVFP4 / 4 ≈ 150 GB/GPU (sharded EP/TP), + bf16 excluded modules
- KV per sequence: SWA fixed ≈230 MB (55 × 512 × 16×128×2×2B) + global 45 KB/token
  (11 × 8×128×2×2B) + sconv state ~negligible (window 4)
  → at 16K ctx ≈ 0.97 GB/seq vs ≈5.3 GB uniform-global equivalent: the
  concurrency edge. KV pool sizing and canonical concurrency derive from this.
- Remainder: activations, CUDA graph pools, NCCL buffers.

## Kernel inventory (rev 2)

| Kernel | Notes |
|---|---|
| moe_grouped_gemm | skinny experts (N=3072): many small tiles, device-side work queue, fused sigmoid gate+bias → top-6 → norm_after_topk → permute; shared-expert (sink) GEMM fused in same launch; NVFP4 B-operand per hf_quant_config layout |
| attention_prefill / decode | two KV layouts (paged global 8KV-head, ring-512 SWA 16KV-head); relative bias + log scaling fused pre-softmax; deterministic reduction order (parity gate is batch-sensitive) |
| sconv | window-4 depthwise conv on K/V/attn-out/MoE-out; fuse into adjacent ops; per-layer ring state cached like a virtual window-4 KV |
| fused_rmsnorm | rms_norm_eps 1e-6, embed_norm variant; (RoPE kernel from rev 1 deleted — model has no RoPE) |
| dense_mlp | layer 2 only, FFN 24576, standard fused GEMM path |
| mtp | 8 chained draft layers with own local/global ids; verify pass batched with main model |

## Fairness notes (feed FAIRNESS_AUDIT.md)

- Window-512 SWA on the 55 local layers is the model's trained semantics
  (config `local_layer_ids`, `sliding_window_size: 512`); vLLM implements the
  same. Not a computation-reduction optimization.
- KV dtype: bf16 both engines (`kv_cache_quant_algo: none`).
- Checkpoint identical both engines: local NVFP4 snapshot, exclude-list honored.
- Baseline = official vLLM recipe flags + pinned build g9243e0124; tested at
  TP=4 (×2 replicas) and TP=8; canonical comparison is against vLLM's best.

## Determinism contract

Fixed reduction orders in attention/MoE reductions; batch-invariant outputs.
Parity gate: greedy token match + logprob delta vs HF transformers (≥5.14)
goldens, generated once from this checkpoint.
