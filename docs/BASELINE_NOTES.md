# Baseline notes — vLLM serving Inkling-NVFP4 on 4×B300

Evidence log for the canonical baseline. For every claim here I cite a source:
e.g., committed log (docs/logs/), live-server probe, or vLLM source .
First successful serve: 2026-07-19, build `vllm-0.23.1rc1.dev1270+g9243e0124-tp4`
(system_fingerprint from the first completion response).

## Serve configuration (smoke config — NOT the frozen benchmark config)

- Script: `scripts/serve_vllm_baseline.sh` (official recipe flags, local model path,
  TP=4, max_model_len 16384, port 8106, CUDA 13.0 toolchain for FlashInfer JIT).
- Resolved defaults observed in logs: `gpu_memory_utilization → 0.92`,
  `enable_prefix_caching=True`, chunked prefill on (max_num_batched_tokens=8192),
  kv_cache_dtype `auto` → bf16 (checkpoint declares kv_cache_quant_algo "none").
- **Benchmark variant must add:** `--no-enable-prefix-caching` (both engines),
  and consider pinning KV via `--kv-cache-memory` (log suggested 111231943107
  bytes for the 0.92 target) instead of utilization fraction, for exact parity.

## First tokens

- "The capital of France is" → " Paris." (greedy, correct). 
- Logprobs format (this build): token **strings** in `logprobs.tokens`,
  floats in `logprobs.token_logprobs`; token **IDs** only with request flag
  `"return_token_ids": true` → populated at **choice level** (`choice.token_ids`,
  plus `prompt_token_ids`). Parity gate (`harness/correctness.py`) reads
  choice-level IDs and fails loudly if absent. Our engine must implement the
  same flag.

## Startup anatomy (cold, first run — init total 1748 s)

| Phase | Time |
|---|---|
| Weight load (551.35 GiB, 34 shards, XFS volume) | ~34 s (~15 GB/s) |
| NVFP4 MoE weight prep | ~70 s |
| KV memory profiling | ~18 min (dominant) |
| FlashInfer MoE autotune (gemm1/gemm2/block_scale_moe, 21 profiles each) | ~5 min |
| CuTeDSL `inkling_fa4` attention compile (130 units) | ~170 s |
| CUDA graph capture (51 sizes, piecewise+full) | 56 s, 2.53 GiB |

Caches persisted to `/workspace/.flashinfer` (JIT + autotune live under
`/root/.cache/flashinfer` = ephemeral). Warm restarts are minutes, not ~29.
Methodology note: server init is excluded from all measurements, both engines.

## Per-GPU memory ledger (serving, GPUs 0–3; GPUs 4–7 idle at 4 MiB)

267.69 GiB usable = 137.46 weights + 1.91 peak activation + 2.53 CUDA graphs
+ 0.63 non-torch + **106.27 KV pool** (+ slack). nvidia-smi: 261,144 MiB used.

## KV allocation investigation (the "10.48×" anomaly) — RESOLVED

**Question:** startup log reports `GPU KV cache size: 171,639 tokens` and
`Maximum concurrency for 16,384 tokens per request: 10.48x`, implying ~10 GiB
per 16K request — vs ~1 GiB expected from exact hybrid semantics.

**Spec evidence (source):** the integration is window-aware in *allocation*,
not just computation:
- `vllm/models/inkling/nvidia/attention.py:200-211` — `get_kv_cache_spec()`
  returns `SlidingWindowSpec` for the 55 local layers, `FullAttentionSpec`
  for the 11 global layers.
- `vllm/models/inkling/nvidia/sconv_swa_attn.py` — sconv state emitted as a
  window-4 `SlidingWindowSpec` per layer (padded to the uniform page size).
- `attention.py:209` — compute mask also passes `sliding_window=local_extent`.

**Exact-quotient observation:** 171,639 / 16,384 = 10.477 — the reported
"maximum concurrency" is a worst-case reservation convention (max_model_len
per request), not measured capacity.

**Empirical probe (live server):** one request, ~12K-token prompt + 4,000
forced decode tokens (`ignore_eos`), gauge `vllm:kv_cache_usage_perc` polled
every 3 s: rose 0.00179 → **peak 0.00209 (0.21%)**. Full-length uniform
allocation predicts ~7%; observed is ~35× lower. Response evidence:
`/tmp/kvprobe_response.json` (copy archived in docs/logs/ if referenced).

**Verdict:** vLLM's Inkling baseline is KV-efficient. The hypothesized
"~10× concurrency lever from vLLM over-allocation" **does not exist**.
Our engine's hybrid ring/paged KV is required for *parity*, not advantage.
KV memory will not be the binding constraint at 16K context; canonical
concurrency is set by compute/scheduler saturation, found empirically.

**Fairness-audit line:** baseline measured at vLLM's best available
configuration; its window-aware KV allocation was verified at spec level and
empirically before any comparative claims.

## Orientation numbers (non-canonical, single observations)

- 12K prefill + 4,000-token forced decode completed within a ~24 s window →
  single-stream decode order-of-magnitude ≥ ~170 tok/s. To be superseded by
  the canonical sweep.
