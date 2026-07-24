# Baseline notes — vLLM serving Inkling-NVFP4 on 4×B300

Evidence log for the canonical baseline. Every claim here cites a source:
a committed log (docs/logs/), a live-server probe, or vLLM source (file:line).
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

## Canonical contract freeze (P1, 2026-07-23) — cache_key 8451a604a8849296

NUMA note: container seccomp blocks set_mempolicy (membind); bench server
runs cpunodebind-only to node 0 (capability-probed in serve_vllm_bench.sh;
mode recorded in each bench log). Both engines run under the identical mode.

Three-stage sweep (evidence: docs/logs/2026-07-2{2,3}_sweep_*.log), bench
server (prefix cache OFF, KV pinned 111231943107 B):

| conc | decode-heavy tok/s | prefill-heavy tok/s |
|---|---|---|
| 8    | 882.9   | 813.3  |
| 16   | 1,640.5 | 1,432.3 |
| 32   | 2,910.0 | 2,162.4 |
| 64   | 4,636.9 | 3,016.4 |
| 128  | 7,134.6 / 7,143.0 / 3,865.0* | 3,827.2 |
| 256  | 10,674.9 | 4,583.9 |
| 512  | 13,283.9 / **13,374.1** | 4,995.2 / **5,108.8** |
| 768  | 4,958.9 (COLLAPSE) | 3,999.4 |
| 1024 | 6,036.3 | 3,769.9 |

*128 decode measured in sweeps 1 and 2: 7,134.6 vs 7,143.0 = 0.12% spread;
prefill numbers from their respective sweeps. 512 measured twice:
decode 0.68% spread, prefill 2.2% spread → first measured noise floors
(merge-gate minimum 0.3% is realistic for decode; prefill claims need the
larger floor — P3 repeats will pin both).

Findings:
- TRUE PEAK at conc 512, not a plateau. Collapse past 512 is
  recompute-preemption thrash: per-seq KV ≈ 0.64 GiB (ISL~0.77K avg + 8K
  decode) → 512 seqs ≈ 328 GiB fits the 4×106.27 GiB pool; 768 ≈ 491 GiB
  does not. Consistent with the KV-allocation verdict above: KV not binding
  at ~100 seqs, binding at ~700.
- Canonical concurrency FROZEN at 512 for both workloads (D10).
- Orientation peaks (sweep observations, superseded by P3's formal capture):
  decode-heavy ≈ 13.4K tok/s, prefill-heavy ≈ 5.1K tok/s.
- Engine consequence: the canonical point sits at the KV-capacity edge —
  KV efficiency and 512-sequence scheduling are entry requirements
  (ARCHITECTURE rev 4).
