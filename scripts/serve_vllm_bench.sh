#!/usr/bin/env bash
# Benchmark-mode vLLM server — the ONLY config canonical measurements run against.
# Differs from serve_vllm_baseline.sh (smoke config) in exactly two ways, both
# fairness-mandated:
#   1. --no-enable-prefix-caching   (frozen contract: prefix cache OFF both engines)
#   2. --kv-cache-memory pinned     (exact KV bytes, not a utilization fraction;
#      value = vLLM's own suggestion for the 0.92 target, docs/BASELINE_NOTES.md)
# Any change here changes the canonical cache-key -> all prior comparisons invalid.
set -euo pipefail
source /workspace/venv/bin/activate
export VLLM_USE_V2_MODEL_RUNNER=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"

mkdir -p /workspace/logs
LOG="/workspace/logs/vllm_bench_$(date +%Y%m%d_%H%M%S).log"
echo "[serve-bench] logging to $LOG"

#1. NUMA pin: GPUs 0-3 live on node 0 (cores 0-85) — keep host-side work local
exec numactl --cpunodebind=0 --membind=0 \
  vllm serve /workspace/models/inkling-nvfp4 \
  --tokenizer-mode inkling \
  --reasoning-parser inkling \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --no-enable-prefix-caching \
  --kv-cache-memory 111231943107 \
  --trust-remote-code \
  --port 8106 2>&1 | tee "$LOG"
