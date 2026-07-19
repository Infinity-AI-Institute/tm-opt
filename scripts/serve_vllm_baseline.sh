#!/usr/bin/env bash
# Official vLLM recipe config for Inkling-NVFP4 — the canonical baseline server.
# Source: recipes.vllm.ai/thinkingmachines/Inkling (pinned build g9243e0124)
set -euo pipefail
source /workspace/venv/bin/activate
export VLLM_USE_V2_MODEL_RUNNER=1
export FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"

mkdir -p /workspace/logs
LOG="/workspace/logs/vllm_$(date +%Y%m%d_%H%M%S).log"
echo "[serve] logging to $LOG"

exec vllm serve /workspace/models/inkling-nvfp4 \
  --tokenizer-mode inkling \
  --reasoning-parser inkling \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --trust-remote-code \
  --port 8106 2>&1 | tee "$LOG"