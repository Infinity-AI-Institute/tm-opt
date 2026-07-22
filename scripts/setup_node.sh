#!/usr/bin/env bash
# @brief  One-time node prep: make every GPU a stable measurement instrument,
#         then bring up the vLLM baseline on the reserved slot (GPU 6).
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/workspace/models/inkling-nvfp4}     # local HF checkpoint path
BASELINE_GPU=6
BASELINE_PORT=8106

#1. persistence mode: keep driver loaded, no cold-start jitter between runs
sudo nvidia-smi -pm 1

#2. lock graphics clocks on ALL GPUs so thermal drift can't fake a regression
#   (query supported clocks first; pick the max sustainable, not boost)
MAX_CLOCK=$(nvidia-smi --query-supported-clocks=graphics --format=csv,noheader,nounits -i 0 | head -1)
sudo nvidia-smi -lgc "${MAX_CLOCK},${MAX_CLOCK}"
echo "[setup] clocks locked at ${MAX_CLOCK} MHz on all GPUs"

#3. sanity: confirm 8 GPUs visible and idle
nvidia-smi --query-gpu=index,name,memory.total,clocks.gr --format=csv

#4. python deps for the harness (NOT the engine -- engine builds per-worktree)
pip install --quiet aiohttp requests transformers torch

#5. launch vLLM baseline pinned to the reserved GPU, kept warm permanently
CUDA_VISIBLE_DEVICES=${BASELINE_GPU} nohup vllm serve "${MODEL_DIR}" \
    --port "${BASELINE_PORT}" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    > baseline_vllm.log 2>&1 &
echo "[setup] vLLM baseline starting on GPU ${BASELINE_GPU}, port ${BASELINE_PORT}"
echo "[setup] baseline bench: python harness/benchmark.py \\"
echo "          --endpoint http://localhost:${BASELINE_PORT} \\"
echo "          --model ${MODEL_DIR} --gpu ${BASELINE_GPU}"
