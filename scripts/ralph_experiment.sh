#!/usr/bin/env bash
# Ralph EXPERIMENT loop: one hypothesis per iteration, queue-backpressured.
# Usage: bash scripts/ralph_experiment.sh [max_iters]   (default 6)
# Requires: dispatcher running (python harness/dispatcher.py) to drain the queue.
set -uo pipefail
cd /workspace/tm-opt
MAX=${1:-6}
RUN="/workspace/ralph_logs/exp_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$RUN"
echo "[ralph-exp] up to $MAX iterations; logs in $RUN; stop: touch STOP_RALPH"

for i in $(seq 1 "$MAX"); do
  [ -f STOP_RALPH ] && { echo "[ralph-exp] STOP_RALPH — halting"; break; }
  PENDING=$(ls experiments/queue/*.json 2>/dev/null | wc -l)
  if [ "$PENDING" -ge 2 ]; then
    echo "[ralph-exp] queue has $PENDING pending — backpressure, waiting 10m"
    sleep 600; continue
  fi
  echo "[ralph-exp] === iteration $i/$MAX ($(date +%H:%M:%S)) ==="
  timeout 5400 claude -p "$(cat agent/PROMPT_EXPERIMENT.md)" \
      --dangerously-skip-permissions \
      > "$RUN/iter_$i.log" 2>&1 || echo "[ralph-exp] iter $i rc=$?"
  ls experiments/queue/ 2>/dev/null || true
done
echo "[ralph-exp] done. Ledger tail:"; tail -3 experiments/ledger.jsonl 2>/dev/null || true
