#!/usr/bin/env bash
# Ralph EXPERIMENT loop: one hypothesis per iteration, queue-backpressured.
# Usage: bash scripts/ralph_experiment.sh [max_iters]   (default 6)
# Requires: dispatcher running (python harness/dispatcher.py) to drain the queue.
# Run as the non-root 'ralph' user (CLI refuses --dangerously-skip-permissions
# as root; the cage is also just better design).

export BASH_DEFAULT_TIMEOUT_MS=2700000   # 45 min per foreground tool command
export BASH_MAX_TIMEOUT_MS=3300000       # 55 min hard cap (< the loop's 1h guillotine)

set -uo pipefail
cd /workspace/tm-opt

#0. self-sufficient environment + preflight (never depend on the launching shell)
source /workspace/venv/bin/activate
command -v claude >/dev/null || { echo "[ralph-exp] FATAL: claude CLI not on PATH"; exit 1; }

MAX=${1:-6}
RUN="/workspace/ralph_logs/exp_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$RUN"
echo "[ralph-exp] up to $MAX iterations; logs in $RUN; stop: touch STOP_RALPH"

for i in $(seq 1 "$MAX"); do
  [ -f STOP_RALPH ] && { echo "[ralph-exp] STOP_RALPH — halting"; break; }

  #1. backpressure: never stack specs the dispatcher hasn't drained
  PENDING=$(ls experiments/queue/*.json 2>/dev/null | wc -l)
  if [ "$PENDING" -ge 2 ]; then
    echo "[ralph-exp] queue has $PENDING pending — backpressure, waiting 10m"
    sleep 600; continue
  fi

  echo "[ralph-exp] === iteration $i/$MAX ($(date +%H:%M:%S)) ==="
  BEFORE=$(git rev-parse HEAD)

  #2. one stateless iteration; the prompt file is the whole contract
  timeout 5400 claude -p "$(cat agent/PROMPT_EXPERIMENT.md)" \
      --dangerously-skip-permissions \
      > "$RUN/iter_$i.log" 2>&1 || echo "[ralph-exp] iter $i rc=$?"

  AFTER=$(git rev-parse HEAD)
  #3. landed-report + provenance check: every loop commit carries [ralph].
  #   NOTE: worktree-branch commits (experiments/wt-<id>) are not visible to
  #   HEAD here — the worker enforces the prefix on those when it processes
  #   the spec; this check covers main-checkout commits (specs, DEADENDS.md).
  if [ "$BEFORE" != "$AFTER" ]; then
    echo "[ralph-exp] iter $i landed:"
    git log --oneline "$BEFORE..$AFTER" | sed "s/^/    /"
    if git log --format=%s "$BEFORE..$AFTER" | grep -qv '^\[ralph\]'; then
      echo "[ralph-exp] WARNING: commit(s) missing [ralph] prefix — review"
    fi
  fi

  ls experiments/queue/ 2>/dev/null || true
done
echo "[ralph-exp] done. Ledger tail:"; tail -3 experiments/ledger.jsonl 2>/dev/null || true
