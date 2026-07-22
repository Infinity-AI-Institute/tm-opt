#!/usr/bin/env bash
# Ralph BUILD loop: grind PROGRESS.md one item per iteration.
# Usage: bash scripts/ralph_build.sh [max_iters]   (default 10)
# Stop early: touch /workspace/tm-opt/STOP_RALPH
# Logs: /workspace/ralph_logs/build_<ts>/iter_N.log
set -uo pipefail
cd /workspace/tm-opt
MAX=${1:-10}
RUN="/workspace/ralph_logs/build_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$RUN"
echo "[ralph-build] up to $MAX iterations; logs in $RUN; stop: touch STOP_RALPH"
NOPROG=0

for i in $(seq 1 "$MAX"); do
  [ -f STOP_RALPH ] && { echo "[ralph-build] STOP_RALPH — halting"; break; }
  #1. done? (any unchecked, unblocked item left?)
  if ! grep -E "^- \[ \]" PROGRESS.md | grep -qv "BLOCKED"; then
    echo "[ralph-build] no open items — PROGRESS complete or all BLOCKED"; break
  fi
  BEFORE=$(git rev-parse HEAD)
  echo "[ralph-build] === iteration $i/$MAX ($(date +%H:%M:%S)) ==="
  #2. one stateless iteration; the prompt file is the whole contract
  timeout 3600 claude -p "$(cat agent/PROMPT_BUILD.md)" \
      --dangerously-skip-permissions \
      > "$RUN/iter_$i.log" 2>&1
  RC=$?
  AFTER=$(git rev-parse HEAD)
  #3. verdict from observable repo state, not the agent's claims
  if [ "$BEFORE" = "$AFTER" ]; then
    NOPROG=$((NOPROG+1))
    echo "[ralph-build] iter $i: NO COMMIT (rc=$RC) — see $RUN/iter_$i.log"
    [ "$NOPROG" -ge 2 ] && { echo "[ralph-build] two no-progress iterations — halting for human review"; break; }
  else
    NOPROG=0
    echo "[ralph-build] iter $i landed:"; git log --oneline "$BEFORE..$AFTER" | sed "s/^/    /"
  fi
  grep -E "^- \[.\]" PROGRESS.md | tail -3
done
echo "[ralph-build] done. Review: git log, PROGRESS.md, $RUN/"
