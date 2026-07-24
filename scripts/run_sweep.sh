#!/usr/bin/env bash
# One sweep extension run; edit CONCS between runs. Short invocation exists
# because pasting long commands into tmux mangles (% and " are tmux keys).
set -euo pipefail
cd /workspace/tm-opt
source /workspace/venv/bin/activate
CONCS="${1:-512,768,1024}"
exec python scripts/freeze_canonical.py --endpoint http://localhost:8106 \
  --concs "$CONCS" 2>&1 | tee "/workspace/logs/sweep_$(date +%Y%m%d_%H%M).log"