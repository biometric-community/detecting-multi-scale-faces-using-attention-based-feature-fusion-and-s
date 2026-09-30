#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PROJECT_ROOT="$ROOT"
# shellcheck source=/dev/null
source "$ROOT/../../../.cursor/skills/_shared/project_env.sh"
LOG="$ROOT/outputs/logs/train_full.log"
mkdir -p "$ROOT/outputs/logs"
echo "Logging to $LOG"
exec "$PY" -m sanet.train --config configs/default.yaml "$@" 2>&1 | tee -a "$LOG"
