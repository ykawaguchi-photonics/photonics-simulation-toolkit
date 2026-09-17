#!/bin/bash
# Lightweight dry run confirming cron can reach this repo and the mp conda
# environment before trusting the real overnight sweep (run_mzi_sweep_once.sh)
# to the same mechanism. Self-removes its own crontab entry after firing once.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="$REPO_DIR/logs/cron_dryrun_test.log"
CONDA_BIN="$(command -v conda 2>/dev/null || echo "$HOME/miniconda3/bin/conda")"
CRON_MARKER="mzi-dryrun-test"

mkdir -p "$REPO_DIR/logs"
cd "$REPO_DIR"

{
    echo "=== dryrun_test.sh fired at $(date) ==="
    pwd
    "$CONDA_BIN" run -n mp python -c "import meep; print('meep version:', meep.__version__)"
    echo "=== dryrun_test.sh OK ==="
} >> "$LOG_FILE" 2>&1

crontab -l 2>/dev/null | grep -v "$CRON_MARKER" | crontab -
