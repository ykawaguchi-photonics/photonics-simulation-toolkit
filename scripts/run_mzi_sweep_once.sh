#!/bin/bash
# One-time overnight run of notebooks/06_mzi.ipynb's full delta_L sweep (Section 10),
# fired by a single crontab entry. Bundles cd + the real command into one
# self-contained invocation -- a background process's working directory is not
# guaranteed to match the interactive shell's -- so this script never assumes any
# inherited shell state.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_DIR/logs"
LOG_FILE="$LOG_DIR/mzi_sweep_run.log"
CONDA_BIN="$(command -v conda 2>/dev/null || echo "$HOME/miniconda3/bin/conda")"
CRON_MARKER="mzi-sweep-once"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

{
    echo "=== run_mzi_sweep_once.sh starting at $(date) ==="
    "$CONDA_BIN" run -n mp jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=21600 \
        notebooks/06_mzi.ipynb
    echo "=== run_mzi_sweep_once.sh finished at $(date) ==="
} >> "$LOG_FILE" 2>&1

# One-shot job: remove this script's own crontab entry now that it has fired, so it
# doesn't silently re-fire on the same calendar date next year.
crontab -l 2>/dev/null | grep -v "$CRON_MARKER" | crontab -
