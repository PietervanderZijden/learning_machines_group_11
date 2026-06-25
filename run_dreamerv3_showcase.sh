#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CHECKPOINT="${1:-dreamerv3_models/dreamerv3-reference/latest.pt}"
if [[ $# -gt 0 ]]; then
    shift
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi
if [[ ! -f dreamerv3_reference/dreamer.py ]]; then
    echo "Missing dreamerv3_reference checkout." >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
uv run python -u showcase_dreamerv3_reference.py \
    --checkpoint "$CHECKPOINT" \
    "$@"
