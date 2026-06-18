#!/usr/bin/env bash
set -euo pipefail

# DreamerV4 image shortcut-forcing training on recorded episodes
# Usage:
#   ./run_dreamerv4.sh                                    # train on recorded data
#   ./run_dreamerv4.sh --resume                           # resume from checkpoint

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RECORD_DIR=${RECORD_DIR:-recorded-states/dreamer-v3-states}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-results/dreamer-v4-checkpoints}
TOTAL_STEPS=${TOTAL_STEPS:-50000}
HARDWARE_RECORD_DIR=${HARDWARE_RECORD_DIR:-}

echo "DreamerV4 Training"
echo "  Record dir: $RECORD_DIR"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Total steps: $TOTAL_STEPS"
if [[ -n "$HARDWARE_RECORD_DIR" ]]; then
    echo "  Hardware record dir: $HARDWARE_RECORD_DIR"
fi

extra_args=()
if [[ -n "$HARDWARE_RECORD_DIR" ]]; then
    extra_args+=(--hardware-record-dir "$HARDWARE_RECORD_DIR")
fi

uv run python train_dreamerv4_image.py \
    --offline \
    --record-dir "$RECORD_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --total-steps "$TOTAL_STEPS" \
    "${extra_args[@]}" \
    "$@"
