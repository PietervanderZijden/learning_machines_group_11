#!/usr/bin/env bash
set -euo pipefail

# DreamerV4 training on recorded episodes
# Usage:
#   ./run_dreamerv4.sh                                    # train on recorded data
#   ./run_dreamerv4.sh --use-synthetic                    # test with synthetic data
#   ./run_dreamerv4.sh --resume                           # resume from checkpoint

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

RECORD_DIR=${RECORD_DIR:-recorded_episodes}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-dreamerv4_models}
TOTAL_STEPS=${TOTAL_STEPS:-500000}

echo "DreamerV4 Training"
echo "  Record dir: $RECORD_DIR"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Total steps: $TOTAL_STEPS"

uv run python train_dreamerv4.py \
    --record-dir "$RECORD_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --total-steps "$TOTAL_STEPS" \
    "$@"
