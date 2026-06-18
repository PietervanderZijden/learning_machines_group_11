#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PORT=${COPPELIA_SIM_PORT:-23000}
TOTAL_STEPS=${TOTAL_STEPS:-500000}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-sac_models}

echo "SAC transfer training"
echo "  Port: $PORT"
echo "  Total timesteps: $TOTAL_STEPS"
echo "  Checkpoint dir: $CHECKPOINT_DIR"

uv run python train_sac.py \
    --port "$PORT" \
    --total-timesteps "$TOTAL_STEPS" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    "$@"
