#!/usr/bin/env bash
set -euo pipefail

# SAC + HER training for Robobo food collection
# Records episodes to disk for DreamerV4 offline training.
# Usage:
#   ./run_sac_her.sh                              # fresh training
#   ./run_sac_her.sh --resume                     # resume from checkpoint
#   ./run_sac_her.sh --domain-randomization       # enable DR
#   ./run_sac_her.sh --no-record                  # disable episode recording

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PORT=${COPPELIA_SIM_PORT:-23000}
TOTAL_STEPS=${TOTAL_STEPS:-500000}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-sac_her_models}
RECORD_DIR=${RECORD_DIR:-recorded_episodes}

echo "SAC + HER Training"
echo "  Port: $PORT"
echo "  Total timesteps: $TOTAL_STEPS"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Record dir: $RECORD_DIR"

uv run python train_sac.py \
    --port "$PORT" \
    --total-timesteps "$TOTAL_STEPS" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --record-dir "$RECORD_DIR" \
    "$@"
