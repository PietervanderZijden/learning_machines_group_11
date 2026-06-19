#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PORT=${COPPELIA_SIM_PORT:-23000}
HOST=${COPPELIA_SIM_IP:-127.0.0.1}
TOTAL_STEPS=${TOTAL_STEPS:-500000}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-sac_models}

args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
        --host) HOST="${args[$((i + 1))]}" ;;
        --host=*) HOST="${args[$i]#--host=}" ;;
        --port) PORT="${args[$((i + 1))]}" ;;
        --port=*) PORT="${args[$i]#--port=}" ;;
        --total-timesteps) TOTAL_STEPS="${args[$((i + 1))]}" ;;
        --total-timesteps=*) TOTAL_STEPS="${args[$i]#--total-timesteps=}" ;;
        --checkpoint-dir) CHECKPOINT_DIR="${args[$((i + 1))]}" ;;
        --checkpoint-dir=*) CHECKPOINT_DIR="${args[$i]#--checkpoint-dir=}" ;;
    esac
done

echo "SAC transfer training"
echo "  Simulator: $HOST:$PORT"
echo "  Total timesteps: $TOTAL_STEPS"
echo "  Checkpoint dir: $CHECKPOINT_DIR"

export PYTHONUNBUFFERED=1
uv run python -u train_sac.py \
    --host "$HOST" \
    --port "$PORT" \
    --total-timesteps "$TOTAL_STEPS" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    "$@"
