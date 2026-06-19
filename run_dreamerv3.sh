#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PORT=${COPPELIA_SIM_PORT:-23000}
HOST=${COPPELIA_SIM_IP:-127.0.0.1}
TOTAL_STEPS=${TOTAL_STEPS:-500000}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-dreamerv3_models}

args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
        --port)
            if ((i + 1 < ${#args[@]})); then
                PORT="${args[$((i + 1))]}"
            fi
            ;;
        --port=*)
            PORT="${args[$i]#--port=}"
            ;;
        --host)
            if ((i + 1 < ${#args[@]})); then
                HOST="${args[$((i + 1))]}"
            fi
            ;;
        --host=*)
            HOST="${args[$i]#--host=}"
            ;;
        --total-steps)
            if ((i + 1 < ${#args[@]})); then
                TOTAL_STEPS="${args[$((i + 1))]}"
            fi
            ;;
        --total-steps=*)
            TOTAL_STEPS="${args[$i]#--total-steps=}"
            ;;
        --checkpoint-dir)
            if ((i + 1 < ${#args[@]})); then
                CHECKPOINT_DIR="${args[$((i + 1))]}"
            fi
            ;;
        --checkpoint-dir=*)
            CHECKPOINT_DIR="${args[$i]#--checkpoint-dir=}"
            ;;
    esac
done

echo "DreamerV3 Training"
echo "  Simulator: $HOST:$PORT"
echo "  Total timesteps: $TOTAL_STEPS"
echo "  Checkpoint dir: $CHECKPOINT_DIR"

export PYTHONUNBUFFERED=1
uv run python -u train_dreamerv3.py \
    --host "$HOST" \
    --port "$PORT" \
    --total-steps "$TOTAL_STEPS" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    "$@"
