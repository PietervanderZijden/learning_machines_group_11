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
HOST=${COPPELIA_SIM_IP:-127.0.0.1}
PORT=${COPPELIA_SIM_PORT:-23000}
MODE=offline

args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
        --online) MODE=online ;;
        --offline) MODE=offline ;;
        --host) HOST="${args[$((i + 1))]}" ;;
        --host=*) HOST="${args[$i]#--host=}" ;;
        --port) PORT="${args[$((i + 1))]}" ;;
        --port=*) PORT="${args[$i]#--port=}" ;;
        --record-dir) RECORD_DIR="${args[$((i + 1))]}" ;;
        --record-dir=*) RECORD_DIR="${args[$i]#--record-dir=}" ;;
        --checkpoint-dir) CHECKPOINT_DIR="${args[$((i + 1))]}" ;;
        --checkpoint-dir=*) CHECKPOINT_DIR="${args[$i]#--checkpoint-dir=}" ;;
        --total-steps) TOTAL_STEPS="${args[$((i + 1))]}" ;;
        --total-steps=*) TOTAL_STEPS="${args[$i]#--total-steps=}" ;;
    esac
done

echo "DreamerV4 Training"
echo "  Mode: $MODE"
if [[ "$MODE" == "online" ]]; then
    echo "  Simulator: $HOST:$PORT"
else
    echo "  Simulator connection: disabled (offline dataset training)"
fi
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

export PYTHONUNBUFFERED=1
uv run python -u train_dreamerv4_image.py \
    --offline \
    --host "$HOST" \
    --port "$PORT" \
    --record-dir "$RECORD_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --total-steps "$TOTAL_STEPS" \
    "${extra_args[@]}" \
    "$@"
