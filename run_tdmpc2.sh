#!/usr/bin/env bash
# Launch TD-MPC2 training
set -euo pipefail
cd "$(dirname "$0")"
uv run python tdmpc2/tdmpc2/train.py --config-name=task/robobo-food-collection "$@"
