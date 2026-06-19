#!/bin/bash
set -euo pipefail

# Usage:
#   ./run_hardware_deploy.sh sac sac_models/sac_latest.zip config/calibration/hardware.json [manifest] [options]
#   ./run_hardware_deploy.sh dreamerv3 dreamerv3_models/dreamerv3_latest.pt config/calibration/hardware.json [manifest] [options]
# Options after the optional manifest are forwarded to deploy_hardware.py.

ALGORITHM="${1:?algorithm required: sac, dreamerv3, or dreamerv4}"
CHECKPOINT="${2:?checkpoint path required}"
CALIBRATION="${3:?hardware calibration JSON required}"
shift 3
MANIFEST="$(dirname "$CHECKPOINT")/manifest.json"
if [[ $# -gt 0 && "$1" != --* ]]; then
    MANIFEST="$1"
    shift
fi

for path in "$CHECKPOINT" "$CALIBRATION" "$MANIFEST"; do
    if [ ! -f "$path" ]; then
        echo "Required file not found: $path" >&2
        exit 1
    fi
done

docker build --tag learning_machines .
docker run -it --rm \
    --name robobo-deploy \
    --network host \
    -v "$(pwd)":/workspace \
    --env-file .env \
    --entrypoint bash \
    learning_machines \
    -c '
        source /opt/ros/noetic/setup.bash
        source /root/catkin_ws/devel/setup.bash
        source /root/catkin_ws/setup.bash
        export PYTHONPATH="/workspace/catkin_ws/src/learning_machines/src:/workspace/catkin_ws/src/robobo_interface/src:${PYTHONPATH:-}"
        cd /workspace
        python3 -c "import torch, gymnasium, stable_baselines3; print(\"Deployment dependencies:\", torch.__version__, gymnasium.__version__, stable_baselines3.__version__)"
        exec python3 deploy_hardware.py \
            --algorithm "$1" \
            --checkpoint "$2" \
            --manifest "$3" \
            --calibration "$4" \
            "${@:5}"
    ' bash "$ALGORITHM" "$CHECKPOINT" "$MANIFEST" "$CALIBRATION" "$@"
