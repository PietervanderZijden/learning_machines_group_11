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

docker_ros_env=()
for name in ROS_MASTER_URI ROS_IP ROS_HOSTNAME ROS_XMLRPC_PORT ROS_TCPROS_PORT; do
    if [[ -n "${!name:-}" ]]; then
        docker_ros_env+=(-e "$name")
    fi
done

docker build --tag learning_machines .
docker run -it --rm \
    --name robobo-deploy \
    --network host \
    "${docker_ros_env[@]}" \
    -v "$(pwd)":/workspace \
    --env-file .env \
    -e LEARNING_MACHINES_WORKSPACE=/workspace \
    learning_machines \
    --hardware-deploy \
    --algorithm "$ALGORITHM" \
    --checkpoint "$CHECKPOINT" \
    --manifest "$MANIFEST" \
    --calibration "$CALIBRATION" \
    "$@"
