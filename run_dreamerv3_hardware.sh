#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:?checkpoint required, for example dreamerv3_models/dreamerv3-reference/latest.pt}"
shift

HARDWARE_MANIFEST="config/calibration/hardware.json"
if [[ $# -gt 0 && "$1" != --* ]]; then
    HARDWARE_MANIFEST="$1"
    shift
fi

for path in "$CHECKPOINT" "$HARDWARE_MANIFEST"; do
    if [[ ! -f "$path" ]]; then
        echo "Required file not found: $path" >&2
        exit 1
    fi
done

if [[ ! -f dreamerv3_reference/dreamer.py ]]; then
    echo "Missing dreamerv3_reference checkout." >&2
    echo "Clone https://github.com/NM512/dreamerv3-torch into dreamerv3_reference." >&2
    exit 1
fi

docker_ros_env=()
for name in ROS_MASTER_URI ROS_IP ROS_HOSTNAME ROS_XMLRPC_PORT ROS_TCPROS_PORT; do
    if [[ -n "${!name:-}" ]]; then
        docker_ros_env+=(-e "$name")
    fi
done

env_file_args=()
if [[ -f .env ]]; then
    env_file_args=(--env-file .env)
fi

docker build --tag learning_machines_dreamerv3_hardware .
docker run -it --rm \
    --name robobo-dreamerv3-reference \
    --network host \
    "${docker_ros_env[@]}" \
    "${env_file_args[@]}" \
    -v "$(pwd)":/workspace \
    -e LEARNING_MACHINES_WORKSPACE=/workspace \
    learning_machines_dreamerv3_hardware \
    --dreamerv3-reference-hardware \
    --checkpoint "$CHECKPOINT" \
    --hardware-manifest "$HARDWARE_MANIFEST" \
    "$@"
