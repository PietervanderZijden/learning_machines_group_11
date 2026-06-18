#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f .env ]]; then
    echo "Missing .env file" >&2
    exit 1
fi

docker build --tag learning_machines .
docker run -it --rm \
    --name robobo-hardware-validation \
    --network host \
    -v "$(pwd)":/workspace \
    --env-file .env \
    --entrypoint bash \
    learning_machines \
    -c '
        set -e
        source /opt/ros/noetic/setup.bash
        source /root/catkin_ws/devel/setup.bash
        source /root/catkin_ws/setup.bash
        export ROS_IP="${ROS_IP:-${COPPELIA_SIM_IP:-}}"
        export PYTHONPATH="/workspace/catkin_ws/src/learning_machines/src:/workspace/catkin_ws/src/robobo_interface/src:${PYTHONPATH:-}"
        cd /workspace
        exec python3 validate_hardware.py "$@"
    ' bash "$@"
