#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f .env ]]; then
    echo "Missing .env file" >&2
    exit 1
fi

docker_network_args=()
case "$(uname -s)" in
    Darwin)
        # Docker Desktop runs containers in a Linux VM. Publishing the fixed
        # ROS callback ports lets the robot reach the node through the Mac's
        # advertised ROS_IP.
        docker_network_args=(-p 45100:45100 -p 45101:45101)
        ;;
    Linux)
        docker_network_args=(--network host)
        ;;
    *)
        echo "Unsupported host OS. Use Linux host networking or publish ports 45100/45101." >&2
        exit 1
        ;;
esac

docker build --tag learning_machines .
docker run -it --rm \
    --name robobo-hardware-validation \
    "${docker_network_args[@]}" \
    -v "$(pwd)":/workspace \
    --env-file .env \
    --entrypoint bash \
    learning_machines \
    -c '
        set -e
        source /opt/ros/noetic/setup.bash
        source /root/catkin_ws/devel/setup.bash
        source /root/catkin_ws/setup.bash
        export PYTHONPATH="/workspace/catkin_ws/src/learning_machines/src:/workspace/catkin_ws/src/robobo_interface/src:${PYTHONPATH:-}"
        cd /workspace
        exec python3 validate_hardware.py "$@"
    ' bash "$@"
