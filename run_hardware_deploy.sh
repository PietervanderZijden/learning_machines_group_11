#!/bin/bash
set -euo pipefail

# Usage:
#   ./run_hardware_deploy.sh sac sac_models/sac_latest.zip config/calibration/hardware.json
#   ./run_hardware_deploy.sh dreamerv3 dreamerv3_models/dreamerv3_latest.pt config/calibration/hardware.json
#   ./run_hardware_deploy.sh dreamerv4 dreamerv4_image_models/dreamerv4_image_latest.pt config/calibration/hardware.json

ALGORITHM="${1:?algorithm required: sac, dreamerv3, or dreamerv4}"
CHECKPOINT="${2:?checkpoint path required}"
CALIBRATION="${3:?hardware calibration JSON required}"
MANIFEST="${4:-$(dirname "$CHECKPOINT")/manifest.json}"

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
    -v "$(pwd)":/root/catkin_ws \
    --env-file .env \
    --entrypoint bash \
    learning_machines \
    -c "
        source /opt/ros/noetic/setup.bash
        source /root/catkin_ws/devel/setup.bash 2>/dev/null || true
        cd /root/catkin_ws
        python3 deploy_hardware.py \
            --algorithm '$ALGORITHM' \
            --checkpoint '$CHECKPOINT' \
            --manifest '$MANIFEST' \
            --calibration '$CALIBRATION'
    "
