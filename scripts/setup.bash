#!/usr/bin/env bash

# Physical Robobo ROS master.
export ROS_MASTER_URI="http://100.64.0.9:11311"

# This computer's address on the same network as the Robobo. ROS advertises
# this address so the robot can open topic connections back to this process.
export ROS_IP="100.64.0.11"

# Fixed callback ports used by HardwareRobobo. Docker publishes these ports on
# macOS; Linux uses host networking.
export ROS_XMLRPC_PORT="45100"
export ROS_TCPROS_PORT="45101"

# Simulation only. This does not affect physical-hardware validation.
export COPPELIA_SIM_IP="100.64.0.11"
 
