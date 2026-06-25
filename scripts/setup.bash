#!/usr/bin/env bash
# replace localhost with the port you see on the smartphone
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"

# You want your local IP, usually starting with 192.168, following RFC1918
# Windows powershell:
#    (Get-NetIPAddress | Where-Object { $_.AddressState -eq "Preferred" -and $_.ValidLifetime -lt "24:00:00" }).IPAddress
# linux:
#    hostname -I | awk '{print $1}'
# macOS:
#    ipconfig getifaddr en1
export COPPELIA_SIM_IP="${COPPELIA_SIM_IP:-192.168.2.15}"

# HardwareRobobo uses fixed callback ports. Preserve explicit values supplied
# by the hardware launcher while providing stable defaults.
export ROS_XMLRPC_PORT="${ROS_XMLRPC_PORT:-45100}"
export ROS_TCPROS_PORT="${ROS_TCPROS_PORT:-45101}"
 
