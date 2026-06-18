#!/usr/bin/env bash
# ROS_MASTER_URI points to localhost because SSH tunnels forward traffic
# through the iPad (100.64.0.7) to the robot (10.15.2.56).
# ROS_IP is the iPad's local IP — the robot connects back to this address,
# and the reverse SSH tunnel routes it to this host.
export ROS_MASTER_URI="http://localhost:11311"
export ROS_IP="10.15.2.253"
# You want your local IP, usually starting with 192.168, following RFC1918
# Windows powershell:
#    (Get-NetIPAddress | Where-Object { $_.AddressState -eq "Preferred" -and $_.ValidLifetime -lt "24:00:00" }).IPAddress
# linux:
#    hostname -I | awk '{print $1}'
# macOS:
#    ipconfig getifaddr en0
export COPPELIA_SIM_IP="10.15.2.224"
 
