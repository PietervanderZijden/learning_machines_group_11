#!/usr/bin/env zsh

export DISPLAY=:1
# Pass the scene you want to load as first argument
# Specify the port you want as second argument. Defaults to 23000 (same as default for SimulationRobobo, and CoppeliaSim as a whole)
# If you want to run in headless mode, specify `-h` as a third argument
${1:?"Specify the scene you want to load as a first argument"}

# Presumes you have CoppeliaSim extracted to ./coppeliaSim.app
# coppeliasim "$1" $3 "-GzmqRemoteApi.rpcPort=${2:-23000}"
QT_QPA_PLATFORM=xcb coppeliasim "$1" $3 "-GzmqRemoteApi.rpcPort=${2:-23000}"
