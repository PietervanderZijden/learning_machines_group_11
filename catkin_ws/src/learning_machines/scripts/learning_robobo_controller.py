#!/usr/bin/env python3
import os
import sys

from robobo_interface import HardwareRobobo, SimulationRobobo

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError(
            """To run, we need to know if we are running on hardware of simulation
            Pass `--hardware`, `--simulation`, or `--hardware-deploy` to specify."""
        )
    elif sys.argv[1] == "--hardware-deploy":
        workspace = os.environ.get("LEARNING_MACHINES_WORKSPACE", "/workspace")
        os.chdir(workspace)
        sys.path.insert(0, workspace)
        from deploy_hardware import main as deploy_main

        rob = HardwareRobobo(camera=True)
        deploy_main(rob=rob, argv=sys.argv[2:])
        raise SystemExit(0)
    elif sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")

    from learning_machines import main, test

    main()
    test()
