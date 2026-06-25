# learning_machines_group_11
# Learning Machines - Robobo Project (Group 11)

This is the collaborative repository for our project in the university course Learning Machines. Our goal is to develop an AI-driven controller for the Robobo robot, enabling it to navigate and complete tasks autonomously.

The codebase is designed to seamlessly interface with both the virtual robot in the simulator and the physical hardware on campus.

### DreamerV3 reference policy on hardware

The NM512 reference checkpoint runs on 96x96 RGB camera observations and the
measured IR calibration from `config/calibration/hardware.json`:

```bash
./run_dreamerv3_hardware.sh \
  dreamerv3_models/dreamerv3-reference/latest.pt \
  config/calibration/hardware.json \
  --raised-wheel-test \
  --max-seconds 10
```

Set `ROS_MASTER_URI` and `ROS_IP` (or `ROS_HOSTNAME`) for the Robobo network
before launching. During rollout, enter `e` or `q` and press Enter for an
emergency stop. A missing wheel service or completion reply is retried with
the same command after a five-second timeout.

To showcase the same checkpoint in CoppeliaSim, open
`scenes/arena_push_easy.ttt` on port `23000`, then run:

```bash
./run_dreamerv3_showcase.sh \
  dreamerv3_models/dreamerv3-reference/latest.pt \
  --episodes 3
```

This runs deterministic fixed-layout evaluation at real-time speed without
training or domain randomization. Pass `--no-realtime` for fast evaluation.

### Tech Stack & Environment
* **Language:** Python 3.8 (managed via `uv`)
* **Robot Framework:** ROS1 Noetic (fully encapsulated in Docker via OrbStack / Docker Desktop)
* **Simulator:** CoppeliaSim Edu 

### Key Files
* `learning_robobo_controller.py`: The main switchboard that initializes the robot/simulator connection.
* `test_actions.py`: The original example and reference file provided by the university.
