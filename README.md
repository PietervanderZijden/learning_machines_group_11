# learning_machines_group_11
# Learning Machines - Robobo Project (Group 11)

This is the collaborative repository for our project in the university course Learning Machines. Our goal is to develop an AI-driven controller for the Robobo robot, enabling it to navigate and complete tasks autonomously.

The codebase is designed to seamlessly interface with both the virtual robot in the simulator and the physical hardware on campus.

The current calibrated training/deployment workflow is documented in
[TRANSFER.md](TRANSFER.md).

### Tech Stack & Environment
* **Language:** Python 3.8 (managed via `uv`)
* **Robot Framework:** ROS1 Noetic (fully encapsulated in Docker via OrbStack / Docker Desktop)
* **Simulator:** CoppeliaSim Edu 

### Main workflows

```bash
./run_sac.sh
./run_dreamerv3.sh
./run_dreamerv4.sh
python evaluate_transfer.py --help
python validate_simulation.py --help
```

DreamerV4 defaults to streamed offline training from
`recorded-states/dreamer-v3-states`. Use `--resume` for phase-aware continuation.
Set `HARDWARE_RECORD_DIR` and provide a measured `--hardware-calibration` when
hardware episodes are available.

Training metrics, environment diagnostics, camera/reconstruction samples,
action distributions, KL statistics, and evaluation summaries are logged to
Weights & Biases unless `--no-wandb` is passed.
