# learning_machines_group_11
# Learning Machines - Robobo Project (Group 11)

This is the collaborative repository for our project in the university course Learning Machines. Our goal is to develop an AI-driven controller for the Robobo robot, enabling it to navigate and complete tasks autonomously.

The codebase is designed to seamlessly interface with both the virtual robot in the simulator and the physical hardware on campus.

### Tech Stack & Environment
* **Language:** Python 3.8 (managed via `uv`)
* **Robot Framework:** ROS1 Noetic (fully encapsulated in Docker via OrbStack / Docker Desktop)
* **Simulator:** CoppeliaSim Edu 

### Key Files
* `learning_robobo_controller.py`: The main switchboard that initializes the robot/simulator connection.
* `test_actions.py`: The original example and reference file provided by the university.