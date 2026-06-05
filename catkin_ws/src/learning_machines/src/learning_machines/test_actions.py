import time

import cv2
import wandb
from data_files import FIGURES_DIR
from robobo_interface import (
    Emotion,
    HardwareRobobo,
    IRobobo,
    LedColor,
    LedId,
    SimulationRobobo,
    SoundEmotion,
)
from wandb.sdk.lib import run_moment


def test_emotions(rob: IRobobo):
    rob.set_emotion(Emotion.HAPPY)
    rob.talk("Hello")
    rob.play_emotion_sound(SoundEmotion.PURR)
    rob.set_led(LedId.FRONTCENTER, LedColor.GREEN)


def test_move_and_wheel_reset(rob: IRobobo):
    rob.move_blocking(100, 100, 1000)
    print("before reset: ", rob.read_wheels())
    rob.reset_wheels()
    rob.sleep(1)
    print("after reset: ", rob.read_wheels())


def test_sensors(rob: IRobobo):
    print("IRS data: ", rob.read_irs())
    image = rob.read_image_front()
    cv2.imwrite(str(FIGURES_DIR / "photo.png"), image)
    print("Phone pan: ", rob.read_phone_pan())
    print("Phone tilt: ", rob.read_phone_tilt())
    print("Current acceleration: ", rob.read_accel())
    print("Current orientation: ", rob.read_orientation())


def test_phone_movement(rob: IRobobo):
    rob.set_phone_pan_blocking(20, 100)
    print("Phone pan after move to 20: ", rob.read_phone_pan())
    rob.set_phone_tilt_blocking(50, 100)
    print("Phone tilt after move to 50: ", rob.read_phone_tilt())


def test_sim(rob: SimulationRobobo):
    print("Current simulation time:", rob.get_sim_time())
    print("Is the simulation currently running? ", rob.is_running())
    rob.stop_simulation()
    print("Simulation time after stopping:", rob.get_sim_time())
    print("Is the simulation running after shutting down? ", rob.is_running())
    rob.play_simulation()
    print("Simulation time after starting again: ", rob.get_sim_time())
    print("Current robot position: ", rob.get_position())
    print("Current robot orientation: ", rob.get_orientation())

    pos = rob.get_position()
    orient = rob.get_orientation()
    rob.set_position(pos, orient)
    print("Position the same after setting to itself: ", pos == rob.get_position())
    print("Orient the same after setting to itself: ", orient == rob.get_orientation())


def test_hardware(rob: HardwareRobobo):
    print("Phone battery level: ", rob.phone_battery())
    print("Robot battery level: ", rob.robot_battery())


def test_move_and_return(rob: IRobobo, runNum, sim):
    irs = rob.read_irs()
    timestep = 0
    while True:
        timestep += 1
        if irs != [] and irs[4] <= 100:
            print(f"IRS data: {irs[4]}; moving forward!")
            wandb.log(
                {
                    "Is Simulation": sim,
                    "Run Num": runNum,
                    "Front Center IR Data": irs[4],
                    "Timestep": timestep,
                    "Turn Around": False,
                }
            )
            rob.move_blocking(10, 10, 50)
        else:
            print(f"IRS data: {irs[4]}; turning around!")
            rob.move_blocking(20, -20, 100)
            wandb.log(
                {
                    "Is Simulation": sim,
                    "Run Num": runNum,
                    "Front Center IR Data": irs[4],
                    "Timestep": timestep,
                    "Turn Around": True,
                }
            )
            print("Turned around!")
            break


def run_all_actions(rob: IRobobo):
    run = wandb.init(
        project="learning-machines",
        config={
            "simRuns": 5,
        },
    )
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    test_sensors(rob)
    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
        for runNum in range(run.config["simRuns"]):
            rob.play_simulation()
            test_move_and_return(rob, runNum, True)
            print(f"Run {runNum} completed!")
            time.sleep(5)
            rob.stop_simulation()

    if isinstance(rob, HardwareRobobo):
        for runNum in range(run.config["simRuns"]):
            test_move_and_return(rob, runNum, False)
            reset = input("Reset? (y/n)")
            if reset != "y":
                break

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
