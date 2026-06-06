import time

import cv2
import wandb
from data_files import FIGURES_DIR
from robobo_interface import HardwareRobobo, IRobobo, SimulationRobobo


def test_sensors(rob: IRobobo):
    image = rob.read_image_front()
    cv2.imwrite(str(FIGURES_DIR / "photo.png"), image)
    print("Phone pan: ", rob.read_phone_pan())
    print("Phone tilt: ", rob.read_phone_tilt())
    print("Current acceleration: ", rob.read_accel())
    print("Current orientation: ", rob.read_orientation())


def test_move_and_return(rob: IRobobo, runNum, sim):
    timestep = 0
    while True:
        timestep += 1
        irs = rob.read_irs()
        if irs != [] and irs[4] <= 20:
            print(f"IRS data: {irs[4]}; moving forward!")
            wandb.log(
                {
                    "Is Simulation": sim,
                    "Run Num": runNum,
                    "Front Center IR Data": irs[4],
                    "Back Center IR Data": irs[6],
                    "Timestep": timestep,
                    "Turn Around": False,
                }
            )
            rob.move_blocking(20, 20, 200)

        elif irs != []:
            block_reading = irs[4]
            plant_reading = irs[6]
            print(
                f"Block detected ({block_reading:.2f}). Plant behind ({plant_reading:.2f}). Turning..."
            )

            front_cleared = False  # Phase 1: confirm we've rotated past the block

            while True:
                newIRS = rob.read_irs()
                print(
                    f"Turning... Front: {newIRS[4]:.2f}, Back: {newIRS[6]:.2f}, front_cleared: {front_cleared}"
                )
                wandb.log(
                    {
                        "Is Simulation": sim,
                        "Run Num": runNum,
                        "Front Center IR Data": newIRS[4],
                        "Back Center IR Data": newIRS[6],
                        "Timestep": timestep,
                        "Turn Around": True,
                    }
                )

                # Phase 1: wait until FrontC drops — robot has rotated past the block
                if not front_cleared:
                    if newIRS[4] < block_reading * 0.5:
                        front_cleared = True
                        print("Block no longer in front. Watching for it behind...")

                else:
                    if newIRS[6] > plant_reading * 1.5:
                        print(
                            f"Turned around! Back: {newIRS[6]:.2f} (threshold: {plant_reading * 1.5:.2f})"
                        )
                        break

                rob.move_blocking(20, -20, 100)
            rob.move_blocking(20, 20, 10000)
            break


def run_all_actions(rob: IRobobo):
    run = wandb.init(
        project="learning-machines",
        config={
            "simRuns": 5,
        },
        mode="online",
    )
    if isinstance(rob, SimulationRobobo):
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
            else:
                print("Resetting in 5...", end="")
                time.sleep(1)
                print("4...", end="")
                time.sleep(1)
                print("3...", end="")
                time.sleep(1)
                print("2...", end="")
                time.sleep(1)
                print("1...")
                time.sleep(1)
                print("Resetting!")

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
