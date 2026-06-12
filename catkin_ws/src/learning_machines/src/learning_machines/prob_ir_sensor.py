import csv

import wandb
from data_files import RESULTS_DIR
from robobo_interface import HardwareRobobo, IRobobo, SimulationRobobo

globalData: "list[dict]" = []
fieldNames = [
    "Back Left",
    "Back Right",
    "Front Left",
    "Front Right",
    "Front Center",
    "Front RR",
    "Back Center",
    "Front LL",
    "Movement Type",
]


def log_ir_data(ir_data, movement):
    data = {
        "Back Left": ir_data[0],
        "Back Right": ir_data[1],
        "Front Left": ir_data[2],
        "Front Right": ir_data[3],
        "Front Center": ir_data[4],
        "Front RR": ir_data[5],
        "Back Center": ir_data[6],
        "Front LL": ir_data[7],
        "Movement Type": movement,
    }

    globalData.append(data)
    wandb.log(data)


def move_forward(rob, left_speed, right_speed, duration, movement, half=False):
    for _ in range(123):
        ir_data = rob.read_irs()
        log_ir_data(ir_data, movement)
        rob.move_blocking(left_speed, right_speed, duration)


def save_csv():
    with open(RESULTS_DIR / "results.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldNames)
        writer.writeheader()
        writer.writerows(globalData)


def move(rob: IRobobo):
    print("Moving forwards")
    move_forward(rob, 20, 20, 200, "Moving forwards")

    print("Moving backwards")
    move_forward(rob, -20, -20, 200, "Moving backwards")

    print("Moving forwards (half speed)")
    move_forward(rob, 10, 10, 200, "Moving forwards (half speed)", True)

    print("Moving backwards (half speed)")
    move_forward(rob, -10, -10, 200, "Moving backwards (half speed)", True)

    print("Turning Right")
    move_forward(rob, 20, -20, 200, "Turning One way")

    print("Turning back")
    move_forward(rob, -20, 20, 200, "Turning other way")

    save_csv()


def change_camera(rob):
    # rob.set_phone_tilt_blocking(50, 20)
    print("Set tilt 50")
    # rob.set_phone_tilt_blocking(20, 20)
    print("Set tilt 20")
    rob.set_phone_tilt_blocking(109, 20)
    print("Set tilt 100")
    input()


def run_all_actions(rob: IRobobo):
    run = wandb.init(
        project="learning-machines",
        mode="offline",
    )

    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
        change_camera(rob)
        print("Done")
        # move(rob)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
