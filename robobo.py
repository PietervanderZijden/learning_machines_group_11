#!/usr/bin/env python3
"""Run Robobo training, evaluation, validation, and deployment commands."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

TRAINERS = {
    ("food", "sac"): "train_sac.py",
    ("food", "dreamerv3"): "train_dreamerv3.py",
    ("food", "dreamerv4-full"): "train_dreamerv4_full.py",
    ("push", "sac"): "train_sac_push.py",
    ("push", "dreamerv3"): "train_dreamerv3_push.py",
}


def _run(script: str, arguments: list[str]) -> int:
    """Run a repository command with forwarded arguments."""
    return subprocess.call([sys.executable, str(ROOT / script), *arguments])


def _run_module(module: str, function: str) -> int:
    """Run a legacy learning function in the configured Python environment."""
    source = ROOT / "catkin_ws/src/learning_machines/src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(source), environment.get("PYTHONPATH")) if value
    )
    expression = f"from {module} import {function}; {function}()"
    return subprocess.call([sys.executable, "-c", expression], env=environment)


def _run_reactive() -> int:
    """Run the existing reactive controller in simulation."""
    source = ROOT / "catkin_ws/src/learning_machines/src"
    interface = ROOT / "catkin_ws/src/robobo_interface/src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (
            str(source),
            str(interface),
            environment.get("PYTHONPATH"),
        )
        if value
    )
    expression = (
        "from learning_machines.test_actions import run_all_actions;"
        "from robobo_interface import SimulationRobobo;"
        "run_all_actions(SimulationRobobo())"
    )
    return subprocess.call([sys.executable, "-c", expression], env=environment)


def build_parser() -> argparse.ArgumentParser:
    """Build the unified command-line parser."""
    parser = argparse.ArgumentParser(prog="robobo.py")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train")
    train.add_argument("task", choices=["food", "push", "approach", "evade"])
    train.add_argument(
        "algorithm",
        choices=["reactive", "ddpg", "sac", "dreamerv3", "dreamerv4-full"],
    )
    train.add_argument("arguments", nargs=argparse.REMAINDER)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("arguments", nargs=argparse.REMAINDER)

    validate = commands.add_parser("validate")
    validate.add_argument("target", choices=["simulation", "hardware"])
    validate.add_argument("arguments", nargs=argparse.REMAINDER)

    run = commands.add_parser("run")
    run.add_argument("task", choices=["approach", "evade"])
    run.add_argument("algorithm", choices=["reactive", "ddpg", "sac"])

    deploy = commands.add_parser("deploy")
    deploy.add_argument("task", choices=["food", "push"])
    deploy.add_argument(
        "algorithm", choices=["sac", "dreamerv3", "dreamerv4-full"]
    )
    deploy.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def main() -> int:
    """Dispatch a unified Robobo command."""
    args = build_parser().parse_args()
    if args.command == "train":
        script = TRAINERS.get((args.task, args.algorithm))
        if script:
            return _run(script, args.arguments)
        if args.algorithm == "ddpg":
            return _run_module("learning_machines.train_ddpg", "train_simple")
        if args.algorithm == "sac":
            return _run_module(
                "learning_machines.train_obstacle_avoidance", "main"
            )
        raise SystemExit(f"{args.algorithm} training is unavailable for {args.task}")
    if args.command == "evaluate":
        return _run("evaluate_transfer.py", args.arguments)
    if args.command == "validate":
        script = (
            "validate_simulation.py"
            if args.target == "simulation"
            else "validate_hardware.py"
        )
        return _run(script, args.arguments)
    if args.command == "run":
        if args.algorithm == "reactive":
            return _run_reactive()
        function = "test_simple" if args.algorithm == "ddpg" else "test"
        module = (
            "learning_machines.train_ddpg"
            if args.algorithm == "ddpg"
            else "learning_machines.test_obstacle_avoidance"
        )
        return _run_module(module, function)
    if args.task == "push":
        if args.algorithm != "dreamerv3":
            raise SystemExit("push hardware deployment supports dreamerv3")
        return _run("deploy_dreamerv3_push_hardware.py", args.arguments)
    return _run(
        "deploy_hardware.py", ["--algorithm", args.algorithm, *args.arguments]
    )


if __name__ == "__main__":
    raise SystemExit(main())
