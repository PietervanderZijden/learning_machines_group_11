"""Tests for unified command dispatch."""

from __future__ import annotations

import sys

import pytest

import robobo


def dispatch(monkeypatch, arguments: list[str]) -> tuple[str, list[str]]:
    """Run the launcher while capturing script dispatch."""
    captured: list[tuple[str, list[str]]] = []

    def fake_run(script: str, forwarded: list[str]) -> int:
        """Capture one script invocation."""
        captured.append((script, forwarded))
        return 0

    monkeypatch.setattr(robobo, "_run", fake_run)
    monkeypatch.setattr(sys, "argv", ["robobo.py", *arguments])
    assert robobo.main() == 0
    return captured[0]


def test_record_dispatch(monkeypatch) -> None:
    """Dispatch food recording."""
    assert dispatch(
        monkeypatch, ["record", "food", "--steps", "4"]
    ) == ("record_food.py", ["--steps", "4"])


@pytest.mark.parametrize(
    ("task", "algorithm", "script"),
    [
        ("food", "sac", "train_sac.py"),
        ("food", "dreamerv3", "train_dreamerv3.py"),
        ("food", "dreamerv4-full", "train_dreamerv4_full.py"),
        ("push", "sac", "train_sac_push.py"),
        ("push", "dreamerv3", "train_dreamerv3_push.py"),
    ],
)
def test_training_matrix(monkeypatch, task, algorithm, script) -> None:
    """Dispatch every supported learned trainer."""
    assert dispatch(
        monkeypatch, ["train", task, algorithm, "--no-wandb"]
    ) == (script, ["--no-wandb"])


def test_push_evaluation_dispatch(monkeypatch) -> None:
    """Pass task and algorithm to push evaluation."""
    script, arguments = dispatch(
        monkeypatch,
        ["evaluate", "push", "dreamerv3", "--checkpoint", "model.pt"],
    )
    assert script == "evaluate_transfer.py"
    assert arguments[:4] == [
        "--task",
        "push",
        "--algorithm",
        "dreamerv3",
    ]


def test_push_hardware_uses_local_deployer(monkeypatch) -> None:
    """Deploy local DreamerV3 push checkpoints."""
    script, arguments = dispatch(
        monkeypatch,
        ["deploy", "push", "dreamerv3", "--checkpoint", "model.pt"],
    )
    assert script == "deploy_hardware.py"
    assert arguments[:4] == [
        "--task",
        "push",
        "--algorithm",
        "dreamerv3",
    ]


def test_unsupported_combination_fails(monkeypatch) -> None:
    """Reject task and algorithm combinations without an implementation."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["robobo.py", "train", "push", "dreamerv4-full"],
    )
    with pytest.raises(SystemExit, match="unavailable"):
        robobo.main()
