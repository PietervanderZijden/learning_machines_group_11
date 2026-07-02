import os

from evaluate_transfer import _load_repository_env


def test_repository_env_loads_missing_values_without_overriding_shell(
    tmp_path, monkeypatch
):
    path = tmp_path / ".env"
    path.write_text(
        "WANDB_API_KEY='file-key'\n"
        "WANDB_ENTITY=robot-team # comment\n"
        "export WANDB_MODE=offline\n"
    )
    monkeypatch.setenv("WANDB_API_KEY", "shell-key")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_MODE", raising=False)

    loaded = _load_repository_env(path)

    assert os.environ["WANDB_API_KEY"] == "shell-key"
    assert os.environ["WANDB_ENTITY"] == "robot-team"
    assert os.environ["WANDB_MODE"] == "offline"
    assert loaded == ["WANDB_ENTITY", "WANDB_MODE"]
