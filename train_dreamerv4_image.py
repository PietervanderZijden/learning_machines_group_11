"""Training script for image-based DreamerV4.

Collects image episodes from the environment, trains the tokenizer,
then trains the dynamics model and actor-critic via imagination.

Usage:
    python train_dreamerv4_image.py
    python train_dreamerv4_image.py --total-steps 100000
    python train_dreamerv4_image.py --resume
"""
from __future__ import annotations
import argparse
import atexit
import datetime
import json
import sys
import time
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
import tqdm


PHASE_BAR_FORMAT = "{desc:<10} {percentage:3.0f}%|{bar:28}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
PHASES = ("tokenizer", "dynamics", "mtp", "pmpo")


@dataclass(frozen=True)
class EpisodeRef:
    path: Path
    source: str
    calibration_profile: str
    transitions: int
    episode_return: float
    has_ir: bool


@dataclass
class DatasetScan:
    episodes: list[EpisodeRef] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def transitions(self) -> int:
        return sum(episode.transitions for episode in self.episodes)


@dataclass
class TrainingProgress:
    tokenizer: int = 0
    dynamics: int = 0
    mtp: int = 0
    pmpo: int = 0
    global_log_step: int = 0

    @classmethod
    def load(cls, path: Path) -> "TrainingProgress":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text())
        return cls(**{
            key: int(data.get(key, 0))
            for key in (*PHASES, "global_log_step")
        })

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2) + "\n")
        temporary.replace(path)


class _NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def _numeric_metrics(metrics: dict[str, object]) -> dict[str, float]:
    numeric = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.detach().item()
        elif isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (int, float, np.number)):
            numeric[key] = float(value)
    return numeric


def _format_metrics(metrics: dict[str, float]) -> str:
    numeric = _numeric_metrics(metrics)
    return " ".join(f"{key}={value:.4f}" for key, value in sorted(numeric.items()))


def _metric_postfix(metrics: dict[str, object], preferred: tuple[str, ...]) -> str:
    numeric = _numeric_metrics(metrics)
    postfix = []
    keys = [key for key in preferred if key in numeric]
    if not keys:
        keys = list(sorted(numeric))[:3]
    for key in keys[:3]:
        label = key.split("/")[-1]
        postfix.append(f"{label}={numeric[key]:.4f}")
    return " ".join(postfix)


def _write_metrics(phase: str, step: int, metrics: dict[str, object]) -> None:
    numeric = _numeric_metrics(metrics)
    if not numeric:
        return
    rows = sorted(numeric.items())
    lines = ["", f"{phase} metrics @ step {step:,}", "-" * 52]
    name_width = 30
    value_width = 12
    for name, value in rows:
        lines.append(f"|    {name:<{name_width}} | {value:>{value_width}.4f} |")
    lines.append("-" * 52)
    tqdm.tqdm.write("\n".join(lines))


def _progress(iterable, desc: str):
    return tqdm.tqdm(
        iterable,
        desc=desc,
        dynamic_ncols=True,
        smoothing=0.05,
        bar_format=PHASE_BAR_FORMAT,
    )


def _log_metrics(writer, wandb_run, metrics: dict[str, float], step: int, phase: str) -> None:
    numeric = _numeric_metrics(metrics)
    for key, value in numeric.items():
        writer.add_scalar(key, value, step)
    if wandb_run is not None:
        wandb_run.log({**numeric, "phase": phase, "global_step": step})


def _episode_metadata(data) -> dict:
    if "manifest" in data:
        import json
        raw = data["manifest"].item()
        return json.loads(str(raw))
    return {
        key: data[key].item()
        for key in (
            "observation_contract",
            "reward_contract",
            "control_interval_seconds",
            "calibration_profile",
            "phone_tilt",
        )
        if key in data
    }


def _episode_from_npz(
    path: Path,
    expected_calibration: str | None,
    expected_phone_tilt: int,
    source: str = "simulation",
) -> dict | None:
    with np.load(path, allow_pickle=False) as data:
        metadata = _episode_metadata(data)
        if metadata.get("observation_contract") != "robobo-obs-v2":
            raise ValueError(f"{path} is missing the robobo-obs-v2 data contract")
        if metadata.get("reward_contract") != "robobo-reward-v4":
            raise ValueError(f"{path} is missing the robobo-reward-v4 data contract")
        if not np.isclose(metadata.get("control_interval_seconds", -1.0), 0.4):
            raise ValueError(f"{path} was not recorded with the 400 ms control contract")
        calibration = str(metadata.get("calibration_profile", ""))
        if expected_calibration is not None and calibration != expected_calibration:
            raise ValueError(
                f"{path} calibration profile {calibration!r} "
                f"does not match {expected_calibration!r}"
            )
        if abs(int(metadata.get("phone_tilt", -999)) - expected_phone_tilt) > 8:
            raise ValueError(f"{path} was recorded with an incompatible phone tilt")
        image_key = "images" if "images" in data else (
            "observations" if "observations" in data else None
        )
        if image_key is None:
            return None

        images = data[image_key]
        actions = data["actions"].astype(np.float32)
        rewards = data["rewards"].astype(np.float32)
        dones = (
            data["dones"].astype(bool)
            if "dones" in data
            else np.zeros(len(rewards), dtype=bool)
        )
        irs = data["irs"].astype(np.float32) if "irs" in data else None
        if irs is None and "observations" in data:
            vectors = data["observations"]
            if vectors.ndim == 2 and vectors.shape[1] >= 12:
                irs = vectors[:, 4:12].astype(np.float32)
        max_transitions = min(
            len(actions), len(rewards), len(dones), len(images) - 1
        )
        if max_transitions <= 0:
            return None
        episode = {
            "images": images[: max_transitions + 1].copy(),
            "actions": actions[:max_transitions],
            "rewards": rewards[:max_transitions],
            "dones": dones[:max_transitions],
            "source": source,
            "calibration_profile": calibration,
            "path": str(path),
        }
        if irs is not None and len(irs) >= max_transitions + 1:
            episode["irs"] = irs[: max_transitions + 1]
        elif irs is not None and len(irs) >= max_transitions:
            aligned = irs[:max_transitions]
            episode["irs"] = np.concatenate([aligned, aligned[-1:]], axis=0)
        return episode


def scan_recorded_episodes(
    record_dir: str,
    expected_calibration: str | None,
    expected_phone_tilt: int = 100,
    source: str = "simulation",
) -> DatasetScan:
    episode_dir = Path(record_dir) / "episodes"
    result = DatasetScan()
    if not episode_dir.exists():
        return result
    for path in sorted(episode_dir.glob("*.npz")):
        try:
            episode = _episode_from_npz(
                path,
                expected_calibration=expected_calibration,
                expected_phone_tilt=expected_phone_tilt,
                source=source,
            )
            if episode is None:
                result.skipped.append((str(path), "no aligned transitions"))
                continue
            result.episodes.append(EpisodeRef(
                path=path,
                source=source,
                calibration_profile=episode["calibration_profile"],
                transitions=len(episode["actions"]),
                episode_return=float(np.sum(episode["rewards"])),
                has_ir="irs" in episode,
            ))
        except (OSError, KeyError, ValueError) as exc:
            result.skipped.append((str(path), str(exc)))
    return result


def load_recorded_episodes(
    record_dir: str,
    expected_calibration: str | None,
    expected_phone_tilt: int = 100,
    source: str = "simulation",
    strict: bool = False,
) -> list[dict]:
    scan = scan_recorded_episodes(
        record_dir,
        expected_calibration,
        expected_phone_tilt,
        source,
    )
    if strict and scan.skipped:
        raise ValueError(scan.skipped[0][1])
    episodes = []
    for ref in scan.episodes:
        episode = _episode_from_npz(
            ref.path,
            expected_calibration=expected_calibration,
            expected_phone_tilt=expected_phone_tilt,
            source=source,
        )
        if episode is not None:
            episodes.append(episode)
    return episodes


class StreamingEpisodeDataset:
    """Disk-backed episode index with source-aware sampling."""

    def __init__(
        self,
        simulation: Iterable[EpisodeRef],
        hardware_train: Iterable[EpisodeRef] = (),
        hardware_validation: Iterable[EpisodeRef] = (),
        hardware_sample_ratio: float = 0.25,
        expected_phone_tilt: int = 100,
    ):
        self.simulation = list(simulation)
        self.hardware_train = list(hardware_train)
        self.hardware_validation = list(hardware_validation)
        self.hardware_sample_ratio = float(hardware_sample_ratio)
        self.expected_phone_tilt = expected_phone_tilt
        if not 0.0 <= self.hardware_sample_ratio <= 1.0:
            raise ValueError("hardware_sample_ratio must be in [0, 1]")
        if not self.simulation and not self.hardware_train:
            raise ValueError("dataset contains no training episodes")

    @property
    def training_refs(self) -> list[EpisodeRef]:
        return self.simulation + self.hardware_train

    def sample_ref(self, rng: random.Random = random) -> EpisodeRef:
        use_hardware = (
            bool(self.hardware_train)
            and (
                not self.simulation
                or rng.random() < self.hardware_sample_ratio
            )
        )
        pool = self.hardware_train if use_hardware else self.simulation
        return rng.choice(pool)

    def load(self, ref: EpisodeRef) -> dict:
        episode = _episode_from_npz(
            ref.path,
            expected_calibration=ref.calibration_profile,
            expected_phone_tilt=self.expected_phone_tilt,
            source=ref.source,
        )
        if episode is None:
            raise ValueError(f"{ref.path} no longer contains a usable episode")
        return episode

    def sample_observation_batch(
        self,
        batch_size: int,
        use_ir: bool,
        device: torch.device,
        rng: random.Random = random,
    ) -> tuple[torch.Tensor, torch.Tensor | None, float]:
        selections: list[tuple[EpisodeRef, int]] = []
        hardware = 0
        for _ in range(batch_size):
            ref = self.sample_ref(rng)
            index = rng.randrange(ref.transitions + 1)
            selections.append((ref, index))
            hardware += int(ref.source == "hardware")
        loaded: dict[Path, dict] = {}
        images, irs = [], []
        for ref, index in selections:
            if ref.path not in loaded:
                loaded[ref.path] = self.load(ref)
            episode = loaded[ref.path]
            images.append(episode["images"][index])
            if use_ir:
                if "irs" not in episode:
                    raise ValueError(f"{ref.path} does not contain IR observations")
                irs.append(episode["irs"][index])
        image_tensor = torch.from_numpy(np.stack(images)).float().to(device) / 255.0
        ir_tensor = (
            torch.from_numpy(np.stack(irs)).float().to(device)
            if use_ir else None
        )
        return image_tensor, ir_tensor, hardware / max(1, batch_size)


def collect_episodes(
    env,
    agent,
    device: torch.device,
    steps: int,
    image_size: int,
    use_ir: bool,
    writer=None,
    wandb_run=None,
) -> list[dict]:
    episodes = []
    obs_dict, info = env.reset()
    current = {
        "images": [obs_dict["image"].copy()],
        "actions": [],
        "rewards": [],
        "dones": [],
    }
    if use_ir:
        current["irs"] = [obs_dict["ir"].astype(np.float32).copy()]
    episode_return = 0.0
    episode_collisions = 0
    episode_safety = 0
    episode_action_change = 0.0
    episode_saturation = 0.0
    episode_index = 0

    for _ in _progress(range(steps), "Collecting"):
        image = obs_dict["image"]
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).to(device) / 255.0
        ir_tensor = None
        if use_ir:
            ir_tensor = torch.from_numpy(obs_dict["ir"].astype(np.float32)).unsqueeze(0).to(device)

        with torch.no_grad():
            latent = agent.tokenizer.encode(image_tensor, ir_tensor)
            action = agent.act(latent, deterministic=False)
        action_np = action.squeeze(0).cpu().numpy()

        obs_dict, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

        executed_action = np.asarray(info.get("executed_action", action_np), dtype=np.float32)
        current["actions"].append(executed_action.copy())
        current["rewards"].append(float(reward))
        current["dones"].append(bool(done))
        current["images"].append(obs_dict["image"].copy())
        if use_ir:
            current["irs"].append(obs_dict["ir"].astype(np.float32).copy())
        episode_return += float(reward)
        episode_collisions += int(bool(info.get("collision", False)))
        episode_safety += int(info.get("safety_override") is not None)
        episode_action_change += float(info.get("action_change", 0.0))
        episode_saturation += float(info.get("action_saturation", 0.0))

        if done:
            episodes.append({k: np.array(v) for k, v in current.items()})
            length = len(current["actions"])
            metrics = {
                "collection/episode_return": episode_return,
                "collection/episode_length": length,
                "collection/elapsed_seconds": float(info.get("elapsed_seconds", length * 0.4)),
                "collection/food_collected": float(info.get("food_collected", 0)),
                "collection/food_per_minute": float(info.get("food_per_minute", 0.0)),
                "collection/collisions": episode_collisions,
                "collection/safety_overrides": episode_safety,
                "collection/mean_action_change": episode_action_change / max(1, length),
                "collection/action_saturation_rate": episode_saturation / max(1, length),
            }
            if writer is not None:
                for key, value in metrics.items():
                    writer.add_scalar(key, value, episode_index)
            if wandb_run is not None:
                import wandb
                metrics["diagnostics/camera_frame"] = wandb.Image(
                    np.transpose(obs_dict["image"], (1, 2, 0))
                )
                metrics["diagnostics/ir_histogram"] = wandb.Histogram(obs_dict["ir"])
                if "raw_ir" in info:
                    metrics["diagnostics/raw_ir_histogram"] = wandb.Histogram(
                        info["raw_ir"]
                    )
                metrics["global_step"] = episode_index
                wandb_run.log(metrics)
            episode_index += 1
            obs_dict, info = env.reset()
            current = {
                "images": [obs_dict["image"].copy()],
                "actions": [],
                "rewards": [],
                "dones": [],
            }
            if use_ir:
                current["irs"] = [obs_dict["ir"].astype(np.float32).copy()]
            episode_return = 0.0
            episode_collisions = 0
            episode_safety = 0
            episode_action_change = 0.0
            episode_saturation = 0.0

    if len(current["actions"]) > 0:
        episodes.append({k: np.array(v) for k, v in current.items()})
    return episodes


def encode_episode_latents(
    agent,
    dataset: StreamingEpisodeDataset,
    refs: list[EpisodeRef],
    device: torch.device,
    batch_size: int,
    use_ir: bool,
):
    """Stream episodes through the tokenizer and retain compact latents on CPU."""
    encoded = []
    with torch.no_grad():
        for ref in _progress(refs, "Encoding"):
            ep = dataset.load(ref)
            images = torch.from_numpy(ep["images"]).float() / 255.0
            irs = (
                torch.from_numpy(ep["irs"]).float()
                if use_ir and "irs" in ep else None
            )
            latents = []
            for i in range(0, len(images), batch_size):
                image_batch = images[i:i + batch_size].to(device)
                ir_batch = (
                    irs[i:i + batch_size].to(device)
                    if irs is not None else None
                )
                latents.append(
                    agent.tokenizer.encode(image_batch, ir_batch).cpu().half()
                )
            encoded.append({
                "latents": torch.cat(latents, dim=0),
                "actions": torch.from_numpy(ep["actions"]).float(),
                "rewards": torch.from_numpy(ep["rewards"]).float(),
                "dones": torch.from_numpy(ep["dones"].astype(np.float32)).float(),
                "source": ref.source,
            })
    return encoded


def sample_latent_batch(
    encoded_episodes: list[dict],
    batch_size: int,
    context_length: int,
    device: torch.device,
    hardware_sample_ratio: float = 0.25,
):
    valid = [ep for ep in encoded_episodes if len(ep["actions"]) >= 1]
    if not valid:
        raise ValueError("No episodes with usable transitions")
    simulation = [ep for ep in valid if ep.get("source") != "hardware"]
    hardware = [ep for ep in valid if ep.get("source") == "hardware"]

    latents, actions, rewards, dones, masks = [], [], [], [], []
    hardware_samples = 0
    for _ in range(batch_size):
        use_hardware = bool(hardware) and (
            not simulation or random.random() < hardware_sample_ratio
        )
        ep = random.choice(hardware if use_hardware else simulation)
        hardware_samples += int(use_hardware)
        n = len(ep["actions"])
        length = min(context_length, n)
        start = 0 if n == length else random.randint(0, n - length)
        end = start + length

        lat = ep["latents"][start:end + 1]
        act = ep["actions"][start:end]
        rew = ep["rewards"][start:end]
        done = ep["dones"][start:end]
        mask = torch.ones(length)

        if length < context_length:
            pad = context_length - length
            lat = torch.cat([lat, torch.zeros(pad, lat.shape[-1])], dim=0)
            act = torch.cat([act, torch.zeros(pad, act.shape[-1])], dim=0)
            rew = torch.cat([rew, torch.zeros(pad)], dim=0)
            done = torch.cat([done, torch.ones(pad)], dim=0)
            mask = torch.cat([mask, torch.zeros(pad)], dim=0)

        latents.append(lat)
        actions.append(act)
        rewards.append(rew)
        dones.append(done)
        masks.append(mask)

    return (
        torch.stack(latents).float().to(device),
        torch.stack(actions).to(device),
        torch.stack(rewards).to(device),
        torch.stack(dones).to(device),
        torch.stack(masks).to(device),
        hardware_samples / max(1, batch_size),
    )


def evaluate_tokenizer(
    agent,
    dataset: StreamingEpisodeDataset,
    refs: list[EpisodeRef],
    device: torch.device,
    use_ir: bool,
    max_episodes: int = 8,
) -> dict[str, float]:
    if not refs:
        return {}
    metrics = []
    with torch.no_grad():
        for ref in refs[:max_episodes]:
            episode = dataset.load(ref)
            indices = np.linspace(
                0, len(episode["images"]) - 1, num=min(8, len(episode["images"])),
                dtype=int,
            )
            images = torch.from_numpy(episode["images"][indices]).float().to(device) / 255.0
            ir = (
                torch.from_numpy(episode["irs"][indices]).float().to(device)
                if use_ir else None
            )
            latent = agent.tokenizer.encode(images, ir)
            reconstruction = agent.tokenizer.decode(latent)
            metrics.append(F.mse_loss(reconstruction, images).item())
    return {"validation/hardware_tokenizer_mse": float(np.mean(metrics))}


def evaluate_dynamics(
    agent,
    encoded_validation: list[dict],
    device: torch.device,
    context_length: int,
    batches: int = 4,
) -> dict[str, float]:
    from learning_machines.distributional import logits_to_value

    if not encoded_validation:
        return {}
    aggregate: dict[str, list[float]] = {}
    with torch.no_grad():
        for _ in range(batches):
            lat, act, rew, done, mask, _ = sample_latent_batch(
                encoded_validation,
                min(8, len(encoded_validation)),
                context_length,
                device,
                hardware_sample_ratio=1.0,
            )
            batch, transitions = act.shape[:2]
            tau = torch.full(
                (batch, transitions), 0.5, device=device
            )
            d = torch.full(
                (batch, transitions), 0.25, device=device
            )
            predictions = agent.dynamics(lat, act, tau, d, mask)
            latent_per_step = F.mse_loss(
                predictions["latent_pred"], lat[:, 1:], reduction="none"
            ).mean(-1)
            latent_loss = (latent_per_step * mask).sum() / mask.sum().clamp(min=1)
            reward_prediction = logits_to_value(
                predictions["reward_logits"]
            )
            reward_loss = (
                (reward_prediction - rew).square() * mask
            ).sum() / mask.sum().clamp(min=1)
            done_prediction = torch.sigmoid(
                predictions["done_pred"].squeeze(-1)
            )
            done_loss = (
                (done_prediction - done).square() * mask
            ).sum() / mask.sum().clamp(min=1)
            metrics = {
                "dyn_latent_mse": latent_loss.item(),
                "dyn_reward_mse": reward_loss.item(),
                "dyn_done_mse": done_loss.item(),
            }
            for key, value in metrics.items():
                aggregate.setdefault(f"validation/{key}", []).append(value)
    return {
        key: float(np.mean(values)) for key, values in aggregate.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Image-based DreamerV4 training")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--checkpoint-dir", type=str, default="dreamerv4_image_models")
    parser.add_argument("--record-dir", type=str, default="recorded_episodes")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--imagination-horizon", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--calibration", default="config/calibration/simulation.json")
    parser.add_argument("--hardware-calibration", default=None)
    parser.add_argument("--hardware-record-dir", default=None)
    parser.add_argument("--hardware-sample-ratio", type=float, default=0.25)
    parser.add_argument("--hardware-validation-fraction", type=float, default=0.2)
    parser.add_argument("--dataset-seed", type=int, default=0)
    parser.add_argument("--validation-every", type=int, default=1000)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Also write local TensorBoard events; W&B logging is independent.",
    )
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--tokenizer-steps", type=int, default=5000)
    parser.add_argument("--dynamics-steps", type=int, default=None,
                        help="Shortcut dynamics training steps; defaults to --total-steps")
    parser.add_argument("--mtp-steps", type=int, default=5000,
                        help="MTP behavior-prior pretraining steps")
    parser.add_argument("--rl-steps", type=int, default=None,
                        help="PMPO imagination RL steps; defaults to --total-steps // 5")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--offline", action="store_true",
                        help="Train from recorded episodes only (no env interaction)")
    parser.add_argument("--online", action="store_true",
                        help="Collect online episodes before training instead of using recorded episodes")
    parser.add_argument("--no-ir", action="store_true",
                        help="Disable IR input and train from camera images only")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args = parser.parse_args()
    if args.online:
        args.offline = False
    else:
        args.offline = True
    dynamics_steps = args.dynamics_steps if args.dynamics_steps is not None else args.total_steps
    rl_steps = args.rl_steps if args.rl_steps is not None else max(1, args.total_steps // 5)

    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root / "catkin_ws" / "src" / "learning_machines" / "src"))
    sys.path.insert(0, str(project_root / "catkin_ws" / "src" / "robobo_interface" / "src"))

    import os
    os.environ["COPPELIA_SIM_PORT"] = str(args.port)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = checkpoint_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(log_dir))
    else:
        writer = _NullSummaryWriter()

    wandb_run = None
    if not args.no_wandb:
        import wandb
        run_name = args.wandb_run_name or f"dreamerv4-image-{datetime.datetime.now().strftime('%m%d-%H%M%S')}"
        wandb_run = wandb.init(
            project="learning-machines",
            entity="Learningmachine",
            name=run_name,
            config=vars(args),
            sync_tensorboard=False,
            resume="allow",
        )
        wandb_run.define_metric("global_step")
        wandb_run.define_metric("*", step_metric="global_step")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    from learning_machines.dreamerv4.dreamerv4_image import ImageDreamerV4Agent
    from learning_machines.rl_robobo_compact_env import RoboboCompactEnv, RoboboCompactEnvConfig
    from learning_machines.domain_randomization import DomainRandomizationWrapper
    from learning_machines.domain_randomization import RandomizationRanges
    from learning_machines.transfer import CalibrationProfile, CheckpointManifest

    use_ir = not args.no_ir
    agent = ImageDreamerV4Agent(
        latent_dim=args.latent_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        context_length=args.context_length,
        imagination_horizon=args.imagination_horizon,
        lr=args.lr,
        image_size=args.image_size,
        ir_dim=8 if use_ir else 0,
    ).to(device)
    if wandb_run is not None:
        wandb_run.config.update({
            "observation_contract": "robobo-obs-v2",
            "reward_contract": "robobo-reward-v4",
            "control_interval_seconds": 0.4,
            "phone_tilt": 100,
        }, allow_val_change=True)

    model_path = checkpoint_dir / "dreamerv4_image_latest.pt"
    progress_path = checkpoint_dir / "training_progress.json"
    progress = TrainingProgress.load(progress_path) if args.resume else TrainingProgress()

    if args.resume and model_path.exists():
        manifest_path = checkpoint_dir / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(
                "refusing to resume a DreamerV4 checkpoint without a manifest"
            )
        manifest = CheckpointManifest.load(manifest_path)
        manifest.validate(
            "dreamerv4",
            CalibrationProfile.load(args.calibration).name,
            args.image_size,
            100,
        )
        resume_expected = {
            "latent_dim": args.latent_dim,
            "d_model": args.d_model,
            "use_ir": use_ir,
            "context_length": args.context_length,
            "imagination_horizon": args.imagination_horizon,
        }
        mismatches = {
            key: (manifest.algorithm_config.get(key), expected)
            for key, expected in resume_expected.items()
            if manifest.algorithm_config.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"DreamerV4 resume architecture is incompatible: {mismatches}"
            )
        print(f"Resuming from {model_path}")
        agent = ImageDreamerV4Agent.load(str(model_path), device)
        print(f"Resuming phase progress: {asdict(progress)}")

    if args.tokenizer_steps > progress.tokenizer and any(
        (progress.dynamics, progress.mtp, progress.pmpo)
    ):
        print("Tokenizer target increased; resetting downstream phase counters")
        progress.dynamics = progress.mtp = progress.pmpo = 0
    elif dynamics_steps > progress.dynamics and any((progress.mtp, progress.pmpo)):
        print("Dynamics target increased; resetting MTP and PMPO phase counters")
        progress.mtp = progress.pmpo = 0
    elif args.mtp_steps > progress.mtp and progress.pmpo:
        print("MTP target increased; resetting PMPO phase counter")
        progress.pmpo = 0

    def save_checkpoint() -> None:
        agent.save(str(model_path))
        progress.save(progress_path)

    def save_manifest() -> None:
        CheckpointManifest(
            algorithm="dreamerv4",
            calibration_profile=CalibrationProfile.load(args.calibration).name,
            image_size=args.image_size,
            algorithm_config={
                "latent_dim": args.latent_dim,
                "d_model": args.d_model,
                "use_ir": use_ir,
                "ir_dim": 8 if use_ir else 0,
                "context_length": args.context_length,
                "imagination_horizon": args.imagination_horizon,
                "mtp_length": agent.mtp_length,
                "gamma": agent.gamma,
                "hardware_calibration": args.hardware_calibration,
                "hardware_record_dir": args.hardware_record_dir,
                "hardware_sample_ratio": args.hardware_sample_ratio,
                "hardware_validation_fraction": args.hardware_validation_fraction,
                "dataset_seed": args.dataset_seed,
            },
        ).save(checkpoint_dir / "manifest.json")

    if args.offline:
        expected_calibration = CalibrationProfile.load(args.calibration).name
        simulation_scan = scan_recorded_episodes(
            args.record_dir,
            expected_calibration=expected_calibration,
            expected_phone_tilt=100,
            source="simulation",
        )
        if not simulation_scan.episodes:
            raise RuntimeError(f"No usable .npz episodes found in {Path(args.record_dir) / 'episodes'}")
        hardware_scan = DatasetScan()
        if args.hardware_record_dir:
            expected_hardware_calibration = (
                CalibrationProfile.load(args.hardware_calibration).name
                if args.hardware_calibration else None
            )
            hardware_scan = scan_recorded_episodes(
                args.hardware_record_dir,
                expected_calibration=expected_hardware_calibration,
                expected_phone_tilt=100,
                source="hardware",
            )
        split_rng = random.Random(args.dataset_seed)
        hardware_refs = hardware_scan.episodes.copy()
        split_rng.shuffle(hardware_refs)
        validation_count = (
            max(1, round(len(hardware_refs) * args.hardware_validation_fraction))
            if hardware_refs and args.hardware_validation_fraction > 0 else 0
        )
        dataset = StreamingEpisodeDataset(
            simulation_scan.episodes,
            hardware_train=hardware_refs[validation_count:],
            hardware_validation=hardware_refs[:validation_count],
            hardware_sample_ratio=args.hardware_sample_ratio,
        )
        if use_ir and not all(ref.has_ir for ref in dataset.training_refs):
            raise RuntimeError(
                "Recorded episodes must contain IR data by default. "
                "Pass --no-ir for image-only fallback."
            )
        skipped = simulation_scan.skipped + hardware_scan.skipped
        if skipped:
            print(
                f"Skipped {len(skipped)} incompatible/corrupt episodes; "
                f"first reason: {skipped[0][1]}"
            )
    else:
        print("Collecting image episodes...")
        env_config = RoboboCompactEnvConfig(
            max_episode_steps=150,
            randomize_food_positions=True,
            return_image=True,
            image_obs_size=(args.image_size, args.image_size),
            calibration_path=args.calibration,
        )
        rob_env = RoboboCompactEnv(config=env_config)
        randomization_ranges = None
        if args.hardware_calibration:
            randomization_ranges = RandomizationRanges.from_calibration_profiles(
                CalibrationProfile.load(args.calibration),
                CalibrationProfile.load(args.hardware_calibration),
            )
            if wandb_run is not None:
                wandb_run.config.update({
                    "derived_randomization_ranges": randomization_ranges.__dict__,
                }, allow_val_change=True)
        env = DomainRandomizationWrapper(rob_env, ranges=randomization_ranges)
        episodes = collect_episodes(
            env, agent, device, args.total_steps, args.image_size, use_ir,
            writer=writer, wandb_run=wandb_run,
        )
        env.close()
        online_episode_dir = checkpoint_dir / "online_dataset" / "episodes"
        online_episode_dir.mkdir(parents=True, exist_ok=True)
        calibration_name = CalibrationProfile.load(args.calibration).name
        for index, episode in enumerate(episodes):
            np.savez_compressed(
                online_episode_dir / f"ep_{index:06d}.npz",
                **episode,
                observation_contract=np.array("robobo-obs-v2"),
                reward_contract=np.array("robobo-reward-v4"),
                control_interval_seconds=np.array(0.4),
                calibration_profile=np.array(calibration_name),
                phone_tilt=np.array(100),
            )
        simulation_scan = scan_recorded_episodes(
            str(online_episode_dir.parent),
            calibration_name,
            source="simulation",
        )
        hardware_scan = DatasetScan()
        dataset = StreamingEpisodeDataset(simulation_scan.episodes)

    total_transitions = sum(ref.transitions for ref in dataset.training_refs)
    print(
        f"Indexed {len(dataset.training_refs)} training episodes with "
        f"{total_transitions} transitions "
        f"({len(dataset.hardware_train)} hardware train, "
        f"{len(dataset.hardware_validation)} hardware validation)"
    )
    dataset_metrics = {
        "dataset/episodes": len(dataset.training_refs),
        "dataset/transitions": total_transitions,
        "dataset/mean_episode_length": total_transitions / max(1, len(dataset.training_refs)),
        "dataset/mean_return": float(np.mean([
            ref.episode_return for ref in dataset.training_refs
        ])),
        "dataset/simulation_episodes": len(dataset.simulation),
        "dataset/hardware_train_episodes": len(dataset.hardware_train),
        "dataset/hardware_validation_episodes": len(dataset.hardware_validation),
        "dataset/hardware_validation_transitions": sum(
            ref.transitions for ref in dataset.hardware_validation
        ),
        "dataset/skipped_episodes": len(simulation_scan.skipped) + len(hardware_scan.skipped),
    }
    if wandb_run is not None:
        wandb_run.log({**dataset_metrics, "global_step": progress.global_log_step})
    save_manifest()
    atexit.register(save_checkpoint)

    # Phase 1: Train tokenizer
    print(
        f"\nPhase 1: Training tokenizer from {progress.tokenizer} "
        f"to {args.tokenizer_steps} steps..."
    )
    tokenizer_pbar = _progress(
        range(progress.tokenizer, args.tokenizer_steps), "Tokenizer"
    )
    for step in tokenizer_pbar:
        batch_images, batch_irs, hardware_fraction = dataset.sample_observation_batch(
            args.batch_size, use_ir, device
        )
        tok_metrics = agent.update_tokenizer(batch_images, batch_irs)
        tok_metrics["dataset/hardware_batch_fraction"] = hardware_fraction
        _log_metrics(
            writer, wandb_run, tok_metrics, progress.global_log_step, "tokenizer"
        )
        tokenizer_pbar.set_postfix_str(
            _metric_postfix(tok_metrics, ("tok/loss", "tok/mse_loss", "tok/ir_loss")),
            refresh=False,
        )
        progress.tokenizer = step + 1
        progress.global_log_step += 1

        if step % args.log_interval == 0:
            _write_metrics("Tokenizer", step, tok_metrics)
            if wandb_run is not None:
                import wandb
                with torch.no_grad():
                    diagnostic_latent = agent.tokenizer.encoder(batch_images[:4])
                    diagnostic_recon = agent.tokenizer.decode(diagnostic_latent)
                originals = [
                    wandb.Image(image.detach().cpu().permute(1, 2, 0).numpy())
                    for image in batch_images[:4]
                ]
                reconstructions = [
                    wandb.Image(image.detach().cpu().permute(1, 2, 0).numpy())
                    for image in diagnostic_recon
                ]
                wandb_run.log({
                    "diagnostics/tokenizer_original": originals,
                    "diagnostics/tokenizer_reconstruction": reconstructions,
                    "global_step": progress.global_log_step,
                })
        if dataset.hardware_validation and (
            step % args.validation_every == 0 or step + 1 == args.tokenizer_steps
        ):
            validation = evaluate_tokenizer(
                agent, dataset, dataset.hardware_validation, device, use_ir
            )
            _write_metrics("Hardware validation", step, validation)
            _log_metrics(
                writer, wandb_run, validation,
                progress.global_log_step, "validation"
            )
        if progress.tokenizer % args.save_freq == 0:
            save_checkpoint()
    save_checkpoint()

    # Phase 2: Encode all images to latents
    print("\nPhase 2: Encoding images to latents...")
    encoded_episodes = encode_episode_latents(
        agent, dataset, dataset.training_refs, device, args.batch_size, use_ir
    )
    encoded_validation = encode_episode_latents(
        agent, dataset, dataset.hardware_validation, device, args.batch_size, use_ir
    ) if dataset.hardware_validation else []

    print(f"Encoded {sum(len(ep['latents']) for ep in encoded_episodes)} observations to latents")

    # Phase 3: Train dynamics model
    print(f"\nPhase 3: Training dynamics from {progress.dynamics} to {dynamics_steps} steps...")
    dynamics_pbar = _progress(range(progress.dynamics, dynamics_steps), "Dynamics")
    for step in dynamics_pbar:
        batch_latents, batch_actions, batch_rewards, batch_dones, batch_mask, hardware_fraction = sample_latent_batch(
            encoded_episodes, args.batch_size, args.context_length, device,
            args.hardware_sample_ratio,
        )

        dyn_metrics = agent.update_dynamics(
            batch_latents, batch_actions, batch_rewards, batch_dones, batch_mask
        )
        dyn_metrics["dataset/hardware_batch_fraction"] = hardware_fraction
        _log_metrics(writer, wandb_run, dyn_metrics, progress.global_log_step, "dynamics")
        dynamics_pbar.set_postfix_str(
            _metric_postfix(dyn_metrics, ("dyn/total_loss", "dyn/latent_loss", "dyn/rew_loss")),
            refresh=False,
        )
        progress.dynamics = step + 1
        progress.global_log_step += 1

        if step % args.log_interval == 0:
            _write_metrics("Dynamics", step, dyn_metrics)
            if wandb_run is not None:
                import wandb
                wandb_run.log({
                    "diagnostics/dynamics_reward_targets": wandb.Histogram(
                        batch_rewards.detach().cpu().numpy()
                    ),
                    "diagnostics/dynamics_action_targets": wandb.Histogram(
                        batch_actions.detach().cpu().numpy()
                    ),
                    "diagnostics/dynamics_done_rate": float(
                        batch_dones.float().mean().item()
                    ),
                    "global_step": progress.global_log_step,
                })
        if encoded_validation and (
            step % args.validation_every == 0 or step + 1 == dynamics_steps
        ):
            validation = evaluate_dynamics(
                agent, encoded_validation, device, args.context_length
            )
            _write_metrics("Hardware validation", step, validation)
            _log_metrics(
                writer, wandb_run, validation,
                progress.global_log_step, "validation"
            )

        if progress.dynamics % args.save_freq == 0:
            save_checkpoint()
            tqdm.tqdm.write(f"[checkpoint] saved dynamics step {step} -> {model_path}")
    save_checkpoint()

    # Phase 4: MTP behavior-prior pretraining
    print(f"\nPhase 4: Training MTP from {progress.mtp} to {args.mtp_steps} steps...")
    mtp_pbar = _progress(range(progress.mtp, args.mtp_steps), "MTP")
    for step in mtp_pbar:
        batch_latents, batch_actions, batch_rewards, _, batch_mask, hardware_fraction = sample_latent_batch(
            encoded_episodes, args.batch_size, args.context_length, device,
            args.hardware_sample_ratio,
        )
        mtp_metrics = agent.update_mtp_behavior(
            batch_latents, batch_actions, batch_rewards, batch_mask
        )
        mtp_metrics["dataset/hardware_batch_fraction"] = hardware_fraction
        _log_metrics(writer, wandb_run, mtp_metrics, progress.global_log_step, "mtp")
        mtp_pbar.set_postfix_str(
            _metric_postfix(mtp_metrics, ("mtp/total_loss", "mtp/action_loss", "mtp/reward_loss")),
            refresh=False,
        )
        progress.mtp = step + 1
        progress.global_log_step += 1
        if step % args.log_interval == 0:
            _write_metrics("MTP", step, mtp_metrics)
        if progress.mtp % args.save_freq == 0:
            save_checkpoint()
    save_checkpoint()

    agent.freeze_behavior_prior(copy_to_actor=progress.pmpo == 0)

    # Phase 5: PMPO imagination RL with frozen tokenizer/dynamics/prior
    print(f"\nPhase 5: Training PMPO from {progress.pmpo} to {rl_steps} steps...")
    pmpo_pbar = _progress(range(progress.pmpo, rl_steps), "PMPO")
    for step in pmpo_pbar:
        batch_latents, _, _, _, _, hardware_fraction = sample_latent_batch(
            encoded_episodes, args.batch_size, args.context_length, device,
            args.hardware_sample_ratio,
        )
        ac_metrics = agent.update_actor_critic(batch_latents[:, 0])
        ac_metrics["dataset/hardware_batch_fraction"] = hardware_fraction
        _log_metrics(writer, wandb_run, ac_metrics, progress.global_log_step, "pmpo")
        pmpo_pbar.set_postfix_str(
            _metric_postfix(ac_metrics, ("ac/actor_loss", "ac/critic_loss", "ac/return_mean")),
            refresh=False,
        )
        progress.pmpo = step + 1
        progress.global_log_step += 1
        if step % args.log_interval == 0:
            _write_metrics("PMPO", step, ac_metrics)
        if progress.pmpo % args.save_freq == 0:
            save_checkpoint()
            tqdm.tqdm.write(f"[checkpoint] saved PMPO step {step} -> {model_path}")

    save_checkpoint()
    save_manifest()
    atexit.unregister(save_checkpoint)
    writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    print(f"\nTraining complete. Model saved to {model_path}")


if __name__ == "__main__":
    main()
