#!/usr/bin/env python3
'Train the paper-aligned DreamerV4 implementation on recorded Robobo data.'
from __future__ import annotations

import argparse
import datetime
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "catkin_ws/src/learning_machines/src"))
sys.path.insert(0, str(ROOT / "catkin_ws/src/robobo_interface/src"))

from learning_machines.dreamerv4_full import (  # noqa: E402
    DreamerV4FullAgent,
    DreamerV4FullConfig,
)
from learning_machines.distributional import two_hot_loss  # noqa: E402
from learning_machines.transfer import (  # noqa: E402
    CalibrationProfile,
    CheckpointManifest,
)
from train_dreamerv4_image import (  # noqa: E402
    EpisodeRef,
    _episode_from_npz,
    scan_recorded_episodes,
)


@dataclass
class Progress:
    tokenizer: int = 0
    world: int = 0
    world_long: int = 0
    finetune: int = 0
    imagination: int = 0

    @classmethod
    def load(cls, path: Path) -> "Progress":
        return cls(**json.loads(path.read_text())) if path.exists() else cls()

    def save(self, path: Path):
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2) + "\n")
        temporary.replace(path)


def load_episode(ref: EpisodeRef) -> dict:
    episode = _episode_from_npz(
        ref.path,
        expected_calibration=ref.calibration_profile,
        expected_phone_tilt=100,
        source=ref.source,
    )
    if episode is None:
        raise ValueError(f"episode became unavailable: {ref.path}")
    return episode


def choose_ref(refs: list[EpisodeRef], minimum: int) -> EpisodeRef:
    eligible = [ref for ref in refs if ref.transitions >= minimum]
    if not eligible:
        raise ValueError(f"no episode has at least {minimum} transitions")
    return random.choice(eligible)


def sample_video_batch(
    refs: list[EpisodeRef],
    batch_size: int,
    length: int,
    device: torch.device,
    use_ir: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    images, irs = [], []
    for _ in range(batch_size):
        episode = load_episode(choose_ref(refs, length - 1))
        maximum = len(episode["images"]) - length
        start = random.randint(0, maximum)
        images.append(episode["images"][start : start + length])
        if use_ir:
            irs.append(episode["irs"][start : start + length])
    image_tensor = torch.from_numpy(np.stack(images)).float().to(device) / 255.0
    ir_tensor = (
        torch.from_numpy(np.stack(irs)).float().to(device) if use_ir else None
    )
    return image_tensor, ir_tensor


@torch.no_grad()
def encode_dataset(
    agent: DreamerV4FullAgent,
    refs: list[EpisodeRef],
    use_ir: bool,
    chunk_length: int,
) -> list[dict]:
    encoded = []
    context = agent.cfg.context_length
    for ref in tqdm.tqdm(refs, desc="Encoding", dynamic_ncols=True):
        episode = load_episode(ref)
        images = torch.from_numpy(episode["images"]).float() / 255.0
        ir = (
            torch.from_numpy(episode["irs"]).float() if use_ir else None
        )
        chunks = []
        for start in range(0, len(images), chunk_length):
            prefix = max(0, start - context + 1)
            end = min(len(images), start + chunk_length)
            image_chunk = images[prefix:end].unsqueeze(0).to(agent.device)
            ir_chunk = (
                ir[prefix:end].unsqueeze(0).to(agent.device)
                if ir is not None else None
            )
            latent = agent.tokenizer.encode(image_chunk, ir_chunk)[0]
            chunks.append(latent[start - prefix :].cpu().half())
        encoded.append({
            "latents": torch.cat(chunks),
            "actions": torch.from_numpy(episode["actions"]).float(),
            "rewards": torch.from_numpy(episode["rewards"]).float(),
            "dones": torch.from_numpy(episode["dones"].astype(np.float32)),
        })
    return encoded


def sample_latent_batch(
    episodes: list[dict],
    batch_size: int,
    length: int,
    device: torch.device,
    start_frame_fraction: float = 0.0,
) -> tuple[torch.Tensor, ...]:
    eligible = [episode for episode in episodes if len(episode["actions"]) >= length]
    if not eligible:
        raise ValueError(f"no encoded episode has {length} transitions")
    latents, actions, rewards, dones, masks, action_known = [], [], [], [], [], []
    for _ in range(batch_size):
        episode = random.choice(eligible)
        if random.random() < start_frame_fraction:
            index = random.randrange(len(episode["latents"]))
            latent = torch.zeros(
                length + 1,
                *episode["latents"].shape[1:],
                dtype=episode["latents"].dtype,
            )

            latent[1] = episode["latents"][index]
            latents.append(latent)
            actions.append(torch.zeros(length, episode["actions"].shape[-1]))
            rewards.append(torch.zeros(length))
            dones.append(torch.zeros(length))
            mask = torch.zeros(length)
            mask[0] = 1
            masks.append(mask)
            known = torch.zeros(length, dtype=torch.bool)
            action_known.append(known)
            continue
        start = random.randint(0, len(episode["actions"]) - length)
        end = start + length
        latents.append(episode["latents"][start : end + 1])
        actions.append(episode["actions"][start:end])
        rewards.append(episode["rewards"][start:end])
        dones.append(episode["dones"][start:end])
        masks.append(torch.ones(length))
        action_known.append(torch.ones(length, dtype=torch.bool))
    return (
        torch.stack(latents).float().to(device),
        torch.stack(actions).to(device),
        torch.stack(rewards).to(device),
        torch.stack(dones).to(device),
        torch.stack(masks).to(device),
        torch.stack(action_known).to(device),
    )


def phase_length(args, step: int) -> int:
    if random.random() < args.long_batch_probability:
        return args.long_batch_length
    return args.short_batch_length


def run_phase(
    name,
    start,
    stop,
    update,
    save,
    log_every,
    validate=None,
    validation_every=0,
    on_validation=None,
    wandb_run=None,
):
    progress = tqdm.trange(start, stop, desc=name, dynamic_ncols=True)
    for step in progress:
        metrics = update()
        progress.set_postfix({
            key.split("/")[-1]: f"{value:.4f}"
            for key, value in list(metrics.items())[:3]
        })
        if step % log_every == 0:
            tqdm.tqdm.write(
                f"{name} {step}: "
                + " ".join(f"{key}={value:.5f}" for key, value in metrics.items())
            )
            if wandb_run is not None:
                wandb_run.log({**metrics, f"{name.lower()}_step": step})
        if validate is not None and (
            step + 1 == stop
            or (
                validation_every > 0
                and (step + 1) % validation_every == 0
            )
        ):
            validation = validate()
            tqdm.tqdm.write(
                f"{name} validation {step}: "
                + " ".join(
                    f"{key}={value:.5f}"
                    for key, value in validation.items()
                )
            )
            if wandb_run is not None:
                wandb_run.log({**validation, f"{name.lower()}_step": step + 1})
            if on_validation is not None:
                on_validation(validation, step + 1)
        save(step + 1)


@torch.no_grad()
def validate_tokenizer(agent, refs, batch_size, length, use_ir):
    was_training = agent.training
    agent.eval()
    images, ir = sample_video_batch(
        refs, batch_size, length, agent.device, use_ir
    )
    latent = agent.tokenizer.encode(images, ir)
    reconstruction, ir_prediction = agent.tokenizer.decode(latent)
    metrics = {
        "validation/tokenizer_mse": F.mse_loss(
            reconstruction, images
        ).item(),
    }
    if agent.cfg.lpips_weight:
        metrics["validation/tokenizer_lpips"] = agent.tokenizer.lpips(
            reconstruction.flatten(0, 1),
            images.flatten(0, 1),
        ).item()
    if use_ir:
        metrics["validation/tokenizer_ir_mse"] = F.mse_loss(
            ir_prediction, ir
        ).item()
    if was_training:
        agent.train()
    return metrics


@torch.no_grad()
def log_tokenizer_reconstructions(agent, refs, wandb_run, num_samples=4):
    'Log original, reconstructed, and error images to W&B after tokenizer training.'
    import wandb
    was_training = agent.training
    agent.eval()
    try:
        sampled_refs = random.sample(refs, min(num_samples, len(refs)))
        images_list, reconstructions_list, error_list = [], [], []
        for ref in sampled_refs:
            episode = load_episode(ref)
            images = torch.from_numpy(episode["images"]).float().to(agent.device) / 255.0
            length = min(agent.cfg.context_length, images.shape[0])
            images = images[:length]
            ir = None
            if agent.cfg.ir_dim > 0 and "irs" in episode:
                ir = torch.from_numpy(episode["irs"]).float().to(agent.device)[:length]
            images = images.unsqueeze(0)
            if ir is not None:
                ir = ir.unsqueeze(0)
            latent = agent.tokenizer.encode(images, ir)
            reconstruction, _ = agent.tokenizer.decode(latent)
            reconstruction = reconstruction.clamp(0, 1)
            error = (reconstruction - images).abs()

            mid = length // 2
            images_list.append(images[0, mid].cpu())
            reconstructions_list.append(reconstruction[0, mid].cpu())
            error_list.append(error[0, mid].cpu())

        panels = []
        for original, recon, err in zip(images_list, reconstructions_list, error_list):
            panels.append(np.concatenate([
                original.permute(1, 2, 0).numpy(),
                recon.permute(1, 2, 0).numpy(),
                err.permute(1, 2, 0).numpy(),
            ], axis=1))
        grid = np.concatenate(panels, axis=0)
        caption = "Left: original | Middle: reconstructed | Right: absolute error"
        wandb_run.log({
            "tokenizer/reconstructions": wandb.Image(
                (grid * 255).astype(np.uint8), caption=caption
            )
        })
    finally:
        if was_training:
            agent.train()


@torch.no_grad()
def validate_latent_model(agent, episodes, batch_size, length):
    was_training = agent.training
    agent.eval()
    latents, actions, rewards, dones, mask, _ = sample_latent_batch(
        episodes, batch_size, length, agent.device
    )
    target, transition_actions = agent._world_sequence(latents, actions)
    signal = torch.full(target.shape[:2], 0.5, device=agent.device)
    step = torch.full(
        target.shape[:2],
        1 / agent.cfg.shortcut_steps,
        device=agent.device,
    )
    noise = torch.randn_like(target)
    corrupted = 0.5 * noise + 0.5 * target
    prediction = agent.dynamics(
        corrupted, transition_actions, signal, step
    )["latent"]
    latent_per_step = (prediction - target).square().mean((-1, -2))
    latent_mse = (latent_per_step * mask).sum() / mask.sum()

    policy_state, previous_actions = agent._policy_sequence(latents, actions)
    hidden = agent._clean_agent_hidden(
        policy_state, previous_actions, None
    )
    action_nll = -agent.policy.log_prob(hidden, actions)
    reward_nll = two_hot_loss(
        agent.reward(hidden), rewards, reduction="none"
    )
    continue_loss = F.binary_cross_entropy_with_logits(
        agent.continue_head(hidden).squeeze(-1),
        1 - dones,
        reduction="none",
    )
    denominator = mask.sum().clamp_min(1)
    metrics = {
        "validation/latent_mse": latent_mse.item(),
        "validation/action_nll": (
            action_nll * mask
        ).sum().div(denominator).item(),
        "validation/reward_nll": (
            reward_nll * mask
        ).sum().div(denominator).item(),
        "validation/continue_loss": (
            continue_loss * mask
        ).sum().div(denominator).item(),
    }
    if was_training:
        agent.train()
    return metrics


@torch.no_grad()
def validate_imagination(agent, episodes, batch_size, length):
    was_training = agent.training
    agent.eval()
    latents, actions, _, _, _, _ = sample_latent_batch(
        episodes, batch_size, length, agent.device
    )
    rollout = agent.imagine(latents, actions)
    prior_kl = agent.policy.reverse_kl(
        rollout["hidden"], agent.prior_policy
    ).mean()
    metrics = {
        "validation/imagined_return": rollout["rewards"].sum(1).mean().item(),
        "validation/prior_kl": prior_kl.item(),
        "validation/continuation": rollout["continuation"].mean().item(),
    }
    if was_training:
        agent.train()
    return metrics


def average_validation(callback, batches: int):
    aggregate: dict[str, list[float]] = {}
    for _ in range(max(1, batches)):
        for key, value in callback().items():
            aggregate.setdefault(key, []).append(float(value))
    return {
        key: float(np.mean(values)) for key, values in aggregate.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", default="recorded_episodes")
    parser.add_argument("--checkpoint-dir", default="results/dreamer-v4-full-checkpoints")
    parser.add_argument("--calibration", default="config/calibration/simulation.json")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-ir", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tokenizer-steps", type=int, default=10_000)
    parser.add_argument("--world-steps", type=int, default=50_000)
    parser.add_argument(
        "--long-world-steps",
        type=int,
        default=5_000,
        help="Final long-sequence-only world-model finetuning from Section 3.4.",
    )
    parser.add_argument("--finetune-steps", type=int, default=10_000)
    parser.add_argument("--imagination-steps", type=int, default=10_000)
    parser.add_argument(
        "--imagination-horizon",
        type=int,
        default=None,
        help="Override imagination horizon for PMPO rollouts.",
    )
    parser.add_argument("--short-batch-length", type=int, default=40)
    parser.add_argument("--long-batch-length", type=int, default=80)
    parser.add_argument("--long-batch-probability", type=float, default=0.1)
    parser.add_argument(
        "--start-frame-fraction",
        type=float,
        default=0.3,
        help="Fraction of world-model samples trained as standalone images.",
    )
    parser.add_argument("--encode-chunk-length", type=int, default=64)
    parser.add_argument(
        "--reencode",
        action="store_true",
        help="Ignore cached tokenizer latents and rebuild them.",
    )
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--validation-every", type=int, default=1000)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="learning-machines")
    parser.add_argument("--wandb-entity", type=str, default="Learningmachine")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--validation-batches", type=int, default=4)
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--latent-tokens", type=int, default=16)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--tokenizer-layers", type=int, default=4)
    parser.add_argument("--dynamics-layers", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.1)
    parser.add_argument(
        "--agent-grad-clip",
        type=float,
        default=None,
        help="Optional separate gradient clip for agent/policy heads. "
             "If omitted, all parameters use --grad-clip.",
    )
    parser.add_argument("--pmpo-alpha", type=float, default=0.5)
    parser.add_argument("--prior-kl-weight", type=float, default=0.3)
    parser.add_argument("--entropy-weight", type=float, default=1e-4)
    parser.add_argument("--no-advantage-normalization", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument(
        "--mixed-precision-dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
    )
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if args.short_batch_length <= args.context_length:
        parser.error("--short-batch-length must exceed --context-length")
    if not 0 <= args.start_frame_fraction <= 1:
        parser.error("--start-frame-fraction must be in [0, 1]")
    if not 0 <= args.validation_fraction < 1:
        parser.error("--validation-fraction must be in [0, 1)")
    expected_calibration = CalibrationProfile.load(args.calibration).name
    scan = scan_recorded_episodes(args.record_dir, expected_calibration)
    if not scan.episodes:
        raise SystemExit("no compatible recorded episodes found")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_path = checkpoint_dir / "dreamerv4_full_latest.pt"
    progress_path = checkpoint_dir / "training_progress.json"

    wandb_run = None
    if not args.no_wandb:
        import wandb
        run_name = args.wandb_run_name or (
            f"dreamerv4-full-{datetime.datetime.now().strftime('%m%d-%H%M%S')}"
        )
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
            resume="allow",
        )

    if args.resume and model_path.exists():
        agent = DreamerV4FullAgent.load(model_path)
        progress = Progress.load(progress_path)
    else:
        config = DreamerV4FullConfig(
            ir_dim=0 if args.no_ir else 8,
            model_dim=args.model_dim,
            latent_tokens=args.latent_tokens,
            latent_channels=args.latent_channels,
            tokenizer_layers=args.tokenizer_layers,
            dynamics_layers=args.dynamics_layers,
            context_length=args.context_length,
            learning_rate=args.lr,
            warmup_fraction=args.warmup_fraction,
            minimum_lr_ratio=args.minimum_lr_ratio,
            agent_grad_clip=args.agent_grad_clip,
            pmpo_alpha=args.pmpo_alpha,
            prior_kl_weight=args.prior_kl_weight,
            entropy_weight=args.entropy_weight,
            normalize_advantages=not args.no_advantage_normalization,
            imagination_horizon=(
                args.imagination_horizon
                if args.imagination_horizon is not None else 16
            ),
            mixed_precision=not args.no_mixed_precision,
            mixed_precision_dtype=args.mixed_precision_dtype,
        )
        agent = DreamerV4FullAgent(config)
        progress = Progress()
    use_ir = agent.cfg.ir_dim > 0
    shuffled_refs = list(scan.episodes)
    random.Random(0).shuffle(shuffled_refs)
    validation_count = (
        max(1, round(len(shuffled_refs) * args.validation_fraction))
        if args.validation_fraction and len(shuffled_refs) > 1
        else 0
    )
    validation_refs = (
        shuffled_refs[-validation_count:]
        if validation_count else shuffled_refs[:1]
    )
    refs = (
        shuffled_refs[:-validation_count]
        if validation_count else shuffled_refs
    )
    print(
        f"device={agent.device} train_episodes={len(refs)} "
        f"validation_episodes={len(validation_refs)} "
        f"transitions={scan.transitions}"
    )
    if not agent.schedulers:
        agent.configure_schedulers({
            "tokenizer": args.tokenizer_steps,
            "world": args.world_steps + args.long_world_steps,
            "finetune": args.finetune_steps,
            "imagination": args.imagination_steps,
        })
    CheckpointManifest(
        algorithm="dreamerv4-full",
        calibration_profile=expected_calibration,
        image_size=agent.cfg.image_size,
        algorithm_config=agent.cfg.to_dict(),
    ).save(checkpoint_dir / "manifest.json")
    if wandb_run is not None:
        wandb_run.config.update({
            "algorithm_config": agent.cfg.to_dict(),
            "train_episodes": len(refs),
            "validation_episodes": len(validation_refs),
            "transitions": scan.transitions,
        })
    best_path = checkpoint_dir / "best_metrics.json"
    best_metrics = (
        json.loads(best_path.read_text()) if best_path.exists() else {}
    )

    def checkpoint():
        agent.save(model_path)
        progress.save(progress_path)

    def record_best(phase, validation, score):
        previous = best_metrics.get(phase, {}).get("score", float("inf"))
        if score >= previous:
            return
        best_metrics[phase] = {
            "score": float(score),
            **{key: float(value) for key, value in validation.items()},
        }
        best_path.write_text(json.dumps(best_metrics, indent=2) + "\n")
        agent.save(checkpoint_dir / f"dreamerv4_full_best_{phase}.pt")

    def tokenizer_update():
        length = phase_length(args, progress.tokenizer)
        images, ir = sample_video_batch(
            refs, args.batch_size, length, agent.device, use_ir
        )
        return agent.update_tokenizer(images, ir)

    def tokenizer_save(value):
        progress.tokenizer = value
        if value % args.save_every == 0:
            checkpoint()

    run_phase(
        "Tokenizer", progress.tokenizer, args.tokenizer_steps,
        tokenizer_update, tokenizer_save, args.log_every,
        validate=lambda: average_validation(
            lambda: validate_tokenizer(
                agent,
                validation_refs,
                min(args.batch_size, len(validation_refs)),
                args.short_batch_length,
                use_ir,
            ),
            args.validation_batches,
        ),
        validation_every=args.validation_every,
        on_validation=lambda metrics, _step: record_best(
            "tokenizer",
            metrics,
            metrics["validation/tokenizer_mse"]
            + agent.cfg.lpips_weight
            * metrics.get("validation/tokenizer_lpips", 0.0)
            + metrics.get("validation/tokenizer_ir_mse", 0.0),
        ),
        wandb_run=wandb_run,
    )
    if wandb_run is not None and len(validation_refs) > 0:
        log_tokenizer_reconstructions(agent, validation_refs, wandb_run)

    checkpoint()

    latent_cache = checkpoint_dir / "encoded_latents.pt"
    if latent_cache.exists() and not args.reencode:
        cached = torch.load(latent_cache, map_location="cpu", weights_only=False)
        if cached.get("tokenizer_step") == progress.tokenizer:
            encoded = cached["training"]
            encoded_validation = cached["validation"]
            print(f"Loaded encoded latent cache: {latent_cache}")
        else:
            cached = None
    else:
        cached = None
    if cached is None:
        encoded = encode_dataset(
            agent, refs, use_ir, args.encode_chunk_length
        )
        encoded_validation = encode_dataset(
            agent, validation_refs, use_ir, args.encode_chunk_length
        )
        temporary_cache = latent_cache.with_suffix(".tmp")
        torch.save({
            "tokenizer_step": progress.tokenizer,
            "training": encoded,
            "validation": encoded_validation,
        }, temporary_cache)
        temporary_cache.replace(latent_cache)

    def latent_batch(length=None, start_frame_fraction=0.0):
        if length is None:
            length = phase_length(args, progress.world)
        return sample_latent_batch(
            encoded,
            args.batch_size,
            length,
            agent.device,
            start_frame_fraction,
        )

    def world_update():
        latents, actions, _, _, mask, action_known = latent_batch(
            start_frame_fraction=args.start_frame_fraction
        )
        return agent.update_world_model(
            latents, actions, mask, action_known
        )

    def world_save(value):
        progress.world = value
        if value % args.save_every == 0:
            checkpoint()

    run_phase(
        "World", progress.world, args.world_steps,
        world_update, world_save, args.log_every,
        validate=lambda: average_validation(
            lambda: validate_latent_model(
                agent,
                encoded_validation,
                min(args.batch_size, len(encoded_validation)),
                args.short_batch_length,
            ),
            args.validation_batches,
        ),
        validation_every=args.validation_every,
        on_validation=lambda metrics, _step: record_best(
            "world", metrics, metrics["validation/latent_mse"]
        ),
        wandb_run=wandb_run,
    )
    checkpoint()

    def long_world_update():
        latents, actions, _, _, mask, action_known = latent_batch(
            args.long_batch_length,
            args.start_frame_fraction,
        )
        return agent.update_world_model(
            latents, actions, mask, action_known
        )

    def long_world_save(value):
        progress.world_long = value
        if value % args.save_every == 0:
            checkpoint()

    run_phase(
        "WorldLong",
        progress.world_long,
        args.long_world_steps,
        long_world_update,
        long_world_save,
        args.log_every,
        validate=lambda: average_validation(
            lambda: validate_latent_model(
                agent,
                encoded_validation,
                min(args.batch_size, len(encoded_validation)),
                args.long_batch_length,
            ),
            args.validation_batches,
        ),
        validation_every=args.validation_every,
        on_validation=lambda metrics, _step: record_best(
            "world_long", metrics, metrics["validation/latent_mse"]
        ),
        wandb_run=wandb_run,
    )
    checkpoint()

    if progress.finetune == 0:
        agent.prepare_agent_finetune()

    def finetune_update():
        length = phase_length(args, progress.finetune)
        batch = sample_latent_batch(encoded, args.batch_size, length, agent.device)
        return agent.update_agent_finetune(*batch[:5])

    def finetune_save(value):
        progress.finetune = value
        if value % args.save_every == 0:
            checkpoint()

    run_phase(
        "Finetune", progress.finetune, args.finetune_steps,
        finetune_update, finetune_save, args.log_every,
        validate=lambda: average_validation(
            lambda: validate_latent_model(
                agent,
                encoded_validation,
                min(args.batch_size, len(encoded_validation)),
                args.short_batch_length,
            ),
            args.validation_batches,
        ),
        validation_every=args.validation_every,
        on_validation=lambda metrics, _step: record_best(
            "finetune",
            metrics,
            metrics["validation/latent_mse"]
            + metrics["validation/action_nll"]
            + metrics["validation/reward_nll"]
            + metrics["validation/continue_loss"],
        ),
        wandb_run=wandb_run,
    )
    agent.freeze_behavior_prior()
    if progress.imagination == 0:
        agent.prepare_imagination()
    checkpoint()

    def imagination_update():
        latents, actions, _, _, _, _ = sample_latent_batch(
            encoded, args.batch_size, args.context_length, agent.device
        )
        return agent.update_imagination(latents, actions)

    def imagination_save(value):
        progress.imagination = value
        if value % args.save_every == 0:
            checkpoint()

    run_phase(
        "Imagination", progress.imagination, args.imagination_steps,
        imagination_update, imagination_save, args.log_every,
        validate=lambda: average_validation(
            lambda: validate_imagination(
                agent,
                encoded_validation,
                min(args.batch_size, len(encoded_validation)),
                args.context_length,
            ),
            args.validation_batches,
        ),
        validation_every=args.validation_every,
        on_validation=lambda metrics, _step: record_best(
            "imagination",
            metrics,
            -metrics["validation/imagined_return"]
            + agent.cfg.prior_kl_weight * metrics["validation/prior_kl"],
        ),
        wandb_run=wandb_run,
    )
    checkpoint()
    if wandb_run is not None:
        wandb_run.finish()
    print(f"Training complete: {model_path}")


if __name__ == "__main__":
    main()
