from __future__ import annotations

import ast
import inspect
import numpy as np
import torch
import pytest

from learning_machines.reference_checkpoint import (
    REFERENCE_CHECKPOINT_VERSION,
    collect_optimizer_state_dicts,
    load_optimizer_state_dicts,
    save_checkpoint_atomic,
    validate_checkpoint,
)
from train_dreamerv3_reference_push import (
    REFERENCE_BATCH_LENGTH,
    REFERENCE_BATCH_SIZE,
    REFERENCE_CNN_DEPTH,
    REFERENCE_CNN_MINRES,
    REFERENCE_DYN_DETER,
    REFERENCE_DYN_HIDDEN,
    REFERENCE_IMAGE_SIZE,
    REFERENCE_MLP_UNITS,
    REFERENCE_TRAIN_RATIO,
)
import train_dreamerv3_reference_push


class _ExplodingDescriptor:
    def __getattr__(self, _name):
        raise RuntimeError("runtime logger object must not be traversed")


class _OptimizerWrapper:
    def __init__(self, parameter):
        self._opt = torch.optim.Adam([parameter], lr=1e-3)


class _Agent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self._model_opt = _OptimizerWrapper(self.weight)
        self._logger = _ExplodingDescriptor()
        self._config = _ExplodingDescriptor()
        self._dataset = (value for value in ())


def test_reference_push_uses_resource_conscious_model_defaults():
    assert REFERENCE_IMAGE_SIZE == (96, 96)
    assert REFERENCE_DYN_HIDDEN == 512
    assert REFERENCE_DYN_DETER == 1024
    assert REFERENCE_MLP_UNITS == 512
    assert REFERENCE_CNN_DEPTH == 32
    assert REFERENCE_CNN_MINRES == 3
    assert REFERENCE_BATCH_SIZE == 4
    assert REFERENCE_BATCH_LENGTH == 48
    assert REFERENCE_TRAIN_RATIO == 128


def test_reference_cnn_round_trips_96_pixel_images():
    import networks

    encoder = networks.ConvEncoder(
        (96, 96, 3),
        depth=REFERENCE_CNN_DEPTH,
        minres=REFERENCE_CNN_MINRES,
    )
    encoded = encoder(torch.zeros(1, 1, 96, 96, 3))
    assert encoded.shape[-1] == encoder.outdim

    decoder = networks.ConvDecoder(
        feat_size=32,
        shape=(3, 96, 96),
        depth=REFERENCE_CNN_DEPTH,
        act="ELU",
        minres=REFERENCE_CNN_MINRES,
    )
    decoded = decoder(torch.zeros(1, 1, 32))
    assert decoded.shape == (1, 1, 96, 96, 3)


def test_nm512_wrapper_advertises_actual_96_pixel_image_shape():
    import gymnasium as gym
    from learning_machines.robobo_env_wrapper import RoboboNM512Wrapper

    class ImageEnv(gym.Env):
        observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(
                low=0, high=255, shape=(3, 96, 96), dtype=np.uint8
            ),
            "ir": gym.spaces.Box(
                low=0.0, high=1.0, shape=(8,), dtype=np.float32
            ),
        })
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    wrapper = RoboboNM512Wrapper(ImageEnv(), include_ir=True)

    assert wrapper.observation_space["image"].shape == (96, 96, 3)


def test_reference_multimodal_encoder_declared_width_matches_96_pixel_output():
    import networks

    encoder = networks.MultiEncoder(
        {"image": (96, 96, 3), "ir": (8,)},
        mlp_keys="ir",
        cnn_keys="image",
        act="SiLU",
        norm=True,
        cnn_depth=REFERENCE_CNN_DEPTH,
        kernel_size=4,
        minres=REFERENCE_CNN_MINRES,
        mlp_layers=5,
        mlp_units=REFERENCE_MLP_UNITS,
        symlog_inputs=True,
    )
    output = encoder({
        "image": torch.zeros(1, 1, 96, 96, 3),
        "ir": torch.zeros(1, 1, 8),
    })

    assert output.shape == (1, 1, encoder.outdim)


def test_reference_trainer_constructs_only_one_simulator_environment():
    tree = ast.parse(inspect.getsource(train_dreamerv3_reference_push.main))
    constructor_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "make_robobo_env"
    ]

    assert len(constructor_calls) == 1


def test_nm512_wrapper_applies_curriculum_stage_and_domain_randomization():
    import gymnasium as gym
    from learning_machines.domain_randomization import DomainRandomizationWrapper
    from learning_machines.robobo_env_wrapper import RoboboNM512Wrapper

    class BaseEnv(gym.Env):
        observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(
                low=0, high=255, shape=(3, 96, 96), dtype=np.uint8
            ),
            "ir": gym.spaces.Box(
                low=0.0, high=1.0, shape=(8,), dtype=np.float32
            ),
        })
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        def __init__(self):
            self.config = type("Config", (), {"push_curriculum_stage": 0})()

    base = BaseEnv()
    randomized = DomainRandomizationWrapper(base, enabled=False)
    wrapper = RoboboNM512Wrapper(randomized, include_ir=True)

    wrapper.set_push_curriculum_stage(2, domain_randomization=True)

    assert base.config.push_curriculum_stage == 2
    assert randomized.enabled is True


def test_reference_wrapper_stack_forwards_curriculum_stage_updates():
    import gymnasium as gym
    from envs import wrappers
    from learning_machines.domain_randomization import DomainRandomizationWrapper
    from learning_machines.robobo_env_wrapper import RoboboNM512Wrapper
    from parallel import Damy

    class BaseEnv(gym.Env):
        observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(
                low=0, high=255, shape=(3, 96, 96), dtype=np.uint8
            ),
            "ir": gym.spaces.Box(
                low=0.0, high=1.0, shape=(8,), dtype=np.float32
            ),
        })
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        def __init__(self):
            self.config = type("Config", (), {"push_curriculum_stage": 0})()

    base = BaseEnv()
    randomized = DomainRandomizationWrapper(base, enabled=False)
    env = RoboboNM512Wrapper(randomized, include_ir=True)
    env = wrappers.NormalizeActions(env)
    env = wrappers.TimeLimit(env, 200)
    env = wrappers.SelectAction(env, key="action")
    env = Damy(wrappers.UUID(env))

    env.set_push_curriculum_stage(2, True)

    assert base.config.push_curriculum_stage == 2
    assert randomized.enabled is True


def test_nm512_wrapper_promotes_curriculum_before_next_reset(tmp_path):
    import gymnasium as gym
    from learning_machines.domain_randomization import DomainRandomizationWrapper
    from learning_machines.push_curriculum import (
        PushCurriculumConfig,
        PushCurriculumController,
    )
    from learning_machines.robobo_env_wrapper import RoboboNM512Wrapper

    class SuccessEnv(gym.Env):
        observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(
                low=0, high=255, shape=(3, 96, 96), dtype=np.uint8
            ),
            "ir": gym.spaces.Box(
                low=0.0, high=1.0, shape=(8,), dtype=np.float32
            ),
        })
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        def __init__(self):
            self.config = type("Config", (), {"push_curriculum_stage": 0})()

        def step(self, _action):
            return (
                {
                    "image": np.zeros((3, 96, 96), dtype=np.uint8),
                    "ir": np.zeros(8, dtype=np.float32),
                },
                1.0,
                True,
                False,
                {"block_goal_distance": 0.0},
            )

    controller = PushCurriculumController(
        PushCurriculumConfig(window=1, min_stage_steps=0)
    )
    base = SuccessEnv()
    randomized = DomainRandomizationWrapper(base, enabled=False)
    wrapper = RoboboNM512Wrapper(
        randomized,
        include_ir=True,
        curriculum_controller=controller,
        curriculum_state_path=tmp_path / "curriculum_state.json",
        domain_randomization=True,
    )
    promotions = []
    wrapper.set_curriculum_promotion_callback(promotions.append)

    wrapper.step(np.zeros(2, dtype=np.float32))

    assert controller.stage == 1
    assert base.config.push_curriculum_stage == 1
    assert randomized.enabled is False
    assert len(promotions) == 1
    assert (tmp_path / "curriculum_state.json").exists()

    wrapper.step(np.zeros(2, dtype=np.float32))

    assert controller.stage == 2
    assert base.config.push_curriculum_stage == 2
    assert randomized.enabled is True
    assert len(promotions) == 2


def test_nm512_wrapper_excludes_evaluation_from_curriculum_counts():
    import gymnasium as gym
    from learning_machines.domain_randomization import DomainRandomizationWrapper
    from learning_machines.push_curriculum import (
        PushCurriculumConfig,
        PushCurriculumController,
    )
    from learning_machines.robobo_env_wrapper import RoboboNM512Wrapper

    class SuccessEnv(gym.Env):
        observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(
                low=0, high=255, shape=(3, 96, 96), dtype=np.uint8
            ),
            "ir": gym.spaces.Box(
                low=0.0, high=1.0, shape=(8,), dtype=np.float32
            ),
        })
        action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        def __init__(self):
            self.config = type("Config", (), {"push_curriculum_stage": 0})()

        def step(self, _action):
            return (
                {
                    "image": np.zeros((3, 96, 96), dtype=np.uint8),
                    "ir": np.zeros(8, dtype=np.float32),
                },
                1.0,
                True,
                False,
                {},
            )

    controller = PushCurriculumController(
        PushCurriculumConfig(window=1, min_stage_steps=0)
    )
    wrapper = RoboboNM512Wrapper(
        DomainRandomizationWrapper(SuccessEnv(), enabled=False),
        curriculum_controller=controller,
    )
    wrapper.set_curriculum_tracking_enabled(False)

    wrapper.step(np.zeros(2, dtype=np.float32))

    assert controller.transition_count == 0
    assert controller.stage == 0


def test_reference_checkpoint_collects_optimizer_without_traversing_logger():
    states = collect_optimizer_state_dicts(_Agent())

    assert states.keys() == {"_model_opt._opt"}
    assert states["_model_opt._opt"]["param_groups"][0]["lr"] == 1e-3


def test_reference_checkpoint_handles_cycles():
    agent = _Agent()
    agent.cycle = agent

    states = collect_optimizer_state_dicts(agent)

    assert states.keys() == {"_model_opt._opt"}


def test_reference_checkpoint_handles_torch_compiled_modules():
    compiled = torch.compile(_Agent())
    root = torch.nn.Module()
    root._wm = compiled

    states = collect_optimizer_state_dicts(root)

    assert len(states) == 1
    assert next(iter(states)).endswith("._model_opt._opt")


def test_reference_checkpoint_restores_optimizer_state():
    source = _Agent()
    source._model_opt._opt.param_groups[0]["lr"] = 4e-5
    states = collect_optimizer_state_dicts(source)
    target = _Agent()

    load_optimizer_state_dicts(target, states)

    assert target._model_opt._opt.param_groups[0]["lr"] == 4e-5


def test_reference_checkpoint_is_atomic_and_validated(tmp_path):
    contract = {"reward_contract": "robobo-push-sparse-v1"}
    path = tmp_path / "latest.pt"
    checkpoint = {
        "checkpoint_version": REFERENCE_CHECKPOINT_VERSION,
        "agent_state_dict": {"weight": torch.ones(())},
        "optims_state_dict": {},
        "training_step": 4000,
        "reward_contract": contract,
        "wandb_run_id": "run-id",
    }

    save_checkpoint_atomic(checkpoint, path)
    loaded = torch.load(path, weights_only=False)
    validate_checkpoint(
        loaded,
        reward_contract=contract,
        replay_step=4000,
    )

    assert path.exists()
    assert not (tmp_path / ".latest.pt.tmp").exists()


def test_reference_checkpoint_rejects_replay_step_mismatch():
    checkpoint = {
        "checkpoint_version": REFERENCE_CHECKPOINT_VERSION,
        "agent_state_dict": {},
        "optims_state_dict": {},
        "training_step": 4000,
        "reward_contract": {"reward_contract": "robobo-push-sparse-v1"},
    }

    with pytest.raises(ValueError, match="checkpoint/replay step mismatch"):
        validate_checkpoint(
            checkpoint,
            reward_contract=checkpoint["reward_contract"],
            replay_step=3999,
        )


def test_reference_checkpoint_allows_one_partial_episode_of_replay_lag():
    checkpoint = {
        "checkpoint_version": REFERENCE_CHECKPOINT_VERSION,
        "agent_state_dict": {},
        "optims_state_dict": {},
        "training_step": 4000,
        "reward_contract": {"reward_contract": "robobo-push-sparse-v1"},
    }

    validate_checkpoint(
        checkpoint,
        reward_contract=checkpoint["reward_contract"],
        replay_step=3801,
        max_replay_lag=199,
    )
    with pytest.raises(ValueError, match="checkpoint/replay step mismatch"):
        validate_checkpoint(
            checkpoint,
            reward_contract=checkpoint["reward_contract"],
            replay_step=3800,
            max_replay_lag=199,
        )


def test_reference_checkpoint_allows_replay_ahead_within_training_block():
    checkpoint = {
        "checkpoint_version": REFERENCE_CHECKPOINT_VERSION,
        "agent_state_dict": {},
        "optims_state_dict": {},
        "training_step": 25_000,
        "reward_contract": {"reward_contract": "robobo-push-sparse-v1"},
    }

    validate_checkpoint(
        checkpoint,
        reward_contract=checkpoint["reward_contract"],
        replay_step=26_595,
        max_replay_lag=199,
        max_replay_lead=5_000,
    )


def test_reference_checkpoint_rejects_replay_ahead_beyond_training_block():
    checkpoint = {
        "checkpoint_version": REFERENCE_CHECKPOINT_VERSION,
        "agent_state_dict": {},
        "optims_state_dict": {},
        "training_step": 25_000,
        "reward_contract": {"reward_contract": "robobo-push-sparse-v1"},
    }

    with pytest.raises(ValueError, match="allowed replay lead is 5000"):
        validate_checkpoint(
            checkpoint,
            reward_contract=checkpoint["reward_contract"],
            replay_step=30_001,
            max_replay_lag=199,
            max_replay_lead=5_000,
        )
