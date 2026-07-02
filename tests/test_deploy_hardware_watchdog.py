import numpy as np

from deploy_hardware import (
    SACPolicy,
    exception_summary,
    sensor_acquisition_seconds,
)


def test_sensor_watchdog_excludes_blocking_wheel_duration():
    timing = {
        "wheel_step": 0.82,
        "observation": 0.09,
        "image_read": 0.08,
        "ir_read": 0.001,
    }

    assert sensor_acquisition_seconds(timing, fallback=0.91) == 0.09


def test_sensor_watchdog_uses_safe_fallback_for_invalid_timing():
    assert sensor_acquisition_seconds({}, fallback=1.2) == 1.2
    assert sensor_acquisition_seconds(
        {"observation": float("nan")}, fallback=1.2
    ) == 1.2


def test_exception_summary_preserves_message_on_one_line():
    error = RuntimeError("service failed\nconnection reset")
    assert exception_summary(error) == (
        "RuntimeError: service failed connection reset"
    )


def test_push_sac_policy_builds_eighteen_value_observation():
    """Flatten push observations with previous executed actions."""

    class Model:
        """Capture policy input vectors."""

        def predict(self, observation, deterministic):
            """Return a fixed action for one observation."""
            self.observation = observation
            return np.zeros(2, dtype=np.float32), None

    policy = object.__new__(SACPolicy)
    policy.model = Model()
    policy.observation_dim = 18
    policy.previous_executed = np.array([0.25, -0.25], dtype=np.float32)
    policy.act(
        {
            "red_block": np.zeros(4, dtype=np.float32),
            "green_goal": np.ones(4, dtype=np.float32),
            "ir": np.full(8, 0.5, dtype=np.float32),
        }
    )
    assert policy.model.observation.shape == (18,)
    np.testing.assert_allclose(
        policy.model.observation[-2:], policy.previous_executed
    )
