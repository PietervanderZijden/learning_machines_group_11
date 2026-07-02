from deploy_hardware import exception_summary, sensor_acquisition_seconds


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
