# Camera tilt reset fix for SAC / Dreamer V3 / Dreamer V4

**Date:** 2026-06-18
**Scope:** `RoboboCompactEnv.reset()` used by `train_sac.py`, `train_dreamerv3.py` and `train_dreamerv4_image.py`.

---

## 1. The error

When starting a new episode, all three training scripts crash at `env.reset()` with:

```text
Exception: 354: in sim.callScriptFunction: script is not initialized (or has already ended)
...
File ".../rl_robobo_compact_env.py", line 320, in _initialize_camera_pose
    raise RuntimeError("failed to command the Robobo phone tilt") from exc
RuntimeError: failed to command the Robobo phone tilt
```

The full traceback points to `SimulationRobobo.set_phone_tilt()` calling
`sim.callScriptFunction("moveTiltTo", self._tilt_motor_script, ...)`.

---

## 2. Root-cause analysis

`RoboboCompactEnv.reset()` restarts CoppeliaSim every episode via the following
flow (see `_reset_simulation()` in `rl_robobo_compact_env.py`):

1. `self.rob.stop_simulation()`
2. `self.rob.configure_simulation_timing()`
3. randomize food positions while stopped
4. `self.rob.play_simulation()`
5. `_fix_lifted_food()`
6. `_initialize_camera_pose()` → calls `set_phone_tilt()`

`SimulationRobobo.play_simulation()` only waits until `getSimulationState()` is
*running*:

```python
def play_simulation(self):
    self._sim.startSimulation()
    for _ in range(100):
        if self.is_running():
            return
        time.sleep(0.002)
```

It does **not** wait for child scripts to finish `sysCall_init`. CoppeliaSim
initializes child scripts on the first simulation step, but because the ZMQ
remote API is running in **stepping mode** (`sim.setStepping(True)`), no step
has happened yet after `play_simulation()` returns.

So when `_initialize_camera_pose()` immediately calls `set_phone_tilt()`, the
tilt-motor Lua script is not ready and the remote API throws:

```text
script is not initialized (or has already ended)
```

This is a race condition on every episode restart and therefore hits SAC,
Dreamer V3 and Dreamer V4 equally.

---

## 3. Possible fixes considered

| Option | What it does | Invasiveness | Speed impact |
|--------|--------------|--------------|--------------|
| **A. Add wall-clock sleeps in the env** | Retry `set_phone_tilt()` with `self.rob.sleep(...)` until the script initializes. | Minimal — only `rl_robobo_compact_env.py`. | 1 extra sim step per retry, plus the existing 1.5 s settle. |
| **B. Replace sleeps with explicit simulation steps (chosen)** | Retry by calling `self.rob.step_simulation(1)` until the script responds, then step instead of sleeping during tilt polling. | Minimal — only `rl_robobo_compact_env.py`. | Same wall-clock/simulation cost as A, but deterministic and no `.sleep` calls. |
| **C. Fix `SimulationRobobo.play_simulation()`** | Make `play_simulation()` step until scripts are ready before returning. | Medium — changes shared sim interface used by other envs. | Every caller pays the step, even if it does not need scripts yet. |
| **D. Direct joint position set while stopped** | Bypass the Lua script: get the tilt joint handle and call `sim.setJointPosition(...)` before `play_simulation()`. | High — requires joint-handle discovery + angle-unit calibration. | Fastest — zero extra simulation steps, but fragile and scene-specific. |

### Why option B was selected

- It is the **most non-invasive** way to fix the crash: only one file changes
  and the public robot interface (`SimulationRobobo`) is untouched.
- It is **as fast as option A** in stepping mode, because `self.rob.sleep()`
  already advances one 400 ms simulation step per call.
- It removes all `.sleep` calls from the camera-tilt logic, making the reset
  deterministic rather than wall-clock based.
- Option C is more invasive and does not speed things up. Option D is fastest
  but requires calibrating the tilt-script units to raw joint radians, which is
  error-prone without the CoppeliaSim scene file in version control.

---

## 4. Exact changes (option B)

**File changed:**

```
catkin_ws/src/learning_machines/src/learning_machines/rl_robobo_compact_env.py
```

### 4.1 New helper methods

Inserted after `_read_phone_tilt()` (now at lines 309–342):

```python
    def _settle(self, seconds: float) -> None:
        """Advance the simulation to let physics/scripts settle.

        In stepping mode this is equivalent to the requested amount of
        simulation time without relying on wall-clock sleep.
        """
        if self._is_simulation:
            steps = max(1, round(seconds / 0.4))
            self.rob.step_simulation(steps)
        else:
            self.rob.sleep(seconds)

    def _wait_for_tilt_script_ready(self, timeout: float) -> int | None:
        """Wait until the tilt motor script has initialized after a restart.

        CoppeliaSim child scripts initialize on the first simulation step, so
        this helper steps the simulation until read_phone_tilt() succeeds.
        """
        if not self._is_simulation:
            return self._read_phone_tilt()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                return self.rob.read_phone_tilt()
            except Exception as exc:
                msg = str(exc)
                if "script is not initialized" in msg or "has already ended" in msg:
                    self.rob.step_simulation(1)
                    continue
                raise
        raise RuntimeError(
            "Tilt motor script did not initialize after simulation restart"
        )
```

### 4.2 Modified `_initialize_camera_pose()`

Previous body (was lines 309–337):

```python
    def _initialize_camera_pose(self) -> None:
        """Point the camera at the arena before the first observation."""
        if not (self.config.return_image or self.config.detect_blob_from_camera):
            self.rob.sleep(self.config.reset_settle_time)
            return
        try:
            self.rob.set_phone_tilt(
                self.config.phone_tilt,
                self.config.phone_tilt_speed,
            )
        except Exception as exc:
            raise RuntimeError("failed to command the Robobo phone tilt") from exc

        deadline = time.monotonic() + self.config.phone_tilt_timeout
        actual = self._read_phone_tilt()
        while (
            actual is not None
            and abs(actual - self.config.phone_tilt) > self.config.phone_tilt_tolerance
            and time.monotonic() < deadline
        ):
            self.rob.sleep(0.1)
            actual = self._read_phone_tilt()

        if actual is not None and abs(actual - self.config.phone_tilt) > self.config.phone_tilt_tolerance:
            raise RuntimeError(
                f"phone tilt did not reach ground-facing target {self.config.phone_tilt}; "
                f"actual={actual}"
            )
        self.rob.sleep(self.config.reset_settle_time)
```

New body (now lines 344–376):

```python
    def _initialize_camera_pose(self) -> None:
        """Point the camera at the arena before the first observation."""
        if not (self.config.return_image or self.config.detect_blob_from_camera):
            self._settle(self.config.reset_settle_time)
            return

        actual = self._wait_for_tilt_script_ready(self.config.phone_tilt_timeout)
        try:
            self.rob.set_phone_tilt(
                self.config.phone_tilt,
                self.config.phone_tilt_speed,
            )
        except Exception as exc:
            raise RuntimeError("failed to command the Robobo phone tilt") from exc

        deadline = time.monotonic() + self.config.phone_tilt_timeout
        while (
            actual is not None
            and abs(actual - self.config.phone_tilt) > self.config.phone_tilt_tolerance
            and time.monotonic() < deadline
        ):
            if self._is_simulation:
                self.rob.step_simulation(1)
            else:
                self.rob.sleep(0.1)
            actual = self._read_phone_tilt()

        if actual is not None and abs(actual - self.config.phone_tilt) > self.config.phone_tilt_tolerance:
            raise RuntimeError(
                f"phone tilt did not reach ground-facing target {self.config.phone_tilt}; "
                f"actual={actual}"
            )
        self._settle(self.config.reset_settle_time)
```

### 4.3 Why this fixes the crash

1. `_wait_for_tilt_script_ready()` issues a harmless `read_phone_tilt()`.
2. The first attempt fails with the "script is not initialized" error.
3. The helper calls `self.rob.step_simulation(1)`, which executes one CoppeliaSim
   step and initializes all child scripts.
4. The next `read_phone_tilt()` succeeds, proving the tilt script is ready.
5. Only then does `set_phone_tilt()` run, so it no longer hits the race.

### 4.4 What stays the same

- The camera still tilts down to `config.phone_tilt` at the start of every
  episode.
- The tilt target/tolerance/timeout logic is unchanged.
- The `time.sleep(0.1)` inside `_reset_simulation()` (used while waiting for the
  simulation to fully stop) is untouched because it happens while CoppeliaSim is
  stopped and cannot be replaced by a simulation step.

---

## 5. Expected effect

- `train_sac.py`, `train_dreamerv3.py` and `train_dreamerv4_image.py` should no
  longer crash on `env.reset()`.
- Reset behaviour remains deterministic.
- No new sleeps are introduced on the simulation path.

---

## 6. Validation steps

1. Syntax check the modified file:
   ```bash
   python -m py_compile catkin_ws/src/learning_machines/src/learning_machines/rl_robobo_compact_env.py
   ```
2. Run the relevant unit tests (these use a fake robot and exercise the reset
   and camera-tilt logic):
   ```bash
   PYTHONPATH="catkin_ws/src/learning_machines/src:catkin_ws/src/robobo_interface/src" \
       uv run pytest tests/test_transfer_contract.py tests/test_dreamer_math.py -v
   ```
3. Run one of the affected training scripts far enough to survive several
   episode resets, e.g.:
   ```bash
   python train_sac.py --no-wandb  # or train_dreamerv3.py / train_dreamerv4_image.py
   ```
4. Confirm the first reset succeeds and the logged `info["phone_tilt"]` value
   moves toward the configured `phone_tilt`.
