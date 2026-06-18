# Robobo transfer contract

All new checkpoints use a fixed 400 ms simulation/policy interval and a
5 ms internal dynamics timestep. CoppeliaSim performs 80 physics substeps
inside each policy transition; Python issues one synchronized simulator step
per transition. Observations and rewards follow the
`robobo-obs-v2`/`robobo-reward-v4` contracts. Legacy checkpoints without a
`manifest.json` are intentionally rejected by hardware deployment.
The default episode remains 60 seconds, now represented by 150 decisions.
The camera is commanded to the ground-facing tilt value `100` before the
first observation, and that value is part of checkpoint compatibility.

## Calibration

Create separate simulation and hardware profiles:

```bash
python -m learning_machines.calibrate_ir \
  --output config/calibration/hardware.json \
  --raw-output calibration_runs/hardware.csv
```

Collect open-space and each prompted wall-distance phase for every sensor.
Repeat representative rollouts across at least ten combinations of battery,
surface, lighting, and start orientation before final training.

Each normalized sensor is:

```text
clip(polarity * (raw - free_space) /
     (polarity * (near_obstacle - free_space)), 0, 1) ** exponent
```

## Training

SAC receives only the 12 deployable blob+IR values. Dreamer receives calibrated
IR alongside its image input. SAC replay stores the policy-requested action,
because smoothing and safety are environment dynamics; it separately logs the
executed action. Dreamer sequence data stores executed actions for world-model
prediction.

SAC uses bounded potential-difference shaping from blob alignment and apparent
proximity, scales training rewards by `0.01`, caps the completion bonus, uses a
fixed entropy coefficient, and clips actor/critic gradients. The raw task
reward remains available in logs. HER is not used because food annotations and
food count are not policy inputs.

```bash
python train_sac.py --calibration config/calibration/simulation.json
python train_dreamerv3.py --calibration config/calibration/simulation.json
python train_dreamerv4_image.py --online --calibration config/calibration/simulation.json
```

Offline DreamerV4 training streams images from disk and retains encoded
latents on CPU instead of allocating the complete dataset on the GPU. The
default launcher uses the DreamerV3 recordings and 50,000 dynamics updates.

When hardware episodes become available:

```bash
HARDWARE_RECORD_DIR=recorded-states/hardware \
./run_dreamerv4.sh \
  --hardware-calibration config/calibration/hardware.json \
  --hardware-sample-ratio 0.25 \
  --hardware-validation-fraction 0.2
```

Legacy or corrupt episodes are reported and skipped. `--resume` tracks the
tokenizer, dynamics, MTP, and PMPO phases independently. Extending an upstream
phase resets its dependent downstream phase counters.

Use a measured hardware-profile name for final injected-profile training so the
resulting checkpoint manifest matches deployment.

W&B tracks rollout reward components, elapsed time, food/minute, collisions,
safety interventions, requested/executed actions, saturation, calibrated and
raw IR distributions, camera frames, tokenizer/world-model reconstructions,
reward targets and predictions, actor statistics, gradient norms, return
percentiles, and raw/unclamped KL statistics. Local TensorBoard event files are
disabled by default; pass `--tensorboard` only when they are specifically
needed.

## Simulation validation and evaluation

```bash
python validate_simulation.py --port 23000 --steps 2000 --task-events

python evaluate_transfer.py \
  --algorithm sac \
  --checkpoint sac_models/sac_latest.zip \
  --domain fixed
```

Run evaluation separately for `fixed`, `training`, `heldout`, and
`calibration`. Results include completion rate, completion time, food/minute,
collisions, safety overrides, action changes, and saturation, and can be sent
to W&B with `--wandb-project`.

## Hardware deployment

Before deploying a policy, run the read-only hardware diagnostics:

```bash
./run_hardware_validation.sh
```

The launcher requires `ROS_MASTER_URI` and `ROS_IP` (or `ROS_HOSTNAME`) to be
set by `scripts/setup.bash` or the container environment. It checks ROS
services, IR sensors, camera, phone pose, IMU, wheel encoders, and battery
readings, then saves a timestamped report and camera frame under
`hardware_logs/diagnostics/`. It does not move the robot by default.

Create the measured IR profile while the robot remains stationary:

```bash
./run_hardware_validation.sh --calibrate-ir
```

Follow every placement prompt; obstacles must exercise all eight sensor
directions. Inspect `ir_calibration_summary.json` for weak sensor spans before
using the generated `hardware.json`.

Phone tilt actuation is opt-in:

```bash
./run_hardware_validation.sh --test-tilt --tilt-position 100
```

Only test motors with the robot physically raised and every wheel clear:

```bash
./run_hardware_validation.sh --test-wheels --wheels-raised
```

The wheel test requires typing `WHEELS RAISED`, caps speed at 20, caps each
command at 500 ms, and sends stop commands before, between, and after tests.

```bash
./run_hardware_deploy.sh \
  sac \
  sac_models/sac_latest.zip \
  config/calibration/hardware.json
```

During rollout, enter `f` to annotate a food event and `e` or `q` for emergency
stop. Logs contain aligned T+1 observations and T requested/executed actions,
rewards, dones, timing, safety, and food annotations.

Roll out in stages: wheels raised, empty arena, obstacle-only arena, then food
collection. Promote only checkpoints with no watchdog failures and acceptable
collision/safety metrics.
