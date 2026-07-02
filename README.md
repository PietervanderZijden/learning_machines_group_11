# Robobo learning machines

## Available tasks and algorithms

| Task | Algorithms |
| --- | --- |
| Approach and evade | Reactive, DDPG, SAC |
| Food collection | SAC, DreamerV3, DreamerV4-full |
| Pushing | SAC, DreamerV3 |

## Setup

```sh
uv sync
./scripts/start_coppelia_sim.sh ./scenes/arena_approach.ttt
```

Use `python robobo.py --help` and append `--help` after a command for its
options. Arguments after the selected task and algorithm are forwarded to the
underlying command.

## Training

```sh
python robobo.py train food sac --total-timesteps 500000
python robobo.py train food dreamerv3 --total-steps 500000
python robobo.py train food dreamerv4-full --total-steps 500000
python robobo.py train push sac --total-timesteps 500000
python robobo.py train push dreamerv3 --total-steps 500000
python robobo.py train evade ddpg
python robobo.py train evade sac
```

## Evaluation and validation

```sh
python robobo.py evaluate --algorithm sac --checkpoint PATH
python robobo.py validate simulation --steps 2000
python robobo.py validate hardware
python robobo.py run evade ddpg
python robobo.py run evade sac
```

## Hardware deployment

```sh
python robobo.py deploy food sac --checkpoint PATH --calibration config/calibration/hardware.json
python robobo.py deploy food dreamerv3 --checkpoint PATH --calibration config/calibration/hardware.json
python robobo.py deploy food dreamerv4-full --checkpoint PATH --calibration config/calibration/hardware.json
python robobo.py deploy push dreamerv3 --checkpoint PATH
```
