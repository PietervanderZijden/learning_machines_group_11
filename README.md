# Robobo learning machines

## Algorithms

| Task | Training | Running and evaluation | Hardware deployment |
| --- | --- | --- | --- |
| Approach/evade | DDPG, SAC | Reactive, DDPG, SAC | — |
| Food collection | SAC, DreamerV3, DreamerV4-full | SAC, DreamerV3, DreamerV4-full | SAC, DreamerV3, DreamerV4-full |
| Pushing | SAC, DreamerV3 | SAC, DreamerV3 | DreamerV3 |

## Setup

```sh
uv sync
```

```sh
./scripts/start_coppelia_sim.sh ./scenes/arena_obstacles.ttt 23000 -h
./scripts/start_coppelia_sim.sh ./scenes/arena_approach.ttt 23000 -h
./scripts/start_coppelia_sim.sh ./scenes/arena_push_easy.ttt 23000 -h
```

## Recording

```sh
python robobo.py record food --steps 50000 --output recorded_episodes
```

## Training

```sh
python robobo.py train approach-evade ddpg
python robobo.py train approach-evade sac
python robobo.py train food sac --total-timesteps 500000
python robobo.py train food dreamerv3 --total-steps 500000
python robobo.py train food dreamerv4-full --record-dir recorded_episodes --tokenizer-steps 10000 --world-steps 50000 --finetune-steps 10000 --imagination-steps 10000
python robobo.py train push sac --total-timesteps 500000
python robobo.py train push dreamerv3 --total-steps 500000
```

## Evaluation and running

```sh
python robobo.py evaluate food sac --checkpoint PATH
python robobo.py evaluate food dreamerv3 --checkpoint PATH
python robobo.py evaluate food dreamerv4-full --checkpoint PATH
python robobo.py evaluate push sac --checkpoint PATH
python robobo.py evaluate push dreamerv3 --checkpoint PATH
python robobo.py run approach-evade reactive
python robobo.py run approach-evade ddpg
python robobo.py run approach-evade sac
python robobo.py run food dreamerv3 --checkpoint PATH
python robobo.py run push dreamerv3 --checkpoint PATH
```

## Validation

```sh
python robobo.py validate simulation --steps 2000
python robobo.py validate hardware
```

## Hardware deployment

```sh
python robobo.py deploy food sac --checkpoint PATH --calibration config/calibration/hardware.json
python robobo.py deploy food dreamerv3 --checkpoint PATH --calibration config/calibration/hardware.json
python robobo.py deploy food dreamerv4-full --checkpoint PATH --calibration config/calibration/hardware.json
python robobo.py deploy push dreamerv3 --checkpoint PATH --calibration config/calibration/hardware.json
```
