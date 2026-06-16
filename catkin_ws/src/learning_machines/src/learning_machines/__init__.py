from .multi_robobo_env import (
    DomainRandomizationConfig,
    MultiRoboboObstacleAvoidanceEnv,
    RoboboDomainRandomizationWrapper,
)
from .rl_robobo_env import RoboboObstacleAvoidanceEnv, RoboboObstacleEnvConfig
from .robobo_sac_policy import RoboboCombinedExtractor
from .test_obstacle_avoidance import test
from .train_obstacle_avoidance import main

# 1. Voeg hier de import van jouw eigen functies toe:
from .train_ddpg import train_simple, test_simple

__all__ = (
    "main",
    "test",
    "RoboboObstacleEnvConfig",
    "RoboboObstacleAvoidanceEnv",
    "RoboboCombinedExtractor",
    "DomainRandomizationConfig",
    "MultiRoboboObstacleAvoidanceEnv",
    "RoboboDomainRandomizationWrapper",
    # 2. Voeg ze hier toe aan de export-lijst:
    "train_simple",
    "test_simple",
)