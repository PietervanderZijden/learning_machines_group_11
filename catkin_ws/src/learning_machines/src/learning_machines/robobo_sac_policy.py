from __future__ import annotations

import gymnasium as gym
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class RoboboCombinedExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for SAC with MultiInputPolicy.

    Inputs:
        observations["image"]:
            Shape: (batch, 1, 84, 84)
        observations["ir"]:
            Shape: (batch, 8)

    Output:
        One combined feature vector used by the SAC actor and critic.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_features_dim: int = 128,
        ir_features_dim: int = 32,
        combined_features_dim: int = 256,
    ) -> None:
        super().__init__(observation_space, features_dim=combined_features_dim)

        image_space = observation_space.spaces["image"]
        ir_space = observation_space.spaces["ir"]

        n_input_channels = image_space.shape[0]
        n_ir_features = ir_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample_image = th.as_tensor(image_space.sample()[None]).float()
            n_flatten = self.cnn(sample_image).shape[1]

        self.image_net = nn.Sequential(
            nn.Linear(n_flatten, cnn_features_dim),
            nn.ReLU(),
        )

        self.ir_net = nn.Sequential(
            nn.Linear(n_ir_features, 64),
            nn.ReLU(),
            nn.Linear(64, ir_features_dim),
            nn.ReLU(),
        )

        self.combined_net = nn.Sequential(
            nn.Linear(cnn_features_dim + ir_features_dim, combined_features_dim),
            nn.ReLU(),
        )

        self._features_dim = combined_features_dim

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        image_features = self.image_net(self.cnn(observations["image"]))
        ir_features = self.ir_net(observations["ir"])

        features = th.cat([image_features, ir_features], dim=1)

        return self.combined_net(features)
