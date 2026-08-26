"""Convolutional actor-critic encoder for the AoR observation tensor.

The architecture follows the "Controller Variants" paragraph: three 3x3
convolutions with 32, 64 and 64 channels (stride 1, padding 1) and ReLU
activations, a 2x2 max pool after the second and third convolution, then a
flatten and a linear projection into a 256-dimensional latent shared by the
actor and critic heads.  With the paper's 8 x 64 x 32 input this is ~2.2M
parameters.

Convolutions are what let the policy exploit the spatial correlation between
targets, sensor footprints and the partially observed cloud field without any
hand-crafted features.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

__all__ = ["AoRCNN", "count_parameters", "POLICY_KWARGS"]


class AoRCNN(BaseFeaturesExtractor):
    """CNN encoder over the multi-channel area-of-regard tensor.

    Parameters
    ----------
    observation_space:
        A ``Box`` of shape ``(C, H, W)``.
    features_dim:
        Width of the shared latent representation.
    channels:
        Output channels of the three convolutional layers.
    pool_after:
        Indices of convolutions followed by a 2x2 max pool.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        channels: Sequence[int] = (32, 64, 64),
        pool_after: Sequence[int] = (1, 2),
    ) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        layers: list[nn.Module] = []
        in_channels = n_input_channels
        for index, out_channels in enumerate(channels):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            layers.append(nn.ReLU())
            if index in pool_after:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)

        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def count_parameters(module: nn.Module) -> int:
    """Number of trainable parameters, for comparison with the paper's ~2.2M."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def policy_kwargs(features_dim: int = 256) -> dict:
    """``policy_kwargs`` for :class:`stable_baselines3.PPO`.

    ``net_arch=[]`` keeps the actor and critic as single linear heads on the
    shared 256-dimensional latent, as described in the paper.  For a
    ``MultiDiscrete`` action space Stable-Baselines3 emits one concatenated
    logit vector of width ``|A_move| + |A_sense|`` and splits it into two
    independently sampled categorical distributions.
    """
    return {
        "features_extractor_class": AoRCNN,
        "features_extractor_kwargs": {"features_dim": features_dim},
        "net_arch": [],
    }


#: Default policy keyword arguments.
POLICY_KWARGS = policy_kwargs()
