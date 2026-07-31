"""Minimal vectorized environment interface for the local RSL-RL runner."""

from __future__ import annotations

from abc import ABC, abstractmethod


class VecEnv(ABC):
    """Interface expected by the local ``OnPolicyRunner``."""

    num_envs: int
    num_actions: int
    max_episode_length: int
    device: str

    @abstractmethod
    def get_observations(self):
        """Return policy observations and extras."""

    @abstractmethod
    def step(self, actions):
        """Step the vectorized environment."""

    def reset(self):
        """Reset the environment if the wrapper exposes reset."""
        return None
