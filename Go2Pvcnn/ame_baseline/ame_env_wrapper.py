"""RSL-RL VecEnv adapter for AME map/state observation groups."""

from __future__ import annotations

import gymnasium as gym
import torch

from rsl_rl.env import VecEnv


class AmeRslRlEnvWrapper(VecEnv):
    def __init__(self, env, clip_actions: float | None = 100.0):
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = gym.spaces.flatdim(self.unwrapped.single_action_space)
        self.env.reset()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def cfg(self):
        return self.unwrapped.cfg

    @property
    def episode_length_buf(self):
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        self.unwrapped.episode_length_buf = value

    @staticmethod
    def _flatten(obs_dict: dict[str, torch.Tensor], names: tuple[str, ...]) -> torch.Tensor:
        values = [obs_dict[name].reshape(obs_dict[name].shape[0], -1) for name in names]
        return torch.cat(values, dim=-1)

    def _format_observations(self, obs_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        policy = self._flatten(
            obs_dict, ("policy_elevation_semantic_map", "policy_state")
        )
        critic = self._flatten(
            obs_dict, ("critic_elevation_semantic_map", "critic_state")
        )
        return policy, {"observations": {"critic": critic}}

    def get_observations(self):
        return self._format_observations(self.unwrapped.observation_manager.compute())

    def reset(self):
        obs_dict, _ = self.env.reset()
        return self._format_observations(obs_dict)

    def step(self, actions):
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        obs_dict, rewards, terminated, truncated, extras = self.env.step(actions)
        policy, obs_extras = self._format_observations(obs_dict)
        extras["time_outs"] = truncated
        extras["observations"] = obs_extras["observations"]
        return policy, rewards, (terminated | truncated).to(dtype=torch.long), extras

    def close(self):
        return self.env.close()


__all__ = ["AmeRslRlEnvWrapper"]
