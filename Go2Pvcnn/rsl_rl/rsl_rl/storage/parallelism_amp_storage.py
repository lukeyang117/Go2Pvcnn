"""Rollout storage extensions for masked AMP returns."""

from __future__ import annotations

import torch
from torch import Tensor

from .rollout_storage import RolloutStorage


def combine_advantages(base_norm: Tensor, amp_norm: Tensor, active: Tensor, amp_weight: float) -> Tensor:
    mask = torch.as_tensor(active, device=base_norm.device, dtype=base_norm.dtype)
    return base_norm + float(amp_weight) * amp_norm * mask


class ParallelismAMPStorage(RolloutStorage):
    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device="cpu"):
        super().__init__(num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device)
        shape = (num_transitions_per_env, num_envs, 1)
        self.amp_rewards = torch.zeros(shape, device=device)
        self.amp_values = torch.zeros(shape, device=device)
        self.amp_active = torch.zeros(shape, device=device)
        self.history_ratio = torch.zeros(shape, device=device)
        self.amp_returns = torch.zeros(shape, device=device)
        self.amp_advantages = torch.zeros(shape, device=device)
        self.base_advantages = torch.zeros(shape, device=device)
        self.amp_expert_windows = None
        self.amp_agent_windows = None

    def add_transitions(self, transition: RolloutStorage.Transition):
        super().add_transitions(transition)
        index = self.step - 1
        for name, target in (("amp_reward", self.amp_rewards), ("amp_value", self.amp_values), ("amp_active", self.amp_active), ("history_ratio", self.history_ratio)):
            value = getattr(transition, name, None)
            if value is not None:
                target[index].copy_(torch.as_tensor(value, device=self.device).reshape(-1, 1))
        expert = getattr(transition, "amp_expert_window", None)
        agent = getattr(transition, "amp_agent_window", None)
        if expert is not None and agent is not None:
            if self.amp_expert_windows is None:
                self.amp_expert_windows = torch.zeros((self.num_transitions_per_env, self.num_envs, *expert.shape[1:]), device=self.device)
                self.amp_agent_windows = torch.zeros_like(self.amp_expert_windows)
            self.amp_expert_windows[index].copy_(expert)
            self.amp_agent_windows[index].copy_(agent)

    def compute_returns(self, last_base_values: Tensor, last_amp_values: Tensor, gamma: float, lam: float, last_amp_active: Tensor | None = None):
        last_base_values = torch.as_tensor(last_base_values, device=self.device).reshape(self.num_envs, 1)
        last_amp_values = torch.as_tensor(last_amp_values, device=self.device).reshape(self.num_envs, 1)
        advantage = torch.zeros_like(last_base_values)
        amp_advantage = torch.zeros_like(last_amp_values)
        terminal_amp_mask = self.amp_active[-1] if last_amp_active is None else torch.as_tensor(last_amp_active, device=self.device).reshape(self.num_envs, 1)
        for step in reversed(range(self.num_transitions_per_env)):
            next_base = last_base_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
            next_amp = last_amp_values if step == self.num_transitions_per_env - 1 else self.amp_values[step + 1]
            next_amp_mask = terminal_amp_mask if step == self.num_transitions_per_env - 1 else self.amp_active[step + 1]
            nonterminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + gamma * nonterminal * next_base - self.values[step]
            advantage = delta + gamma * lam * nonterminal * advantage
            self.returns[step] = advantage + self.values[step]
            active = self.amp_active[step]
            amp_delta = active * (self.amp_rewards[step] + gamma * nonterminal * next_amp_mask * next_amp - self.amp_values[step])
            amp_advantage = amp_delta + gamma * lam * nonterminal * active * next_amp_mask * amp_advantage
            self.amp_advantages[step] = amp_advantage
            self.amp_returns[step] = amp_advantage + self.amp_values[step]
        self.base_advantages = self.returns - self.values
        self.advantages = (self.base_advantages - self.base_advantages.mean()) / (self.base_advantages.std() + 1.0e-8)
        active = self.amp_active.bool()
        if bool(active.any()):
            values = self.amp_advantages[active]
            # Population statistics remain defined when only one environment
            # has a valid AMP window in the rollout.
            mean, std = values.mean(), values.std(unbiased=False)
            self.amp_advantages = torch.where(active, (self.amp_advantages - mean) / (std + 1.0e-8), torch.zeros_like(self.amp_advantages))
        else:
            self.amp_advantages.zero_()
