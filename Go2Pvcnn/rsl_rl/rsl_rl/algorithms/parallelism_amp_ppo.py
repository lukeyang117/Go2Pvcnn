"""PPO with a masked AMP return channel."""

from __future__ import annotations

import torch
from torch import nn

from rsl_rl.storage import ParallelismAMPStorage, combine_advantages
from .ppo import PPO


class ParallelismAMPPPO(PPO):
    def __init__(self, actor_critic, amp_reward_weight=0.1, amp_value_loss_coef=1.0, **kwargs):
        super().__init__(actor_critic, **kwargs)
        self.amp_reward_weight = float(amp_reward_weight)
        self.amp_value_loss_coef = float(amp_value_loss_coef)
        self.amp_discriminator = None
        self._amp_active = torch.zeros(1, device=self.device)
        self._history_ratio = torch.zeros(1, device=self.device)

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = ParallelismAMPStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def act(self, obs, critic_obs, env=None, amp_context=None):
        if amp_context is not None:
            self._amp_active, self._history_ratio = amp_context
        active = self._amp_active.expand(obs.shape[0]) if self._amp_active.numel() == 1 else self._amp_active
        ratio = self._history_ratio.expand(obs.shape[0]) if self._history_ratio.numel() == 1 else self._history_ratio
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.amp_value = self.actor_critic.evaluate_amp(critic_obs, active, ratio).detach()
        self.transition.amp_active = active.detach()
        self.transition.history_ratio = ratio.detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        amp_info = infos.get("amp", {}) if isinstance(infos, dict) else {}
        if amp_info:
            self.transition.amp_reward = torch.as_tensor(amp_info.get("amp_reward", 0.0), device=self.device).reshape(-1)
            self.transition.amp_active = torch.as_tensor(amp_info.get("amp_active", self.transition.amp_active), device=self.device).reshape(-1)
            self.transition.history_ratio = torch.as_tensor(amp_info.get("history_ratio", self.transition.history_ratio), device=self.device).reshape(-1)
            if "expert_window" in amp_info and "agent_window" in amp_info:
                self.transition.amp_expert_window = amp_info["expert_window"]
                self.transition.amp_agent_window = amp_info["agent_window"]
        super().process_env_step(rewards, dones, infos)

    def compute_returns(self, last_critic_obs):
        base_last = self.actor_critic.evaluate(last_critic_obs).detach()
        active = self._amp_active.expand(last_critic_obs.shape[0]) if self._amp_active.numel() == 1 else self._amp_active
        ratio = self._history_ratio.expand(last_critic_obs.shape[0]) if self._history_ratio.numel() == 1 else self._history_ratio
        amp_last = self.actor_critic.evaluate_amp(last_critic_obs, active, ratio).detach()
        self.storage.compute_returns(base_last, amp_last, self.gamma, self.lam, active)

    def update(self):
        storage = self.storage
        batch_size = storage.num_envs * storage.num_transitions_per_env
        observations = storage.observations.reshape(batch_size, -1)
        critic_obs = storage.privileged_observations.reshape(batch_size, -1) if storage.privileged_observations is not None else observations
        actions = storage.actions.reshape(batch_size, -1)
        old_log_prob = storage.actions_log_prob.reshape(batch_size, 1)
        old_values = storage.values.reshape(batch_size, 1)
        returns = storage.returns.reshape(batch_size, 1)
        advantages = storage.advantages.reshape(batch_size, 1)
        amp_advantages = storage.amp_advantages.reshape(batch_size, 1)
        amp_returns = storage.amp_returns.reshape(batch_size, 1)
        amp_values = storage.amp_values.reshape(batch_size, 1)
        active = storage.amp_active.reshape(batch_size, 1)
        ratio = storage.history_ratio.reshape(batch_size, 1)
        total_value, total_surrogate, total_amp_value = 0.0, 0.0, 0.0
        for _ in range(self.num_learning_epochs):
            self.actor_critic.act(observations)
            log_prob = self.actor_critic.get_actions_log_prob(actions).reshape(-1, 1)
            base_value = self.actor_critic.evaluate(critic_obs)
            amp_value = self.actor_critic.evaluate_amp(critic_obs, active, ratio)
            combined = combine_advantages(advantages, amp_advantages, active, self.amp_reward_weight)
            prob_ratio = torch.exp(log_prob - old_log_prob)
            surrogate = -combined * prob_ratio
            clipped = -combined * torch.clamp(prob_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, clipped).mean()
            value_clipped = old_values + (base_value - old_values).clamp(-self.clip_param, self.clip_param)
            base_loss = torch.max((base_value - returns).square(), (value_clipped - returns).square()).mean()
            amp_loss = ((amp_value - amp_returns).square() * active).sum() / active.sum().clamp_min(1.0)
            loss = surrogate_loss + self.value_loss_coef * base_loss + self.amp_value_loss_coef * amp_loss - self.entropy_coef * self.actor_critic.entropy.mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total_value += float(base_loss.detach())
            total_amp_value += float(amp_loss.detach())
            total_surrogate += float(surrogate_loss.detach())
        storage.clear()
        return {"value_loss": total_value / self.num_learning_epochs, "amp_value_loss": total_amp_value / self.num_learning_epochs, "surrogate_loss": total_surrogate / self.num_learning_epochs, "discriminator_loss": 0.0}
