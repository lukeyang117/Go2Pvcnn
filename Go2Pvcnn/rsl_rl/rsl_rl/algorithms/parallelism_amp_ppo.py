"""PPO with a masked AMP return channel."""

from __future__ import annotations

import torch
from torch import nn

from rsl_rl.storage import ParallelismAMPStorage, combine_advantages
from rsl_rl.modules import AMPDiscriminator
from .ppo import PPO


class ParallelismAMPPPO(PPO):
    def __init__(
        self,
        actor_critic,
        amp_reward_weight=0.1,
        amp_value_loss_coef=1.0,
        amp_window_frames=24,
        amp_warmup_iterations=500,
        amp_weight_ramp_iterations=100,
        disc_learning_rate=1.0e-4,
        disc_epochs=2,
        disc_batch_size=4096,
        disc_replay_capacity=32768,
        **kwargs,
    ):
        super().__init__(actor_critic, **kwargs)
        self.amp_reward_weight = float(amp_reward_weight)
        self.amp_value_loss_coef = float(amp_value_loss_coef)
        self.amp_window_frames = int(amp_window_frames)
        self.amp_warmup_iterations = max(int(amp_warmup_iterations), 0)
        self.amp_weight_ramp_iterations = max(int(amp_weight_ramp_iterations), 0)
        self._iteration = 0
        self._actor_amp_reward_weight = 0.0
        self.disc_epochs = int(disc_epochs)
        self.disc_batch_size = int(disc_batch_size)
        self.disc_replay_capacity = int(disc_replay_capacity)
        self.amp_discriminator = AMPDiscriminator(learning_rate=float(disc_learning_rate)).to(self.device)
        self._amp_active = torch.zeros(1, device=self.device)
        self._history_ratio = torch.zeros(1, device=self.device)
        self._amp_replay_expert = None
        self._amp_replay_agent = None

    @property
    def actor_amp_reward_weight(self) -> float:
        """Current AMP advantage weight used by the actor update."""

        return self._actor_amp_reward_weight

    def set_iteration(self, iteration: int, *_args, **_kwargs) -> None:
        """Update the global AMP-to-actor schedule, including resumed runs."""

        self._iteration = max(int(iteration), 0)
        if self._iteration < self.amp_warmup_iterations:
            self._actor_amp_reward_weight = 0.0
            return
        if self.amp_weight_ramp_iterations == 0:
            self._actor_amp_reward_weight = self.amp_reward_weight
            return
        progress = min(
            1.0,
            (self._iteration - self.amp_warmup_iterations)
            / float(self.amp_weight_ramp_iterations),
        )
        self._actor_amp_reward_weight = self.amp_reward_weight * progress

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
            active = torch.as_tensor(
                amp_info.get("amp_active", self._amp_active),
                device=self.device,
                dtype=torch.bool,
            ).reshape(-1)
            ratio = torch.as_tensor(
                amp_info.get("history_ratio", self._history_ratio),
                device=self.device,
                dtype=torch.float32,
            ).reshape(-1)
            self._amp_active = active.detach()
            self._history_ratio = ratio.detach()
            self.transition.amp_active = active
            self.transition.history_ratio = ratio
            if "expert_window" in amp_info and "agent_window" in amp_info:
                self.transition.amp_expert_window = amp_info["expert_window"]
                self.transition.amp_agent_window = amp_info["agent_window"]
                self.transition.amp_reward = self.amp_discriminator.reward(
                    self.transition.amp_agent_window, active
                )
            else:
                self.transition.amp_reward = torch.zeros_like(ratio)
        else:
            self.transition.amp_reward = torch.zeros_like(torch.as_tensor(rewards, device=self.device).reshape(-1).float())
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
        expert_windows = storage.amp_expert_windows
        agent_windows = storage.amp_agent_windows
        if expert_windows is not None and agent_windows is not None:
            flat_active = storage.amp_active.reshape(batch_size).bool()
            expert_flat = expert_windows.reshape(batch_size, -1)
            agent_flat = agent_windows.reshape(batch_size, -1)
            if bool(flat_active.any()):
                self._append_amp_replay(expert_flat[flat_active], agent_flat[flat_active])
        total_value, total_surrogate, total_amp_value, updates = 0.0, 0.0, 0.0, 0
        for _ in range(self.num_learning_epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            mini_batch_size = max(1, batch_size // self.num_mini_batches)
            for start in range(0, batch_size, mini_batch_size):
                indices = permutation[start : min(start + mini_batch_size, batch_size)]
                obs_mb, critic_mb, actions_mb = observations[indices], critic_obs[indices], actions[indices]
                old_log_mb, old_values_mb = old_log_prob[indices], old_values[indices]
                returns_mb, advantages_mb = returns[indices], advantages[indices]
                amp_adv_mb, amp_returns_mb = amp_advantages[indices], amp_returns[indices]
                active_mb, ratio_mb = active[indices], ratio[indices]
                self.actor_critic.act(obs_mb)
                log_prob = self.actor_critic.get_actions_log_prob(actions_mb).reshape(-1, 1)
                base_value = self.actor_critic.evaluate(critic_mb)
                amp_value = self.actor_critic.evaluate_amp(critic_mb, active_mb, ratio_mb)
                combined = combine_advantages(
                    advantages_mb,
                    amp_adv_mb,
                    active_mb,
                    self.actor_amp_reward_weight,
                )
                prob_ratio = torch.exp(log_prob - old_log_mb)
                surrogate = -combined * prob_ratio
                clipped = -combined * torch.clamp(prob_ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, clipped).mean()
                value_clipped = old_values_mb + (base_value - old_values_mb).clamp(-self.clip_param, self.clip_param)
                base_loss = torch.max((base_value - returns_mb).square(), (value_clipped - returns_mb).square()).mean()
                amp_loss = ((amp_value - amp_returns_mb).square() * active_mb).sum() / active_mb.sum().clamp_min(1.0)
                loss = surrogate_loss + self.value_loss_coef * base_loss + self.amp_value_loss_coef * amp_loss - self.entropy_coef * self.actor_critic.entropy.mean()
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                total_value += float(base_loss.detach())
                total_amp_value += float(amp_loss.detach())
                total_surrogate += float(surrogate_loss.detach())
                updates += 1
        discriminator_metrics = self._update_discriminator()
        storage.clear()
        denominator = max(1, updates)
        return {
            "value_loss": total_value / denominator,
            "amp_value_loss": total_amp_value / denominator,
            "surrogate_loss": total_surrogate / denominator,
            "amp_actor_reward_weight": self.actor_amp_reward_weight,
            **discriminator_metrics,
        }

    def _append_amp_replay(self, expert: torch.Tensor, agent: torch.Tensor) -> None:
        """Append valid windows to a bounded device-local replay queue."""

        expert = expert.detach()
        agent = agent.detach()
        if self._amp_replay_expert is None:
            self._amp_replay_expert, self._amp_replay_agent = expert, agent
        else:
            self._amp_replay_expert = torch.cat((self._amp_replay_expert, expert), dim=0)
            self._amp_replay_agent = torch.cat((self._amp_replay_agent, agent), dim=0)
        if self._amp_replay_expert.shape[0] > self.disc_replay_capacity:
            self._amp_replay_expert = self._amp_replay_expert[-self.disc_replay_capacity :]
            self._amp_replay_agent = self._amp_replay_agent[-self.disc_replay_capacity :]

    def _update_discriminator(self) -> dict[str, float]:
        if self._amp_replay_expert is None or self._amp_replay_expert.shape[0] == 0:
            return {
                "discriminator_loss": 0.0,
                "discriminator_gradient_penalty": 0.0,
                "discriminator_expert_accuracy": 0.0,
                "discriminator_agent_accuracy": 0.0,
            }
        expert, agent = self._amp_replay_expert, self._amp_replay_agent
        count = expert.shape[0]
        totals = {"loss": 0.0, "gradient_penalty": 0.0, "expert_accuracy": 0.0, "agent_accuracy": 0.0}
        updates = 0
        for _ in range(max(1, self.disc_epochs)):
            order = torch.randperm(count, device=self.device)
            for start in range(0, count, max(1, self.disc_batch_size)):
                idx = order[start : start + max(1, self.disc_batch_size)]
                metrics = self.amp_discriminator.update(
                    expert.index_select(0, idx),
                    agent.index_select(0, idx),
                    torch.ones(idx.numel(), dtype=torch.bool, device=self.device),
                )
                for key in totals:
                    totals[key] += float(metrics[key])
                updates += 1
        denominator = max(1, updates)
        return {
            "discriminator_loss": totals["loss"] / denominator,
            "discriminator_gradient_penalty": totals["gradient_penalty"] / denominator,
            "discriminator_expert_accuracy": totals["expert_accuracy"] / denominator,
            "discriminator_agent_accuracy": totals["agent_accuracy"] / denominator,
        }
