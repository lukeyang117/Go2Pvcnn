#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch

from rsl_rl.utils import split_and_pad_trajectories


class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.privileged_actions = None
            self.ppo_active = None
            self.imitation_weight = None
            self.plan_valid = None
            # For PVCNN semantic supervision
            self.point_cloud = None
            self.semantic_labels = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, privileged_obs_shape, actions_shape, device="cpu"):
        self.device = device

        self.obs_shape = obs_shape

        # Flag for training with transitions (used in mini_batch_generator)
        self.train_with_transitions = False
        self.privileged_obs_shape = privileged_obs_shape
        self.actions_shape = actions_shape

        # Core
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        if privileged_obs_shape[0] is not None:
            self.privileged_observations = torch.zeros(
                num_transitions_per_env, num_envs, *privileged_obs_shape, device=self.device
            )
        else:
            self.privileged_observations = None
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.privileged_actions = torch.zeros(
            num_transitions_per_env, num_envs, *actions_shape, device=self.device
        )
        # Teacher-controlled samples are excluded from the student PPO actor loss.
        # Default to active so legacy PPO callers keep their original behavior.
        self.ppo_active_masks = torch.ones(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.imitation_weights = torch.ones(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.plan_valid_masks = torch.ones(
            num_transitions_per_env, num_envs, 1, device=self.device
        )
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # For PPO
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        # For PVCNN semantic supervision
        self.point_clouds = None  # Will be initialized when first point cloud is received
        self.semantic_labels = None  # Will be initialized when first labels are received
        self.has_semantic_labels = False

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            privileged_obs = transition.critic_observations
            if privileged_obs is None:
                privileged_obs = getattr(transition, "privileged_observations", None)
            self.privileged_observations[self.step].copy_(privileged_obs)
        self.actions[self.step].copy_(transition.actions)
        if transition.privileged_actions is not None:
            self.privileged_actions[self.step].copy_(transition.privileged_actions)
        if transition.ppo_active is not None:
            self.ppo_active_masks[self.step].copy_(
                transition.ppo_active.to(device=self.device).view(-1, 1)
            )
        else:
            self.ppo_active_masks[self.step].fill_(1.0)
        if transition.imitation_weight is not None:
            self.imitation_weights[self.step].copy_(
                transition.imitation_weight.to(device=self.device).view(-1, 1)
            )
        else:
            self.imitation_weights[self.step].fill_(1.0)
        if transition.plan_valid is not None:
            self.plan_valid_masks[self.step].copy_(
                transition.plan_valid.to(device=self.device).view(-1, 1)
            )
        else:
            self.plan_valid_masks[self.step].fill_(1.0)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        if transition.values is not None:
            self.values[self.step].copy_(transition.values)
        if transition.actions_log_prob is not None:
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        if transition.action_mean is not None:
            self.mu[self.step].copy_(transition.action_mean)
        if transition.action_sigma is not None:
            self.sigma[self.step].copy_(transition.action_sigma)
        self._save_hidden_states(transition.hidden_states)
        
        # Store point cloud and semantic labels if available
        if transition.point_cloud is not None: 
            # Initialize storage on first valid point cloud
            if self.point_clouds is None:
                pc_shape = transition.point_cloud.shape[1:]  # (num_points, 3)
                self.point_clouds = torch.zeros(
                    self.num_transitions_per_env, self.num_envs, *pc_shape, device=self.device
                )
            self.point_clouds[self.step].copy_(transition.point_cloud)
        
        if transition.semantic_labels is not None:
            # Initialize storage on first valid labels
            if self.semantic_labels is None:
                label_shape = transition.semantic_labels.shape[1:]  # (num_points,)
                self.semantic_labels = torch.zeros(
                    self.num_transitions_per_env, self.num_envs, *label_shape, 
                    dtype=torch.long, device=self.device
                )
                self.has_semantic_labels = True
            self.semantic_labels[self.step].copy_(transition.semantic_labels)
        
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0])
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(
        self,
        num_mini_batches,
        num_epochs=8,
        include_privileged_actions=False,
        include_ppo_mask=False,
        include_imitation_context=False,
    ):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations.flatten(0, 1)
        else:
            critic_observations = observations

        actions = self.actions.flatten(0, 1)
        privileged_actions = self.privileged_actions.flatten(0, 1)
        ppo_active_masks = self.ppo_active_masks.flatten(0, 1)
        imitation_weights = self.imitation_weights.flatten(0, 1)
        plan_valid_masks = self.plan_valid_masks.flatten(0, 1)
        values = self.values.flatten(0, 1)
        if self.train_with_transitions:
            transitions = self.transitions.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        # Flatten point clouds and semantic labels if available
        if self.point_clouds is not None:
            point_clouds = self.point_clouds.flatten(0, 1)
        else:
            point_clouds = None
        
        if self.semantic_labels is not None:
            semantic_labels = self.semantic_labels.flatten(0, 1)
        else:
            semantic_labels = None

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                critic_observations_batch = critic_observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_values_batch = values[batch_idx]
                returns_batch = returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                advantages_batch = advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                
                # Get point cloud and semantic labels batch
                point_cloud_batch = point_clouds[batch_idx] if point_clouds is not None else None
                semantic_labels_batch = semantic_labels[batch_idx] if semantic_labels is not None else None

                base_batch = (obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    None,
                    None,
                ), None, point_cloud_batch, semantic_labels_batch)
                if include_privileged_actions:
                    batch = base_batch + (privileged_actions[batch_idx],)
                else:
                    batch = base_batch
                if include_ppo_mask:
                    batch = batch + (ppo_active_masks[batch_idx],)
                if include_imitation_context:
                    batch = batch + (
                        imitation_weights[batch_idx],
                        plan_valid_masks[batch_idx],
                    )
                yield batch

    def distillation_generator(self, num_mini_batches, num_epochs=1):
        observations = self.observations.flatten(0, 1)
        if self.privileged_observations is not None:
            privileged_observations = self.privileged_observations.flatten(0, 1)
        else:
            privileged_observations = observations
        actions = self.actions.flatten(0, 1)
        privileged_actions = self.privileged_actions.flatten(0, 1)
        dones = self.dones.flatten(0, 1)
        batch_size = observations.shape[0]
        mini_batch_size = max(1, batch_size // num_mini_batches)
        indices = torch.randperm(batch_size, requires_grad=False, device=self.device)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = min((i + 1) * mini_batch_size, batch_size)
                batch_idx = indices[start:end]
                yield (
                    observations[batch_idx],
                    privileged_observations[batch_idx],
                    actions[batch_idx],
                    privileged_actions[batch_idx],
                    dones[batch_idx],
                )

    # for RNNs only
    def reccurent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None:
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else:
            padded_critic_obs_trajectories = padded_obs_trajectories

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]
                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                yield obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    hid_a_batch,
                    hid_c_batch,
                ), masks_batch

                first_traj = last_traj
