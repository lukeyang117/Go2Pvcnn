from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import StudentTeacherCNN
from rsl_rl.storage import RolloutStorage


class HybridDistillationPPO:
    """Student PPO with a frozen teacher action target.

    Teacher and student are both evaluated on every step. A scheduled,
    batch-wide ratio selects which policy controls each environment. The
    teacher supplies imitation targets for every sample, while PPO actor
    terms use only samples actually controlled by the student.
    """

    policy: StudentTeacherCNN

    def __init__(
        self,
        policy,
        num_learning_epochs=3,
        num_mini_batches=4,
        learning_rate=3e-4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.001,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        ppo_coef=1.0,
        teacher_coef=0.5,
        teacher_coef_min=0.1,
        teacher_ratio_warmup_pct=0.30,
        teacher_coef_decay_end_pct=0.80,
        teacher_ratio_decay_end_pct=0.80,
        teacher_ratio_min=0.0,
        teacher_ratio_start=None,
        teacher_ratio_end=None,
        device="cpu",
        clip_min_std=1.0e-6,
        multi_gpu_cfg: dict | None = None,
    ):
        self.device = device
        self.policy = policy
        self.actor_critic = policy
        self.policy.to(self.device)

        self.is_multi_gpu = multi_gpu_cfg is not None
        self.gpu_world_size = (
            int(multi_gpu_cfg.get("world_size", 1)) if multi_gpu_cfg is not None else 1
        )
        self.distributed = self.is_multi_gpu
        self.dist = torch.distributed if self.distributed else None

        self.num_learning_epochs = int(num_learning_epochs)
        self.num_mini_batches = int(num_mini_batches)
        self.learning_rate = float(learning_rate)
        self.clip_param = float(clip_param)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.value_loss_coef = float(value_loss_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.use_clipped_value_loss = bool(use_clipped_value_loss)
        self.schedule = str(schedule)
        self.desired_kl = desired_kl
        self.ppo_coef = float(ppo_coef)
        self.teacher_coef = float(teacher_coef)
        self.teacher_coef_min = float(teacher_coef_min)
        self.teacher_ratio_warmup_pct = float(teacher_ratio_warmup_pct)
        self.teacher_coef_decay_end_pct = float(teacher_coef_decay_end_pct)
        self.teacher_ratio_decay_end_pct = float(teacher_ratio_decay_end_pct)
        self.teacher_ratio_min = float(teacher_ratio_min)
        self.teacher_ratio_start = (
            None if teacher_ratio_start is None else float(teacher_ratio_start)
        )
        self.teacher_ratio_end = (
            None if teacher_ratio_end is None else float(teacher_ratio_end)
        )
        self.clip_min_std = clip_min_std

        trainable_parameters = list(self.policy.student.parameters()) + list(
            self.policy.student_critic.parameters()
        )
        self.optimizer = optim.Adam(trainable_parameters, lr=self.learning_rate)
        self.storage = None
        self.transition = RolloutStorage.Transition()
        self.current_iteration = 0
        self.total_iterations = 1
        self.schedule_start_iteration = 0
        self.last_teacher_coef = self.teacher_coef
        self.last_teacher_ratio = 1.0
        self.last_student_ratio = 0.0
        self.last_teacher_action_share = 1.0
        self.last_ppo_active_ratio = 0.0
        self._teacher_control_mask = None
        self._needs_control_assignment = None
        self.num_updates = 0

    def init_storage(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        student_obs_shape,
        teacher_obs_shape,
        actions_shape,
    ):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            self.device,
        )

    def set_iteration(
        self,
        iteration: int,
        total_iterations: int,
        schedule_start_iteration: int | None = None,
    ) -> None:
        self.current_iteration = max(int(iteration), 0)
        self.total_iterations = max(int(total_iterations), 1)
        if schedule_start_iteration is not None:
            self.schedule_start_iteration = max(int(schedule_start_iteration), 0)

    def _compute_teacher_coef(self) -> float:
        # Keep the imitation-loss weight fixed. Only teacher_ratio controls
        # how many newly started episodes are assigned to the teacher.
        self.last_teacher_coef = float(self.teacher_coef)
        return self.last_teacher_coef

    def _compute_teacher_ratio(self) -> float:
        if self.teacher_ratio_start is not None or self.teacher_ratio_end is not None:
            if self.teacher_ratio_start is None or self.teacher_ratio_end is None:
                raise ValueError(
                    "teacher_ratio_start and teacher_ratio_end must be provided together"
                )
            start = min(max(self.teacher_ratio_start, 0.0), 1.0)
            end = min(max(self.teacher_ratio_end, 0.0), 1.0)
            segment_start = min(self.schedule_start_iteration, self.total_iterations)
            segment_length = max(self.total_iterations - segment_start, 1)
            progress = min(
                max(
                    (float(self.current_iteration) - float(segment_start))
                    / float(segment_length),
                    0.0,
                ),
                1.0,
            )
            return start + (end - start) * progress

        progress = min(
            max(float(self.current_iteration) / float(self.total_iterations), 0.0),
            1.0,
        )
        warmup = min(max(self.teacher_ratio_warmup_pct, 0.0), 1.0)
        decay_end = min(max(self.teacher_ratio_decay_end_pct, warmup), 1.0)
        min_ratio = min(max(self.teacher_ratio_min, 0.0), 1.0)
        if progress < warmup:
            return 1.0
        if progress < decay_end:
            ratio = 1.0 - (progress - warmup) / max(decay_end - warmup, 1.0e-6)
            return max(min_ratio, min(1.0, ratio))
        return min_ratio

    def _assign_control_sources(self, env_ids, teacher_ratio: float) -> None:
        """Assign a fixed controller to each newly started episode."""

        env_ids = env_ids.to(device=self.device, dtype=torch.long).flatten()
        if env_ids.numel() == 0:
            return
        teacher_count = int(round(env_ids.numel() * teacher_ratio))
        permutation = torch.randperm(env_ids.numel(), device=self.device)
        self._teacher_control_mask[env_ids] = False
        if teacher_count > 0:
            self._teacher_control_mask[env_ids[permutation[:teacher_count]]] = True
        self._needs_control_assignment[env_ids] = False

    def act(self, obs, teacher_obs, critic_obs=None, distillation_context=None):
        """Select teacher/student actions for the current environment batch."""

        if critic_obs is None:
            critic_obs = teacher_obs
        if distillation_context is None:
            imitation_weight = torch.ones(obs.shape[0], device=obs.device)
            plan_valid = torch.ones(obs.shape[0], device=obs.device)
        else:
            context = torch.as_tensor(distillation_context, device=obs.device)
            if context.ndim != 2 or context.shape[0] != obs.shape[0] or context.shape[1] < 2:
                raise ValueError(
                    "distillation_context must have shape [num_envs, 2] with multiplier and plan_valid"
                )
            imitation_weight = context[:, 0].float()
            plan_valid = context[:, 1].float()
        student_action = self.policy.student.act(obs)
        teacher_action = self.policy.evaluate(teacher_obs).detach()
        value = self.policy.evaluate_value(critic_obs).detach()

        teacher_ratio = self._compute_teacher_ratio()
        num_envs = obs.shape[0]
        if (
            self._teacher_control_mask is None
            or self._teacher_control_mask.numel() != num_envs
        ):
            self._teacher_control_mask = torch.zeros(
                num_envs, dtype=torch.bool, device=obs.device
            )
            self._needs_control_assignment = torch.ones(
                num_envs, dtype=torch.bool, device=obs.device
            )
            self._assign_control_sources(
                torch.arange(num_envs, device=obs.device), teacher_ratio
            )
        elif torch.any(self._needs_control_assignment):
            self._assign_control_sources(
                torch.nonzero(self._needs_control_assignment, as_tuple=False).flatten(),
                teacher_ratio,
            )

        teacher_mask = self._teacher_control_mask
        teacher_mask_action = teacher_mask.unsqueeze(-1)
        env_action = torch.where(teacher_mask_action, teacher_action, student_action)
        ppo_active = (~teacher_mask).to(student_action.dtype)

        self.transition.actions = env_action.detach()
        self.transition.values = value
        self.transition.actions_log_prob = self.policy.student.get_actions_log_prob(
            student_action
        ).detach()
        self.transition.action_mean = self.policy.student.action_mean.detach()
        self.transition.action_sigma = self.policy.student.action_std.detach()
        self.transition.privileged_actions = teacher_action
        self.transition.ppo_active = ppo_active
        self.transition.imitation_weight = imitation_weight.detach()
        self.transition.plan_valid = plan_valid.detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        self.last_teacher_ratio = teacher_ratio
        self.last_student_ratio = 1.0 - teacher_ratio
        self.last_teacher_action_share = float(teacher_mask.float().mean().item())
        self.last_ppo_active_ratio = float(ppo_active.mean().item())
        return env_action

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device),
                1,
            )
        self.storage.add_transitions(self.transition)
        if self._needs_control_assignment is not None:
            done_ids = torch.nonzero(dones.view(-1), as_tuple=False).flatten()
            if done_ids.numel() > 0:
                self._needs_control_assignment[done_ids] = True
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        last_values = self.policy.evaluate_value(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        self.num_updates += 1
        teacher_coef = self._compute_teacher_coef()
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_imitation_loss_unweighted = 0.0
        mean_imitation_loss_weighted = 0.0
        mean_imitation_contribution = 0.0
        mean_effective_teacher_coef = 0.0
        mean_plan_valid_ratio = 0.0
        mean_imitation_to_surrogate_ratio = 0.0
        mean_action_l1 = 0.0
        mean_entropy = 0.0
        update_count = 0

        generator = self.storage.mini_batch_generator(
            self.num_mini_batches,
            self.num_learning_epochs,
            include_privileged_actions=True,
            include_ppo_mask=True,
            include_imitation_context=True,
        )
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            _hid_states_batch,
            _masks_batch,
            _point_cloud_batch,
            _semantic_labels_batch,
            privileged_actions_batch,
            ppo_active_mask_batch,
            imitation_weight_batch,
            plan_valid_batch,
        ) in generator:
            self.policy.student.act(obs_batch)
            actions_log_prob_batch = self.policy.student.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate_value(critic_obs_batch)
            mu_batch = self.policy.student.action_mean
            sigma_batch = self.policy.student.action_std
            entropy_batch = self.policy.student.entropy

            ratio = torch.exp(
                actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
            )
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio,
                1.0 - self.clip_param,
                1.0 + self.clip_param,
            )
            ppo_active = ppo_active_mask_batch.reshape(-1) > 0.5
            if torch.any(ppo_active):
                surrogate_loss = torch.max(
                    surrogate[ppo_active], surrogate_clipped[ppo_active]
                ).mean()
                entropy_loss = entropy_batch.reshape(-1)[ppo_active].mean()
            else:
                surrogate_loss = actions_log_prob_batch.new_zeros(())
                entropy_loss = actions_log_prob_batch.new_zeros(())

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (
                    value_batch - target_values_batch
                ).clamp(-self.clip_param, self.clip_param)
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            sample_mse = torch.mean((mu_batch - privileged_actions_batch).pow(2), dim=-1)
            imitation_loss_unweighted = sample_mse.mean()
            imitation_loss_weighted = torch.mean(
                sample_mse * imitation_weight_batch.reshape(-1)
            )
            imitation_contribution = teacher_coef * imitation_loss_weighted
            action_l1 = torch.mean(torch.abs(mu_batch - privileged_actions_batch))
            surrogate_scale = surrogate_loss.detach().abs() + 1.0e-6
            imitation_to_surrogate_ratio = imitation_contribution.detach().abs() / surrogate_scale

            loss = (
                self.ppo_coef
                * (
                    surrogate_loss
                    + self.value_loss_coef * value_loss
                    - self.entropy_coef * entropy_loss
                )
                + imitation_contribution
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.distributed:
                self._reduce_gradients()
            nn.utils.clip_grad_norm_(
                list(self.policy.student.parameters())
                + list(self.policy.student_critic.parameters()),
                self.max_grad_norm,
            )
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_imitation_loss_unweighted += imitation_loss_unweighted.item()
            mean_imitation_loss_weighted += imitation_loss_weighted.item()
            mean_imitation_contribution += imitation_contribution.detach().item()
            mean_effective_teacher_coef += (
                teacher_coef * imitation_weight_batch.reshape(-1).mean().item()
            )
            mean_plan_valid_ratio += plan_valid_batch.reshape(-1).mean().item()
            mean_imitation_to_surrogate_ratio += imitation_to_surrogate_ratio.item()
            mean_action_l1 += action_l1.item()
            mean_entropy += entropy_loss.item()
            update_count += 1

        count = max(update_count, 1)
        self.storage.clear()
        self.policy.student.clip_std(min=self.clip_min_std)
        return {
            "ppo_coef": self.ppo_coef,
            "teacher_coef": teacher_coef,
            "teacher_ratio": self.last_teacher_ratio,
            "student_ratio": self.last_student_ratio,
            "teacher_action_share": self.last_teacher_action_share,
            "ppo_active_ratio": self.last_ppo_active_ratio,
            "value_loss": mean_value_loss / count,
            "surrogate_loss": mean_surrogate_loss / count,
            "imitation_loss": mean_imitation_loss_unweighted / count,
            "imitation_loss_unweighted": mean_imitation_loss_unweighted / count,
            "imitation_loss_weighted": mean_imitation_loss_weighted / count,
            "imitation_contribution": mean_imitation_contribution / count,
            "effective_teacher_coef_mean": mean_effective_teacher_coef / count,
            "plan_valid_ratio": mean_plan_valid_ratio / count,
            "imitation_to_surrogate_ratio": mean_imitation_to_surrogate_ratio / count,
            "action_l1": mean_action_l1 / count,
            "entropy": mean_entropy / count,
        }

    def _reduce_gradients(self):
        parameters = [
            parameter
            for parameter in list(self.policy.student.parameters())
            + list(self.policy.student_critic.parameters())
            if parameter.grad is not None
        ]
        for parameter in parameters:
            self.dist.all_reduce(parameter.grad.data, op=self.dist.ReduceOp.SUM)
            parameter.grad.data /= self.gpu_world_size

    def broadcast_parameters(self):
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0], strict=False)
