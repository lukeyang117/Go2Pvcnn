from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.modules import StudentTeacherCNN
from rsl_rl.storage import RolloutStorage


class Distillation:
    """Distillation algorithm for training a student policy to mimic a teacher policy."""

    policy: StudentTeacherCNN

    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=4,
        gradient_length=15,
        learning_rate=1e-3,
        loss_type="mse",
        device="cpu",
        multi_gpu_cfg: dict | None = None,
    ):
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.rnd = None
        self.policy = policy
        self.actor_critic = policy
        self.policy.to(self.device)
        self.storage = None
        self.optimizer = optim.Adam(self.policy.student.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = None

        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.gradient_length = gradient_length
        self.learning_rate = learning_rate

        if loss_type == "mse":
            self.loss_fn = nn.functional.mse_loss
        elif loss_type == "huber":
            self.loss_fn = nn.functional.huber_loss
        else:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported types are: mse, huber")

        self.num_updates = 0

    def init_storage(self, training_type, num_envs, num_transitions_per_env, student_obs_shape, teacher_obs_shape, actions_shape):
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            self.device,
        )

    def act(self, obs, teacher_obs):
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.privileged_actions = self.policy.evaluate(teacher_obs).detach()
        self.transition.observations = obs
        self.transition.critic_observations = teacher_obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards
        self.transition.dones = dones
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def update(self):
        self.num_updates += 1
        mean_behavior_loss = 0.0
        mean_action_l1 = 0.0
        mean_action_max = 0.0
        loss = 0.0
        cnt = 0
        pending_steps = 0

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, _, _, privileged_actions, dones in self.storage.distillation_generator(
                self.num_mini_batches, num_epochs=1
            ):
                actions = self.policy.act_inference(obs)
                behavior_loss = self.loss_fn(actions, privileged_actions)
                action_l1 = torch.mean(torch.abs(actions - privileged_actions))
                action_max = torch.max(torch.abs(actions - privileged_actions))
                loss = loss + behavior_loss
                pending_steps += 1
                mean_behavior_loss += behavior_loss.item()
                mean_action_l1 += action_l1.item()
                mean_action_max += action_max.item()
                cnt += 1

                if pending_steps >= self.gradient_length:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0.0
                    pending_steps = 0

                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        if pending_steps > 0:
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            self.optimizer.step()

        mean_behavior_loss /= max(cnt, 1)
        mean_action_l1 /= max(cnt, 1)
        mean_action_max /= max(cnt, 1)
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()
        return {
            "behavior": mean_behavior_loss,
            "action_mse": mean_behavior_loss,
            "action_l1": mean_action_l1,
            "action_error_max": mean_action_max,
        }

    def broadcast_parameters(self):
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
