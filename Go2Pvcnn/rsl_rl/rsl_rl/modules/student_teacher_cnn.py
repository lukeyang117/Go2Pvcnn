from __future__ import annotations

import torch
import torch.nn as nn

from .actor_critic_cnn import ActorCriticCNN


class StudentTeacherCNN(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        num_critic_obs=None,
        student_hidden_dims=[256, 256, 128],
        teacher_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 128],
        activation="elu",
        init_noise_std=1.0,
        cost_map_channels=2,
        cost_map_size=16,
        actor_cnn_cfg=None,
        critic_cnn_cfg=None,
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentTeacherCNN.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        self.loaded_teacher = False
        num_critic_obs = num_teacher_obs if num_critic_obs is None else num_critic_obs
        self.student_proprio_dim = int(num_student_obs) - int(cost_map_channels * cost_map_size * cost_map_size)
        self.student = ActorCriticCNN(
            num_student_obs,
            num_student_obs,
            num_actions,
            cost_map_channels=cost_map_channels,
            cost_map_size=cost_map_size,
            actor_hidden_dims=student_hidden_dims,
            critic_hidden_dims=[1],
            activation=activation,
            init_noise_std=init_noise_std,
            actor_cnn_cfg=actor_cnn_cfg,
            critic_cnn_cfg=critic_cnn_cfg,
        )
        self.student_critic = ActorCriticCNN(
            num_critic_obs,
            num_critic_obs,
            num_actions,
            cost_map_channels=cost_map_channels,
            cost_map_size=cost_map_size,
            actor_hidden_dims=[1],
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            actor_cnn_cfg=actor_cnn_cfg,
            critic_cnn_cfg=critic_cnn_cfg,
        )
        self.teacher = ActorCriticCNN(
            num_teacher_obs,
            num_teacher_obs,
            num_actions,
            cost_map_channels=cost_map_channels,
            cost_map_size=cost_map_size,
            actor_hidden_dims=teacher_hidden_dims,
            critic_hidden_dims=[1],
            activation=activation,
            init_noise_std=init_noise_std,
            actor_cnn_cfg=actor_cnn_cfg,
            critic_cnn_cfg=critic_cnn_cfg,
        )
        self.teacher.eval()
        self.teacher.requires_grad_(False)

    def reset(self, dones=None, hidden_states=None):
        return None

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        return None

    def act(self, observations):
        return self.student.act(observations)

    def act_inference(self, observations):
        return self.student.act_inference(observations)

    def evaluate(self, teacher_observations):
        """Return the frozen teacher's deterministic action target."""

        with torch.no_grad():
            return self.teacher.act_inference(teacher_observations)

    def evaluate_value(self, critic_observations):
        """Evaluate the trainable critic on PPO-compatible observations."""

        return self.student_critic.evaluate(critic_observations)

    @property
    def action_mean(self):
        return self.student.action_mean

    @property
    def action_std(self):
        return self.student.action_std

    @property
    def std(self):
        return self.student.std

    @property
    def entropy(self):
        return self.student.entropy

    def load_state_dict(self, state_dict, strict=True):
        state_dict = dict(state_dict)
        self._adapt_legacy_student_input_weights(state_dict)
        if any(key.startswith("student.") or key.startswith("teacher.") for key in state_dict.keys()):
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher.requires_grad_(False)
            return True

        teacher_state = {
            key: value
            for key, value in state_dict.items()
            if key == "std"
            or key.startswith("actor.")
            or key.startswith("actor_cnns.")
            or key.startswith("cnn_encoder.")
        }
        self.teacher.load_state_dict(teacher_state, strict=False)
        self.loaded_teacher = True
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        return False

    def load_student_state_dict(self, state_dict, keep_std=True):
        """Load student actor and critic weights without embedded teacher weights."""

        state_dict = dict(state_dict)
        self._adapt_legacy_student_input_weights(state_dict)
        has_student_namespace = any(
            key.startswith("student.") or key.startswith("student_critic.")
            for key in state_dict
        )
        if not has_student_namespace:
            legacy_prefixes = (
                ("actor.", "student.actor."),
                ("actor_cnns.", "student.actor_cnns."),
                ("critic.", "student_critic.critic."),
                ("critic_cnns.", "student_critic.critic_cnns."),
            )
            legacy_state = {}
            for key, value in state_dict.items():
                if key == "std":
                    legacy_state["student.std"] = value
                    continue
                for source_prefix, target_prefix in legacy_prefixes:
                    if key.startswith(source_prefix):
                        legacy_state[target_prefix + key[len(source_prefix):]] = value
                        break
            state_dict = legacy_state
        if not keep_std:
            state_dict.pop("student.std", None)
            state_dict.pop("student_critic.std", None)
        student_state = {
            key: value
            for key, value in state_dict.items()
            if key.startswith("student.") or key.startswith("student_critic.")
        }
        result = super().load_state_dict(student_state, strict=False)
        self.teacher.eval()
        self.teacher.requires_grad_(False)
        return result

    def _adapt_legacy_student_input_weights(self, state_dict):
        """Drop legacy base_lin_vel columns when loading pre-removal student checkpoints."""

        for key in ("student.actor.0.weight", "student.critic.0.weight"):
            weight = state_dict.get(key)
            if weight is None or weight.ndim != 2:
                continue
            current_weight = dict(self.named_parameters()).get(key)
            if current_weight is None:
                continue
            if weight.shape[1] != current_weight.shape[1] + 3:
                continue
            old_proprio_dim = self.student_proprio_dim + 3
            state_dict[key] = torch.cat(
                (weight[:, : self.student_proprio_dim], weight[:, old_proprio_dim:]),
                dim=1,
            )
