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
        student_hidden_dims=[256, 256, 128],
        teacher_hidden_dims=[256, 256, 128],
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
        with torch.no_grad():
            return self.teacher.act_inference(teacher_observations)

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
