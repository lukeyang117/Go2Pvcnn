#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Definitions for neural-network components for RL-agents."""

from .actor_critic import ActorCritic
from .actor_critic_cnn import ActorCriticCNN
from .actor_critic_recurrent import ActorCriticRecurrent
from .normalizer import EmpiricalNormalization
from .student_teacher_cnn import StudentTeacherCNN
from .amp_discriminator import AMPDiscriminator, AMPObservationNormalizer
from .amp_actor_critic_cnn import AmpActorCriticCNN

__all__ = [
    "ActorCritic",
    "ActorCriticCNN",
    "ActorCriticRecurrent",
    "StudentTeacherCNN",
    "AMPDiscriminator",
    "AMPObservationNormalizer",
    "AmpActorCriticCNN",
]
