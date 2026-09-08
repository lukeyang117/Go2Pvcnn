"""Isolated AME-XYZ-Semantic baseline for Go2 locomotion."""

from .actor_critic_ame import ActorCriticAME
from .ame_train_cfg import get_ame_train_cfg

__all__ = ["ActorCriticAME", "get_ame_train_cfg"]
