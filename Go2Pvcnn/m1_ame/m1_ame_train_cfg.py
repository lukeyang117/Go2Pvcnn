"""M1 AME pure-PPO training configuration."""

from __future__ import annotations

import copy

from ame_baseline.ame_train_cfg import get_ame_train_cfg


def get_m1_ame_train_cfg() -> dict:
    """Return the Go2 AME PPO config unchanged for the M1 environment."""

    return copy.deepcopy(get_ame_train_cfg())


__all__ = ["get_m1_ame_train_cfg"]
