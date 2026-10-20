"""M1 AME pure-PPO training configuration."""

from __future__ import annotations

import copy

from ame_baseline.ame_train_cfg import get_ame_train_cfg


M1_ACTION_JOINT_NAMES = (
    "FBL_ABAD_JOINT", "FBL_HIP_JOINT", "FBL_KNEE_JOINT", "FBL_FOOT_JOINT",
    "FAR_ABAD_JOINT", "FAR_HIP_JOINT", "FAR_KNEE_JOINT", "FAR_FOOT_JOINT",
    "RBL_ABAD_JOINT", "RBL_HIP_JOINT", "RBL_KNEE_JOINT", "RBL_FOOT_JOINT",
    "RAR_ABAD_JOINT", "RAR_HIP_JOINT", "RAR_KNEE_JOINT", "RAR_FOOT_JOINT",
)


M1_LEG_INIT_STD = 0.20
M1_WHEEL_INIT_STD = 0.10


def get_m1_ame_train_cfg() -> dict:
    """Return the Go2 AME PPO config with M1 action-noise and KL adaptation."""

    cfg = copy.deepcopy(get_ame_train_cfg())
    cfg["algorithm"]["normalize_kl_by_action_dim"] = True
    cfg["policy"]["init_noise_std"] = [
        M1_WHEEL_INIT_STD if name.endswith("_FOOT_JOINT") else M1_LEG_INIT_STD
        for name in M1_ACTION_JOINT_NAMES
    ]
    return cfg


__all__ = ["get_m1_ame_train_cfg"]
