#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Implementation of different RL agents."""

from .ppo import PPO
from .distillation import Distillation
from .hybrid_distillation_ppo import HybridDistillationPPO

__all__ = ["PPO", "Distillation", "HybridDistillationPPO"]
