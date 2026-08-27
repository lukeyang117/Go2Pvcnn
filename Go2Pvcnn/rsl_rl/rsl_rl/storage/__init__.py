#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Implementation of transitions storage for RL-agent."""

from .rollout_storage import RolloutStorage

__all__ = ["RolloutStorage"]
from .parallelism_amp_storage import ParallelismAMPStorage, combine_advantages

__all__ = ["RolloutStorage", "ParallelismAMPStorage", "combine_advantages"]
