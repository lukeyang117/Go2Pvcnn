"""Fixed inputs shared by the final LQ and hard-safe line search."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from extension.joint_mpc_rti.model.gait_schedule import FixedTrotSchedule
from extension.joint_mpc_rti.types import JointMpcPerceptiveField, JointMpcTerrainField


@dataclass(frozen=True)
class LossContext:
    command_body: Tensor
    touchdown_reference_w: Tensor
    schedule: FixedTrotSchedule
    terrain: JointMpcTerrainField
    stance_anchor_w: Tensor
    support_height: Tensor
    perceptive_field: JointMpcPerceptiveField | None = None


__all__ = ["LossContext"]
