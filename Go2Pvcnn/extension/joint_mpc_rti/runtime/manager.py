"""Trajectory-manager skeleton for the joint MPC RTI backend."""

from __future__ import annotations

from extension.joint_mpc_rti.config import JointMpcRtiCfg


class JointMpcRtiManager:
    planner_backend = "joint_mpc_rti"

    def __init__(self, task_cfg, *, device) -> None:
        self._device = device
        self._cfg = getattr(task_cfg, "joint_mpc_rti_cfg", JointMpcRtiCfg())

    def horizon_steps(self) -> int:
        return int(self._cfg.runtime.horizon_steps)


__all__ = ["JointMpcRtiManager"]
