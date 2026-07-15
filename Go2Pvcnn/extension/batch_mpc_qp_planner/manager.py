"""Trajectory manager for the experimental MPC-QP backend."""

from __future__ import annotations

from extension.batch_mpc_planner.manager import MpcTrajectoryManager

from .adapter import mpc_result_to_reference_cache
from .config import MpcQpPlannerCfg, planner_cfg_from_task_cfg
from .planner import plan_segment_qp


class MpcQpTrajectoryManager(MpcTrajectoryManager):
    """Planner-owned cache manager for the MPC-QP backend."""

    planner_backend = "mpc_qp"

    def _planner_cfg(self) -> MpcQpPlannerCfg:
        return planner_cfg_from_task_cfg(self._cfg)

    def _plan_segment(self, terrain, states, command, cfg):
        return plan_segment_qp(terrain, states, command, cfg=cfg)

    def _result_to_reference_cache(self, result):
        return mpc_result_to_reference_cache(result)


__all__ = ["MpcQpTrajectoryManager"]
