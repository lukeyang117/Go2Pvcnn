"""Fixed-address CUDA Graph runner for steady-state rolling RTI calls."""

from __future__ import annotations

import torch

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.planner import step as planner_step
from extension.joint_mpc_rti.types import (
    JointMpcRtiSolverState,
    JointMpcRtiState,
    JointMpcRtiStepResult,
    JointMpcTerrainField,
)


def _clone_state(state: JointMpcRtiState) -> JointMpcRtiState:
    return JointMpcRtiState(
        root_pos_w=state.root_pos_w.clone(),
        root_rpy_w=state.root_rpy_w.clone(),
        joint_pos=state.joint_pos.clone(),
        root_lin_vel_b=state.root_lin_vel_b.clone(),
        root_ang_vel_b=state.root_ang_vel_b.clone(),
        joint_vel=state.joint_vel.clone(),
    )


class JointMpcCudaGraphRunner:
    def __init__(
        self,
        measured_state: JointMpcRtiState,
        command_body: torch.Tensor,
        terrain_field: JointMpcTerrainField,
        solver_state: JointMpcRtiSolverState,
        cfg: JointMpcRtiCfg,
    ) -> None:
        if measured_state.device.type != "cuda":
            raise ValueError("JointMpcCudaGraphRunner requires CUDA tensors")
        self._cfg = cfg
        self._state = _clone_state(measured_state)
        self._command = torch.as_tensor(command_body, dtype=torch.float32, device=measured_state.device).clone()
        self._field = terrain_field
        self._field_height_ptr = int(terrain_field.height_w.data_ptr())
        self._solver_state = JointMpcRtiSolverState(
            state=solver_state.state.clone(),
            control=solver_state.control.clone(),
            dual=solver_state.dual.clone(),
            previous_control=solver_state.previous_control.clone(),
            gait_phase=None if solver_state.gait_phase is None else solver_state.gait_phase.clone(),
        )
        warm = planner_step(self._state, self._command, self._field, self._solver_state, self._cfg)
        torch.cuda.synchronize(device=measured_state.device)
        del warm
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._result = planner_step(
                self._state,
                self._command,
                self._field,
                self._solver_state,
                self._cfg,
            )
            self._solver_state.state.copy_(self._result.solver_state.state)
            self._solver_state.control.copy_(self._result.solver_state.control)
            self._solver_state.dual.copy_(self._result.solver_state.dual)
            self._solver_state.previous_control.copy_(self._result.solver_state.previous_control)
            if self._solver_state.gait_phase is not None and self._result.solver_state.gait_phase is not None:
                self._solver_state.gait_phase.copy_(self._result.solver_state.gait_phase)

    @property
    def solver_state(self) -> JointMpcRtiSolverState:
        return self._solver_state

    @property
    def captured_result(self) -> JointMpcRtiStepResult:
        return self._result

    def matches_field(self, terrain_field: JointMpcTerrainField) -> bool:
        return int(terrain_field.height_w.data_ptr()) == self._field_height_ptr

    def run(
        self,
        measured_state: JointMpcRtiState,
        command_body: torch.Tensor,
        terrain_field: JointMpcTerrainField,
    ) -> JointMpcRtiStepResult:
        if not self.matches_field(terrain_field):
            raise ValueError("terrain field storage changed; CUDA graph must be rebuilt")
        self._state.root_pos_w.copy_(measured_state.root_pos_w)
        self._state.root_rpy_w.copy_(measured_state.root_rpy_w)
        self._state.joint_pos.copy_(measured_state.joint_pos)
        self._state.root_lin_vel_b.copy_(measured_state.root_lin_vel_b)
        self._state.root_ang_vel_b.copy_(measured_state.root_ang_vel_b)
        self._state.joint_vel.copy_(measured_state.joint_vel)
        self._command.copy_(command_body)
        self._graph.replay()
        return self._result


__all__ = ["JointMpcCudaGraphRunner"]
