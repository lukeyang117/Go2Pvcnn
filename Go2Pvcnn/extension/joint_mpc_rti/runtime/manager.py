"""Trajectory-manager skeleton for the joint MPC RTI backend."""

from __future__ import annotations

import torch

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.integration.field_sync import JointMpcRayCasterFieldSync
from extension.joint_mpc_rti.integration.reference_adapter import trajectory_to_reference_cache
from extension.joint_mpc_rti.integration.isaaclab_adapter import command_from_env, scanner_from_env, state_from_env
from extension.joint_mpc_rti.planner import step as planner_step
from extension.joint_mpc_rti.runtime.reference_buffer import PendingReferenceBuffer
from extension.joint_mpc_rti.runtime.cuda_graph import JointMpcCudaGraphRunner
from extension.joint_mpc_rti.types import JointMpcRtiSolverState, JointMpcRtiState, JointMpcRtiStepResult, JointMpcTerrainField


class JointMpcRtiManager:
    planner_backend = "joint_mpc_rti"

    def __init__(self, task_cfg, *, device) -> None:
        self._device = device
        self._cfg = getattr(task_cfg, "joint_mpc_rti_cfg", JointMpcRtiCfg())
        self._solver_state: JointMpcRtiSolverState | None = None
        self._last_result: JointMpcRtiStepResult | None = None
        self._buffer: PendingReferenceBuffer | None = None
        self._cache = None
        self._field_sync: JointMpcRayCasterFieldSync | None = None
        self._last_step_token = None
        self._graph_runner: JointMpcCudaGraphRunner | None = None
        self._num_envs: int | None = None

    @classmethod
    def from_config(
        cls,
        cfg: JointMpcRtiCfg,
        *,
        num_envs: int,
        device,
    ) -> "JointMpcRtiManager":
        instance = cls.__new__(cls)
        instance._device = device
        instance._cfg = cfg
        instance._solver_state = None
        instance._last_result = None
        instance._buffer = PendingReferenceBuffer(num_envs=num_envs, device=device)
        instance._cache = None
        instance._field_sync = None
        instance._last_step_token = None
        instance._graph_runner = None
        instance._num_envs = int(num_envs)
        return instance

    def horizon_steps(self) -> int:
        return int(self._cfg.runtime.horizon_steps)

    @property
    def pending_valid(self):
        if self._buffer is None:
            raise RuntimeError("pending reference buffer is not initialized")
        return self._buffer.valid

    def plan_from_tensors(
        self,
        measured_state: JointMpcRtiState,
        command_body,
        terrain_field: JointMpcTerrainField,
    ) -> JointMpcRtiStepResult:
        batch_size = measured_state.batch_size
        if self._num_envs is None:
            self._num_envs = batch_size
        elif batch_size != self._num_envs:
            raise ValueError(
                f"joint_mpc_rti batch changed from {self._num_envs} to {batch_size}; "
                "rebuild the manager/cache/CUDA graph for the new environment count"
            )
        if self._buffer is None:
            self._buffer = PendingReferenceBuffer(num_envs=batch_size, device=measured_state.device)
        use_graph = (
            bool(self._cfg.solver.use_cuda_graph)
            and measured_state.device.type == "cuda"
            and self._solver_state is not None
        )
        if use_graph and self._graph_runner is None:
            self._graph_runner = JointMpcCudaGraphRunner(
                measured_state,
                command_body,
                terrain_field,
                self._solver_state,
                self._cfg,
            )
            result = self._graph_runner.captured_result
        elif use_graph and self._graph_runner is not None:
            if not self._graph_runner.matches_field(terrain_field):
                self._graph_runner = JointMpcCudaGraphRunner(
                    measured_state,
                    command_body,
                    terrain_field,
                    self._solver_state,
                    self._cfg,
                )
                result = self._graph_runner.captured_result
            else:
                result = self._graph_runner.run(measured_state, command_body, terrain_field)
        else:
            result = planner_step(measured_state, command_body, terrain_field, self._solver_state, self._cfg)
        self._last_result = result
        self._solver_state = self._graph_runner.solver_state if use_graph and self._graph_runner is not None else result.solver_state
        self._buffer.update(result.pending_reference)
        self._cache = trajectory_to_reference_cache(result.full_trajectory)
        return result

    def reset_envs(self, env_mask) -> None:
        if self._buffer is not None:
            self._buffer.reset_rows(env_mask)
        if self._solver_state is not None and self._solver_state.gait_phase is not None:
            mask = torch.as_tensor(env_mask, dtype=torch.bool, device=self._solver_state.gait_phase.device)
            self._solver_state.gait_phase.masked_fill_(mask, 0)

    def current_reference(self):
        if self._buffer is None or self._buffer.reference is None:
            raise RuntimeError("no pending joint MPC RTI reference is available")
        reference = self._buffer.reference
        return {
            "root_pos_w": reference.root_pos_w,
            "root_rpy_w": reference.root_rpy_w,
            "joint_angles": reference.joint_angles,
            "foot_pos_w": reference.foot_pos_w,
            "contact_state": reference.contact_state,
            "valid": self._buffer.valid,
        }

    def current_frame_ids(self):
        if self._buffer is None:
            raise RuntimeError("pending reference buffer is not initialized")
        import torch

        return torch.ones_like(self._buffer.valid, dtype=torch.long)

    def latest_trajectory(self):
        if self._last_result is None:
            raise RuntimeError("no joint MPC RTI trajectory is available")
        return self._last_result.full_trajectory

    def refresh_from_env(self, env, *, command_body=None, force: bool = False):
        root = getattr(env, "unwrapped", env)
        step_token = getattr(root, "common_step_counter", None)
        if not force and step_token is not None and self._cache is not None and step_token == self._last_step_token:
            return self._cache
        measured_state = state_from_env(env, device=self._device)
        if command_body is None:
            command_name = str(getattr(getattr(root, "cfg", None), "reference_command_name", "base_velocity"))
            command = command_from_env(env, device=self._device, command_name=command_name)
        else:
            command = torch.as_tensor(command_body, dtype=torch.float32, device=measured_state.device)
        scanner_name = str(
            getattr(getattr(root, "cfg", None), "reference_height_scanner_name", "semantic_height_scanner")
        )
        scanner = scanner_from_env(env, scanner_name=scanner_name)
        if self._field_sync is None:
            ray_hits = torch.as_tensor(scanner.data.ray_hits_w)
            grid_size = int(round(float(ray_hits.shape[1]) ** 0.5))
            pattern_cfg = getattr(getattr(scanner, "cfg", None), "pattern_cfg", None)
            resolution = float(getattr(pattern_cfg, "resolution", 0.01))
            self._field_sync = JointMpcRayCasterFieldSync(
                num_envs=measured_state.batch_size,
                grid_size=grid_size,
                device=self._device,
                resolution=resolution,
            )
            self._field_sync.attach(scanner)
            self._field_sync.on_raycaster_update(
                scanner,
                torch.arange(measured_state.batch_size, dtype=torch.long, device=measured_state.device),
            )
        field = self._field_sync.latest_field()
        self.plan_from_tensors(measured_state, command, field)
        self._last_step_token = step_token
        root._trajectory_reference_cache = self._cache
        return self._cache


__all__ = ["JointMpcRtiManager"]
