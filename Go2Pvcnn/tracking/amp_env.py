"""Isaac environment adapter that exposes AMP payloads without new observations."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from tracking.env import ParallelismTrackingEnv
from tracking.managers.parallelism_amp_manager import ParallelismAmpManager
from tracking.managers.parallelism_reference_manager import get_parallelism_reference_manager


def _frame_from_robot(env) -> torch.Tensor:
    robot = env.scene["robot"]
    root_pos = robot.data.root_pos_w
    joint_pos = robot.data.joint_pos
    frame = torch.zeros((env.num_envs, 39), dtype=torch.float32, device=env.device)
    frame[:, : min(12, joint_pos.shape[-1])] = joint_pos[:, :12]
    joint_vel = torch.as_tensor(robot.data.joint_vel, dtype=torch.float32, device=env.device)
    frame[:, 12 : 12 + min(12, joint_vel.shape[-1])] = joint_vel[:, :12]
    frame[:, 24:27] = root_pos
    quat = robot.data.root_quat_w
    w, x, y, z = quat.unbind(-1)
    rotation = torch.empty((env.num_envs, 3, 3), dtype=torch.float32, device=env.device)
    rotation[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotation[:, 0, 1] = 2 * (x * y - w * z)
    rotation[:, 0, 2] = 2 * (x * z + w * y)
    rotation[:, 1, 0] = 2 * (x * y + w * z)
    rotation[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotation[:, 1, 2] = 2 * (y * z - w * x)
    rotation[:, 2, 0] = 2 * (x * z - w * y)
    rotation[:, 2, 1] = 2 * (y * z + w * x)
    rotation[:, 2, 2] = 1 - 2 * (x * x + y * y)
    frame[:, 27:33] = rotation[:, :, :2].transpose(-1, -2).reshape(env.num_envs, 6)
    frame[:, 33:36] = torch.matmul(
        rotation.transpose(-1, -2), torch.as_tensor(robot.data.root_lin_vel_w, device=env.device).unsqueeze(-1)
    ).squeeze(-1)
    frame[:, 36:39] = torch.matmul(
        rotation.transpose(-1, -2), torch.as_tensor(robot.data.root_ang_vel_w, device=env.device).unsqueeze(-1)
    ).squeeze(-1)
    return frame


def _rpy_to_rotation_6d(rpy: torch.Tensor) -> torch.Tensor:
    roll, pitch, yaw = rpy.unbind(-1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    matrix = torch.empty(rpy.shape[:-1] + (3, 3), dtype=rpy.dtype, device=rpy.device)
    matrix[..., 0, 0] = cy * cp
    matrix[..., 0, 1] = cy * sp * sr - sy * cr
    matrix[..., 0, 2] = cy * sp * cr + sy * sr
    matrix[..., 1, 0] = sy * cp
    matrix[..., 1, 1] = sy * sp * sr + cy * cr
    matrix[..., 1, 2] = sy * sp * cr - cy * sr
    matrix[..., 2, 0] = -sp
    matrix[..., 2, 1] = cp * sr
    matrix[..., 2, 2] = cp * cr
    return matrix[..., :, :2].transpose(-1, -2).reshape(*rpy.shape[:-1], 6)


def _reference_frame(manager, *, start: bool) -> torch.Tensor:
    """Build a complete 39-D planner frame in the same layout as the robot."""

    frame = torch.zeros((manager.num_envs, 39), dtype=torch.float32, device=manager.device)
    if start:
        frame[:, :12] = manager.joint_pos[:, 0, :12]
        frame[:, 24:27] = manager.root_pos_w[:, 0]
        frame[:, 27:33] = _rpy_to_rotation_6d(manager.root_rpy_w[:, 0])
        frame[:, 12:24] = (manager.joint_pos[:, 1, :12] - manager.joint_pos[:, 0, :12]) / max(manager.dt, 1.0e-6)
        frame[:, 33:36] = manager._root_velocity_b_policy(manager.root_pos_w, manager.root_rpy_w, manager.root_rpy_w.new_zeros(manager.num_envs, dtype=torch.long), manager.root_rpy_w.new_ones(manager.num_envs, dtype=torch.long))
        frame[:, 36:39] = manager._angular_velocity_b_policy(manager.root_rpy_w, manager.root_rpy_w.new_zeros(manager.num_envs, dtype=torch.long), manager.root_rpy_w.new_ones(manager.num_envs, dtype=torch.long))
    else:
        frame[:, :12] = manager.step_joint_pos[:, :12]
        frame[:, 12:24] = manager.step_joint_vel[:, :12]
        frame[:, 24:27] = manager.step_root_pos_w
        frame[:, 27:33] = _rpy_to_rotation_6d(manager.step_root_rpy_w)
        frame[:, 33:36] = manager.step_root_lin_vel_b_policy
        frame[:, 36:39] = manager.step_root_ang_vel_b_policy
    return frame


class ParallelismAmpEnv(ParallelismTrackingEnv):
    """Attach a batched AMP manager and append payloads to step infos."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.cfg
        self.amp_manager = ParallelismAmpManager(self.num_envs, self.device, getattr(cfg, "amp_window_frames", 24), getattr(cfg, "amp_dt", 0.02))
        self._amp_previous_agent = _frame_from_robot(self)
        self._amp_previous_expert = self._amp_previous_agent.clone()
        self._amp_plan_start = self._amp_previous_expert.clone()

    def step(self, action):
        reference_manager = get_parallelism_reference_manager(self)
        agent_start = _frame_from_robot(self)
        result = super().step(action)
        agent_target = _frame_from_robot(self)
        expert_target = _reference_frame(reference_manager, start=False)
        # A newly planned segment starts at its measured B0.  The first
        # transition is therefore B0 -> B1; no A23 -> B1 teleport is inserted.
        new_plan = reference_manager.phase == 0
        plan_start = _reference_frame(reference_manager, start=True)
        self._amp_plan_start = torch.where(new_plan[:, None], plan_start, self._amp_plan_start)
        expert_start = torch.where(new_plan[:, None], self._amp_plan_start, self._amp_previous_expert)
        valid = torch.as_tensor(reference_manager.step_plan_valid, device=self.device, dtype=torch.bool)
        payload = self.amp_manager.push_transition(agent_start, agent_target, expert_start, expert_target, valid)
        self._amp_previous_agent = agent_target
        self._amp_previous_expert = expert_target
        obs, rewards, dones, truncated, infos = result
        infos.setdefault("amp", {})
        infos["amp"].update({"expert_window": payload.expert_window, "agent_window": payload.agent_window, "amp_active": payload.amp_active, "history_ratio": payload.history_ratio})
        return obs, rewards, dones, truncated, infos

    def _reset_idx(self, env_ids: Sequence[int]) -> None:
        super()._reset_idx(env_ids)
        # Isaac Lab may reset during ManagerBasedRLEnv.__init__, before the
        # AMP buffers are constructed by this subclass.
        if not hasattr(self, "amp_manager"):
            return
        if isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
            if ids.dtype == torch.bool:
                ids = ids.nonzero(as_tuple=False).flatten()
            else:
                ids = ids.to(dtype=torch.long)
        if ids.numel() == 0:
            return
        mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        mask[ids] = True
        self.amp_manager.reset(mask)
        current = _frame_from_robot(self)
        self._amp_previous_agent[mask] = current[mask]
        self._amp_previous_expert[mask] = current[mask]
        self._amp_plan_start[mask] = current[mask]
