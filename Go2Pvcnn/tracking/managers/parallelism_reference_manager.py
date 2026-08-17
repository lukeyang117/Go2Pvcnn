"""Reference cache manager for flat parallelism RL tracking."""

from __future__ import annotations

from collections.abc import Sequence
import functools

import torch
from torch import Tensor

from extension.convention import extract_roll_pitch_batch, extract_yaw_batch
from extension.parallelism.config import ParallelismCfg
from extension.parallelism.kinematics import fk_go2
from extension.parallelism.planner import plan_trajectory
from extension.parallelism.terrain import query_height_semantic_valid
from extension.parallelism.types import ParallelismState, ParallelismTerrain

_PLANNER_JOINT_ORDER = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)
_PLANNER_FOOT_ORDER = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")


def _env_root(env):
    return getattr(env, "unwrapped", env)


def _as_env_ids(env_ids, *, num_envs: int, device: torch.device) -> Tensor:
    if env_ids is None:
        return torch.arange(num_envs, dtype=torch.long, device=device)
    tensor = torch.as_tensor(env_ids, device=device)
    if tensor.dtype == torch.bool:
        return tensor.nonzero(as_tuple=False).flatten().to(dtype=torch.long)
    return tensor.to(dtype=torch.long).flatten()


def _normalize_name(name: str) -> str:
    normalized = str(name).split("/")[-1]
    normalized = normalized.split(":")[-1]
    return normalized.lower()


def _order_indices(source_order: Sequence[str], target_order: Sequence[str], *, device: torch.device) -> Tensor | None:
    source_to_index = {_normalize_name(name): idx for idx, name in enumerate(source_order)}
    indices: list[int] = []
    for name in target_order:
        idx = source_to_index.get(_normalize_name(name))
        if idx is None:
            return None
        indices.append(idx)
    return torch.tensor(indices, dtype=torch.long, device=device)


def _reorder_last(values: Tensor, *, source_order: Sequence[str] | None, target_order: Sequence[str]) -> Tensor:
    tensor = torch.as_tensor(values)
    if not source_order or not target_order:
        return tensor
    indices = _order_indices(source_order, target_order, device=tensor.device)
    if indices is None or int(tensor.shape[-1]) != len(tuple(source_order)):
        return tensor
    return tensor.index_select(-1, indices)


def _reorder_joint_from_planner(values: Tensor, robot_joint_names) -> Tensor:
    return _reorder_last(values, source_order=_PLANNER_JOINT_ORDER, target_order=tuple(robot_joint_names or ()))


def _reorder_joint_to_planner(values: Tensor, robot_joint_names) -> Tensor:
    return _reorder_last(values, source_order=tuple(robot_joint_names or ()), target_order=_PLANNER_JOINT_ORDER)


def _quat_to_matrix_wxyz(quat: Tensor) -> Tensor:
    """Convert batched wxyz quaternions to world-from-body rotation matrices."""

    q = torch.as_tensor(quat)
    w, x, y, z = q.unbind(dim=-1)
    two = q.new_tensor(2.0)
    matrix = torch.empty(q.shape[:-1] + (3, 3), dtype=q.dtype, device=q.device)
    matrix[..., 0, 0] = 1 - two * (y * y + z * z)
    matrix[..., 0, 1] = two * (x * y - w * z)
    matrix[..., 0, 2] = two * (x * z + w * y)
    matrix[..., 1, 0] = two * (x * y + w * z)
    matrix[..., 1, 1] = 1 - two * (x * x + z * z)
    matrix[..., 1, 2] = two * (y * z - w * x)
    matrix[..., 2, 0] = two * (x * z - w * y)
    matrix[..., 2, 1] = two * (y * z + w * x)
    matrix[..., 2, 2] = 1 - two * (x * x + y * y)
    return matrix


def _rpy_to_matrix_wxyz(rpy: Tensor) -> Tensor:
    """Convert roll-pitch-yaw values to world-from-reference-root matrices."""

    roll, pitch, yaw = torch.as_tensor(rpy).unbind(dim=-1)
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
    return matrix


def _rpy_to_quat_wxyz(rpy: Tensor) -> Tensor:
    """Convert batched XYZ roll-pitch-yaw values to wxyz quaternions."""

    roll, pitch, yaw = torch.as_tensor(rpy).unbind(dim=-1)
    half = rpy.new_tensor(0.5)
    cr, sr = torch.cos(roll * half), torch.sin(roll * half)
    cp, sp = torch.cos(pitch * half), torch.sin(pitch * half)
    cy, sy = torch.cos(yaw * half), torch.sin(yaw * half)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )


def _quat_inverse_wxyz(quat: Tensor) -> Tensor:
    q = torch.as_tensor(quat)
    norm_sq = torch.sum(torch.square(q), dim=-1, keepdim=True).clamp_min(1.0e-12)
    conjugate = q * q.new_tensor([1.0, -1.0, -1.0, -1.0])
    return conjugate / norm_sq


def _quat_mul_wxyz(lhs: Tensor, rhs: Tensor) -> Tensor:
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _axis_angle_from_quat_wxyz(quat: Tensor) -> Tensor:
    """Convert normalized-or-nearly-normalized wxyz quaternions to axis-angle."""

    q = torch.as_tensor(quat)
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1.0e-12)
    q = torch.where(q[..., :1] < 0.0, -q, q)
    w = q[..., :1].clamp(-1.0, 1.0)
    xyz = q[..., 1:]
    angle = 2.0 * torch.acos(w)
    sin_half = torch.sqrt(torch.clamp(1.0 - torch.square(w), min=1.0e-12))
    regular = xyz / sin_half * angle
    small = torch.abs(angle) < 1.0e-5
    return torch.where(small, 2.0 * xyz, regular)


def _rotate_inverse_wxyz(quat: Tensor, vectors: Tensor) -> Tensor:
    """Rotate world-frame vectors into the inverse of a wxyz quaternion frame."""

    matrix_w = _quat_to_matrix_wxyz(quat)
    return torch.matmul(matrix_w.transpose(-1, -2), vectors.unsqueeze(-1)).squeeze(-1)


class ParallelismReferenceManager:
    """Owns 24-frame parallelism references and exposes the current phase frame."""

    def __init__(
        self,
        env,
        cfg: ParallelismCfg | None = None,
        *,
        command_name: str = "base_velocity",
        plan_batch_size: int | None = None,
        terrain_grid_size: int = 151,
        terrain_resolution: float = 0.01,
        autostart: bool = True,
    ) -> None:
        self.env = _env_root(env)
        self.cfg = cfg or ParallelismCfg()
        self.command_name = str(command_name)
        self.plan_batch_size = int(plan_batch_size or getattr(getattr(self.env, "cfg", None), "parallelism_plan_batch_size", 64))
        self.device = torch.device(getattr(self.env, "device", "cpu"))
        self.num_envs = int(getattr(self.env, "num_envs"))
        self.horizon = int(self.cfg.horizon)
        self.dt = float(self.cfg.dt)
        self.terrain_grid_size = int(terrain_grid_size)
        self.terrain_resolution = float(terrain_resolution)

        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._cached_cycle = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.plan_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.standstill_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.standstill_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.plan_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.plan_valid_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.plan_reject_counts = torch.zeros(self.num_envs, 6, dtype=torch.long, device=self.device)
        self.plan_collision_counts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.plan_per_leg_valid_count = torch.zeros(self.num_envs, 4, dtype=torch.long, device=self.device)
        self.plan_per_leg_collision_count = torch.zeros(self.num_envs, 4, dtype=torch.long, device=self.device)
        self._manual_episode_length = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._step_reference_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.root_pos_w = torch.zeros(self.num_envs, self.horizon, 3, dtype=torch.float32, device=self.device)
        self.root_rpy_w = torch.zeros_like(self.root_pos_w)
        self.joint_pos = torch.zeros(self.num_envs, self.horizon, 12, dtype=torch.float32, device=self.device)
        self.foot_pos_w = torch.zeros(self.num_envs, self.horizon, 4, 3, dtype=torch.float32, device=self.device)
        self.contact_state = torch.ones(self.num_envs, self.horizon, 4, dtype=torch.bool, device=self.device)
        self.valid = torch.zeros(self.num_envs, self.horizon, dtype=torch.bool, device=self.device)
        self._step_joint_pos = torch.zeros(self.num_envs, 12, dtype=torch.float32, device=self.device)
        self._step_joint_vel = torch.zeros_like(self._step_joint_pos)
        self._step_foot_pos_w = torch.zeros(self.num_envs, 4, 3, dtype=torch.float32, device=self.device)
        self._step_root_pos_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self._step_root_rpy_w = torch.zeros_like(self._step_root_pos_w)
        self._step_root_lin_vel_b_policy = torch.zeros_like(self._step_root_pos_w)
        self._step_root_ang_vel_b_policy = torch.zeros_like(self._step_root_pos_w)
        self._install_env_reset_hook()

        if autostart:
            self.reset()

    def refresh(self) -> None:
        episode_length = self._episode_length()
        planning_stride = max(self.horizon - 1, 1)
        cycle = torch.div(episode_length, planning_stride, rounding_mode="floor")
        phase = torch.remainder(episode_length, planning_stride)
        reset_mask = (episode_length == 0) & ((~self._initialized) | (self._cached_cycle != 0))
        needs_plan = (~self._initialized) | reset_mask | (cycle != self._cached_cycle)
        env_ids = needs_plan.nonzero(as_tuple=False).flatten()
        if int(env_ids.numel()) > 0:
            self._plan(env_ids, cycle.index_select(0, env_ids))
        self.phase.copy_(phase.to(dtype=torch.long))

    def reset(self, env_ids: Sequence[int] | Tensor | None = None) -> None:
        ids = _as_env_ids(env_ids, num_envs=self.num_envs, device=self.device)
        if int(ids.numel()) == 0:
            return
        self.on_environment_reset(ids)
        self._plan(ids, torch.zeros_like(ids, dtype=torch.long, device=self.device))

    def on_environment_reset(self, env_ids: Sequence[int] | Tensor | None = None) -> None:
        """Invalidate references and clear episode state after an environment reset."""

        ids = _as_env_ids(env_ids, num_envs=self.num_envs, device=self.device)
        if int(ids.numel()) == 0:
            return
        self._manual_episode_length[ids] = 0
        self.phase[ids] = 0
        self.standstill_latched[ids] = False
        self.standstill_count[ids] = 0
        self._cached_cycle[ids] = -1
        self._initialized[ids] = False
        self._step_reference_valid[ids] = False

    def mark_command_changed(self, env_mask: Sequence[int] | Tensor | None = None, *_, **__) -> None:
        ids = _as_env_ids(env_mask, num_envs=self.num_envs, device=self.device)
        if int(ids.numel()) == 0:
            return
        self._cached_cycle[ids] = -1
        self._initialized[ids] = False
        self._step_reference_valid[ids] = False

    def _install_env_reset_hook(self) -> None:
        original = getattr(self.env, "reset", None)
        if original is None or not callable(original) or getattr(original, "_parallelism_reference_reset_hook_wrapped", False):
            return

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            result = original(*args, **kwargs)
            env_ids = kwargs.get("env_ids", None)
            if env_ids is None and args:
                env_ids = args[0]
            self.reset(env_ids)
            return result

        wrapped._parallelism_reference_reset_hook_wrapped = True
        self.env.reset = wrapped

    def step(self) -> None:
        self._manual_episode_length += 1
        self.refresh()

    def prepare_step_reference(self) -> None:
        """Cache the target frame and current-to-next velocities before physics."""

        self.refresh()
        start_phase = self.phase
        target_phase = torch.clamp(start_phase + 1, max=self.horizon - 1)
        self._step_joint_pos.copy_(self._take(self.joint_pos, target_phase))
        start_joint = self._take(self.joint_pos, start_phase)
        target_joint = self._take(self.joint_pos, target_phase)
        self._step_joint_vel.copy_((target_joint - start_joint) / max(self.dt, 1.0e-6))
        self._step_foot_pos_w.copy_(self._take(self.foot_pos_w, target_phase))
        self._step_root_pos_w.copy_(self._take(self.root_pos_w, target_phase))
        self._step_root_rpy_w.copy_(self._take(self.root_rpy_w, target_phase))
        self._step_root_lin_vel_b_policy.copy_(
            self._root_velocity_b_policy(self.root_pos_w, self.root_rpy_w, start_phase, target_phase)
        )
        self._step_root_ang_vel_b_policy.copy_(
            self._angular_velocity_b_policy(self.root_rpy_w, start_phase, target_phase)
        )
        self._step_reference_valid[:] = True

    @property
    def next_joint_pos(self) -> Tensor:
        self.refresh()
        next_phase = torch.clamp(self.phase + 1, max=self.horizon - 1)
        return self._take(self.joint_pos, next_phase)

    @property
    def step_joint_pos(self) -> Tensor:
        if not bool(torch.all(self._step_reference_valid)):
            self.prepare_step_reference()
        return self._step_joint_pos

    @property
    def step_joint_vel(self) -> Tensor:
        if not bool(torch.all(self._step_reference_valid)):
            self.prepare_step_reference()
        return self._step_joint_vel

    @property
    def step_foot_pos_w(self) -> Tensor:
        if not bool(torch.all(self._step_reference_valid)):
            self.prepare_step_reference()
        return self._step_foot_pos_w

    @property
    def step_root_pos_w(self) -> Tensor:
        if not bool(torch.all(self._step_reference_valid)):
            self.prepare_step_reference()
        return self._step_root_pos_w

    @property
    def step_root_rpy_w(self) -> Tensor:
        if not bool(torch.all(self._step_reference_valid)):
            self.prepare_step_reference()
        return self._step_root_rpy_w

    @property
    def step_root_lin_vel_b_policy(self) -> Tensor:
        if not bool(torch.all(self._step_reference_valid)):
            self.prepare_step_reference()
        return self._step_root_lin_vel_b_policy

    @property
    def step_root_ang_vel_b_policy(self) -> Tensor:
        if not bool(torch.all(self._step_reference_valid)):
            self.prepare_step_reference()
        return self._step_root_ang_vel_b_policy

    @property
    def current_root_pos_b_policy(self) -> Tensor:
        """Next reference root position relative to the live policy root frame."""

        self.refresh()
        target_phase = torch.clamp(self.phase + 1, max=self.horizon - 1)
        reference_pos_w = self._take(self.root_pos_w, target_phase)
        robot = self._robot()
        policy_pos_w = torch.as_tensor(robot.data.root_pos_w, dtype=reference_pos_w.dtype, device=reference_pos_w.device)
        policy_quat_w = torch.as_tensor(robot.data.root_quat_w, dtype=reference_pos_w.dtype, device=reference_pos_w.device)
        return _rotate_inverse_wxyz(policy_quat_w, reference_pos_w - policy_pos_w)

    @property
    def current_root_rot_b_policy(self) -> Tensor:
        """Next reference root orientation relative to the live policy root frame."""

        self.refresh()
        target_phase = torch.clamp(self.phase + 1, max=self.horizon - 1)
        reference_rpy_w = self._take(self.root_rpy_w, target_phase)
        reference_quat_w = _rpy_to_quat_wxyz(reference_rpy_w)
        robot = self._robot()
        policy_quat_w = torch.as_tensor(robot.data.root_quat_w, dtype=reference_quat_w.dtype, device=reference_quat_w.device)
        relative_quat = _quat_mul_wxyz(_quat_inverse_wxyz(policy_quat_w), reference_quat_w)
        return _axis_angle_from_quat_wxyz(relative_quat)

    @property
    def current_joint_pos(self) -> Tensor:
        self.refresh()
        return self._current_take(self.joint_pos)

    @property
    def current_joint_vel(self) -> Tensor:
        self.refresh()
        next_phase = torch.clamp(self.phase + 1, max=self.horizon - 1)
        current = self._take(self.joint_pos, self.phase)
        nxt = self._take(self.joint_pos, next_phase)
        return (nxt - current) / max(self.dt, 1.0e-6)

    @property
    def current_root_pos_w(self) -> Tensor:
        self.refresh()
        return self._current_take(self.root_pos_w)

    @property
    def current_root_rpy_w(self) -> Tensor:
        self.refresh()
        return self._current_take(self.root_rpy_w)

    @property
    def current_foot_pos_w(self) -> Tensor:
        self.refresh()
        return self._current_take(self.foot_pos_w)

    @property
    def current_contact_state(self) -> Tensor:
        self.refresh()
        return self._current_take(self.contact_state)

    @property
    def current_root_lin_vel_b(self) -> Tensor:
        """Backward-compatible alias for the live-policy-frame reference velocity."""

        return self.current_root_lin_vel_b_policy

    @property
    def current_root_lin_vel_b_policy(self) -> Tensor:
        """Reference linear velocity expressed in the current policy root frame."""

        self.refresh()
        next_phase = torch.clamp(self.phase + 1, max=self.horizon - 1)
        return self._root_velocity_b_policy(self.root_pos_w, self.root_rpy_w, self.phase, next_phase)

    @property
    def current_root_ang_vel_b(self) -> Tensor:
        """Backward-compatible alias for the live-policy-frame reference angular velocity."""

        return self.current_root_ang_vel_b_policy

    @property
    def current_root_ang_vel_b_policy(self) -> Tensor:
        """Reference angular velocity expressed in the current policy root frame."""

        self.refresh()
        next_phase = torch.clamp(self.phase + 1, max=self.horizon - 1)
        return self._angular_velocity_b_policy(self.root_rpy_w, self.phase, next_phase)

    def _episode_length(self) -> Tensor:
        value = getattr(self.env, "episode_length_buf", None)
        if value is None:
            return self._manual_episode_length
        return torch.as_tensor(value, dtype=torch.long, device=self.device).reshape(self.num_envs)

    def _robot(self):
        return self.env.scene["robot"]

    def _command(self, env_ids: Tensor) -> Tensor:
        command_manager = getattr(self.env, "command_manager", None)
        if command_manager is None:
            return torch.zeros(int(env_ids.numel()), 3, dtype=torch.float32, device=self.device)
        if hasattr(command_manager, "get_command"):
            command = command_manager.get_command(self.command_name)
        else:
            command = getattr(command_manager, self.command_name)
        command = torch.as_tensor(command, dtype=torch.float32, device=self.device)
        if command.ndim != 2 or int(command.shape[-1]) < 3:
            raise ValueError("Parallelism command must have shape [batch, 3 or more]")
        return command[:, :3].index_select(0, env_ids).contiguous()

    def _state(self, env_ids: Tensor) -> ParallelismState:
        robot = self._robot()
        root_pos = torch.as_tensor(robot.data.root_pos_w, dtype=torch.float32, device=self.device).index_select(0, env_ids)
        root_quat = torch.as_tensor(robot.data.root_quat_w, dtype=torch.float32, device=self.device).index_select(0, env_ids)
        roll, pitch = extract_roll_pitch_batch(root_quat)
        yaw = extract_yaw_batch(root_quat)
        root_rpy = torch.stack((roll, pitch, yaw), dim=-1)
        joint = torch.as_tensor(robot.data.joint_pos, dtype=torch.float32, device=self.device)
        joint = _reorder_joint_to_planner(joint, getattr(robot, "joint_names", None)).index_select(0, env_ids)
        foot_pos = self._measured_foot_pos_w(robot, env_ids)
        return ParallelismState(root_pos_w=root_pos, root_rpy_w=root_rpy, joint_pos=joint, foot_pos_w=foot_pos)

    def _standard_stand_state(
        self,
        state: ParallelismState,
        terrain: ParallelismTerrain,
        env_ids: Tensor,
    ) -> ParallelismState:
        """Build the canonical flat standing pose used after a failed plan."""

        root_pos = state.root_pos_w.clone()
        root_rpy = state.root_rpy_w.clone()
        root_rpy[:, :2] = 0.0
        foot_pos = state.foot_pos_w
        if foot_pos is None:
            foot_pos = fk_go2(state.root_pos_w, state.root_rpy_w, state.joint_pos).foot_pos_w
        query = query_height_semantic_valid(terrain, foot_pos[..., :2].reshape(foot_pos.shape[0], -1, 2))
        heights = query.height.reshape_as(foot_pos[..., 2])
        valid = query.valid.reshape_as(foot_pos[..., 2])
        support_height = torch.where(valid, heights, foot_pos[..., 2]).mean(dim=-1)
        root_pos[:, 2] = support_height + float(self.cfg.root_clearance_m)

        robot = self._robot()
        default_joint = getattr(getattr(robot, "data", None), "default_joint_pos", None)
        if default_joint is None:
            joint_pos = state.joint_pos
        else:
            default_joint = torch.as_tensor(default_joint, dtype=state.joint_pos.dtype, device=self.device)
            if int(default_joint.shape[0]) == self.num_envs:
                default_joint = default_joint.index_select(0, env_ids)
            joint_pos = _reorder_joint_to_planner(default_joint, getattr(robot, "joint_names", None))
            if int(joint_pos.shape[0]) != int(state.joint_pos.shape[0]):
                joint_pos = state.joint_pos
        canonical_foot = fk_go2(root_pos, root_rpy, joint_pos).foot_pos_w
        foot_query = query_height_semantic_valid(
            terrain,
            canonical_foot[..., :2].reshape(canonical_foot.shape[0], -1, 2),
        )
        foot_height = foot_query.height.reshape_as(canonical_foot[..., 2])
        foot_valid = foot_query.valid.reshape_as(canonical_foot[..., 2])
        canonical_foot = canonical_foot.clone()
        canonical_foot[..., 2] = torch.where(foot_valid, foot_height, canonical_foot[..., 2])
        root_pos[:, 2] = canonical_foot[..., 2].mean(dim=-1) + float(self.cfg.root_clearance_m)
        return ParallelismState(
            root_pos_w=root_pos,
            root_rpy_w=root_rpy,
            joint_pos=joint_pos,
            foot_pos_w=canonical_foot,
        )

    def _measured_foot_pos_w(self, robot, env_ids: Tensor) -> Tensor | None:
        body_pos = getattr(getattr(robot, "data", None), "body_pos_w", None)
        if body_pos is None:
            return None
        body_pos = torch.as_tensor(body_pos, dtype=torch.float32, device=self.device)
        body_ids = None
        if hasattr(robot, "find_bodies"):
            try:
                body_ids, body_names = robot.find_bodies(".*_foot")
            except Exception:  # noqa: BLE001 - fall back to the common Go2 body layout.
                body_ids = None
        if body_ids is None:
            foot_pos = body_pos[:, -4:]
        else:
            body_ids = torch.as_tensor(body_ids, dtype=torch.long, device=self.device)
            order = _order_indices(body_names, _PLANNER_FOOT_ORDER, device=self.device)
            if order is not None:
                body_ids = body_ids.index_select(0, order)
            foot_pos = body_pos.index_select(1, body_ids)
        return foot_pos.index_select(0, env_ids)

    def _terrain(self, root_pos: Tensor, env_ids: Tensor | None = None) -> ParallelismTerrain:
        n = int(root_pos.shape[0])
        ids = (
            torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
            if env_ids is not None
            else None
        )
        scanner = self._semantic_height_scanner()
        data = getattr(scanner, "data", None) if scanner is not None else None
        ray_hits_source = getattr(data, "ray_hits_w", None)
        height_source = getattr(data, "elevation_map", None)
        semantic_source = getattr(getattr(scanner, "data", None), "semantic_map", None) if scanner is not None else None
        valid_source = getattr(getattr(scanner, "data", None), "valid_mask", None) if scanner is not None else None
        if (ray_hits_source is not None or height_source is not None) and semantic_source is not None:
            origin_xy = None
            yaw = None
            if ray_hits_source is not None:
                ray_hits = torch.as_tensor(ray_hits_source, dtype=torch.float32, device=self.device)
                if ray_hits.ndim != 3 or int(ray_hits.shape[-1]) != 3:
                    raise ValueError(f"semantic_height_scanner ray_hits_w must have shape [B,H*W,3], got {tuple(ray_hits.shape)}")
                if ids is not None and int(ray_hits.shape[0]) == self.num_envs:
                    ray_hits = ray_hits.index_select(0, ids)
                side = int(round(float(ray_hits.shape[1]) ** 0.5))
                if side * side != int(ray_hits.shape[1]):
                    raise ValueError(f"semantic_height_scanner ray count {int(ray_hits.shape[1])} is not a square grid")
                ray_grid = ray_hits.reshape(int(ray_hits.shape[0]), side, side, 3)
                finite_ray = torch.isfinite(ray_grid).all(dim=-1)
                height = ray_grid[..., 2]
                origin_xy = ray_grid[:, 0, 0, :2]
                if side > 1:
                    step_xy = ray_grid[:, 0, 1, :2] - ray_grid[:, 0, 0, :2]
                    yaw = torch.atan2(step_xy[:, 1], step_xy[:, 0])
            else:
                height = torch.as_tensor(height_source, dtype=torch.float32, device=self.device)
                finite_ray = torch.isfinite(height)
            semantic = torch.as_tensor(semantic_source, dtype=torch.long, device=self.device)
            if ids is not None and int(height.shape[0]) == self.num_envs:
                height = height.index_select(0, ids)
                finite_ray = finite_ray.index_select(0, ids)
            if ids is not None and int(semantic.shape[0]) == self.num_envs:
                semantic = semantic.index_select(0, ids)
            if valid_source is not None:
                valid_source = torch.as_tensor(valid_source, dtype=torch.bool, device=self.device)
                if ids is not None and int(valid_source.shape[0]) == self.num_envs:
                    valid_source = valid_source.index_select(0, ids)
            if height.ndim == 2:
                height = height.unsqueeze(0).expand(n, -1, -1)
            if semantic.ndim == 2:
                semantic = semantic.unsqueeze(0).expand(n, -1, -1)
            side = int(height.shape[-1])
            resolution = self._scanner_resolution(scanner, fallback=self.terrain_resolution)
            valid = (
                torch.as_tensor(valid_source, dtype=torch.bool, device=self.device)
                if valid_source is not None
                else finite_ray
            )
            if valid.ndim == 2:
                valid = valid.unsqueeze(0).expand(n, -1, -1)
            height = torch.nan_to_num(height, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            side = self.terrain_grid_size
            resolution = self.terrain_resolution
            height = torch.zeros(n, side, side, dtype=torch.float32, device=self.device)
            semantic = torch.zeros(n, side, side, dtype=torch.long, device=self.device)
            valid = torch.ones(n, side, side, dtype=torch.bool, device=self.device)
            origin_xy = None
            yaw = None
        half_extent = 0.5 * float(side - 1) * resolution
        origin = torch.zeros(n, 3, dtype=torch.float32, device=self.device)
        if origin_xy is None or int(origin_xy.shape[0]) != n:
            origin[:, 0] = root_pos[:, 0] - half_extent
            origin[:, 1] = root_pos[:, 1] - half_extent
        else:
            origin[:, :2] = origin_xy.to(dtype=origin.dtype, device=origin.device)
        if yaw is None or int(yaw.shape[0]) != n:
            yaw = torch.zeros(n, dtype=torch.float32, device=self.device)
        else:
            yaw = yaw.to(dtype=torch.float32, device=self.device)
        return ParallelismTerrain(
            height_w=height,
            semantic_id=semantic,
            valid_mask=valid,
            origin_w=origin,
            yaw_w=yaw,
            resolution=resolution,
        )

    def _plan(self, env_ids: Tensor, cycle: Tensor) -> None:
        batch_size = max(int(self.plan_batch_size), 1)
        for start in range(0, int(env_ids.numel()), batch_size):
            subset = env_ids[start : start + batch_size]
            subset_cycle = cycle[start : start + batch_size]
            live_state = self._state(subset)
            if live_state.foot_pos_w is None:
                live_state = ParallelismState(
                    root_pos_w=live_state.root_pos_w,
                    root_rpy_w=live_state.root_rpy_w,
                    joint_pos=live_state.joint_pos,
                    foot_pos_w=fk_go2(live_state.root_pos_w, live_state.root_rpy_w, live_state.joint_pos).foot_pos_w,
                )
            terrain = self._terrain(live_state.root_pos_w, subset)
            standard_state = self._standard_stand_state(live_state, terrain, subset)
            latched = self.standstill_latched.index_select(0, subset)
            latch_root = latched[:, None]
            latch_root_rpy = latched[:, None]
            latch_joint = latched[:, None]
            latch_foot = latched[:, None, None]
            state = ParallelismState(
                root_pos_w=torch.where(latch_root, standard_state.root_pos_w, live_state.root_pos_w),
                root_rpy_w=torch.where(latch_root_rpy, standard_state.root_rpy_w, live_state.root_rpy_w),
                joint_pos=torch.where(latch_joint, standard_state.joint_pos, live_state.joint_pos),
                foot_pos_w=torch.where(latch_foot, standard_state.foot_pos_w, live_state.foot_pos_w),
            )
            trajectory = plan_trajectory(
                state,
                self._command(subset),
                terrain,
                self.cfg,
                terrain_following_mask=self._terrain_following_mask(subset),
            )
            self.root_pos_w[subset] = trajectory.root_pos_w
            self.root_rpy_w[subset] = trajectory.root_rpy_w
            self.joint_pos[subset] = _reorder_joint_from_planner(trajectory.joint_pos, getattr(self._robot(), "joint_names", None))
            self.foot_pos_w[subset] = trajectory.foot_pos_w
            self.contact_state[subset] = trajectory.contact_state
            # The first frame of every episode/cycle is the measured robot state.
            # This prevents a reset from displaying a planned pose before the robot
            # has reached the planner's first sample.
            self.root_pos_w[subset, 0] = state.root_pos_w
            self.root_rpy_w[subset, 0] = state.root_rpy_w
            self.joint_pos[subset, 0] = _reorder_joint_from_planner(
                state.joint_pos,
                getattr(self._robot(), "joint_names", None),
            )
            if state.foot_pos_w is not None:
                self.foot_pos_w[subset, 0] = state.foot_pos_w
            self.valid[subset] = trajectory.valid[:, None].expand(-1, self.horizon)
            self.standstill_latched[subset] = ~trajectory.valid
            self._update_standstill_count(subset, trajectory.valid)
            self.plan_valid[subset] = trajectory.valid
            self.plan_valid_count[subset] = trajectory.diagnostics.candidate_valid.sum(dim=(1, 2)).to(dtype=torch.long)
            self.plan_reject_counts[subset] = trajectory.diagnostics.candidate_reject_bits.sum(dim=(1, 2)).to(dtype=torch.long)
            self.plan_collision_counts[subset] = trajectory.diagnostics.candidate_collision_bits.any(dim=-1).sum(dim=(1, 2)).to(dtype=torch.long)
            self.plan_per_leg_valid_count[subset] = trajectory.diagnostics.candidate_valid.sum(dim=2).to(dtype=torch.long)
            self.plan_per_leg_collision_count[subset] = trajectory.diagnostics.candidate_collision_bits.any(dim=-1).sum(dim=2).to(dtype=torch.long)
            self._cached_cycle[subset] = subset_cycle
            self._initialized[subset] = True
            self.plan_count[subset] += 1

    def _update_standstill_count(self, env_ids: Tensor, trajectory_valid: Tensor) -> None:
        """Update consecutive failed-replan counts without changing command state."""

        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        valid = torch.as_tensor(trajectory_valid, dtype=torch.bool, device=self.device).reshape(-1)
        current = self.standstill_count.index_select(0, ids)
        updated = torch.where(valid, torch.zeros_like(current), current + 1)
        self.standstill_count[ids] = updated

    def _semantic_height_scanner(self):
        scene = getattr(self.env, "scene", None)
        if scene is None:
            return None
        sensors = getattr(scene, "sensors", None)
        if sensors is not None:
            try:
                return sensors["semantic_height_scanner"]
            except Exception:  # noqa: BLE001 - Isaac containers and test doubles are both duck-typed.
                scanner = getattr(sensors, "semantic_height_scanner", None)
                if scanner is not None:
                    return scanner
        try:
            return scene["semantic_height_scanner"]
        except Exception:  # noqa: BLE001
            return getattr(scene, "semantic_height_scanner", None)

    def _scene_terrain(self):
        scene = getattr(self.env, "scene", None)
        if scene is None:
            return None
        terrain = getattr(scene, "terrain", None)
        if terrain is not None:
            return terrain
        try:
            return scene["terrain"]
        except Exception:  # noqa: BLE001
            return None

    def _terrain_type_names(self) -> tuple[str, ...] | None:
        terrain = self._scene_terrain()
        generator = getattr(getattr(terrain, "cfg", None), "terrain_generator", None) if terrain is not None else None
        if generator is None:
            cfg = getattr(getattr(self.env, "cfg", None), "scene", None)
            terrain_cfg = getattr(cfg, "terrain", None)
            generator = getattr(terrain_cfg, "terrain_generator", None)
        sub_terrains = getattr(generator, "sub_terrains", None)
        if not isinstance(sub_terrains, dict):
            return None
        return tuple(str(name) for name in sub_terrains.keys())

    def _terrain_following_mask(self, env_ids: Tensor) -> Tensor:
        terrain = self._scene_terrain()
        terrain_types = getattr(terrain, "terrain_types", None) if terrain is not None else None
        names = self._terrain_type_names()
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        if terrain_types is None or names is None:
            return torch.zeros(int(ids.numel()), dtype=torch.bool, device=self.device)
        type_tensor = torch.as_tensor(terrain_types, dtype=torch.long, device=self.device).reshape(-1)
        if int(type_tensor.numel()) == self.num_envs:
            type_tensor = type_tensor.index_select(0, ids)
        elif int(type_tensor.numel()) != int(ids.numel()):
            return torch.zeros(int(ids.numel()), dtype=torch.bool, device=self.device)
        valid_type = (type_tensor >= 0) & (type_tensor < len(names))
        flat_indices = [idx for idx, name in enumerate(names) if str(name).lower() == "flat"]
        if not flat_indices:
            return valid_type
        flat_tensor = torch.tensor(flat_indices, dtype=torch.long, device=self.device)
        is_flat = (type_tensor[:, None] == flat_tensor[None]).any(dim=1)
        return valid_type & ~is_flat

    def _scanner_resolution(self, scanner, *, fallback: float) -> float:
        pattern_cfg = getattr(getattr(scanner, "cfg", None), "pattern_cfg", None)
        resolution = getattr(pattern_cfg, "resolution", None)
        if resolution is None:
            return float(fallback)
        return float(resolution)

    def _take(self, values: Tensor, phase: Tensor) -> Tensor:
        batch = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        return values[batch, phase.to(dtype=torch.long, device=self.device)]

    def _current_take(self, values: Tensor) -> Tensor:
        return self._take(values, self.phase)

    def _root_velocity_b_policy(
        self,
        root_pos_w: Tensor,
        root_rpy_w: Tensor,
        start_phase: Tensor,
        target_phase: Tensor,
    ) -> Tensor:
        start_pos = self._take(root_pos_w, start_phase)
        target_pos = self._take(root_pos_w, target_phase)
        vel_w = (target_pos - start_pos) / max(self.dt, 1.0e-6)
        ref_rpy = self._take(root_rpy_w, start_phase)
        ref_matrix_w = _rpy_to_matrix_wxyz(ref_rpy)
        ref_vel_b = torch.matmul(ref_matrix_w.transpose(-1, -2), vel_w.unsqueeze(-1)).squeeze(-1)
        ref_vel_w = torch.matmul(ref_matrix_w, ref_vel_b.unsqueeze(-1)).squeeze(-1)
        policy_quat_w = torch.as_tensor(
            self._robot().data.root_quat_w,
            dtype=ref_vel_w.dtype,
            device=ref_vel_w.device,
        )
        policy_matrix_w = _quat_to_matrix_wxyz(policy_quat_w)
        return torch.matmul(policy_matrix_w.transpose(-1, -2), ref_vel_w.unsqueeze(-1)).squeeze(-1)

    def _angular_velocity_b_policy(
        self,
        root_rpy_w: Tensor,
        start_phase: Tensor,
        target_phase: Tensor,
    ) -> Tensor:
        rpy = self._take(root_rpy_w, start_phase)
        target_rpy = self._take(root_rpy_w, target_phase)
        ref_rpy_rate = (target_rpy - rpy) / max(self.dt, 1.0e-6)
        ref_matrix_w = _rpy_to_matrix_wxyz(rpy)
        ref_ang_vel_w = torch.matmul(ref_matrix_w, ref_rpy_rate.unsqueeze(-1)).squeeze(-1)
        policy_quat_w = torch.as_tensor(
            self._robot().data.root_quat_w,
            dtype=ref_ang_vel_w.dtype,
            device=ref_ang_vel_w.device,
        )
        policy_matrix_w = _quat_to_matrix_wxyz(policy_quat_w)
        return torch.matmul(policy_matrix_w.transpose(-1, -2), ref_ang_vel_w.unsqueeze(-1)).squeeze(-1)


def get_parallelism_reference_manager(env) -> ParallelismReferenceManager:
    root = _env_root(env)
    manager = getattr(root, "parallelism_reference_manager", None)
    if manager is None:
        manager = ParallelismReferenceManager(root)
        root.parallelism_reference_manager = manager
    return manager
