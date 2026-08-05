"""Termination terms for parallelism tracking."""

from __future__ import annotations

import torch

try:
    from isaaclab.managers import SceneEntityCfg
except Exception:  # noqa: BLE001 - allows lightweight unit tests without an Isaac app.
    class SceneEntityCfg:  # type: ignore[no-redef]
        def __init__(self, name: str, **kwargs) -> None:
            self.name = name
            for key, value in kwargs.items():
                setattr(self, key, value)

from extension.convention import euler_to_quat_batch
from tracking.managers.parallelism_reference_manager import get_parallelism_reference_manager


def _quat_apply_inverse(quat_wxyz: torch.Tensor, vec_w: torch.Tensor) -> torch.Tensor:
    q = torch.as_tensor(quat_wxyz)
    v = torch.as_tensor(vec_w, dtype=q.dtype, device=q.device)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    two = q.new_tensor(2.0)
    rot = torch.empty_like(v)
    rot[:, 0] = (
        (1 - two * (y * y + z * z)) * v[:, 0]
        + two * (x * y + w * z) * v[:, 1]
        + two * (x * z - w * y) * v[:, 2]
    )
    rot[:, 1] = (
        two * (x * y - w * z) * v[:, 0]
        + (1 - two * (x * x + z * z)) * v[:, 1]
        + two * (y * z + w * x) * v[:, 2]
    )
    rot[:, 2] = (
        two * (x * z + w * y) * v[:, 0]
        + two * (y * z - w * x) * v[:, 1]
        + (1 - two * (x * x + y * y)) * v[:, 2]
    )
    return rot


def parallelism_ref_root_z_too_far(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.25,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ref_z = get_parallelism_reference_manager(env).step_root_pos_w[:, 2]
    return torch.abs(asset.data.root_pos_w[:, 2].to(device=ref_z.device, dtype=ref_z.dtype) - ref_z) > float(threshold)


def parallelism_ref_joint_pos_too_far(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.8,
    consecutive_steps: int = 3,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ref = get_parallelism_reference_manager(env).step_joint_pos
    actual = torch.as_tensor(asset.data.joint_pos, dtype=ref.dtype, device=ref.device)
    error = torch.max(torch.abs(actual - ref), dim=-1).values
    violation = error > float(threshold)
    count = getattr(env, "_parallelism_joint_violation_count", None)
    if count is None or count.shape[0] != violation.shape[0]:
        count = torch.zeros(violation.shape[0], dtype=torch.long, device=violation.device)
        env._parallelism_joint_violation_count = count
    required_steps = max(int(consecutive_steps), 1)
    env._parallelism_joint_violation_count = torch.where(
        violation,
        torch.clamp(count + 1, max=required_steps),
        torch.zeros_like(count),
    )
    return env._parallelism_joint_violation_count >= required_steps


def parallelism_ref_foot_z_too_far(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*_foot"),
    threshold: float = 0.25,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ref = get_parallelism_reference_manager(env).step_foot_pos_w[..., 2]
    if getattr(asset_cfg, "body_ids", None) is not None:
        body_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    else:
        body_pos = asset.data.body_pos_w[:, -4:, 2]
    body_pos = torch.as_tensor(body_pos, dtype=ref.dtype, device=ref.device)
    error = torch.max(torch.abs(body_pos - ref), dim=-1).values
    return error > float(threshold)


def parallelism_ref_projected_gravity_too_far(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.8,
    z_only: bool = False,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    manager = get_parallelism_reference_manager(env)
    ref_rpy = manager.step_root_rpy_w
    ref_quat = euler_to_quat_batch(ref_rpy[:, 0], ref_rpy[:, 1], ref_rpy[:, 2])
    root_quat = torch.as_tensor(asset.data.root_quat_w, dtype=ref_quat.dtype, device=ref_quat.device)
    gravity = getattr(asset.data, "GRAVITY_VEC_W", None)
    if gravity is None:
        gravity = torch.tensor((0.0, 0.0, -1.0), dtype=ref_quat.dtype, device=ref_quat.device).expand(root_quat.shape[0], 3)
    gravity = torch.as_tensor(gravity, dtype=ref_quat.dtype, device=ref_quat.device)
    if gravity.ndim == 1:
        gravity = gravity.expand(root_quat.shape[0], 3)
    actual_pg = _quat_apply_inverse(root_quat, gravity)
    ref_pg = _quat_apply_inverse(ref_quat, gravity)
    if bool(z_only):
        diff = torch.abs(actual_pg[:, 2] - ref_pg[:, 2])
    else:
        diff = torch.linalg.vector_norm(actual_pg - ref_pg, dim=-1)
    return diff > float(threshold)
