from __future__ import annotations

import pytest
import torch
from dataclasses import fields, replace

from extension.joint_mpc_rti.config import JointMpcRtiCfg


def _state(*, root_x: float = 0.0) -> torch.Tensor:
    state = torch.tensor([[root_x, 0.0, 0.34, 0.0, 0.0, 0.0, *([0.0, 0.8, -1.5] * 4)]])
    return state


def _field_with_point_obstacle(point_w: torch.Tensor, height_m: float, *, size: int = 151):
    from extension.joint_mpc_rti.terrain.perceptive_field import build_perceptive_field
    from extension.joint_mpc_rti.types import JointMpcFieldFrame

    cfg = JointMpcRtiCfg()
    height = torch.zeros(1, size, size)
    semantic = torch.zeros(1, size, size, dtype=torch.long)
    center = (size - 1) // 2
    index_x = int(round(float(point_w[0]) / cfg.terrain.resolution)) + center
    index_y = int(round(float(point_w[1]) / cfg.terrain.resolution)) + center
    height[0, index_x, index_y] = float(height_m)
    semantic[0, index_x, index_y] = 1
    frame = JointMpcFieldFrame(
        origin_w=torch.zeros(1, 3),
        yaw_w=torch.zeros(1),
        timestamp=torch.zeros(1),
        refresh_id=torch.zeros(1, dtype=torch.long),
    )
    return build_perceptive_field(
        height,
        semantic,
        torch.ones_like(semantic, dtype=torch.bool),
        frame,
        cfg,
    )


def test_collision_geometry_exposes_spheres_capsules_sole_and_base_obb() -> None:
    from extension.joint_mpc_rti.model.go2_kinematics import go2_collision_geometry

    state = _state()
    geometry = go2_collision_geometry(state[:, :3], state[:, 3:6], state[:, 6:])

    assert geometry.foot_center_w.shape == (1, 4, 3)
    assert geometry.sole_corners_w.shape == (1, 4, 4, 3)
    assert geometry.knee_center_w.shape == (1, 4, 3)
    assert geometry.calf_endpoints_w.shape == (1, 4, 2, 3)
    assert geometry.thigh_endpoints_w.shape == (1, 4, 2, 3)
    assert geometry.base_center_w.shape == (1, 3)
    assert geometry.base_rotation_w.shape == (1, 3, 3)
    assert geometry.base_half_extents.shape == (1, 3)


def test_interval_collision_detects_safe_endpoints_with_unsafe_middle() -> None:
    from extension.joint_mpc_rti.terrain.swept_safety import (
        evaluate_nodes,
        evaluate_swept_intervals,
    )

    cfg = JointMpcRtiCfg()
    field = _field_with_point_obstacle(torch.tensor([0.0, 0.0]), 0.35, size=201)
    trajectory = torch.stack((_state(root_x=-0.5)[0], _state(root_x=0.5)[0]), dim=0)[None]

    endpoint = evaluate_nodes(trajectory, field, cfg)
    swept = evaluate_swept_intervals(trajectory, field, cfg)

    assert endpoint.safe.all()
    assert not swept.safe.all()
    assert swept.collision_by_part["base"].any()


def test_small_semantic_is_crossable_in_swing_but_forbidden_for_support_sole() -> None:
    from extension.joint_mpc_rti.model.go2_kinematics import go2_collision_geometry
    from extension.joint_mpc_rti.terrain.swept_safety import evaluate_nodes

    cfg = JointMpcRtiCfg()
    state = _state()
    foot = go2_collision_geometry(state[:, :3], state[:, 3:6], state[:, 6:]).foot_center_w
    field = _field_with_point_obstacle(foot[0, 0], 0.0)
    swing = evaluate_nodes(
        state[:, None],
        field,
        cfg,
        contact_state=torch.zeros(1, 1, 4, dtype=torch.bool),
    )
    contact = torch.zeros(1, 1, 4, dtype=torch.bool)
    contact[..., 0] = True
    support = evaluate_nodes(state[:, None], field, cfg, contact_state=contact)

    assert swing.safe.all()
    assert not support.sole_safe[..., 0].any()
    assert support.collision_by_part["foot"].any()
    assert not support.safe.all()


@pytest.mark.parametrize("part", ("foot", "knee", "calf", "thigh", "base"))
def test_each_part_has_independent_clearance_and_reject_bit(part: str) -> None:
    from extension.joint_mpc_rti.model.go2_kinematics import go2_collision_geometry
    from extension.joint_mpc_rti.terrain.swept_safety import evaluate_nodes

    cfg = JointMpcRtiCfg()
    state = _state()
    geometry = go2_collision_geometry(state[:, :3], state[:, 3:6], state[:, 6:])
    if part == "foot":
        point = geometry.foot_center_w[0, 0]
        obstacle_height = float(point[2] - cfg.terrain.foot_radius_m + 0.01)
    elif part == "knee":
        point = geometry.knee_center_w[0, 0]
        obstacle_height = float(point[2] - cfg.terrain.knee_radius_m + 0.01)
    elif part == "calf":
        point = geometry.calf_endpoints_w[0, 0].mean(dim=0)
        obstacle_height = float(point[2] - cfg.terrain.calf_radius_m + 0.01)
    elif part == "thigh":
        point = geometry.thigh_endpoints_w[0, 0].mean(dim=0)
        obstacle_height = float(point[2] - cfg.terrain.thigh_radius_m + 0.01)
    else:
        point = geometry.base_center_w[0].clone()
        point[2] -= geometry.base_half_extents[0, 2]
        obstacle_height = float(point[2] + 0.01)
    field = _field_with_point_obstacle(point, obstacle_height)

    result = evaluate_nodes(state[:, None], field, cfg)

    assert result.collision_by_part[part].any()
    assert result.minimum_clearance_by_part[part].min() < 0.0
    assert not result.safe.all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity requires a GPU")
def test_swept_safety_cuda_matches_cpu_reject_bits_and_clearance() -> None:
    from extension.joint_mpc_rti.terrain.swept_safety import evaluate_swept_intervals

    cfg = JointMpcRtiCfg()
    field = _field_with_point_obstacle(torch.tensor([0.0, 0.0]), 0.35, size=201)
    trajectory = torch.stack((_state(root_x=-0.5)[0], _state(root_x=0.5)[0]), dim=0)[None]
    cpu = evaluate_swept_intervals(trajectory, field, cfg)
    cuda_field = replace(
        field,
        **{
            item.name: getattr(field, item.name).cuda()
            for item in fields(field)
            if isinstance(getattr(field, item.name), torch.Tensor)
        },
    )
    cuda = evaluate_swept_intervals(trajectory.cuda(), cuda_field, cfg)

    assert torch.equal(cuda.safe.cpu(), cpu.safe)
    assert torch.equal(cuda.underresolved.cpu(), cpu.underresolved)
    for part in ("foot", "knee", "calf", "thigh", "base"):
        assert torch.equal(
            cuda.collision_by_part[part].cpu(), cpu.collision_by_part[part]
        )
        torch.testing.assert_close(
            cuda.minimum_clearance_by_part[part].cpu(),
            cpu.minimum_clearance_by_part[part],
            atol=2.0e-5,
            rtol=0.0,
        )
