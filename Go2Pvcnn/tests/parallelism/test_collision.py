from __future__ import annotations


def test_default_official_collision_shapes_match_go2_usd():
    from extension.parallelism.config import ParallelismCfg

    cfg = ParallelismCfg()
    names = tuple(spec.name for spec in cfg.official_collision_shapes)
    link_types = tuple(spec.link_type for spec in cfg.official_collision_shapes)

    assert names == (
        "thigh_box",
        "fl_calf_upper_cylinder",
        "calf_upper_cylinder",
        "calf_mid_cylinder",
        "calf_lower_cylinder",
        "foot_sphere",
    )
    assert link_types == ("thigh", "calf", "calf", "calf", "calf", "foot")
    assert cfg.collision_margin_m == 0.003
    assert cfg.box_surface_points == 6
    assert cfg.cylinder_layers == 1
    assert cfg.cylinder_angles == 4
    assert cfg.sphere_surface_points == 6


def test_fk_returns_link_poses_for_collision_frames():
    import torch
    from extension.parallelism.kinematics import fk_go2

    root_pos = torch.tensor([[1.0, 2.0, 0.3]], dtype=torch.float32)
    root_rpy = torch.zeros(1, 3)
    joint = torch.tensor([[0.0, 0.8, -1.5] * 4], dtype=torch.float32)

    geometry = fk_go2(root_pos, root_rpy, joint)

    assert geometry.thigh_pos_w.shape == (1, 4, 3)
    assert geometry.thigh_rot_w.shape == (1, 4, 3, 3)
    assert geometry.calf_pos_w.shape == (1, 4, 3)
    assert geometry.calf_rot_w.shape == (1, 4, 3, 3)
    assert geometry.foot_rot_w.shape == (1, 4, 3, 3)
    eye = torch.eye(3).expand(1, 4, 3, 3)
    assert torch.allclose(geometry.thigh_rot_w @ geometry.thigh_rot_w.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(geometry.calf_rot_w @ geometry.calf_rot_w.transpose(-1, -2), eye, atol=1e-5)


def test_official_surface_points_match_default_primitive_samples():
    import torch
    from extension.parallelism.collision import build_official_surface_points_l
    from extension.parallelism.config import ParallelismCfg

    cfg = ParallelismCfg()

    points, mask = build_official_surface_points_l(
        cfg.official_collision_shapes,
        cfg,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    assert points.shape == (6, 6, 3)
    assert mask.shape == (6, 6)
    assert mask.all()
    assert torch.allclose(
        points[0],
        torch.tensor(
            [
                [0.0, 0.0, -0.2130],
                [0.0, 0.0, 0.0],
                [0.0, 0.01225, -0.1065],
                [0.0, -0.01225, -0.1065],
                [0.0170, 0.0, -0.1065],
                [-0.0170, 0.0, -0.1065],
            ]
        ),
        atol=1e-5,
    )
    assert torch.allclose(
        points[5],
        torch.tensor(
            [
                [0.0200, 0.0, 0.0],
                [-0.0240, 0.0, 0.0],
                [-0.0020, 0.0220, 0.0],
                [-0.0020, -0.0220, 0.0],
                [-0.0020, 0.0, 0.0220],
                [-0.0020, 0.0, -0.0220],
            ]
        ),
        atol=1e-5,
    )


def test_official_collision_uses_batched_surface_points():
    import torch
    from types import SimpleNamespace
    from extension.parallelism.collision import official_collision_mask
    from extension.parallelism.config import OfficialCollisionShapeSpec, ParallelismCfg
    from extension.parallelism.types import ParallelismTerrain

    cfg = ParallelismCfg(
        collision_margin_m=0.0,
        contact_tolerant_collision_shape_names=(),
        official_collision_shapes=(
            OfficialCollisionShapeSpec(
                name="foot_sphere",
                leg_name=None,
                link_type="foot",
                shape_type="sphere",
                center_l=(0.0, 0.0, 0.0),
                quat_wxyz_l=(1.0, 0.0, 0.0, 0.0),
                size_l=(0.0, 0.0, 0.0),
                radius_m=0.10,
                height_m=0.0,
            ),
        ),
    )
    terrain = ParallelismTerrain(
        height_w=torch.full((1, 11, 11), 0.10),
        semantic_id=torch.zeros(1, 11, 11, dtype=torch.long),
        valid_mask=torch.ones(1, 11, 11, dtype=torch.bool),
        origin_w=torch.tensor([[-0.5, -0.5, 0.0]]),
        yaw_w=torch.zeros(1),
        resolution=0.1,
    )
    geometry = SimpleNamespace(
        foot_pos_w=torch.zeros(1, 1, 1, 1, 3),
        foot_rot_w=torch.eye(3).view(1, 1, 1, 1, 3, 3),
        thigh_pos_w=torch.zeros(1, 1, 1, 1, 3),
        thigh_rot_w=torch.eye(3).view(1, 1, 1, 1, 3, 3),
        calf_pos_w=torch.zeros(1, 1, 1, 1, 3),
        calf_rot_w=torch.eye(3).view(1, 1, 1, 1, 3, 3),
    )

    ok, bits = official_collision_mask(terrain, geometry, cfg)

    assert ok.shape == (1, 1, 1)
    assert bits.shape == (1, 1, 1, 1)
    assert not bool(ok[0, 0, 0])
    assert bool(bits[0, 0, 0, 0])


def test_swing_collision_mask_preserves_candidate_shape():
    import torch
    from types import SimpleNamespace
    from extension.parallelism.config import ParallelismCfg
    from extension.parallelism.planner import _swing_collision_mask
    from extension.parallelism.root import rollout_root
    from extension.parallelism.types import ParallelismState, ParallelismTerrain

    terrain = ParallelismTerrain(
        height_w=torch.zeros(1, 61, 61),
        semantic_id=torch.zeros(1, 61, 61, dtype=torch.long),
        valid_mask=torch.ones(1, 61, 61, dtype=torch.bool),
        origin_w=torch.tensor([[-3.0, -3.0, 0.0]]),
        yaw_w=torch.zeros(1),
        resolution=0.1,
    )
    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, 0.30]], dtype=torch.float32),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.tensor([[0.0, 0.8, -1.5] * 4], dtype=torch.float32),
    )
    cfg = ParallelismCfg(candidates_per_leg=2)
    root = rollout_root(state, torch.zeros(1, 3), terrain, cfg)
    candidate_w = torch.zeros(1, 4, 2, 3)
    candidate_w[..., 2] = 0.0
    candidates = SimpleNamespace(candidate_w=candidate_w)

    ok, bits = _swing_collision_mask(state, root.root_pos_w, root.root_rpy_w, candidates, terrain, cfg)

    assert ok.shape == (1, 4, 2)
    assert bits.shape == (1, 4, 2, len(cfg.official_collision_shapes))
