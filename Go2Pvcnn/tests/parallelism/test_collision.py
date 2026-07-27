from __future__ import annotations


def test_default_collision_ellipsoid_specs_are_named_and_grouped():
    from extension.parallelism.config import ParallelismCfg

    cfg = ParallelismCfg()
    names = tuple(spec.name for spec in cfg.collision_ellipsoids)
    link_types = tuple(spec.link_type for spec in cfg.collision_ellipsoids)

    assert names == (
        "thigh_body_inner",
        "thigh_body_mid",
        "thigh_body_outer",
        "thigh_outer_cap",
        "calf_knee_cap",
        "calf_upper_bar",
        "calf_mid_bar",
        "calf_lower_bar",
        "calf_ankle_cap",
        "foot_pad",
    )
    assert link_types == ("thigh", "thigh", "thigh", "thigh", "calf", "calf", "calf", "calf", "calf", "foot")
    assert cfg.collision_probe_count == 5
    assert cfg.collision_margin_m == 0.003


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


def test_probe_offset_checks_four_neighbors_and_center():
    import torch
    from extension.parallelism.collision import build_ellipsoid_probe_l
    from extension.parallelism.config import EllipsoidSpec

    specs = (EllipsoidSpec("e", "foot", (1.0, 2.0, 3.0), (0.1, 0.2, 0.3), (0.4, 0.5)),)

    probes = build_ellipsoid_probe_l(specs, dtype=torch.float32, device=torch.device("cpu"))

    assert probes.shape == (1, 5, 3)
    assert torch.allclose(
        probes[0],
        torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [1.4, 2.0, 3.0],
                [0.6, 2.0, 3.0],
                [1.0, 2.5, 3.0],
                [1.0, 1.5, 3.0],
            ]
        ),
    )


def test_ellipsoid_collision_uses_link_local_height_points():
    import torch
    from types import SimpleNamespace
    from extension.parallelism.collision import ellipsoid_collision_mask
    from extension.parallelism.config import EllipsoidSpec, ParallelismCfg
    from extension.parallelism.types import ParallelismTerrain

    cfg = ParallelismCfg(
        collision_margin_m=0.0,
        collision_ellipsoids=(EllipsoidSpec("foot_pad", "foot", (0.0, 0.0, 0.0), (0.10, 0.10, 0.10), (0.05, 0.05)),),
    )
    terrain = ParallelismTerrain(
        height_w=torch.full((1, 11, 11), 0.05),
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

    ok, bits = ellipsoid_collision_mask(terrain, geometry, cfg)

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
    assert bits.shape == (1, 4, 2, len(cfg.collision_ellipsoids))
