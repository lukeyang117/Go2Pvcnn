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
