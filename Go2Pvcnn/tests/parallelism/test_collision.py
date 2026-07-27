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
