from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_m1_curriculum_keeps_go2_semantic_gate_and_excluded_flat_dense_terrain():
    source = (ROOT / "m1_ame/m1_ame_curriculum.py").read_text(encoding="utf-8")

    assert "cfg_name" in source
    assert "excluded_terrain_names" in source
    assert "flat_dense_small_obstacles" in source
    assert "body_name_patterns" in source


def test_m1_curriculum_exposes_root_distance_and_transition_diagnostics():
    source = "\n".join(
        (
            (ROOT / "m1_ame/m1_ame_curriculum.py").read_text(encoding="utf-8"),
            (ROOT / "go2_pvcnn/mdp/curriculums.py").read_text(encoding="utf-8"),
        )
    )

    assert "mean_root_distance" in source
    assert "move_up_fraction" in source
    assert "move_down_fraction" in source
    assert "mean_command_speed" in source
    assert "mean_local_root_x" in source
    assert "mean_local_root_y" in source


def test_m1_diagnostics_separates_leg_and_wheel_statistics():
    source = (ROOT / "m1_ame/m1_diagnostics.py").read_text(encoding="utf-8")

    assert "leg_torque_rms" in source
    assert "wheel_torque_rms" in source
    assert "leg_velocity_rms" in source
    assert "wheel_velocity_rms" in source
    assert "action_rate_leg_rms" in source
    assert "action_rate_wheel_rms" in source
