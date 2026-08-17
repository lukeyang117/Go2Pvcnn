from pathlib import Path


_CONFIG_SOURCE = Path(__file__).resolve().parents[2] / "tracking" / (
    "parallelism_cross_large_complex_distillation_env_cfg.py"
)


def test_distillation_uses_official_linear_velocity_curriculum_only():
    source = _CONFIG_SOURCE.read_text()

    assert "lin_vel_cmd_levels = CurrTerm(go2_mdp.lin_vel_cmd_levels)" in source
    assert "self.curriculum.parallelism_velocity = None" in source


def test_distillation_uses_full_yaw_range_without_yaw_curriculum():
    source = _CONFIG_SOURCE.read_text()

    assert "self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)" in source
    assert "self.commands.base_velocity.limit_ranges.ang_vel_z = (-1.0, 1.0)" in source


def test_distillation_keeps_latest_velocity_rewards():
    source = _CONFIG_SOURCE.read_text()

    assert "weight=2.0" in source
    assert 'params={"command_name": "base_velocity", "std": 0.5}' in source
    assert "weight=1.5" in source
