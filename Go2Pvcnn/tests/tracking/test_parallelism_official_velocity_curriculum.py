from pathlib import Path


_CONFIG_SOURCE = Path(__file__).resolve().parents[2] / "tracking" / (
    "parallelism_cross_large_complex_distillation_env_cfg.py"
)
_PARALLELISM_CONFIG_SOURCE = Path(__file__).resolve().parents[2] / "extension" / "parallelism" / "config.py"


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

    assert "weight=1.5" in source
    assert "weight=0.75" in source
    assert "math.sqrt(0.25)" in source


def test_distillation_keeps_commands_long_but_replans_parallelism_independently():
    source = _CONFIG_SOURCE.read_text()
    parallelism_source = _PARALLELISM_CONFIG_SOURCE.read_text()

    assert "resampling_time_range = (100.0, 100.0)" in source
    assert "self.commands.base_velocity.rel_standing_envs = 0.1" in source
    assert "self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)" in source
    assert "replan_interval_steps: int = 23" in parallelism_source
