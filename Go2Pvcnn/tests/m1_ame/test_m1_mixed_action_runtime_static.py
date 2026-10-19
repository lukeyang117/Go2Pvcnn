from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_m1_mixed_action_runtime_probe_checks_action_targets_and_finiteness():
    source = (ROOT / "tests" / "m1_ame" / "m1_mixed_action_runtime_probe.py").read_text(encoding="utf-8")
    for token in (
        "AppLauncher",
        "--num_envs",
        "--steps",
        "gym.make",
        "M1AmeCrossLargeComplexEnvCfg",
        "action_manager",
        "joint_vel_target",
        "torch.isfinite",
    ):
        assert token in source
