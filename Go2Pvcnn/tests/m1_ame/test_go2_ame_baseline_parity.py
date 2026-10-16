from pathlib import Path


M1_ROOT = Path(__file__).resolve().parents[2]
GO2_ROOT = Path("/share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn/Go2Pvcnn")


def test_pure_ppo_baseline_files_match_go2_reference():
    relative_paths = (
        "ame_baseline/actor_critic_ame.py",
        "ame_baseline/ame_env_cfg.py",
        "ame_baseline/ame_env_wrapper.py",
        "ame_baseline/ame_observations.py",
        "ame_baseline/ame_runner.py",
        "ame_baseline/ame_train_cfg.py",
        "tracking/cross_large_complex_ppo_env_cfg.py",
        "tracking/mdp/gait.py",
        "tracking/mdp/distillation.py",
        "tracking/mdp/policy_geometry_rewards.py",
    )

    for relative_path in relative_paths:
        copied = (M1_ROOT / relative_path).read_text(encoding="utf-8").rstrip()
        reference = (GO2_ROOT / relative_path).read_text(encoding="utf-8").rstrip()
        assert copied == reference, relative_path
