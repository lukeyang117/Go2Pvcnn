from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_policy_benchmark_cli_contract() -> None:
    source = (ROOT / "scripts/policy_benchmark.py").read_text(encoding="utf-8")
    for flag in ("--experiment-type", "--checkpoint", "--suite", "--num-envs", "--layout-manifest", "--smoke-test"):
        assert flag in source
    assert "AppLauncher.add_app_launcher_args(parser)" in source
    assert "load_optimizer=False" in source
    assert "get_inference_policy" in source


def test_run_script_requires_explicit_checkpoint() -> None:
    source = (ROOT / "scripts/run_policy_benchmark.sh").read_text(encoding="utf-8")
    assert "--experiment-type" in source
    assert "--checkpoint" in source
    assert "-f" in source or "is_file" in source
