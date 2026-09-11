from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_policy_benchmark_cli_contract() -> None:
    source = (ROOT / "scripts/policy_benchmark.py").read_text(encoding="utf-8")
    adapter_source = (ROOT / "evaluation/model_adapters.py").read_text(encoding="utf-8")
    for flag in ("--experiment-type", "--checkpoint", "--suite", "--num-envs", "--layout-manifest", "--smoke-test"):
        assert flag in source
    assert "AppLauncher.add_app_launcher_args(parser)" in source
    assert "load_optimizer=False" in adapter_source
    assert "get_inference_policy" in adapter_source


def test_policy_benchmark_supports_ame_adapters() -> None:
    source = (ROOT / "scripts/policy_benchmark.py").read_text(encoding="utf-8")
    assert '"ame"' in source
    assert '"ame_amp"' in source
    assert "build_benchmark_adapter" in source
    assert "adapter.create_env_cfg" in source
    assert "adapter.create_runtime" in source


def test_run_script_requires_explicit_checkpoint() -> None:
    source = (ROOT / "scripts/run_policy_benchmark.sh").read_text(encoding="utf-8")
    assert "--experiment-type" in source
    assert "--checkpoint" in source
    assert "-f" in source or "is_file" in source


def test_launchers_support_ame_and_ame_amp() -> None:
    single = (ROOT / "scripts/run_policy_benchmark.sh").read_text(encoding="utf-8")
    batch = (ROOT / "scripts/run_policy_benchmark_batch.sh").read_text(encoding="utf-8")
    smoke = (ROOT / "tests/evaluation/run_policy_benchmark_1024_smoke.py").read_text(encoding="utf-8")
    for model in ("ame", "ame_amp"):
        assert model in single
        assert model in batch
        assert model in smoke
    assert "Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-AME-Go2-v0" in single


def test_launchers_support_semloco() -> None:
    benchmark = (ROOT / "scripts/policy_benchmark.py").read_text(encoding="utf-8")
    single = (ROOT / "scripts/run_policy_benchmark.sh").read_text(encoding="utf-8")
    batch = (ROOT / "scripts/run_policy_benchmark_batch.sh").read_text(encoding="utf-8")
    smoke = (ROOT / "tests/evaluation/run_policy_benchmark_1024_smoke.py").read_text(encoding="utf-8")

    for source in (benchmark, single, batch, smoke):
        assert "semloco" in source
    assert "Isaac-Go2-Cross-Large-Complex-SemLoco-v0" in single


def test_batch_run_script_shares_one_timestamped_output_directory() -> None:
    source = (ROOT / "scripts/run_policy_benchmark_batch.sh").read_text(encoding="utf-8")
    assert "timestamp=\"$(date +%Y%m%d_%H%M%S)\"" in source
    assert "batch_dir=\"$eval_root/${timestamp}_" in source
    assert '--output-dir "$batch_dir"' in source
    assert 'for suite in "${suites[@]}"' in source


def test_batch_run_script_cleans_process_groups_and_stops_on_failure() -> None:
    source = (ROOT / "scripts/run_policy_benchmark_batch.sh").read_text(encoding="utf-8")
    assert "setsid timeout --foreground" in source
    assert "kill -TERM -- \"-$pid\"" in source
    assert "kill -KILL -- \"-$pid\"" in source
    assert "trap cleanup_on_exit EXIT" in source
    assert "stopping remaining suites" in source
    assert "did not write" in source


def test_batch_run_defaults_to_full_benchmark() -> None:
    source = (ROOT / "scripts/run_policy_benchmark_batch.sh").read_text(encoding="utf-8")
    assert "--max-steps" not in source
    assert '"--smoke-test"' not in source
    assert "suite_timeout=1800" in source


def test_app_close_does_not_mask_active_exception() -> None:
    from scripts.policy_benchmark import _close_simulation_app

    class App:
        def close(self):
            raise SystemExit(0)

    with pytest.raises(RuntimeError, match="root cause"):
        try:
            raise RuntimeError("root cause")
        finally:
            _close_simulation_app(App())


def test_distillation_wrapper_exports_critic_and_context() -> None:
    source = (ROOT / "Go2Pvcnn/scripts/play.py").read_text(encoding="utf-8")
    assert '"critic": critic_obs' in source
    assert '"distillation_context": distillation_context' in source
