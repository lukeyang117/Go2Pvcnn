from __future__ import annotations

import pytest

from evaluation.model_adapters import (
    AmeAmpBenchmarkAdapter,
    AmeBenchmarkAdapter,
    DefaultBenchmarkAdapter,
    build_benchmark_adapter,
    resolve_model_spec,
)


def test_resolve_model_spec_rejects_directory(tmp_path) -> None:
    with pytest.raises(ValueError, match="explicit checkpoint file"):
        resolve_model_spec("ppo", tmp_path)


def test_resolve_model_spec_maps_checkpoint(tmp_path) -> None:
    checkpoint = tmp_path / "model_10.pt"
    checkpoint.write_bytes(b"checkpoint")
    spec = resolve_model_spec("amp", checkpoint)
    assert spec.experiment == "parallelism_tracking_cross_large_complex_amp"
    assert spec.observation_kind == "deployment"
    assert spec.requires_reference_manager is True


def test_resolve_model_spec_maps_ame_adapters(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")

    ame = resolve_model_spec("ame", checkpoint)
    ame_amp = resolve_model_spec("ame_amp", checkpoint)

    assert ame.experiment == "cross_large_complex_ame"
    assert ame.task_id == "Isaac-Go2-Cross-Large-Complex-PPO-v0"
    assert isinstance(build_benchmark_adapter(ame), AmeBenchmarkAdapter)
    assert ame_amp.experiment == "parallelism_tracking_cross_large_complex_ame_amp"
    assert ame_amp.task_id == "Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-AME-Go2-v0"
    assert isinstance(build_benchmark_adapter(ame_amp), AmeAmpBenchmarkAdapter)
    assert isinstance(build_benchmark_adapter(resolve_model_spec("ppo", checkpoint)), DefaultBenchmarkAdapter)


def test_resolve_model_spec_maps_semloco_to_default_adapter(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")

    semloco = resolve_model_spec("semloco", checkpoint)

    assert semloco.experiment == "cross_large_complex_semloco"
    assert semloco.task_id == "Isaac-Go2-Cross-Large-Complex-SemLoco-v0"
    assert semloco.observation_kind == "deployment"
    assert semloco.requires_reference_manager is True
    assert semloco.adapter_kind == "default"
    assert isinstance(build_benchmark_adapter(semloco), DefaultBenchmarkAdapter)


def test_ame_amp_adapter_rejects_non_full_resume() -> None:
    with pytest.raises(ValueError, match="complete AME-AMP checkpoint"):
        AmeAmpBenchmarkAdapter.require_full_amp_resume("legacy_policy_warm_start")
