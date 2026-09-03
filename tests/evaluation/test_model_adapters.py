from __future__ import annotations

import pytest

from evaluation.model_adapters import resolve_model_spec


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
