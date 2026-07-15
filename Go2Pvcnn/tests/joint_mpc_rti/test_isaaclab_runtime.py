from __future__ import annotations

from pathlib import Path


def test_isaaclab_probe_declares_real_joint_backend_acceptance_contract() -> None:
    source = Path(
        "Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_isaaclab_probe.py"
    ).read_text()

    assert 'cfg.planner_backend = "joint_mpc_rti"' in source
    assert '"target_step"' in source
    assert '"field_version_min"' in source
    assert '"reference_finite"' in source
    assert '"planner_ms_mean"' in source
    assert '"--disable-cuda-graph"' in source
    assert '"--trace-stages"' in source
    assert '"--detach-field-observer-after-refresh"' in source
