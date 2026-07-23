from __future__ import annotations

from pathlib import Path


def test_timing_summary_reports_required_acceptance_metrics() -> None:
    from extension.joint_mpc_rti.diagnostics.metrics import timing_summary

    summary = timing_summary([1.0, 2.0, 3.0, 4.0])

    assert summary["total_ms"] == 10.0
    assert summary["mean_ms"] == 2.5
    assert summary["p50_ms"] == 2.5
    assert summary["p95_ms"] >= summary["p50_ms"]
    assert summary["p99_ms"] >= summary["p95_ms"]
    assert summary["max_ms"] == 4.0


def test_performance_entrypoints_declare_frozen_h30_workload() -> None:
    for name in ("joint_mpc_rti_perf_probe.py", "joint_mpc_rti_full_refresh_probe.py"):
        source = Path("Go2Pvcnn/tests/joint_mpc_rti", name).read_text()
        assert 'parser.add_argument("--num-envs", type=int, default=1024)' in source
        assert 'parser.add_argument("--horizon", type=int, default=30)' in source
        assert 'parser.add_argument("--steps", type=int, default=1000)' in source
        assert "torch.cuda.Event" in source


def test_production_pipeline_does_not_use_removed_control_rollout_api() -> None:
    source = Path("Go2Pvcnn/extension/joint_mpc_rti/planner.py").read_text()

    assert "trajectory.control" not in source
    assert "recovery" not in source.lower()
