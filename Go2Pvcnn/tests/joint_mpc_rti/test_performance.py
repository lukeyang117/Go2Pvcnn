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


def test_perf_probe_declares_fixed_shape_cuda_event_contract() -> None:
    source = Path(
        "Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_perf_probe.py"
    ).read_text()

    assert 'parser.add_argument("--num-envs", type=int, default=1024)' in source
    assert 'parser.add_argument("--horizon", type=int, default=16)' in source
    assert 'parser.add_argument("--steps", type=int, default=1000)' in source
    assert 'parser.add_argument("--warmup", type=int, default=100)' in source
    assert '"nonfinite_count"' in source
    assert '"peak_allocated_mib"' in source
    assert '"total_ms"' in source
    assert "torch.cuda.Event" in source


def test_full_refresh_probe_times_exact_field_and_mpc_together() -> None:
    source = Path(
        "Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_full_refresh_probe.py"
    ).read_text()

    assert 'parser.add_argument("--num-envs", type=int, default=1024)' in source
    assert 'parser.add_argument("--horizon", type=int, default=16)' in source
    assert 'parser.add_argument("--steps", type=int, default=1000)' in source
    assert "cache.update_rows" in source
    assert "runner.run" in source
    assert '"field_version_increment"' in source
    assert '"full_total_ms"' in source
    assert '"full_mean_ms"' in source
    assert '"full_p95_ms"' in source
    assert '"full_max_ms"' in source
