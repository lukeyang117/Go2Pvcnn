from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch


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
    assert "semantic[:, 70:81, 70:81] = 1" in source
    assert "semantic[:, 15:56, 105:146] = 2" in source
    assert '"field_version_increment"' in source
    assert '"full_total_ms"' in source
    assert '"full_mean_ms"' in source
    assert '"full_p95_ms"' in source
    assert '"full_max_ms"' in source


def test_production_lq_linearization_has_compiled_fixed_shape_path() -> None:
    source = Path(
        "Go2Pvcnn/extension/joint_mpc_rti/planner.py"
    ).read_text()

    assert "_COMPILED_BUILD_LQ_PROBLEM" in source
    assert "_COMPILED_ADD_LARGE_OBSTACLE_LINEARIZATION" in source
    assert "_COMPILED_ADD_SMALL_OBSTACLE_LINEARIZATION" in source
    assert "_COMPILED_ADD_FOOT_TERRAIN_LINEARIZATION" in source
    assert "_COMPILED_ADD_ROOT_SUPPORT_LINEARIZATION" in source
    assert "_COMPILED_QUERY_LINEARIZATION_GEOMETRY" in source
    assert "_COMPILED_DESIRED_CONTROL" in source
    assert "_COMPILED_STANCE_ANCHOR_TARGETS" in source
    assert "_COMPILED_ROLLOUT_CONTROLS" in source
    assert "_linearization_function(" in source


def test_compile_kernels_true_falls_back_to_eager_on_cpu() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.planner import step
    from .helpers import make_command, make_flat_field, make_state

    eager_cfg = JointMpcRtiCfg()
    compiled_cfg = copy.deepcopy(eager_cfg)
    compiled_cfg.solver.compile_kernels = True
    state = make_state(2)
    command = make_command(2, vx=0.2, vy=0.1, yaw=0.2)
    field = make_flat_field(2)

    eager = step(state, command, field, None, eager_cfg)
    fallback = step(state, command, field, None, compiled_cfg)

    torch.testing.assert_close(fallback.pending_reference.root_pos_w, eager.pending_reference.root_pos_w)
    torch.testing.assert_close(fallback.pending_reference.joint_angles, eager.pending_reference.joint_angles)
    torch.testing.assert_close(fallback.pending_reference.foot_pos_w, eager.pending_reference.foot_pos_w)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compiled/eager parity requires CUDA")
def test_compiled_fixed_shape_matches_eager_first_future_reference() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.planner import step
    from .helpers import make_command, make_flat_field, make_state

    eager_cfg = JointMpcRtiCfg()
    compiled_cfg = copy.deepcopy(eager_cfg)
    compiled_cfg.solver.compile_kernels = True
    state = make_state(2, device="cuda")
    command = make_command(2, vx=0.2, vy=0.1, yaw=0.2, device="cuda")
    field = make_flat_field(2, device="cuda")

    eager = step(state, command, field, None, eager_cfg)
    compiled = step(state, command, field, None, compiled_cfg)

    torch.testing.assert_close(
        compiled.pending_reference.root_pos_w,
        eager.pending_reference.root_pos_w,
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    torch.testing.assert_close(
        compiled.pending_reference.joint_angles,
        eager.pending_reference.joint_angles,
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    torch.testing.assert_close(
        compiled.pending_reference.foot_pos_w,
        eager.pending_reference.foot_pos_w,
        atol=3.0e-5,
        rtol=3.0e-5,
    )
