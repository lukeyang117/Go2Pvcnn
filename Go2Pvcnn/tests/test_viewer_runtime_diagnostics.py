from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))

from extension.batched_planner.types import LEG_ORDER
from tests.fixtures import viewer_runtime_diagnostics as viewer_diag
from tests.fixtures.viewer_runtime_diagnostics import build_command_cases


def _make_real_runtime_fixture(**kwargs):
    assert hasattr(viewer_diag, "make_real_runtime_fixture")
    return viewer_diag.make_real_runtime_fixture(**kwargs)


@pytest.fixture(scope="module")
def real_runtime():
    runtime = _make_real_runtime_fixture(num_envs=2)
    try:
        yield runtime
    finally:
        runtime.close()


@pytest.fixture(scope="module")
def real_batched_runtime(real_runtime):
    return real_runtime


def test_build_command_cases_includes_forward_command():
    cases = build_command_cases(device=torch.device("cpu"), num_envs=1)

    assert "forward" in cases
    assert cases["forward"].shape == (1, 3)
    assert torch.linalg.vector_norm(cases["forward"]).item() > 0


def test_runtime_resource_error_detection_requires_resource_evidence():
    assert viewer_diag._is_runtime_resource_error(AttributeError("'Articulation' object has no attribute '_data'")) is False
    assert (
        viewer_diag._is_runtime_resource_error(
            RuntimeError(
                "Unable to allocate memory of size 671088640 for mGpuContactPairsDev; "
                "'Articulation' object has no attribute '_data'"
            )
        )
        is True
    )


def test_runtime_app_launcher_init_failure_closes_partial_app_and_clears_state(monkeypatch):
    closed = {"value": False}

    class FakeApp:
        def close(self):
            closed["value"] = True

    class FakeAppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument("--headless", action="store_true", default=False)
            parser.add_argument("--device", type=str, default="cuda:0")

        def __init__(self, args_cli):
            self.app = FakeApp()
            raise RuntimeError("launcher init failed")

    fake_isaaclab = ModuleType("isaaclab")
    fake_app_module = ModuleType("isaaclab.app")
    fake_app_module.AppLauncher = FakeAppLauncher
    fake_isaaclab.app = fake_app_module

    monkeypatch.setattr(viewer_diag, "_APP_STATE", None)
    monkeypatch.setitem(sys.modules, "isaaclab", fake_isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", fake_app_module)

    with pytest.raises(RuntimeError, match="launcher init failed"):
        viewer_diag._ensure_runtime_app(device="cuda:0")

    assert closed["value"] is True
    assert viewer_diag._APP_STATE is None


def test_viewer_forward_command_changes_plan_motion_metrics(real_runtime):
    standstill = real_runtime.plan_case("standstill")
    forward = real_runtime.plan_case("forward")

    assert standstill.summary["standstill"] is True
    assert forward.summary["standstill"] is False
    assert forward.summary["dx"] > 0.05
    assert abs(forward.summary["dx"]) > abs(forward.summary["dy"]) + 0.03


def test_viewer_lateral_command_changes_plan_motion_metrics(real_runtime):
    standstill = real_runtime.plan_case("standstill")
    lateral = real_runtime.plan_case("lateral_left")

    assert standstill.summary["standstill"] is True
    assert lateral.summary["standstill"] is False
    assert lateral.summary["dy"] > 0.05
    assert abs(lateral.summary["dy"]) > abs(lateral.summary["dx"]) + 0.03


def test_viewer_yaw_command_changes_yaw_and_touchdown_metrics(real_runtime):
    yaw_left = real_runtime.plan_case("yaw_left")

    assert yaw_left.summary["standstill"] is False
    assert yaw_left.summary["dyaw"] > 0.05
    assert yaw_left.left_touchdown_mean_y < -0.01
    assert yaw_left.right_touchdown_mean_y > 0.01


def test_viewer_playback_matches_reference_frame_numeric(real_runtime):
    forward = real_runtime.plan_case("forward")
    frame_idx = min(7, forward.result.num_frames - 1)

    readback = real_runtime.playback_sync_authoritative_readback(forward.result, frame_idx=frame_idx)

    torch.testing.assert_close(readback.root_pos_w, forward.result.root_pos_w[:, frame_idx], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(readback.joint_pos, forward.result.joint_angles[:, frame_idx], atol=1e-4, rtol=1e-4)


def test_viewer_standstill_has_no_single_leg_outlier(real_runtime):
    standstill = real_runtime.plan_case("standstill")

    assert standstill.touchdown_xy_delta_norms.max().item() < 1e-5
    assert (standstill.touchdown_xy_delta_norms.max() - standstill.touchdown_xy_delta_norms.min()).item() < 1e-5


def test_viewer_leg_order_matches_planner_contract(real_runtime):
    assert real_runtime.foot_names == LEG_ORDER


def test_viewer_batched_runtime_smoke_preserves_parallel_path(real_batched_runtime):
    batched = real_batched_runtime.plan_batched_cases(["batched_forward", "batched_lateral_left"])

    assert batched.root_pos_w.shape[0] == 2
    assert abs(batched.path_deltas[0, 0].item()) > abs(batched.path_deltas[0, 1].item()) + 0.03
    assert abs(batched.path_deltas[1, 1].item()) > abs(batched.path_deltas[1, 0].item()) + 0.03
