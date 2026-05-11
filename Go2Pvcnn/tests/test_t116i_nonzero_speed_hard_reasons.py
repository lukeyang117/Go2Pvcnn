from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"
for _path in (REPO_ROOT, GO2PVCNN_ROOT):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from extension.batched_together_planner import TogetherPlannerConfig, TogetherPlannerTerrain, TogetherRobotState, plan_segment
from extension.batched_together_planner.parameterization import (
    T116_MODE_APPROACH_SMALL,
    T116_MODE_BYPASS_OBSTACLE,
    T116_MODE_CROSS_SMALL,
    T116_MODE_CRUISE,
)
from extension.batched_together_planner.planner import _t116_candidate_tables
from extension.batched_together_planner.types import T116_CANDIDATE_COUNT, TogetherPlannerStatus, TogetherTerrainSemanticId
from extension.viz import go2_foostep_planner as viewer
from tests.fixtures.viewer_runtime_diagnostics import format_hard_reason_summary


def _flat_fixture(
    device: torch.device,
    *,
    batch_size: int = 1,
    semantic_id: int | None = None,
) -> tuple[TogetherPlannerTerrain, TogetherRobotState]:
    dtype = torch.float32
    heightmap = torch.zeros((batch_size, 33, 33), device=device, dtype=dtype)
    semantic_map = None
    if semantic_id is not None:
        semantic_map = torch.full((batch_size, 33, 33), int(semantic_id), device=device, dtype=torch.long)
    terrain = TogetherPlannerTerrain.from_heightmap(
        heightmap,
        world_x_range=(-0.8, 0.8),
        world_y_range=(-0.8, 0.8),
        semantic_map=semantic_map,
    )
    root_pos = torch.zeros((batch_size, 3), device=device, dtype=dtype)
    root_pos[:, 2] = 0.30
    root_rpy = torch.zeros((batch_size, 3), device=device, dtype=dtype)
    foot_template = torch.tensor(
        (
            (0.1934, 0.0465, 0.0),
            (0.1934, -0.0465, 0.0),
            (-0.1934, 0.0465, 0.0),
            (-0.1934, -0.0465, 0.0),
        ),
        device=device,
        dtype=dtype,
    )
    foot_pos = foot_template.unsqueeze(0).expand(batch_size, -1, -1).clone()
    joint_angles = torch.zeros((batch_size, 12), device=device, dtype=dtype)
    return terrain, TogetherRobotState(root_pos=root_pos, root_rpy=root_rpy, foot_pos=foot_pos, joint_angles=joint_angles)


def test_t116i_nonzero_command_beta_tables_do_not_include_zero() -> None:
    modes = torch.tensor(
        [T116_MODE_CRUISE, T116_MODE_APPROACH_SMALL, T116_MODE_CROSS_SMALL, T116_MODE_BYPASS_OBSTACLE],
        dtype=torch.long,
    )

    betas, routes, _signs = _t116_candidate_tables(
        modes,
        device=torch.device("cpu"),
        dtype=torch.float32,
        cfg=TogetherPlannerConfig(),
    )

    expected = torch.tensor(
        [
            [1.00, 0.80, 0.60, 0.40, 0.20],
            [0.80, 0.65, 0.50, 0.35, 0.20],
            [0.60, 0.50, 0.40, 0.30, 0.20],
            [0.60, 0.40, 0.60, 0.40, 0.20],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(betas, expected)
    assert torch.all(betas > 0.0)
    assert routes.shape == (4, T116_CANDIDATE_COUNT)


def test_t116i_zero_command_still_uses_hold_standstill_path() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    terrain, state = _flat_fixture(device, batch_size=1)
    command = torch.zeros((1, 3), device=device)

    result = plan_segment(terrain, state, command, TogetherPlannerConfig())

    torch.testing.assert_close(result.root_pos, state.root_pos[:, None, :].expand_as(result.root_pos))
    torch.testing.assert_close(result.foot_pos, state.foot_pos[:, None, :, :].expand_as(result.foot_pos))
    assert torch.all(result.contact_state == 1.0)
    assert torch.all(result.feasible)
    assert result.selected_beta is not None
    assert result.selected_beta.item() >= 0.0


def test_t116i_result_exposes_fixed_shape_hard_reason_schema() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    terrain, state = _flat_fixture(device, batch_size=2)
    command = torch.tensor([[0.4, 0.0, 0.0], [0.0, 0.25, 0.0]], device=device)

    result = plan_segment(terrain, state, command, TogetherPlannerConfig())

    hard_reason_count = len(result.selected_hard_reason_mask[0])
    assert result.candidate_hard_reason_mask.shape == (2, T116_CANDIDATE_COUNT, hard_reason_count)
    assert result.selected_hard_reason_mask.shape == (2, hard_reason_count)
    assert result.candidate_hard_rank_cost.shape == (2, T116_CANDIDATE_COUNT)
    assert result.selected_hard_rank_cost.shape == (2,)
    assert result.selected_candidate_index.shape == (2,)
    assert result.candidate_hard_reason_mask.dtype == torch.bool
    assert result.selected_hard_reason_mask.dtype == torch.bool


def test_t116i_all_hard_selection_uses_hard_rank_and_keeps_infeasible_status() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    terrain, state = _flat_fixture(device, batch_size=1, semantic_id=TogetherTerrainSemanticId.LARGE)
    command = torch.tensor([[0.4, 0.0, 0.0]], device=device)

    result = plan_segment(terrain, state, command, TogetherPlannerConfig())

    assert int(result.status.item()) == int(TogetherPlannerStatus.ALL_INFEASIBLE)
    assert result.selected_beta is not None
    assert float(result.selected_beta.item()) > 0.0
    expected_idx = torch.argmin(result.candidate_hard_rank_cost, dim=1)
    assert int(result.selected_candidate_index.item()) == int(expected_idx.item())
    expected_reason = result.candidate_hard_reason_mask.gather(
        1,
        expected_idx.view(1, 1, 1).expand(1, 1, result.candidate_hard_reason_mask.shape[-1]),
    ).squeeze(1)
    torch.testing.assert_close(result.selected_hard_reason_mask, expected_reason)
    assert bool(result.selected_hard_reason_mask.any().item())


def test_t116i_infeasible_terminal_format_includes_selected_and_candidate_reasons() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    terrain, state = _flat_fixture(device, batch_size=1, semantic_id=TogetherTerrainSemanticId.LARGE)
    command = torch.tensor([[0.4, 0.0, 0.0]], device=device)

    result = plan_segment(terrain, state, command, TogetherPlannerConfig())
    formatted = format_hard_reason_summary(result)

    assert "selected_hard_reasons=" in formatted
    assert "candidate_hard_rank=" in formatted
    assert "candidate_hard_reasons=" in formatted
    assert "none" not in formatted.split("selected_hard_reasons=", 1)[1].split(" ", 1)[0]


def test_t116i_production_viewer_plan_line_includes_hard_reason_fields_when_infeasible() -> None:
    device = torch.device("cpu")
    terrain, state = _flat_fixture(device, batch_size=1, semantic_id=TogetherTerrainSemanticId.LARGE)
    command = torch.tensor([[0.4, 0.0, 0.0]], device=device)
    result = viewer._adapt_together_result_for_viewer(plan_segment(terrain, state, command, TogetherPlannerConfig()))

    line = viewer._format_viewer_plan_line(
        backend="together",
        cycle=7,
        command=command,
        result=result,
        semantic_diagnostics={},
    )

    assert line.startswith("[Viewer][Plan] ")
    assert "status=ALL_INFEASIBLE" in line
    assert "selected_hard_reasons=" in line
    assert "selected_hard_rank_cost=" in line
    assert "candidate_hard_rank=" in line
    assert "candidate_hard_reasons=" in line


def test_t116i_flat_nonzero_commands_are_not_standstill_and_have_no_hard_reason() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    command_values = (
        (0.3, 0.0, 0.0),
        (-0.3, 0.0, 0.0),
        (0.0, 0.25, 0.0),
        (0.0, -0.25, 0.0),
        (0.0, 0.0, 0.3),
        (0.0, 0.0, -0.3),
    )
    terrain, state = _flat_fixture(device, batch_size=len(command_values))
    command = torch.tensor(command_values, device=device)

    result = plan_segment(terrain, state, command, TogetherPlannerConfig())
    root_delta = result.root_pos[:, -1, :2] - result.root_pos[:, 0, :2]
    yaw_delta = result.root_rpy[:, -1, 2] - result.root_rpy[:, 0, 2]
    motion = torch.linalg.vector_norm(root_delta, dim=-1) + yaw_delta.abs()

    assert torch.all(motion > 0.05)
    assert result.selected_beta is not None
    assert torch.all(result.selected_beta > 0.0)
    assert result.mode is not None
    assert torch.all(result.mode == int(T116_MODE_CRUISE))
    assert result.selected_route is not None
    assert torch.all(result.selected_route == 0)
    assert result.command_direction_violation is not None
    assert not torch.any(result.command_direction_violation)
    assert not torch.any(result.selected_hard_reason_mask)


def test_t116i_headless_small_obstacle_crossing_runtime_all_directions() -> None:
    pytest.importorskip("isaaclab.app")
    from tests.fixtures import viewer_runtime_diagnostics as viewer_diag

    runtime = viewer_diag.make_real_runtime_fixture(num_envs=2, planner_backend="together")
    try:
        reports = runtime.grounded_crossing_runtime_sequences_by_command(
            command_names=("forward", "backward", "lateral_left", "lateral_right"),
            semantic_class="small",
        )
    finally:
        runtime.close()

    assert set(reports) == {"forward", "backward", "lateral_left", "lateral_right"}
    for name, report in reports.items():
        assert all(beta > 0.0 for beta in report.selected_beta_sequence), name
        assert T116_MODE_CROSS_SMALL in report.mode_sequence, name
        assert report.cross_small_success_count > 0, name
        assert report.touchdown_on_small_count == 0, name
        assert report.per_leg_touchdown_on_small_count == (0, 0, 0, 0), name
        assert report.foot_small_collision_count == 0, name
        assert report.per_leg_foot_small_collision_count == (0, 0, 0, 0), name
        assert report.base_small_penetration_count == 0, name
        assert report.base_path_crosses_small_flag == 0, name
        assert report.base_min_clearance_to_small_m >= 0.0, name
        assert min(report.per_leg_min_clearance_to_small_m) >= 0.0, name
        for plan in report.sampled_plans:
            grounded = plan.grounded_crossing
            if grounded is None or not bool(torch.as_tensor(grounded.cross_small_success).reshape(-1)[0].item()):
                continue
            if grounded.selected_hard_reason_mask is not None:
                assert not torch.any(torch.as_tensor(grounded.selected_hard_reason_mask, dtype=torch.bool)), name
