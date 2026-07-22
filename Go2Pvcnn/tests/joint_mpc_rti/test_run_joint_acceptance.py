from __future__ import annotations

import pytest


def test_stage_a_runner_builds_all_signed_command_cells() -> None:
    from .run_joint_acceptance import build_stage_a_cells

    cells = build_stage_a_cells("small", shape="cuboid")

    assert len(cells) == 275
    assert len({cell.command for cell in cells}) == 275
    assert cells[0].key[:2] == ("small", "cuboid")


def test_stage_a_runner_rejects_missing_universal_metric_cell() -> None:
    from .run_joint_acceptance import StageAReport, require_complete_stage_a_matrix

    report = StageAReport(cells=())

    with pytest.raises(AssertionError, match="missing applicable command cells"):
        require_complete_stage_a_matrix(report, scenarios=("flat",), commands=((0.0, 0.0, 0.0),))


def test_stage_a_runner_preserves_raw_command_key_and_metric_numerators() -> None:
    from .run_joint_acceptance import MetricCell, StageAReport

    cell = MetricCell(
        key=("small", "sphere", 0.2, -0.3, 0.5),
        command=(0.2, -0.3, 0.5),
        values={"foot_collision_frame_rate": 0.0},
        numerators={"foot_collision_frame_rate": 0},
        denominators={"foot_collision_frame_rate": 10},
        valid_count=10,
        na_reason=None,
    )
    report = StageAReport(cells=(cell,))

    assert report.cells[0].command == (0.2, -0.3, 0.5)
    assert report.cells[0].numerators["foot_collision_frame_rate"] == 0
    assert report.cells[0].denominators["foot_collision_frame_rate"] == 10
