from __future__ import annotations

import json


def test_formal_command_matrix_has_all_275_signed_combinations() -> None:
    from .scenario_matrix import COMMANDS, VX, VY, YAW

    assert len(VX) == 11
    assert len(VY) == 5
    assert len(YAW) == 5
    assert len(COMMANDS) == 275
    assert (-1.0, -0.5, -1.0) in COMMANDS
    assert (1.0, 0.5, 1.0) in COMMANDS


def test_flat_and_small_use_the_same_metric_registry() -> None:
    from .joint_metrics import applicable_metrics
    from .run_joint_acceptance import metric_registry

    assert metric_registry("flat") == applicable_metrics("flat")
    assert metric_registry("small") == applicable_metrics("small")


def test_all_command_flat_cells_preserve_every_signed_command() -> None:
    from .run_joint_acceptance import build_cells
    from .scenario_matrix import COMMANDS

    cells = build_cells(stage="flat", all_commands=True)

    assert len(cells) == 275
    assert tuple(cell.command for cell in cells) == COMMANDS
    assert all(cell.scenario == "flat" for cell in cells)


def test_acceptance_report_round_trips_complete_metric_metadata() -> None:
    from .joint_metrics import MetricResult
    from .run_joint_acceptance import AcceptanceCell, AcceptanceReport, CellReport

    cell = AcceptanceCell(scenario="flat", command=(0.2, 0.0, 0.0))
    metric = MetricResult(
        name="stance_ground_gap",
        value=0.001,
        numerator=0.001,
        denominator=31,
        valid_count=31,
        applicable=True,
        na_reason=None,
        threshold=0.012,
        passed=True,
        worst_case_key=cell.key,
    )
    report = AcceptanceReport(
        stage="flat",
        code_ref="abc123",
        cells=(CellReport(cell=cell, metrics={metric.name: metric}, passed=True),),
    )

    payload = json.loads(report.to_json())

    assert payload["stage"] == "flat"
    assert payload["passed"] is True
    assert payload["cells"][0]["cell"]["command"] == [0.2, 0.0, 0.0]
    assert payload["cells"][0]["metrics"]["stance_ground_gap"]["valid_count"] == 31
    assert payload["cells"][0]["metrics"]["stance_ground_gap"]["worst_case_key"] == list(cell.key)


def test_cell_progress_is_one_machine_readable_line(capsys) -> None:
    from .run_joint_acceptance import AcceptanceCell, emit_cell_progress

    cell = AcceptanceCell(scenario="small", command=(0.2, 0.0, 0.0), phase=7, shape="sphere", offset=0.04)

    emit_cell_progress(cell=cell, index=2, total=9, passed=False)

    event = json.loads(capsys.readouterr().out)
    assert event == {
        "event": "cell_complete",
        "index": 2,
        "total": 9,
        "key": list(cell.key),
        "passed": False,
    }


def test_cli_accepts_every_planned_matrix_selector() -> None:
    from .run_joint_acceptance import parse_args

    args = parse_args(
        [
            "--stage", "small",
            "--all-commands",
            "--all-shapes",
            "--all-phases",
            "--all-offsets",
            "--steps", "160",
            "--heartbeat-seconds", "5",
            "--report-json", "/tmp/report.json",
        ]
    )

    assert args.stage == "small"
    assert args.all_commands and args.all_shapes and args.all_phases and args.all_offsets
    assert args.steps == 160
    assert args.heartbeat_seconds == 5.0
    assert str(args.report_json) == "/tmp/report.json"


def test_small_formal_selector_expands_shape_phase_and_offset_dimensions() -> None:
    from .run_joint_acceptance import build_cells
    from .scenario_matrix import SMALL_OFFSETS, SMALL_PHASES, SMALL_SHAPES

    cells = build_cells(
        stage="small",
        all_commands=False,
        all_shapes=True,
        all_phases=True,
        all_offsets=True,
    )

    assert len(cells) == len(SMALL_SHAPES) * len(SMALL_PHASES) * len(SMALL_OFFSETS) * 3
