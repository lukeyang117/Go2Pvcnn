from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def test_formal_command_matrix_has_all_19_signed_axis_commands() -> None:
    from .scenario_matrix import COMMANDS, VX, VY, YAW

    assert len(VX) == 11
    assert len(VY) == 5
    assert len(YAW) == 5
    assert len(COMMANDS) == 19
    assert (-1.0, 0.0, 0.0) in COMMANDS
    assert (0.0, 0.5, 0.0) in COMMANDS
    assert (0.0, 0.0, 1.0) in COMMANDS
    assert all(sum(value != 0.0 for value in command) <= 1 for command in COMMANDS)


def test_flat_and_small_use_the_same_metric_registry() -> None:
    from .joint_metrics import applicable_metrics
    from .run_joint_acceptance import metric_registry

    assert metric_registry("flat") == applicable_metrics("flat")
    assert metric_registry("small") == applicable_metrics("small")


def test_all_command_flat_cells_preserve_every_signed_command() -> None:
    from .run_joint_acceptance import build_cells
    from .scenario_matrix import COMMANDS

    cells = build_cells(stage="flat", all_commands=True)

    assert len(cells) == 19
    assert tuple(cell.command for cell in cells) == COMMANDS
    assert all(cell.scenario == "flat" for cell in cells)


def test_ranked_commands_cover_zero_and_signed_axis_extremes_without_mixing() -> None:
    from .run_joint_acceptance import RANKED_COMMANDS

    assert RANKED_COMMANDS == (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, -0.5, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    )


def test_flat_acceptance_chunks_cells_without_dropping_or_reordering(monkeypatch) -> None:
    from . import joint_metrics, run_joint_acceptance

    cells = run_joint_acceptance.build_cells(stage="flat", all_commands=True)[:5]
    batch_sizes: list[int] = []

    def fake_simulate(commands, *, steps, device):
        batch_sizes.append(int(commands.shape[0]))
        return list(range(int(commands.shape[0])))

    monkeypatch.setattr(run_joint_acceptance, "simulate_flat_trace", fake_simulate)
    monkeypatch.setattr(run_joint_acceptance, "_slice_trace", lambda trace, index: trace[index])
    monkeypatch.setattr(
        joint_metrics,
        "evaluate_trace",
        lambda trace, *, scenario, key: SimpleNamespace(metrics={}, passed=True),
    )

    report = run_joint_acceptance.run_flat_acceptance(
        cells=cells,
        steps=1,
        device="cpu",
        cell_batch_size=2,
    )

    assert batch_sizes == [2, 2, 1]
    assert tuple(item.cell for item in report.cells) == cells


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
        source="P+A+M",
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
    assert payload["cells"][0]["metrics"]["stance_ground_gap"]["source"] == "P+A+M"


def test_acceptance_report_deserializes_complete_metric_metadata() -> None:
    from .joint_metrics import MetricResult
    from .run_joint_acceptance import (
        AcceptanceCell,
        AcceptanceReport,
        CellReport,
        acceptance_report_from_dict,
    )

    cell = AcceptanceCell(scenario="small", command=(-0.2, 0.0, 0.0), phase=3, shape="cone", offset=0.04)
    metric = MetricResult(
        name="strict_cross_success",
        value=1.0,
        numerator=1.0,
        denominator=161,
        valid_count=161,
        applicable=True,
        na_reason=None,
        threshold=0.95,
        passed=True,
        worst_case_key=cell.key,
        source="P+A+M",
    )
    original = AcceptanceReport(
        stage="small",
        code_ref="abc123",
        cells=(CellReport(cell=cell, metrics={metric.name: metric}, passed=True),),
    )

    restored = acceptance_report_from_dict(json.loads(original.to_json()))

    assert restored == original


def test_cell_shards_form_a_deterministic_complete_partition() -> None:
    from .run_joint_acceptance import build_cells, select_cell_shard

    cells = build_cells(stage="small", all_shapes=True, all_phases=True)[:37]
    shards = tuple(
        select_cell_shard(cells, shard_count=6, shard_index=index)
        for index in range(6)
    )

    boundaries = tuple((len(cells) * index) // 6 for index in range(7))
    assert all(
        shard == cells[boundaries[index] : boundaries[index + 1]]
        for index, shard in enumerate(shards)
    )
    assert {cell.key for shard in shards for cell in shard} == {cell.key for cell in cells}
    assert sum(len(shard) for shard in shards) == len(cells)


def _shard_payloads(cells, *, shard_count: int = 3, code_ref: str = "abc123"):
    from dataclasses import asdict

    from .run_joint_acceptance import (
        AcceptanceReport,
        CellReport,
        ShardMetadata,
        select_cell_shard,
    )

    payloads = []
    for index in range(shard_count):
        selected = select_cell_shard(cells, shard_count=shard_count, shard_index=index)
        report = AcceptanceReport(
            stage="small",
            code_ref=code_ref,
            cells=tuple(CellReport(cell=cell, metrics={}, passed=True) for cell in selected),
        )
        metadata = ShardMetadata(
            index=index,
            count=shard_count,
            total_cells=len(cells),
            selected_cells=len(selected),
        )
        payloads.append({**report.to_dict(), "shard": asdict(metadata)})
    return payloads


def test_merge_shards_restores_exact_formal_cell_order() -> None:
    from .run_joint_acceptance import build_cells, merge_acceptance_shard_payloads

    cells = build_cells(stage="small", all_shapes=True)[:17]

    merged = merge_acceptance_shard_payloads(
        tuple(reversed(_shard_payloads(cells))),
        expected_cells=cells,
        stage="small",
    )

    assert tuple(item.cell for item in merged.cells) == cells
    assert merged.code_ref == "abc123"


@pytest.mark.parametrize("mutation, message", [
    ("missing", "missing shard indices"),
    ("code_ref", "same code_ref"),
    ("wrong_cells", "cell keys do not match"),
])
def test_merge_shards_rejects_incomplete_or_inconsistent_reports(mutation: str, message: str) -> None:
    from .run_joint_acceptance import build_cells, merge_acceptance_shard_payloads

    cells = build_cells(stage="small", all_shapes=True)[:17]
    payloads = _shard_payloads(cells)
    if mutation == "missing":
        payloads.pop()
    elif mutation == "code_ref":
        payloads[1]["code_ref"] = "different"
    else:
        payloads[1]["cells"] = payloads[1]["cells"][1:]

    with pytest.raises(ValueError, match=message):
        merge_acceptance_shard_payloads(payloads, expected_cells=cells, stage="small")


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
            "--cell-batch-size", "40",
            "--shard-count", "96",
            "--shard-index", "7",
            "--report-json", "/tmp/report.json",
        ]
    )

    assert args.stage == "small"
    assert args.all_commands and args.all_shapes and args.all_phases and args.all_offsets
    assert args.steps == 160
    assert args.heartbeat_seconds == 5.0
    assert args.cell_batch_size == 40
    assert args.shard_count == 96
    assert args.shard_index == 7
    assert str(args.report_json) == "/tmp/report.json"


def test_cli_accepts_shard_report_merge_mode() -> None:
    from .run_joint_acceptance import parse_args

    args = parse_args(
        [
            "--stage", "small",
            "--formal",
            "--merge-shard-reports", "/tmp/shard-0.json", "/tmp/shard-1.json",
            "--report-json", "/tmp/formal.json",
        ]
    )

    assert tuple(map(str, args.merge_shard_reports)) == (
        "/tmp/shard-0.json",
        "/tmp/shard-1.json",
    )


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

    assert len(cells) == len(SMALL_SHAPES) * len(SMALL_PHASES) * len(SMALL_OFFSETS) * 7
