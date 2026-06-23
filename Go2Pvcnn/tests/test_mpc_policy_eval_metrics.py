from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/mpc_policy_eval.py"


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("mpc_policy_eval_under_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


eval_module = _load_eval_module()
SmallCollisionRoundAccumulator = eval_module.SmallCollisionRoundAccumulator
TrackingRoundAccumulator = eval_module.TrackingRoundAccumulator
command_for_step = eval_module.command_for_step
make_run_output_dir = eval_module.make_run_output_dir
parse_command_sweep = eval_module.parse_command_sweep
tracking_foot_metrics = eval_module.tracking_foot_metrics
aggregate_small_collision_rounds = eval_module.aggregate_small_collision_rounds
build_controlled_crossing_commands = eval_module.build_controlled_crossing_commands
ControlledCrossingAccumulator = eval_module.ControlledCrossingAccumulator


def test_tracking_foot_metrics_report_mean_p95_and_per_leg() -> None:
    actual = torch.zeros((2, 4, 3), dtype=torch.float32)
    reference = torch.zeros((2, 4, 3), dtype=torch.float32)
    actual[0, 0, 0] = 0.10
    actual[1, 1, 1] = 0.20

    metrics = tracking_foot_metrics(actual, reference)

    assert metrics["foot_tracking_error_mean_m"] == pytest.approx(0.0375)
    assert metrics["foot_tracking_error_p95_m"] >= 0.10
    assert metrics["per_leg_foot_error_mean_m"][0] == pytest.approx(0.05)
    assert metrics["per_leg_foot_error_mean_m"][1] == pytest.approx(0.10)
    assert metrics["per_leg_foot_error_mean_m"][2] == pytest.approx(0.0)
    assert metrics["per_leg_foot_error_mean_m"][3] == pytest.approx(0.0)


def test_small_collision_accumulator_counts_each_env_once_per_round() -> None:
    acc = SmallCollisionRoundAccumulator(num_envs=4, threshold=1.0, device=torch.device("cpu"))
    force = torch.zeros((4, 2, 3, 3), dtype=torch.float32)
    force[1, 0, 0, 0] = 2.0
    acc.update(step=0, force_matrix_w=force, body_names=("base", "foot"))
    force.zero_()
    force[1, 1, 2, 1] = 3.0
    force[3, 0, 1, 2] = 4.0
    acc.update(step=5, force_matrix_w=force, body_names=("base", "foot"))

    summary = acc.summary()

    assert summary["collided_env_count"] == 2
    assert summary["num_envs"] == 4
    assert summary["small_collision_env_rate_per_round"] == pytest.approx(0.5)
    assert summary["first_collision_step_by_env"] == {"1": 0, "3": 5}
    assert summary["collision_body_names_by_env"]["1"] == ["base", "foot"]
    assert summary["round_small_force_max"] == pytest.approx(4.0)


def test_aggregate_small_collision_rounds_uses_env_denominator() -> None:
    summary = aggregate_small_collision_rounds(
        [
            {"collided_env_count": 2, "num_envs": 4},
            {"collided_env_count": 1, "num_envs": 4},
        ]
    )

    assert summary["aggregate_small_collision_env_rate"] == pytest.approx(3 / 8)
    assert summary["round_count"] == 2
    assert summary["total_collided_envs"] == 3
    assert summary["total_env_rounds"] == 8


def test_tracking_accumulator_averages_mean_and_valid_ratio_but_maxes_p95() -> None:
    acc = TrackingRoundAccumulator()

    acc.update(
        {
            "foot_tracking_error_mean_m": 0.10,
            "foot_tracking_error_p95_m": 0.20,
            "per_leg_foot_error_mean_m": [0.10, 0.20, 0.30, 0.40],
            "reference_valid_ratio": 1.0,
        }
    )
    acc.update(
        {
            "foot_tracking_error_mean_m": 0.30,
            "foot_tracking_error_p95_m": 0.15,
            "per_leg_foot_error_mean_m": [0.30, 0.20, 0.10, 0.00],
            "reference_valid_ratio": 0.0,
        }
    )

    summary = acc.summary()

    assert summary["tracking_step_count"] == 2
    assert summary["foot_tracking_error_mean_m"] == pytest.approx(0.20)
    assert summary["foot_tracking_error_p95_m"] == pytest.approx(0.20)
    assert summary["per_leg_foot_error_mean_m"] == pytest.approx([0.20, 0.20, 0.20, 0.20])
    assert summary["reference_valid_ratio"] == pytest.approx(0.5)


def test_tracking_accumulator_reports_empty_reference_as_zero_valid_ratio() -> None:
    acc = TrackingRoundAccumulator()
    acc.update({"reference_valid_ratio": 0.0})

    summary = acc.summary()

    assert summary["tracking_step_count"] == 1
    assert summary["reference_valid_ratio"] == pytest.approx(0.0)
    assert summary["foot_tracking_error_mean_m"] is None
    assert summary["foot_tracking_error_p95_m"] is None
    assert summary["per_leg_foot_error_mean_m"] is None


def test_make_run_output_dir_avoids_same_second_collisions(tmp_path: Path) -> None:
    first = make_run_output_dir(tmp_path)
    second = make_run_output_dir(tmp_path)

    assert first != second
    assert first.exists()
    assert second.exists()


def test_command_for_step_supports_fixed_and_sweep_modes() -> None:
    fixed = SimpleNamespace(command_mode="fixed", command="0.4 0.0 0.1", command_sweep="", random_command_interval=100)
    out = command_for_step(fixed, step=0, env_count=2, device=torch.device("cpu"))
    assert out.tolist() == [[0.4, 0.0, 0.1], [0.4, 0.0, 0.1]]

    sweep = SimpleNamespace(
        command_mode="sweep",
        command="0.0 0.0 0.0",
        command_sweep="0.1 0 0;0 0.2 0",
        random_command_interval=100,
    )
    assert parse_command_sweep(sweep.command_sweep) == [(0.1, 0.0, 0.0), (0.0, 0.2, 0.0)]
    assert command_for_step(sweep, step=0, env_count=1, device=torch.device("cpu")).tolist() == [[0.1, 0.0, 0.0]]
    assert command_for_step(sweep, step=1, env_count=1, device=torch.device("cpu")).tolist() == [[0.0, 0.2, 0.0]]


def test_controlled_crossing_commands_group_speed_and_lateral_offsets() -> None:
    commands, groups = build_controlled_crossing_commands(
        env_count=9,
        speeds=(0.6, 0.8, 1.0),
        lateral_offsets=(-0.08, 0.0, 0.08),
        device=torch.device("cpu"),
    )

    assert commands.shape == (9, 3)
    assert commands[:, 1].tolist() == pytest.approx([-0.08, 0.0, 0.08, -0.08, 0.0, 0.08, -0.08, 0.0, 0.08])
    assert commands[:, 0].tolist() == pytest.approx([0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0])
    assert groups["speed_by_env"] == pytest.approx([0.6, 0.6, 0.6, 0.8, 0.8, 0.8, 1.0, 1.0, 1.0])
    assert groups["lateral_offset_by_env"] == pytest.approx([-0.08, 0.0, 0.08, -0.08, 0.0, 0.08, -0.08, 0.0, 0.08])


def test_controlled_crossing_accumulator_summarizes_success_by_speed_and_lateral() -> None:
    acc = ControlledCrossingAccumulator(
        num_envs=4,
        speed_by_env=[0.6, 0.6, 0.8, 0.8],
        lateral_offset_by_env=[0.0, 0.08, 0.0, 0.08],
        device=torch.device("cpu"),
    )
    acc.opportunity_seen[:] = torch.tensor([True, True, True, False])
    acc.root_crossed[:] = torch.tensor([True, True, True, False])
    acc.foot_over[:] = torch.tensor([True, False, True, False])
    acc.touchdown_on_small[:] = torch.tensor([False, False, True, False])
    acc.done_seen[:] = torch.tensor([False, False, False, False])
    contact = torch.tensor([False, True, False, False])

    summary = acc.summary(contact_collided=contact)

    assert summary["opportunity_env_count"] == 3
    assert summary["root_crossed_count"] == 3
    assert summary["foot_over_count"] == 2
    assert summary["touchdown_on_small_env_count"] == 1
    assert summary["small_contact_env_count"] == 1
    assert summary["small_overpass_success_count"] == 1
    assert summary["small_overpass_success_rate_over_opportunities"] == pytest.approx(1 / 3)
    assert summary["success_by_speed"]["0.6"]["success_count"] == 1
    assert summary["success_by_speed"]["0.6"]["opportunity_count"] == 2
    assert summary["success_by_speed"]["0.8"]["success_count"] == 0
    assert summary["success_by_lateral_offset"]["0.0"]["success_count"] == 1
    assert summary["success_by_lateral_offset"]["0.08"]["success_count"] == 0


def test_controlled_crossing_accumulator_records_reset_stage_after_foot_over() -> None:
    acc = ControlledCrossingAccumulator(
        num_envs=4,
        speed_by_env=[0.6, 0.6, 0.8, 0.8],
        lateral_offset_by_env=[0.0, 0.08, 0.0, 0.08],
        device=torch.device("cpu"),
    )
    acc.opportunity_seen[:] = torch.tensor([True, True, True, False])
    acc.root_crossed[:] = torch.tensor([False, True, True, False])
    acc.foot_over[:] = torch.tensor([True, True, False, False])
    acc.update_reset_diagnostics(
        step=12,
        done_mask=torch.tensor([True, True, True, False]),
        termination_terms={
            "bad_orientation": torch.tensor([True, False, False, False]),
            "base_contact": torch.tensor([False, True, False, False]),
            "time_out": torch.tensor([False, False, True, False]),
        },
    )

    summary = acc.summary(contact_collided=torch.zeros(4, dtype=torch.bool))

    assert summary["reset_env_count"] == 3
    assert summary["reset_after_foot_over_count"] == 2
    assert summary["reset_after_root_crossed_count"] == 2
    assert summary["reset_reason_counts"] == {
        "bad_orientation": 1,
        "base_contact": 1,
        "time_out": 1,
        "unknown": 0,
    }
    assert summary["reset_stage_counts"] == {
        "before_opportunity": 0,
        "before_foot_over": 1,
        "after_foot_over_before_root_cross": 1,
        "after_root_cross": 1,
    }
    assert summary["first_reset_step_by_env"] == {"0": 12, "1": 12, "2": 12}
    assert summary["first_reset_reason_by_env"] == {"0": "bad_orientation", "1": "base_contact", "2": "time_out"}
    assert summary["first_reset_stage_by_env"] == {
        "0": "after_foot_over_before_root_cross",
        "1": "after_root_cross",
        "2": "before_foot_over",
    }
