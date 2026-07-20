from __future__ import annotations


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
