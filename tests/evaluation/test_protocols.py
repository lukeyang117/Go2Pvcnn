from __future__ import annotations

import pytest

from evaluation.protocols import mixed_velocity_commands, replan_boundary, runway_velocity_commands


def test_mixed_protocol_has_exact_nonzero_velocity_grid() -> None:
    commands = mixed_velocity_commands()
    assert len(commands) == 120
    assert sorted({row[0] for row in commands}) == [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]
    assert sorted({row[1] for row in commands}) == [-0.5, -0.25, 0.25, 0.5]
    assert sorted({row[2] for row in commands}) == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert 0.0 not in {row[0] for row in commands}
    assert 0.0 not in {row[1] for row in commands}


def test_replan_boundary_uses_23_transitions() -> None:
    assert replan_boundary(0)
    assert replan_boundary(23)
    assert replan_boundary(46)
    assert not replan_boundary(24)
    with pytest.raises(ValueError):
        replan_boundary(-1)


def test_runway_velocity_assignment_is_balanced_and_seeded() -> None:
    first = runway_velocity_commands(9, seed=7)
    second = runway_velocity_commands(9, seed=7)
    assert first == second
    assert len(first) == 9
    assert sorted(row[0] for row in first) == [0.25] * 3 + [0.5] * 3 + [1.0] * 3
    assert all(row[1:] == (0.0, 0.0) for row in first)
