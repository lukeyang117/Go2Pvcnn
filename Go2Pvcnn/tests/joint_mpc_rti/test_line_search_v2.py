from __future__ import annotations

import torch

from extension.joint_mpc_rti.solver.line_search import FILTER_NAMES, parallel_line_search
from extension.joint_mpc_rti.solver.trajectory_qp import JOINT_LOWER, JOINT_UPPER


def _nominal(batch: int) -> torch.Tensor:
    state = torch.zeros(batch, 31, 18)
    state[..., 6:] = torch.tensor((0.0, 0.8, -1.5) * 4)
    return state


def _limits(state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lower = state.new_tensor(JOINT_LOWER)
    upper = state.new_tensor(JOINT_UPPER)
    velocity = state.new_full((12,), 30.0)
    return lower, upper, velocity


def test_line_search_builds_five_state_candidates_and_selects_lowest_loss() -> None:
    nominal = _nominal(2)
    direction = torch.zeros_like(nominal)
    direction[..., 0] = 1.0
    lower, upper, velocity = _limits(nominal)

    def objective(state: torch.Tensor) -> torch.Tensor:
        return (state[:, :, 0] - 0.5).square().mean(dim=1)

    result = parallel_line_search(
        nominal,
        direction,
        objective,
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity_limit=velocity,
        dt=0.02,
        tie_tolerance=1.0e-7,
    )

    assert result.candidates.shape == (2, 5, 31, 18)
    assert result.alphas.tolist() == [1.0, 0.5, 0.25, 0.125, 0.0]
    torch.testing.assert_close(result.selected_loss, result.candidate_loss.min(dim=1).values)
    torch.testing.assert_close(result.alpha, torch.full((2,), 0.5))
    torch.testing.assert_close(result.state[..., 0], torch.full((2, 31), 0.5))


def test_line_search_evaluates_all_five_candidates_in_one_objective_call() -> None:
    nominal = _nominal(3)
    lower, upper, velocity = _limits(nominal)
    calls: list[tuple[int, ...]] = []

    def objective(state: torch.Tensor) -> torch.Tensor:
        calls.append(tuple(state.shape))
        return state.square().mean(dim=(1, 2))

    parallel_line_search(
        nominal,
        torch.zeros_like(nominal),
        objective,
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity_limit=velocity,
        dt=0.02,
    )
    assert calls == [(15, 31, 18)]


def test_line_search_filters_only_nonfinite_joint_position_and_velocity() -> None:
    assert FILTER_NAMES == ("finite", "joint_position", "joint_velocity")


def test_nonfinite_loss_is_not_selectable() -> None:
    nominal = _nominal(1)
    direction = torch.zeros_like(nominal)
    direction[..., 0] = 1.0
    lower, upper, velocity = _limits(nominal)

    def objective(state: torch.Tensor) -> torch.Tensor:
        loss = -state[:, :, 0].mean(dim=1)
        return torch.where(state[:, 0, 0] == 1.0, torch.full_like(loss, torch.nan), loss)

    result = parallel_line_search(
        nominal,
        direction,
        objective,
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity_limit=velocity,
        dt=0.02,
    )
    assert result.alpha.item() == 0.5


def test_joint_position_filter_selects_largest_valid_alpha() -> None:
    nominal = _nominal(1)
    direction = torch.zeros_like(nominal)
    direction[..., 6] = 3.0
    lower, upper, velocity = _limits(nominal)
    result = parallel_line_search(
        nominal,
        direction,
        objective=lambda state: -state[:, :, 6].mean(dim=1),
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity_limit=velocity,
        dt=0.02,
    )
    assert result.alpha.item() == 0.25


def test_joint_velocity_filter_selects_largest_valid_alpha() -> None:
    nominal = _nominal(1)
    direction = torch.zeros_like(nominal)
    direction[:, 1:, 6] = 1.0
    lower, upper, velocity = _limits(nominal)
    result = parallel_line_search(
        nominal,
        direction,
        objective=lambda state: -state[:, :, 6].mean(dim=1),
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity_limit=velocity,
        dt=0.02,
    )
    assert result.alpha.item() == 0.5


def test_equal_loss_prefers_larger_alpha() -> None:
    nominal = _nominal(1)
    lower, upper, velocity = _limits(nominal)
    result = parallel_line_search(
        nominal,
        torch.zeros_like(nominal),
        objective=lambda state: torch.zeros(state.shape[0], device=state.device),
        joint_lower=lower,
        joint_upper=upper,
        joint_velocity_limit=velocity,
        dt=0.02,
        tie_tolerance=1.0e-7,
    )
    assert result.alpha.eq(1.0).all()
