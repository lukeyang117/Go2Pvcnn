from __future__ import annotations

import pytest
import torch

from extension.joint_mpc_rti.config import JointMpcRtiCfg
from extension.joint_mpc_rti.runtime.manager import JointMpcRtiManager

from .helpers import make_command, make_flat_field, make_state


def test_manager_publishes_x1_and_tracks_full_horizon_from_measured_x0() -> None:
    manager = JointMpcRtiManager.from_config(JointMpcRtiCfg(), num_envs=2, device="cpu")
    measured_t = make_state(batch=2)

    step = manager.plan_from_tensors(measured_t, make_command(batch=2), make_flat_field(batch=2))

    torch.testing.assert_close(step.full_trajectory.state[:, 0], measured_t.as_vector())
    torch.testing.assert_close(step.pending_reference.joint_angles, step.full_trajectory.state[:, 1, 6:])
    torch.testing.assert_close(step.pending_reference.root_pos_w, step.full_trajectory.state[:, 1, :3])
    assert step.pending_reference.target_step == 1
    assert torch.all(step.pending_reference.valid)


def test_second_plan_reinjects_new_measured_state_instead_of_old_prediction() -> None:
    manager = JointMpcRtiManager.from_config(JointMpcRtiCfg(), num_envs=2, device="cpu")
    first_state = make_state(batch=2)
    manager.plan_from_tensors(first_state, make_command(batch=2), make_flat_field(batch=2))
    second_state = make_state(batch=2)
    second_state.root_pos_w[:, 0] = torch.tensor([0.7, -0.4])

    second = manager.plan_from_tensors(second_state, make_command(batch=2), make_flat_field(batch=2))

    torch.testing.assert_close(second.full_trajectory.state[:, 0], second_state.as_vector())


def test_reset_clears_only_selected_pending_rows() -> None:
    manager = JointMpcRtiManager.from_config(JointMpcRtiCfg(), num_envs=4, device="cpu")
    manager.plan_from_tensors(make_state(4), make_command(4), make_flat_field(4))

    manager.reset_envs(torch.tensor([False, True, False, True]))

    assert torch.equal(manager.pending_valid, torch.tensor([True, False, True, False]))
    assert torch.equal(manager._solver_state.initialized, torch.tensor([True, False, True, False]))


def test_reset_rows_cold_start_once_while_other_rows_remain_warm(monkeypatch) -> None:
    from extension.joint_mpc_rti import planner

    manager = JointMpcRtiManager.from_config(JointMpcRtiCfg(), num_envs=2, device="cpu")
    state = make_state(2)
    command = make_command(2)
    field = make_flat_field(2)
    manager.plan_from_tensors(state, command, field)
    manager.reset_envs(torch.tensor([False, True]))

    sources: list[tuple[torch.Tensor, torch.Tensor]] = []
    original = planner.build_nominal

    def spy(*args, **kwargs):
        nominal = original(*args, **kwargs)
        sources.append((nominal.used_cold_start.clone(), nominal.used_warm_start.clone()))
        return nominal

    monkeypatch.setattr(planner, "build_nominal", spy)
    manager.plan_from_tensors(state, command, field)
    manager.plan_from_tensors(state, command, field)

    assert torch.equal(sources[0][0], torch.tensor([False, True]))
    assert torch.equal(sources[0][1], torch.tensor([True, False]))
    assert not sources[1][0].any()
    assert sources[1][1].all()


def test_current_reference_is_always_first_future_frame() -> None:
    manager = JointMpcRtiManager.from_config(JointMpcRtiCfg(), num_envs=3, device="cpu")
    step = manager.plan_from_tensors(make_state(3), make_command(3), make_flat_field(3))

    current = manager.current_reference()

    torch.testing.assert_close(current["joint_angles"], step.full_trajectory.state[:, 1, 6:])
    assert torch.equal(manager.current_frame_ids(), torch.ones(3, dtype=torch.long))


def test_fixed_trot_scheduler_advances_one_phase_step_per_replan() -> None:
    manager = JointMpcRtiManager.from_config(JointMpcRtiCfg(), num_envs=2, device="cpu")
    state = make_state(2)
    field = make_flat_field(2)

    first = manager.plan_from_tensors(state, make_command(2), field)
    second = manager.plan_from_tensors(state, make_command(2), field)

    assert torch.equal(second.full_trajectory.contact_state[:, :-1], first.full_trajectory.contact_state[:, 1:])


def test_cuda_graph_runtime_flag_falls_back_cleanly_on_cpu() -> None:
    cfg = JointMpcRtiCfg()
    cfg.solver.use_cuda_graph = True
    manager = JointMpcRtiManager.from_config(cfg, num_envs=2, device="cpu")

    result = manager.plan_from_tensors(make_state(2), make_command(2), make_flat_field(2))

<<<<<<< HEAD
    assert result.full_trajectory.state.shape == (2, cfg.runtime.horizon_steps + 1, 18)
=======
    assert result.full_trajectory.state.shape == (2, 31, 18)
>>>>>>> 156a6c0 (refactor: route joint mpc through pure kinematic rti)
    assert manager._graph_runner is None


def test_cuda_graph_capture_materializes_the_first_result_before_return() -> None:
    from pathlib import Path

    source = Path("Go2Pvcnn/extension/joint_mpc_rti/runtime/cuda_graph.py").read_text()
    assert source.count("self._graph.replay()") == 2
    assert "solver_state.stance_anchor_w.clone()" in source
    assert "self._solver_state.stance_anchor_w.copy_" in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_runner_captures_and_replays_the_kinematic_rti_step() -> None:
    from extension.joint_mpc_rti.planner import step as planner_step
    from extension.joint_mpc_rti.runtime.cuda_graph import JointMpcCudaGraphRunner

    cfg = JointMpcRtiCfg()
    measured = make_state(1, device="cuda")
    command = make_command(1, device="cuda")
    field = make_flat_field(1, device="cuda")
    cold = planner_step(measured, command, field, None, cfg)
    assert torch.isfinite(cold.full_trajectory.state).all()

    runner = JointMpcCudaGraphRunner(measured, command, field, cold.solver_state, cfg)
    assert torch.isfinite(runner.captured_result.full_trajectory.state).all()
    replayed = runner.run(measured, command, field)

    assert torch.isfinite(replayed.full_trajectory.state).all()
    assert replayed.full_trajectory.state.shape == (1, 31, 18)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_graph_runner_captures_and_replays_planner_step() -> None:
    from extension.joint_mpc_rti.planner import step
    from extension.joint_mpc_rti.runtime.cuda_graph import JointMpcCudaGraphRunner

    device = torch.device("cuda")
    cfg = JointMpcRtiCfg()
    cfg.solver.compile_kernels = False
    cfg.solver.emit_loss_breakdown = False
    measured = make_state(1, device=device)
    command = make_command(1, device=device)
    field = make_flat_field(1, device=device)
    first = step(measured, command, field, None, cfg)
    warm = step(measured, command, field, first.solver_state, cfg)

    runner = JointMpcCudaGraphRunner(measured, command, field, warm.solver_state, cfg)
    replayed = runner.run(measured, command, field)
    torch.cuda.synchronize()

    assert runner.solver_state.stance_dual is not None
    assert runner.solver_state.command_start_age is not None
    assert runner.solver_state.command_start_origin_w is not None
    assert runner.solver_state.previous_command_body is not None
    assert torch.isfinite(replayed.full_trajectory.state).all()
    assert torch.isfinite(replayed.full_trajectory.control).all()


def test_manager_requires_rebuild_when_environment_batch_size_changes() -> None:
    manager = JointMpcRtiManager.from_config(JointMpcRtiCfg(), num_envs=4, device="cpu")

    with pytest.raises(ValueError, match="rebuild"):
        manager.plan_from_tensors(make_state(2), make_command(2), make_flat_field(2))
