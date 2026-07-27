from __future__ import annotations

import inspect

import torch


def _terrain(batch: int = 1, *, invalid: bool = False):
    from extension.parallelism import ParallelismTerrain

    valid = torch.ones(batch, 61, 61, dtype=torch.bool)
    if invalid:
        valid[:] = False
    return ParallelismTerrain(
        height_w=torch.zeros(batch, 61, 61),
        semantic_id=torch.zeros(batch, 61, 61, dtype=torch.long),
        valid_mask=valid,
        origin_w=torch.tensor([[-3.0, -3.0, 0.0]], dtype=torch.float32).repeat(batch, 1),
        yaw_w=torch.zeros(batch),
        resolution=0.1,
    )


def _state(batch: int = 1):
    from extension.parallelism import ParallelismState

    return ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, 0.30]], dtype=torch.float32).repeat(batch, 1),
        root_rpy_w=torch.zeros(batch, 3),
        joint_pos=torch.tensor([[0.0, 0.8, -1.5] * 4], dtype=torch.float32).repeat(batch, 1),
    )


def test_full_flat_trajectory_contract():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.planner import plan_trajectory

    traj = plan_trajectory(_state(), torch.tensor([[0.2, 0.0, 0.0]]), _terrain(), ParallelismCfg())

    assert traj.root_pos_w.shape == (1, 24, 3)
    assert traj.joint_pos.shape == (1, 24, 12)
    assert traj.foot_pos_w.shape == (1, 24, 4, 3)
    assert traj.contact_state.shape == (1, 24, 4)
    assert traj.valid.shape == (1,)
    assert traj.diagnostics.candidate_w.shape == (1, 4, 50, 3)
    assert traj.diagnostics.candidate_reject_bits.shape == (1, 4, 50, 4)


def test_invalid_map_makes_trajectory_invalid_single_pass():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.planner import plan_trajectory

    traj = plan_trajectory(_state(), torch.zeros(1, 3), _terrain(invalid=True), ParallelismCfg())

    assert not bool(traj.valid[0])
    assert not traj.diagnostics.candidate_valid.any()


def test_filter_score_source_uses_torch_conditions():
    import extension.parallelism.planner as planner

    source = inspect.getsource(planner)
    assert "torch.where" in source
    assert ".argmin(" in source
    assert "reject_bits = torch.stack" in source
    assert "for candidate" not in source


def test_parallel_batch_contract():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.planner import plan_trajectory

    traj = plan_trajectory(_state(8), torch.zeros(8, 3), _terrain(8), ParallelismCfg())

    assert traj.root_pos_w.shape[0] == 8
    assert traj.diagnostics.candidate_score.shape == (8, 4, 50)


def test_rl_adapter_shape_contract():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.planner import plan_trajectory
    from extension.parallelism.rl_adapter import trajectory_to_reference

    traj = plan_trajectory(_state(2), torch.zeros(2, 3), _terrain(2), ParallelismCfg())
    ref = trajectory_to_reference(traj)

    assert ref.root_pos_w.shape == (2, 24, 3)
    assert ref.foot_pos_w.shape == (2, 24, 4, 3)
    assert torch.equal(ref.valid, traj.valid)
