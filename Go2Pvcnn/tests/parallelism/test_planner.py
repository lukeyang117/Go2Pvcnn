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


def _semantic_terrain(batch: int = 1, *, semantic_id: int):
    terrain = _terrain(batch)
    return type(terrain)(
        height_w=terrain.height_w,
        semantic_id=torch.full_like(terrain.semantic_id, int(semantic_id)),
        valid_mask=terrain.valid_mask,
        origin_w=terrain.origin_w,
        yaw_w=terrain.yaw_w,
        resolution=terrain.resolution,
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

    cfg = ParallelismCfg()
    traj = plan_trajectory(_state(), torch.tensor([[0.2, 0.0, 0.0]]), _terrain(), cfg)

    assert traj.root_pos_w.shape == (1, 24, 3)
    assert traj.joint_pos.shape == (1, 24, 12)
    assert traj.foot_pos_w.shape == (1, 24, 4, 3)
    assert traj.contact_state.shape == (1, 24, 4)
    assert traj.valid.shape == (1,)
    assert traj.diagnostics.candidate_w.shape == (1, 4, 50, 3)
    assert traj.diagnostics.candidate_reject_bits.shape == (1, 4, 50, 6)
    assert traj.diagnostics.candidate_collision_bits.shape == (1, 4, 50, len(cfg.official_collision_shapes))
    assert traj.diagnostics.collision_shape_names == tuple(spec.name for spec in cfg.official_collision_shapes)
    assert traj.diagnostics.collision_surface_point_count == 6
    assert traj.diagnostics.fk_touchdown_semantic.shape == (1, 4, 50)
    assert traj.diagnostics.candidate_valid.any()


def test_invalid_map_makes_trajectory_invalid_single_pass():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.planner import plan_trajectory

    traj = plan_trajectory(_state(), torch.zeros(1, 3), _terrain(invalid=True), ParallelismCfg())

    assert not bool(traj.valid[0])
    assert not traj.diagnostics.candidate_valid.any()


def test_invalid_plan_holds_current_state():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.kinematics import fk_go2
    from extension.parallelism.planner import plan_trajectory

    state = _state()
    current_foot = fk_go2(state.root_pos_w, state.root_rpy_w, state.joint_pos).foot_pos_w
    traj = plan_trajectory(state, torch.tensor([[0.7, 0.2, 0.3]]), _terrain(invalid=True), ParallelismCfg())

    assert not bool(traj.valid[0])
    assert torch.allclose(traj.root_pos_w, state.root_pos_w[:, None].expand(-1, 24, -1))
    assert torch.allclose(traj.root_rpy_w, state.root_rpy_w[:, None].expand(-1, 24, -1))
    assert torch.allclose(traj.joint_pos, state.joint_pos[:, None].expand(-1, 24, -1))
    assert torch.allclose(traj.foot_pos_w, current_foot[:, None].expand(-1, 24, -1, -1), atol=1e-6)
    assert torch.equal(traj.selected_foothold_w, current_foot)
    assert traj.contact_state.all()


def test_invalid_plan_can_disable_standstill_fallback():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.planner import plan_trajectory

    state = _state()
    traj = plan_trajectory(
        state,
        torch.tensor([[0.7, 0.2, 0.3]]),
        _terrain(invalid=True),
        ParallelismCfg(standstill_fallback_enabled=False),
    )

    assert not bool(traj.valid[0])
    assert not torch.allclose(traj.root_pos_w, state.root_pos_w[:, None].expand(-1, 24, -1))


def test_obstacle_semantic_touchdowns_are_hard_rejected():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.planner import plan_trajectory

    for semantic_id in (1, 2):
        traj = plan_trajectory(_state(), torch.zeros(1, 3), _semantic_terrain(semantic_id=semantic_id), ParallelismCfg())

        assert not bool(traj.valid[0])
        assert not traj.diagnostics.candidate_valid.any()
        assert traj.diagnostics.candidate_reject_bits[..., 4].all()
        assert traj.diagnostics.candidate_reject_bits[..., 5].all()
        assert torch.equal(
            traj.diagnostics.fk_touchdown_semantic,
            torch.full((1, 4, 50), semantic_id, dtype=torch.long),
        )


def test_semantic_touchdown_margin_rejects_nearby_obstacle_cells():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.planner import plan_trajectory

    terrain = _terrain()
    semantic = terrain.semantic_id.clone()
    semantic[:, 29:32, 29:32] = 1
    terrain = type(terrain)(
        height_w=terrain.height_w,
        semantic_id=semantic,
        valid_mask=terrain.valid_mask,
        origin_w=terrain.origin_w,
        yaw_w=terrain.yaw_w,
        resolution=terrain.resolution,
    )

    no_margin = plan_trajectory(
        _state(),
        torch.zeros(1, 3),
        terrain,
        ParallelismCfg(semantic_touchdown_margin_m=0.0),
    )
    with_margin = plan_trajectory(
        _state(),
        torch.zeros(1, 3),
        terrain,
        ParallelismCfg(semantic_touchdown_margin_m=0.2),
    )

    assert int(with_margin.diagnostics.candidate_reject_bits[..., 4].sum().item()) > int(
        no_margin.diagnostics.candidate_reject_bits[..., 4].sum().item()
    )
    assert int(with_margin.diagnostics.candidate_reject_bits[..., 5].sum().item()) >= int(
        no_margin.diagnostics.candidate_reject_bits[..., 5].sum().item()
    )


def test_filter_score_source_uses_torch_conditions():
    import extension.parallelism.planner as planner

    source = inspect.getsource(planner)
    assert "torch.where" in source
    assert ".argmin(" in source
    assert "reject_bits = torch.stack" in source
    assert "for candidate" not in source


def test_planner_uses_terrain_aware_swing_for_collision_and_output(monkeypatch):
    import extension.parallelism.planner as planner
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.swing import swing_curve

    calls: list[tuple[int, float, float]] = []

    def fake_terrain_aware_swing(start_w, touchdown_w, terrain, *, frames: int, clearance_m: float, min_apex_m: float):
        calls.append((int(frames), float(clearance_m), float(min_apex_m)))
        return swing_curve(start_w, touchdown_w, frames=int(frames), height_m=float(min_apex_m))

    monkeypatch.setattr(planner, "terrain_aware_swing_curve", fake_terrain_aware_swing, raising=False)

    planner.plan_trajectory(
        _state(),
        torch.tensor([[0.2, 0.0, 0.0]], dtype=torch.float32),
        _terrain(),
        ParallelismCfg(swing_clearance_m=0.07, min_swing_apex_m=0.08),
    )

    assert len(calls) >= 6
    assert all(call == (12, 0.07, 0.08) for call in calls)


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
