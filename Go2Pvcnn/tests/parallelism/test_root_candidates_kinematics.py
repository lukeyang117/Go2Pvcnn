from __future__ import annotations

import torch


def _terrain(batch: int = 1):
    from extension.parallelism import ParallelismTerrain

    return ParallelismTerrain(
        height_w=torch.zeros(batch, 41, 41),
        semantic_id=torch.zeros(batch, 41, 41, dtype=torch.long),
        valid_mask=torch.ones(batch, 41, 41, dtype=torch.bool),
        origin_w=torch.tensor([[-2.0, -2.0, 0.0]], dtype=torch.float32).repeat(batch, 1),
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


def _semantic_two_terrain(
    *,
    obstacle_cells: tuple[tuple[int, int], ...],
    batch: int = 1,
):
    terrain = _terrain(batch)
    semantic = terrain.semantic_id.clone()
    for row, col in obstacle_cells:
        semantic[:, row, col] = 2
    return type(terrain)(
        height_w=terrain.height_w,
        semantic_id=semantic,
        valid_mask=terrain.valid_mask,
        origin_w=terrain.origin_w,
        yaw_w=terrain.yaw_w,
        resolution=terrain.resolution,
    )


def test_large_obstacle_avoidance_keeps_command_without_semantic_two():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.root import _large_obstacle_avoidance_command

    command = torch.tensor([[0.6, 0.0, 0.0]])
    result = _large_obstacle_avoidance_command(_state(), command, _terrain(), ParallelismCfg())

    torch.testing.assert_close(result, command)


def test_large_obstacle_avoidance_is_stronger_for_nearer_obstacle():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.root import _large_obstacle_avoidance_command

    terrain = _semantic_two_terrain(
        obstacle_cells=((22, 23),),
        batch=2,
    )
    terrain = type(terrain)(
        height_w=terrain.height_w,
        semantic_id=terrain.semantic_id.clone(),
        valid_mask=terrain.valid_mask,
        origin_w=terrain.origin_w,
        yaw_w=terrain.yaw_w,
        resolution=terrain.resolution,
    )
    terrain.semantic_id[1].zero_()
    terrain.semantic_id[1, 22, 29] = 2
    result = _large_obstacle_avoidance_command(
        _state(batch=2),
        torch.tensor([[0.6, 0.0, 0.0], [0.6, 0.0, 0.0]]),
        terrain,
        ParallelismCfg(),
    )

    assert result[0, 1] < 0.0
    assert result[1, 1] < 0.0
    assert result[0, 1].abs() > result[1, 1].abs()


def test_large_obstacle_mean_l_only_selects_left_or_right_direction():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.root import _large_obstacle_avoidance_command

    left = _semantic_two_terrain(obstacle_cells=((22, 23),))
    right = _semantic_two_terrain(obstacle_cells=((18, 23),))
    states = _state(batch=2)
    terrain = type(left)(
        height_w=torch.cat((left.height_w, right.height_w), dim=0),
        semantic_id=torch.cat((left.semantic_id, right.semantic_id), dim=0),
        valid_mask=torch.cat((left.valid_mask, right.valid_mask), dim=0),
        origin_w=torch.cat((left.origin_w, right.origin_w), dim=0),
        yaw_w=torch.cat((left.yaw_w, right.yaw_w), dim=0),
        resolution=left.resolution,
    )
    result = _large_obstacle_avoidance_command(
        states,
        torch.tensor([[0.6, 0.0, 0.0], [0.6, 0.0, 0.0]]),
        terrain,
        ParallelismCfg(),
    )

    assert result[0, 1] < 0.0
    assert result[1, 1] > 0.0


def test_large_obstacle_default_side_controls_symmetric_obstacle_direction():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.root import _large_obstacle_avoidance_command

    terrain = _semantic_two_terrain(obstacle_cells=((19, 23), (21, 23)))
    command = torch.tensor([[0.6, 0.0, 0.0]])

    left_result = _large_obstacle_avoidance_command(
        _state(),
        command,
        terrain,
        ParallelismCfg(large_obstacle_default_side=1),
    )
    right_result = _large_obstacle_avoidance_command(
        _state(),
        command,
        terrain,
        ParallelismCfg(large_obstacle_default_side=-1),
    )

    assert left_result[0, 1] > 0.0
    assert right_result[0, 1] < 0.0


def test_rollout_root_applies_large_obstacle_avoidance_per_environment():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.root import rollout_root

    obstacle_terrain = _semantic_two_terrain(obstacle_cells=((22, 23),))
    empty_terrain = _terrain()
    terrain = type(obstacle_terrain)(
        height_w=torch.cat((obstacle_terrain.height_w, empty_terrain.height_w), dim=0),
        semantic_id=torch.cat((obstacle_terrain.semantic_id, empty_terrain.semantic_id), dim=0),
        valid_mask=torch.cat((obstacle_terrain.valid_mask, empty_terrain.valid_mask), dim=0),
        origin_w=torch.cat((obstacle_terrain.origin_w, empty_terrain.origin_w), dim=0),
        yaw_w=torch.cat((obstacle_terrain.yaw_w, empty_terrain.yaw_w), dim=0),
        resolution=obstacle_terrain.resolution,
    )
    result = rollout_root(
        _state(batch=2),
        torch.tensor([[0.6, 0.0, 0.0], [0.6, 0.0, 0.0]]),
        terrain,
        ParallelismCfg(),
    )

    assert result.root_pos_w.shape == (2, 24, 3)
    assert result.clamped_command_body[0, 1] < 0.0
    assert torch.isclose(result.clamped_command_body[1, 1], torch.tensor(0.0))


def test_large_obstacle_avoidance_applies_to_flat_and_terrain_following_rollouts():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.root import rollout_root

    terrain = _semantic_two_terrain(obstacle_cells=((22, 23),), batch=2)
    cfg = ParallelismCfg(
        terrain_following_vx_soft_limit=1.0,
        terrain_following_vy_soft_limit=1.0,
        terrain_following_vyaw_soft_limit=1.0,
        terrain_following_vx_excess_scale=1.0,
        terrain_following_vy_excess_scale=1.0,
        terrain_following_vyaw_excess_scale=1.0,
    )
    result = rollout_root(
        _state(batch=2),
        torch.tensor([[0.6, 0.0, 0.0], [0.6, 0.0, 0.0]]),
        terrain,
        cfg,
        terrain_following_mask=torch.tensor([False, True]),
    )

    torch.testing.assert_close(result.clamped_command_body[0], result.clamped_command_body[1])
    assert result.clamped_command_body[0, 1] < 0.0
    assert torch.allclose(result.root_rpy_w[0, :, :2], torch.zeros(24, 2), atol=1e-6)
    assert result.root_pos_w.shape == (2, 24, 3)


def test_root_rollout_body_command_half_cycle_displacement():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.root import rollout_root

    cfg = ParallelismCfg()
    result = rollout_root(_state(), torch.tensor([[1.0, 0.0, 0.0]]), _terrain(), cfg)
    first_half = result.root_pos_w[0, 11, 0] - result.root_pos_w[0, 0, 0]
    full = result.root_pos_w[0, 23, 0] - result.root_pos_w[0, 0, 0]

    assert result.root_pos_w.shape == (1, 24, 3)
    assert torch.isclose(first_half, torch.tensor(12 * cfg.dt), atol=1e-5)
    assert torch.isclose(full, torch.tensor(24 * cfg.dt), atol=1e-5)
    assert torch.allclose(result.root_pos_w[..., 2], torch.full((1, 24), 0.30), atol=1e-6)


def test_root_rollout_levels_tilted_root_before_second_half_cycle():
    from extension.parallelism import ParallelismCfg, ParallelismState
    from extension.parallelism.root import rollout_root

    state = _state()
    state = ParallelismState(
        root_pos_w=state.root_pos_w,
        root_rpy_w=torch.tensor([[0.20, 0.15, 0.40]]),
        joint_pos=state.joint_pos,
    )
    cfg = ParallelismCfg(root_leveling_frames=12)
    result = rollout_root(state, torch.tensor([[1.0, 0.0, 0.2]]), _terrain(), cfg)

    assert torch.allclose(result.root_rpy_w[0, 0, :2], torch.tensor([0.20, 0.15]), atol=1e-6)
    assert torch.allclose(result.root_rpy_w[0, 12:, :2], torch.zeros(12, 2), atol=1e-6)
    assert torch.allclose(result.root_rpy_w[0, 0, 2], torch.tensor(0.40), atol=1e-6)
    assert torch.all(result.root_rpy_w[0, 1:, :2].abs() <= result.root_rpy_w[0, :-1, :2].abs() + 1e-6)
    assert torch.isclose(
        torch.linalg.vector_norm(result.root_pos_w[0, 11, :2] - result.root_pos_w[0, 0, :2]),
        torch.tensor(12 * cfg.dt),
        atol=1e-5,
    )


def test_terrain_root_sampling_is_symmetric_and_includes_center():
    from extension.parallelism.root import _uniform_symmetric_samples

    samples = _uniform_symmetric_samples(0.35, 7, dtype=torch.float32, device=torch.device("cpu"))

    assert torch.allclose(samples, torch.linspace(-0.35, 0.35, 7))
    assert torch.isclose(samples[3], torch.tensor(0.0))
    assert torch.allclose(samples, -torch.flip(samples, dims=(0,)))


def test_terrain_root_sampling_fits_linear_height_slope():
    from extension.parallelism.root import _fit_height_slope

    offsets = torch.linspace(-0.35, 0.35, 7).reshape(1, 7, 1)
    heights = 0.2 * offsets + 0.43
    slope = _fit_height_slope(heights, offsets)

    assert slope.shape == (1, 1)
    assert torch.allclose(slope, torch.full((1, 1), 0.2), atol=1e-5)


def test_terrain_root_sampling_flat_profile_and_deadband_are_level():
    from extension.parallelism.root import _fit_height_slope

    offsets = torch.linspace(-0.35, 0.35, 7).reshape(1, 7, 1)
    flat_slope = _fit_height_slope(torch.full_like(offsets, 0.43), offsets)
    small_slope = _fit_height_slope(0.01 * offsets + 0.43, offsets)

    assert torch.allclose(flat_slope, torch.zeros(1, 1), atol=1e-6)
    assert torch.allclose(torch.atan(small_slope), torch.full((1, 1), 0.01).atan(), atol=1e-5)


def test_terrain_root_sampling_slope_sign_convention():
    from extension.parallelism.root import _fit_height_slope

    offsets = torch.linspace(-0.35, 0.35, 7).reshape(1, 7, 1)
    slope = _fit_height_slope(-0.2 * offsets + 0.43, offsets)

    assert torch.allclose(slope, torch.full((1, 1), -0.2), atol=1e-5)


def test_nonflat_root_rollout_uses_multi_point_pitch_fit_and_deadband():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.root import rollout_root

    side = 81
    x = (torch.arange(side, dtype=torch.float32) - 40.0) * 0.01
    height = (0.2 * x).view(1, 1, side).expand(1, side, side).clone()
    terrain = _terrain()
    terrain = type(terrain)(
        height_w=height,
        semantic_id=torch.zeros_like(height, dtype=torch.long),
        valid_mask=torch.ones_like(height, dtype=torch.bool),
        origin_w=torch.tensor([[-0.40, -0.40, 0.0]], dtype=torch.float32),
        yaw_w=torch.zeros(1),
        resolution=0.01,
    )
    cfg = ParallelismCfg(
        terrain_following_pitch_sample_range_m=0.35,
        terrain_following_pitch_sample_count=7,
        terrain_following_roll_sample_range_m=0.35,
        terrain_following_roll_sample_count=5,
        terrain_following_rpy_deadband_rad=0.02,
        terrain_following_rpy_smoothing=1.0,
        terrain_following_rpy_rate_limit_rad=10.0,
        terrain_following_pitch_limit_rad=0.35,
        terrain_following_roll_limit_rad=0.25,
    )

    root = rollout_root(
        _state(),
        torch.zeros(1, 3),
        terrain,
        cfg,
        terrain_following_mask=torch.tensor([True]),
    )

    assert torch.allclose(root.root_rpy_w[0, 0, :2], torch.zeros(2), atol=1e-6)
    assert torch.allclose(root.root_rpy_w[0, 1:, 1], torch.full((23,), -torch.atan(torch.tensor(0.2))), atol=0.02)
    assert torch.allclose(root.root_rpy_w[0, 1:, 0], torch.zeros(23), atol=1e-6)


def test_candidate_shape_and_reference_hips():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.candidates import build_candidates
    from extension.parallelism.root import rollout_root

    cfg = ParallelismCfg()
    state = _state()
    terrain = _terrain()
    root = rollout_root(state, torch.zeros(1, 3), terrain, cfg)
    candidates = build_candidates(root, state, torch.zeros(1, 3), terrain, cfg)
    radius = torch.linalg.vector_norm(candidates.offset_body, dim=-1)

    assert candidates.candidate_w.shape == (1, 4, 50, 3)
    assert candidates.score_target_body.shape == (1, 4, 2)
    assert torch.all(radius <= cfg.candidate_radius_m + 1e-6)
    assert candidates.hip_ref_w.shape == (1, 4, 3)


def test_candidates_place_foot_center_above_terrain_by_contact_offset():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.candidates import build_candidates
    from extension.parallelism.root import rollout_root

    cfg = ParallelismCfg(foot_contact_offset_m=0.022)
    state = _state()
    terrain = _terrain()
    terrain = type(terrain)(
        height_w=torch.full_like(terrain.height_w, 0.3),
        semantic_id=terrain.semantic_id,
        valid_mask=terrain.valid_mask,
        origin_w=terrain.origin_w,
        yaw_w=terrain.yaw_w,
        resolution=terrain.resolution,
    )
    root = rollout_root(state, torch.zeros(1, 3), terrain, cfg)

    candidates = build_candidates(root, state, torch.zeros(1, 3), terrain, cfg)

    torch.testing.assert_close(
        candidates.candidate_w[..., 2],
        torch.full_like(candidates.candidate_w[..., 2], 0.322),
    )


def test_candidate_centers_are_laterally_biased_outward():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.candidates import build_candidates
    from extension.parallelism.root import rollout_root

    cfg = ParallelismCfg(hip_lateral_bias_m=0.0955)
    state = _state()
    terrain = _terrain()
    root = rollout_root(state, torch.zeros(1, 3), terrain, cfg)
    candidates = build_candidates(root, state, torch.zeros(1, 3), terrain, cfg)
    lateral_delta = candidates.candidate_center_w[0, :, 1] - candidates.hip_ref_w[0, :, 1]

    assert torch.allclose(
        lateral_delta,
        torch.tensor([0.0955, -0.0955, 0.0955, -0.0955]),
        atol=1e-6,
    )


def test_score_target_scales_command_displacement():
    from extension.parallelism import ParallelismCfg
    from extension.parallelism.candidates import build_candidates
    from extension.parallelism.root import rollout_root

    cfg = ParallelismCfg(foothold_step_gain=1.5)
    state = _state()
    terrain = _terrain()
    command = torch.tensor([[0.2, 0.1, 0.0]], dtype=torch.float32)
    root = rollout_root(state, command, terrain, cfg)
    candidates = build_candidates(root, state, command, terrain, cfg)
    expected = command[:, None, :2] * (cfg.half_cycle * cfg.dt) * cfg.foothold_step_gain

    assert torch.allclose(candidates.score_target_body, expected.expand(-1, 4, -1), atol=1e-6)


def test_ik_fk_round_trip_default_pose():
    from extension.parallelism.kinematics import fk_go2
    from extension.parallelism.ik import ik_go2

    state = _state()
    geometry = fk_go2(state.root_pos_w, state.root_rpy_w, state.joint_pos)
    joint, reachable = ik_go2(state.root_pos_w, state.root_rpy_w, geometry.foot_pos_w)
    round_trip = fk_go2(state.root_pos_w, state.root_rpy_w, joint.reshape(1, 12))

    assert reachable.all()
    assert torch.allclose(round_trip.foot_pos_w, geometry.foot_pos_w, atol=1e-5)
