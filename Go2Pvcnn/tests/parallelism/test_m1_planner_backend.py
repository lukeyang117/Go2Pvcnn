from __future__ import annotations

import inspect

import torch


def _terrain():
    from extension.parallelism import ParallelismTerrain

    return ParallelismTerrain(
        height_w=torch.zeros(1, 121, 121),
        semantic_id=torch.zeros(1, 121, 121, dtype=torch.long),
        valid_mask=torch.ones(1, 121, 121, dtype=torch.bool),
        origin_w=torch.tensor([[-6.0, -6.0, 0.0]]),
        yaw_w=torch.zeros(1),
        resolution=0.1,
    )


def _m1_stair_boundary_case():
    from extension.parallelism import ParallelismState, ParallelismTerrain

    origin_x, origin_y = 28.7188, 19.25
    resolution = 0.01
    axis_x = origin_x + torch.arange(151, dtype=torch.float32) * resolution
    grid_x = axis_x.view(1, -1).expand(151, -1)
    step = torch.zeros_like(grid_x)
    step = torch.where(grid_x >= 29.2088, step + 1.0, step)
    step = torch.where(grid_x >= 29.5088, step + 1.0, step)
    step = torch.where(grid_x >= 29.8088, step + 1.0, step)
    step = torch.where(grid_x >= 30.1088, step + 1.0, step)
    height = (1.43767 - step * 0.20538).unsqueeze(0)
    terrain = ParallelismTerrain(
        height_w=height,
        semantic_id=torch.zeros(1, 151, 151, dtype=torch.long),
        valid_mask=torch.ones(1, 151, 151, dtype=torch.bool),
        origin_w=torch.tensor([[origin_x, origin_y, 0.0]]),
        yaw_w=torch.zeros(1),
        resolution=resolution,
    )
    state = ParallelismState(
        root_pos_w=torch.tensor([[29.4688, 20.0, 1.5752]]),
        root_rpy_w=torch.tensor([[0.0, 0.35, 0.0]]),
        joint_pos=torch.tensor(
            [[
                0.0797, -1.2595, 1.7280,
                0.0247, -1.2441, 1.7973,
                0.0413, -1.4703, 1.5671,
                -0.1301, -1.8727, 2.4825,
            ]]
        ),
        foot_pos_w=torch.tensor(
            [[
                [29.7834, 20.2332, 1.1231],
                [29.7600, 19.8006, 1.1231],
                [29.2745, 20.2199, 1.3284],
                [29.1965, 19.7741, 1.5338],
            ]]
        ),
    )
    return state, terrain


def test_go2_planner_default_backend_remains_go2():
    from extension.parallelism.planner import plan_trajectory
    from extension.parallelism.robot_backend import get_robot_backend

    assert inspect.signature(plan_trajectory).parameters["robot_backend"].default is None
    assert get_robot_backend().name == "go2"


def test_m1_planner_uses_m1_shape_names_and_12_dof_output():
    from extension.parallelism import ParallelismState
    from extension.parallelism.m1_kinematics import M1_CFG, M1_ROOT_Z_M
    from extension.parallelism.planner import plan_trajectory
    from extension.parallelism.robot_backend import get_robot_backend

    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.zeros(1, 12),
    )
    result = plan_trajectory(
        state,
        torch.zeros(1, 3),
        _terrain(),
        M1_CFG,
        robot_backend=get_robot_backend("m1"),
    )

    assert result.joint_pos.shape[-1] == 12
    assert result.diagnostics.collision_shape_names[0].startswith("FBL_")


def test_m1_swing_start_tolerates_existing_knee_contact_shapes():
    from extension.parallelism.m1_kinematics import M1_CFG

    assert {
        "FBL_knee_box",
        "FAR_knee_box",
        "RBL_knee_box",
        "RAR_knee_box",
    }.issubset(set(M1_CFG.swing_start_tolerant_collision_shape_names))


def test_m1_candidates_use_m1_hip_geometry():
    from extension.parallelism.candidates import build_candidates
    from extension.parallelism.m1_kinematics import M1_ROOT_Z_M
    from extension.parallelism.robot_backend import get_robot_backend
    from extension.parallelism.root import rollout_root
    from extension.parallelism.types import ParallelismState

    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.zeros(1, 12),
    )
    backend = get_robot_backend("m1")
    root = rollout_root(state, torch.zeros(1, 3), _terrain(), backend.cfg)

    candidates = build_candidates(
        root,
        state,
        torch.zeros(1, 3),
        _terrain(),
        backend.cfg,
        robot_backend=backend,
    )

    torch.testing.assert_close(
        candidates.candidate_center_w[0, 0],
        torch.tensor([0.3295, 0.1995, M1_ROOT_Z_M]),
        atol=1.0e-5,
        rtol=0.0,
    )


def test_m1_candidates_follow_root_at_swing_touchdown_phase():
    from extension.parallelism import ParallelismState
    from extension.parallelism.candidates import build_candidates
    from extension.parallelism.m1_kinematics import M1_CFG, M1_DEFAULT_JOINT_POS, M1_ROOT_Z_M
    from extension.parallelism.robot_backend import get_robot_backend
    from extension.parallelism.root import rollout_root

    backend = get_robot_backend("m1")
    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.tensor([M1_DEFAULT_JOINT_POS]),
    )
    command = torch.tensor([[0.0, 0.5, 0.0]])
    terrain = _terrain()
    root = rollout_root(
        state,
        command,
        terrain,
        M1_CFG,
        terrain_following_mask=torch.tensor([True]),
    )

    candidates = build_candidates(root, state, command, terrain, M1_CFG, robot_backend=backend)
    touchdown_frame = M1_CFG.half_cycle - 1
    touchdown_hip = backend.fk(
        root.root_pos_w[:, touchdown_frame],
        root.root_rpy_w[:, touchdown_frame],
        state.joint_pos,
    ).hip_pos_w[:, 0, :2]
    expected_center = touchdown_hip + torch.tensor([0.0, 0.0955])

    torch.testing.assert_close(
        candidates.candidate_center_w[:, 0, :2],
        expected_center,
        atol=1.0e-5,
        rtol=0.0,
    )


def test_m1_planner_accepts_forward_motion_on_flat_terrain():
    from extension.parallelism import ParallelismState
    from extension.parallelism.m1_kinematics import M1_CFG, M1_ROOT_Z_M
    from extension.parallelism.planner import plan_trajectory
    from extension.parallelism.robot_backend import get_robot_backend

    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.zeros(1, 12),
    )
    result = plan_trajectory(
        state,
        torch.tensor([[0.5, 0.0, 0.0]]),
        _terrain(),
        M1_CFG,
        robot_backend=get_robot_backend("m1"),
    )

    assert result.valid.tolist() == [True]
    assert float((result.root_pos_w[0, -1, 0] - result.root_pos_w[0, 0, 0]).item()) > 0.1


def test_m1_stair_boundary_keeps_all_legs_valid_and_follows_slope():
    from extension.parallelism.m1_kinematics import M1_CFG
    from extension.parallelism.planner import plan_trajectory
    from extension.parallelism.robot_backend import get_robot_backend
    from extension.parallelism.terrain import query_height_semantic_valid

    state, terrain = _m1_stair_boundary_case()
    backend = get_robot_backend("m1")
    result = plan_trajectory(
        state,
        torch.tensor([[0.3, 0.0, 0.0]]),
        terrain,
        M1_CFG,
        terrain_following_mask=torch.tensor([True]),
        robot_backend=backend,
    )

    assert M1_CFG.terrain_following_root_clearance_m >= 0.32
    assert result.valid.tolist() == [True]
    assert bool((result.diagnostics.candidate_valid.sum(dim=-1) > 0).all())
    terrain_at_end = query_height_semantic_valid(terrain, result.root_pos_w[:, -1, :2]).height.squeeze(-1)
    torch.testing.assert_close(
        result.root_pos_w[:, -1, 2],
        terrain_at_end + float(M1_CFG.terrain_following_root_clearance_m),
        atol=0.16,
        rtol=0.0,
    )
    assert float(result.root_rpy_w[0, -1, 1].item()) > 0.20


def test_m1_terrain_following_root_does_not_drop_below_current_wheel_support():
    from extension.parallelism import ParallelismState
    from extension.parallelism.m1_kinematics import M1_CFG, M1_DEFAULT_JOINT_POS, M1_ROOT_Z_M
    from extension.parallelism.robot_backend import get_robot_backend
    from extension.parallelism.root import rollout_root

    backend = get_robot_backend("m1")
    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.tensor([M1_DEFAULT_JOINT_POS]),
        foot_pos_w=backend.fk(
            torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
            torch.zeros(1, 3),
            torch.tensor([M1_DEFAULT_JOINT_POS]),
        ).foot_pos_w,
    )
    terrain = _terrain()
    terrain.height_w[:, :, 61:] = -0.20
    terrain.semantic_id[:, 60:62, 61:] = 2

    root = rollout_root(
        state,
        torch.tensor([[0.5, 0.0, 0.0]]),
        terrain,
        M1_CFG,
        terrain_following_mask=torch.tensor([True]),
    )

    assert float(root.root_pos_w[0, :, 2].min()) >= float(state.root_pos_w[0, 2]) - float(
        M1_CFG.terrain_following_root_max_drop_m
    ) - 1.0e-5


def test_m1_terrain_following_preserves_default_wheel_support_height_on_flat_tile():
    from extension.parallelism import ParallelismState
    from extension.parallelism.m1_kinematics import M1_CFG, M1_DEFAULT_JOINT_POS, M1_ROOT_Z_M
    from extension.parallelism.root import rollout_root
    from extension.parallelism.robot_backend import get_robot_backend

    backend = get_robot_backend("m1")
    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.tensor([M1_DEFAULT_JOINT_POS]),
        foot_pos_w=backend.fk(
            torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
            torch.zeros(1, 3),
            torch.tensor([M1_DEFAULT_JOINT_POS]),
        ).foot_pos_w,
    )

    root = rollout_root(
        state,
        torch.tensor([[0.5, 0.0, 0.0]]),
        _terrain(),
        M1_CFG,
        terrain_following_mask=torch.tensor([False]),
    )

    torch.testing.assert_close(
        root.root_pos_w[0, :, 2],
        torch.full((M1_CFG.horizon,), M1_ROOT_Z_M),
        atol=0.01,
        rtol=0.0,
    )


def test_m1_terrain_following_uses_trot_phase_support_wheels_for_root_floor():
    from extension.parallelism import ParallelismState
    from extension.parallelism.m1_kinematics import M1_CFG
    from extension.parallelism.root import rollout_root

    # The first trot half is supported by FR/RL, the second by FL/RR.  A
    # wheel that is about to swing may be higher than the current support pair
    # and must not keep the root elevated for the rest of the horizon.
    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, 1.0]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.zeros(1, 12),
        foot_pos_w=torch.tensor(
            [[[0.0, 0.0, 0.40], [0.0, 0.0, 0.70], [0.0, 0.0, 0.90], [0.0, 0.0, 0.30]]]
        ),
    )

    root = rollout_root(
        state,
        torch.tensor([[0.5, 0.0, 0.0]]),
        _terrain(),
        M1_CFG,
        terrain_following_mask=torch.tensor([True]),
    )

    first_half = root.root_pos_w[0, : M1_CFG.half_cycle, 2]
    second_half = root.root_pos_w[0, M1_CFG.half_cycle :, 2]
    assert float(first_half[-1]) > 1.10
    assert float(second_half[-1]) < float(first_half[-1]) - 0.10


def test_m1_box_height_drop_is_rate_limited_for_fixed_foot_workspace():
    from extension.parallelism import ParallelismState, ParallelismTerrain
    from extension.parallelism.m1_kinematics import M1_CFG, M1_ROOT_Z_M
    from extension.parallelism.root import rollout_root

    # A box tile can leave the root query cell while a stance foot is still
    # above the raised cell.  The root must not drop through the fixed-foot
    # workspace in one planning window.
    height = torch.zeros(1, 121, 121)
    x = -6.0 + torch.arange(121, dtype=torch.float32) * 0.1
    height[:, :, x <= 0.05] = 0.18
    terrain = ParallelismTerrain(
        height_w=height,
        semantic_id=torch.zeros(1, 121, 121, dtype=torch.long),
        valid_mask=torch.ones(1, 121, 121, dtype=torch.bool),
        origin_w=torch.tensor([[-6.0, -6.0, 0.0]]),
        yaw_w=torch.zeros(1),
        resolution=0.1,
    )
    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M + 0.18]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.zeros(1, 12),
    )

    root = rollout_root(
        state,
        torch.tensor([[0.5, 0.0, 0.0]]),
        terrain,
        M1_CFG,
        terrain_following_mask=torch.tensor([True]),
    )

    assert float(root.root_pos_w[0, :, 2].min()) >= float(state.root_pos_w[0, 2]) - 0.0601


def test_m1_high_footprint_boundary_does_not_standstill_on_existing_body_contact():
    """A measured wheel at a terrain edge must not invalidate every FL candidate."""
    from extension.parallelism import ParallelismState, ParallelismTerrain
    from extension.parallelism.m1_kinematics import M1_CFG, M1_DEFAULT_JOINT_POS
    from extension.parallelism.planner import plan_trajectory
    from extension.parallelism.robot_backend import get_robot_backend

    backend = get_robot_backend("m1")
    size = 151
    resolution = 0.01
    origin = torch.tensor([[27.25, -44.75, 0.0]])
    axis_x = origin[0, 0] + torch.arange(size) * resolution
    axis_y = origin[0, 1] + torch.arange(size) * resolution
    grid_y, grid_x = torch.meshgrid(axis_y, axis_x, indexing="ij")
    high = (
        (grid_x >= 28.02)
        & (grid_x <= 28.46)
        & (grid_y >= -43.91)
        & (grid_y <= -43.47)
    )
    height = torch.where(high, torch.full_like(grid_x, 1.18), torch.full_like(grid_x, 0.645)).unsqueeze(0)
    terrain = ParallelismTerrain(
        height_w=height,
        semantic_id=torch.where(high, torch.full((1, size, size), 2, dtype=torch.long), torch.zeros(1, size, size, dtype=torch.long)),
        valid_mask=torch.ones(1, size, size, dtype=torch.bool),
        origin_w=origin,
        yaw_w=torch.zeros(1),
        resolution=resolution,
    )
    root_pos = torch.tensor([[28.0, -44.0, 1.2944565]])
    root_rpy = torch.zeros(1, 3)
    joint_pos = torch.tensor([M1_DEFAULT_JOINT_POS])
    foot_pos = backend.fk(root_pos, root_rpy, joint_pos).foot_pos_w
    state = ParallelismState(root_pos, root_rpy, joint_pos, foot_pos)

    result = plan_trajectory(
        state,
        torch.tensor([[0.5, 0.0, 0.0]]),
        terrain,
        M1_CFG,
        terrain_following_mask=torch.tensor([True]),
        robot_backend=backend,
    )

    assert result.valid.tolist() == [True]
    assert result.diagnostics.candidate_valid[0, 0].any()
    assert float((result.root_pos_w[0, -1, 1] - root_pos[0, 1]).abs()) > 0.05


def test_m1_terrain_following_root_uses_body_footprint_height():
    from extension.parallelism import ParallelismState, ParallelismTerrain
    from extension.parallelism.m1_kinematics import M1_CFG, M1_ROOT_Z_M
    from extension.parallelism.root import rollout_root

    size = 151
    resolution = 0.01
    origin = torch.tensor([[27.25, -44.75, 0.0]])
    axis_x = origin[0, 0] + torch.arange(size) * resolution
    axis_y = origin[0, 1] + torch.arange(size) * resolution
    grid_y, grid_x = torch.meshgrid(axis_y, axis_x, indexing="ij")
    high = (
        (grid_x >= 28.02)
        & (grid_x <= 28.46)
        & (grid_y >= -43.91)
        & (grid_y <= -43.47)
    )
    terrain = ParallelismTerrain(
        height_w=torch.where(high, torch.full_like(grid_x, 1.18), torch.full_like(grid_x, 0.645)).unsqueeze(0),
        semantic_id=torch.zeros(1, size, size, dtype=torch.long),
        valid_mask=torch.ones(1, size, size, dtype=torch.bool),
        origin_w=origin,
        yaw_w=torch.zeros(1),
        resolution=resolution,
    )
    state = ParallelismState(
        root_pos_w=torch.tensor([[28.0, -44.0, 0.645 + M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.zeros(1, 12),
    )

    root = rollout_root(
        state,
        torch.tensor([[0.5, 0.0, 0.0]]),
        terrain,
        M1_CFG,
        terrain_following_mask=torch.tensor([True]),
    )

    assert float(root.root_pos_w[0, -1, 2]) > float(state.root_pos_w[0, 2]) + 0.30


def test_m1_flat_motion_keeps_stance_feet_fixed_and_lets_ik_change_joints():
    from extension.parallelism import ParallelismState
    from extension.parallelism.m1_kinematics import M1_CFG, M1_DEFAULT_JOINT_POS, M1_ROOT_Z_M
    from extension.parallelism.planner import plan_trajectory
    from extension.parallelism.robot_backend import get_robot_backend

    backend = get_robot_backend("m1")
    initial_joint = torch.tensor([M1_DEFAULT_JOINT_POS])
    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=initial_joint,
    )
    initial_foot = backend.fk(state.root_pos_w, state.root_rpy_w, initial_joint).foot_pos_w

    result = plan_trajectory(
        state,
        torch.tensor([[0.5, 0.0, 0.0]]),
        _terrain(),
        M1_CFG,
        robot_backend=backend,
    )

    assert result.valid.tolist() == [True]
    selected = result.selected_foothold_w
    tau = torch.linspace(0.0, 1.0, M1_CFG.half_cycle)
    expected = initial_foot[:, None].expand_as(result.foot_pos_w).clone()
    first_pair = initial_foot[:, None, (0, 3)] * (1.0 - tau[None, :, None, None])
    first_pair = first_pair + selected[:, None, (0, 3)] * tau[None, :, None, None]
    second_pair = initial_foot[:, None, (1, 2)] * (1.0 - tau[None, :, None, None])
    second_pair = second_pair + selected[:, None, (1, 2)] * tau[None, :, None, None]
    expected[:, : M1_CFG.half_cycle, (0, 3)] = first_pair
    expected[:, M1_CFG.half_cycle :, (0, 3)] = selected[:, None, (0, 3)]
    expected[:, : M1_CFG.half_cycle, (1, 2)] = initial_foot[:, None, (1, 2)]
    expected[:, M1_CFG.half_cycle :, (1, 2)] = second_pair
    torch.testing.assert_close(
        result.foot_pos_w,
        expected,
        atol=2.0e-3,
        rtol=0.0,
    )
    assert float((result.joint_pos - initial_joint[:, None]).abs().amax().item()) > 0.1
    assert not bool(result.diagnostics.candidate_needs_swing.any())
    assert not bool(result.diagnostics.selected_needs_swing.any())
    assert bool(result.contact_state.all())


def test_m1_flat_candidate_validation_ignores_only_wheel_support_contact():
    from extension.parallelism import ParallelismState
    from extension.parallelism.m1_kinematics import M1_CFG, M1_DEFAULT_JOINT_POS, M1_ROOT_Z_M
    from extension.parallelism.planner import plan_trajectory
    from extension.parallelism.robot_backend import get_robot_backend

    backend = get_robot_backend("m1")
    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.tensor([M1_DEFAULT_JOINT_POS]),
    )
    result = plan_trajectory(
        state,
        torch.tensor([[0.5, 0.0, 0.0]]),
        _terrain(),
        M1_CFG,
        robot_backend=backend,
    )

    wheel_indices = torch.tensor((3, 7, 11, 15), dtype=torch.long)
    wheel_bits = result.diagnostics.candidate_collision_bits.index_select(-1, wheel_indices)
    assert not bool(wheel_bits.any())


def test_m1_collision_diagnostics_preserve_touchdown_and_path_sources():
    from extension.parallelism import ParallelismState
    from extension.parallelism.m1_kinematics import M1_CFG, M1_DEFAULT_JOINT_POS, M1_ROOT_Z_M
    from extension.parallelism.planner import plan_trajectory
    from extension.parallelism.robot_backend import get_robot_backend

    backend = get_robot_backend("m1")
    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.tensor([M1_DEFAULT_JOINT_POS]),
    )
    result = plan_trajectory(
        state,
        torch.tensor([[0.5, 0.0, 0.0]]),
        _terrain(),
        M1_CFG,
        robot_backend=backend,
    )

    touchdown = result.diagnostics.touchdown_collision_bits
    path = result.diagnostics.swing_collision_bits
    assert touchdown.shape == path.shape == result.diagnostics.candidate_collision_bits.shape
    assert torch.equal(result.diagnostics.candidate_collision_bits, touchdown | path)


def test_m1_rolling_path_over_small_obstacle_requires_swing():
    from extension.parallelism import ParallelismState, ParallelismTerrain
    from extension.parallelism.m1_kinematics import M1_CFG, M1_DEFAULT_JOINT_POS, M1_ROOT_Z_M
    from extension.parallelism.planner import _candidate_needs_swing
    from extension.parallelism.robot_backend import get_robot_backend
    from extension.parallelism.root import rollout_root
    from extension.parallelism.candidates import build_candidates

    backend = get_robot_backend("m1")
    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.tensor([M1_DEFAULT_JOINT_POS]),
    )
    terrain = _terrain()
    command = torch.tensor([[0.5, 0.0, 0.0]])
    root = rollout_root(state, command, terrain, M1_CFG)
    candidates = build_candidates(root, state, command, terrain, M1_CFG, robot_backend=backend)
    rolling_foot = backend.fk(
        root.root_pos_w[:, 0],
        root.root_rpy_w[:, 0],
        state.joint_pos,
    ).foot_pos_w
    obstacle_xy = rolling_foot[0, 3, :2] + torch.tensor([0.12, 0.0])
    row = int(torch.round(obstacle_xy[1] / terrain.resolution - terrain.origin_w[0, 1] / terrain.resolution).item())
    col = int(torch.round(obstacle_xy[0] / terrain.resolution - terrain.origin_w[0, 0] / terrain.resolution).item())
    terrain.semantic_id[:, row, col] = 1

    needs_swing = _candidate_needs_swing(
        state,
        candidates,
        terrain,
        M1_CFG,
        backend,
        root_pos_w=root.root_pos_w,
    )

    assert bool(needs_swing[0, 3].all())


def test_m1_non_swing_candidates_use_rolling_collision_path():
    from extension.parallelism.candidates import build_candidates
    from extension.parallelism.m1_kinematics import M1_CFG, M1_DEFAULT_JOINT_POS, M1_ROOT_Z_M
    from extension.parallelism.planner import _swing_collision_mask
    from extension.parallelism.root import rollout_root
    from extension.parallelism.robot_backend import get_robot_backend
    from extension.parallelism import ParallelismState

    backend = get_robot_backend("m1")
    terrain = _terrain()
    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.tensor([M1_DEFAULT_JOINT_POS]),
    )
    command = torch.tensor([[0.5, 0.0, 0.0]])
    root = rollout_root(state, command, terrain, M1_CFG)
    candidates = build_candidates(root, state, command, terrain, M1_CFG, robot_backend=backend)

    collision_ok, collision_bits = _swing_collision_mask(
        state,
        root.root_pos_w,
        root.root_rpy_w,
        candidates,
        terrain,
        M1_CFG,
        robot_backend=backend,
        candidate_needs_swing=torch.zeros(1, 4, 50, dtype=torch.bool),
    )

    assert bool(collision_ok.all())
    assert not bool(collision_bits.any())


def test_m1_small_obstacle_path_marks_only_the_crossing_leg():
    from extension.parallelism import ParallelismState, ParallelismTerrain
    from extension.parallelism.m1_kinematics import M1_CFG, M1_DEFAULT_JOINT_POS, M1_ROOT_Z_M
    from extension.parallelism.planner import plan_trajectory
    from extension.parallelism.robot_backend import get_robot_backend

    backend = get_robot_backend("m1")
    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.tensor([M1_DEFAULT_JOINT_POS]),
    )
    current_foot = backend.fk(state.root_pos_w, state.root_rpy_w, state.joint_pos).foot_pos_w
    midpoint = (current_foot[0, 0, :2] + torch.tensor([0.10, 0.0]))
    terrain = ParallelismTerrain(
        height_w=torch.zeros(1, 121, 121),
        semantic_id=torch.zeros(1, 121, 121, dtype=torch.long),
        valid_mask=torch.ones(1, 121, 121, dtype=torch.bool),
        origin_w=torch.tensor([[-6.0, -6.0, 0.0]]),
        yaw_w=torch.zeros(1),
        resolution=0.1,
    )
    row = int(torch.round(midpoint[1] / terrain.resolution - terrain.origin_w[0, 1] / terrain.resolution).item())
    col = int(torch.round(midpoint[0] / terrain.resolution - terrain.origin_w[0, 0] / terrain.resolution).item())
    terrain.semantic_id[:, row, col] = 1

    result = plan_trajectory(
        state,
        torch.tensor([[0.5, 0.0, 0.0]]),
        terrain,
        M1_CFG,
        robot_backend=backend,
    )

    assert bool(result.diagnostics.candidate_needs_swing[0, 0].any())
    assert not bool(result.diagnostics.candidate_needs_swing[0, 1:].any())


def test_m1_large_obstacle_path_keeps_terrain_following_leg_in_roll_mode():
    from extension.parallelism import ParallelismState
    from extension.parallelism.candidates import build_candidates
    from extension.parallelism.m1_kinematics import M1_CFG, M1_DEFAULT_JOINT_POS, M1_ROOT_Z_M
    from extension.parallelism.planner import _candidate_needs_swing
    from extension.parallelism.robot_backend import get_robot_backend
    from extension.parallelism.root import rollout_root

    backend = get_robot_backend("m1")
    terrain = _terrain()
    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.tensor([M1_DEFAULT_JOINT_POS]),
    )
    root = rollout_root(
        state,
        torch.tensor([[0.5, 0.0, 0.0]]),
        terrain,
        M1_CFG,
        terrain_following_mask=torch.tensor([True]),
    )
    candidates = build_candidates(
        root,
        state,
        torch.tensor([[0.5, 0.0, 0.0]]),
        terrain,
        M1_CFG,
        robot_backend=backend,
    )

    midpoint = (state.root_pos_w[0, :2] + candidates.candidate_w[0, 0, 0, :2]) * 0.5
    row = int(torch.round((midpoint[1] - terrain.origin_w[0, 1]) / terrain.resolution).item())
    col = int(torch.round((midpoint[0] - terrain.origin_w[0, 0]) / terrain.resolution).item())
    terrain.semantic_id[:, row, col] = 2

    needs_swing = _candidate_needs_swing(
        state,
        candidates,
        terrain,
        M1_CFG,
        backend,
        terrain_following_mask=torch.tensor([True]),
        root_pos_w=root.root_pos_w,
        root_rpy_w=root.root_rpy_w,
    )

    assert not bool(needs_swing[0, 0, 0])
    assert not bool(needs_swing[0, 1, 0])


def test_m1_exposes_candidate_and_selected_swing_metadata():
    from extension.parallelism import ParallelismState
    from extension.parallelism.m1_kinematics import M1_CFG, M1_DEFAULT_JOINT_POS, M1_ROOT_Z_M
    from extension.parallelism.planner import plan_trajectory
    from extension.parallelism.robot_backend import get_robot_backend

    backend = get_robot_backend("m1")
    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.tensor([M1_DEFAULT_JOINT_POS]),
    )
    result = plan_trajectory(
        state,
        torch.tensor([[0.5, 0.0, 0.0]]),
        _terrain(),
        M1_CFG,
        robot_backend=backend,
    )

    assert result.diagnostics.candidate_needs_swing.shape == (1, 4, 50)
    assert result.diagnostics.selected_needs_swing.shape == (1, 4)
    expected = result.diagnostics.candidate_needs_swing.gather(
        -1, result.diagnostics.selected_index.unsqueeze(-1)
    ).squeeze(-1)
    assert torch.equal(result.diagnostics.selected_needs_swing, expected)


def test_m1_selected_false_leg_uses_ground_trot_target():
    from extension.parallelism import ParallelismState
    from extension.parallelism.m1_kinematics import M1_CFG, M1_DEFAULT_JOINT_POS, M1_ROOT_Z_M
    from extension.parallelism.planner import _assemble_foot_targets
    from extension.parallelism.robot_backend import get_robot_backend

    backend = get_robot_backend("m1")
    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.tensor([M1_DEFAULT_JOINT_POS]),
    )
    root_pos = state.root_pos_w[:, None].expand(-1, 24, -1).clone()
    root_pos[..., 0] += torch.linspace(0.0, 0.25, 24)
    root_rpy = state.root_rpy_w[:, None].expand(-1, 24, -1)
    current_foot = backend.fk(state.root_pos_w, state.root_rpy_w, state.joint_pos).foot_pos_w
    selected = current_foot.clone()
    selected[..., 0] += 0.10

    targets = _assemble_foot_targets(
        state,
        root_pos,
        root_rpy,
        selected,
        _terrain(),
        M1_CFG,
        robot_backend=backend,
        leg_swing=torch.zeros(1, 4, dtype=torch.bool),
    )
    tau = torch.linspace(0.0, 1.0, M1_CFG.half_cycle)
    expected_first_pair = current_foot[:, None, (0, 3)] * (1.0 - tau[None, :, None, None])
    expected_first_pair = expected_first_pair + selected[:, None, (0, 3)] * tau[None, :, None, None]
    expected_second_pair = current_foot[:, None, (1, 2)] * (1.0 - tau[None, :, None, None])
    expected_second_pair = expected_second_pair + selected[:, None, (1, 2)] * tau[None, :, None, None]
    torch.testing.assert_close(
        targets[:, : M1_CFG.half_cycle, (0, 3)],
        expected_first_pair,
        atol=2.0e-3,
        rtol=0.0,
    )
    torch.testing.assert_close(
        targets[:, M1_CFG.half_cycle :, (0, 3)],
        selected[:, None, (0, 3)].expand(1, M1_CFG.half_cycle, 2, 3),
        atol=2.0e-3,
        rtol=0.0,
    )
    torch.testing.assert_close(
        targets[:, : M1_CFG.half_cycle, (1, 2)],
        current_foot[:, None, (1, 2)].expand(1, M1_CFG.half_cycle, 2, 3),
        atol=2.0e-3,
        rtol=0.0,
    )
    torch.testing.assert_close(
        targets[:, M1_CFG.half_cycle :, (1, 2)],
        expected_second_pair,
        atol=2.0e-3,
        rtol=0.0,
    )


def test_m1_trot_non_swing_legs_move_xy_on_ground_then_hold_stance():
    from extension.parallelism import ParallelismState
    from extension.parallelism.m1_kinematics import M1_CFG, M1_DEFAULT_JOINT_POS, M1_ROOT_Z_M
    from extension.parallelism.planner import _assemble_foot_targets
    from extension.parallelism.robot_backend import get_robot_backend

    backend = get_robot_backend("m1")
    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.tensor([M1_DEFAULT_JOINT_POS]),
    )
    root_pos = state.root_pos_w[:, None].expand(-1, 24, -1).clone()
    root_rpy = state.root_rpy_w[:, None].expand(-1, 24, -1)
    current_foot = backend.fk(state.root_pos_w, state.root_rpy_w, state.joint_pos).foot_pos_w
    selected = current_foot.clone()
    selected[..., 0] += 0.10

    targets = _assemble_foot_targets(
        state,
        root_pos,
        root_rpy,
        selected,
        _terrain(),
        M1_CFG,
        robot_backend=backend,
        leg_swing=torch.zeros(1, 4, dtype=torch.bool),
    )

    tau = torch.linspace(0.0, 1.0, M1_CFG.half_cycle)
    expected_first_pair = current_foot[:, None, (0, 3)] * (1.0 - tau[None, :, None, None])
    expected_first_pair = expected_first_pair + selected[:, None, (0, 3)] * tau[None, :, None, None]
    expected_second_pair = current_foot[:, None, (1, 2)] * (1.0 - tau[None, :, None, None])
    expected_second_pair = expected_second_pair + selected[:, None, (1, 2)] * tau[None, :, None, None]
    torch.testing.assert_close(
        targets[:, : M1_CFG.half_cycle, (0, 3)],
        expected_first_pair,
        atol=2.0e-3,
        rtol=0.0,
    )
    torch.testing.assert_close(
        targets[:, M1_CFG.half_cycle :, (0, 3)],
        selected[:, None, (0, 3)].expand(1, M1_CFG.half_cycle, 2, 3),
        atol=2.0e-3,
        rtol=0.0,
    )
    torch.testing.assert_close(
        targets[:, : M1_CFG.half_cycle, (1, 2)],
        current_foot[:, None, (1, 2)].expand(1, M1_CFG.half_cycle, 2, 3),
        atol=2.0e-3,
        rtol=0.0,
    )
    torch.testing.assert_close(
        targets[:, M1_CFG.half_cycle :, (1, 2)],
        expected_second_pair,
        atol=2.0e-3,
        rtol=0.0,
    )


def test_m1_semantic_swing_lifts_while_ground_trot_swing_stays_at_contact_z():
    from extension.parallelism import ParallelismState
    from extension.parallelism.m1_kinematics import M1_CFG, M1_DEFAULT_JOINT_POS, M1_ROOT_Z_M
    from extension.parallelism.planner import _assemble_foot_targets
    from extension.parallelism.robot_backend import get_robot_backend

    backend = get_robot_backend("m1")
    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.tensor([M1_DEFAULT_JOINT_POS]),
    )
    root_pos = state.root_pos_w[:, None].expand(-1, 24, -1)
    root_rpy = state.root_rpy_w[:, None].expand(-1, 24, -1)
    current_foot = backend.fk(state.root_pos_w, state.root_rpy_w, state.joint_pos).foot_pos_w
    selected = current_foot.clone()
    selected[..., 0] += 0.20

    flat_targets = _assemble_foot_targets(
        state,
        root_pos,
        root_rpy,
        selected,
        _terrain(),
        M1_CFG,
        robot_backend=backend,
        leg_swing=torch.zeros(1, 4, dtype=torch.bool),
    )

    obstacle = _terrain()
    midpoint = (current_foot[0, 0, :2] + selected[0, 0, :2]) / 2.0
    row = int(torch.round(midpoint[1] / obstacle.resolution + 60.0).item())
    col = int(torch.round(midpoint[0] / obstacle.resolution + 60.0).item())
    obstacle.height_w[0, row, col] = 0.20
    obstacle.semantic_id[0, row, col] = 1
    semantic_targets = _assemble_foot_targets(
        state,
        root_pos,
        root_rpy,
        selected,
        obstacle,
        M1_CFG,
        robot_backend=backend,
        leg_swing=torch.tensor([[True, False, False, False]]),
    )

    ground_z = current_foot[0, 0, 2]
    assert torch.allclose(
        flat_targets[0, : M1_CFG.half_cycle, 0, 2],
        ground_z.expand(M1_CFG.half_cycle),
        atol=2.0e-3,
        rtol=0.0,
    )
    assert float(semantic_targets[0, : M1_CFG.half_cycle, 0, 2].max()) > float(ground_z + 0.05)


def test_m1_selected_second_pair_leg_holds_stance_foot_before_swing():
    from extension.parallelism import ParallelismState
    from extension.parallelism.m1_kinematics import M1_CFG, M1_DEFAULT_JOINT_POS, M1_ROOT_Z_M
    from extension.parallelism.planner import _assemble_foot_targets
    from extension.parallelism.robot_backend import get_robot_backend

    backend = get_robot_backend("m1")
    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.tensor([M1_DEFAULT_JOINT_POS]),
    )
    root_pos = state.root_pos_w[:, None].expand(-1, 24, -1).clone()
    root_pos[..., 0] += torch.linspace(0.0, 0.144, 24)
    root_rpy = state.root_rpy_w[:, None].expand(-1, 24, -1)
    current_foot = backend.fk(state.root_pos_w, state.root_rpy_w, state.joint_pos).foot_pos_w
    selected = current_foot.clone()
    selected[..., 0] += 0.10

    targets = _assemble_foot_targets(
        state,
        root_pos,
        root_rpy,
        selected,
        _terrain(),
        M1_CFG,
        robot_backend=backend,
        leg_swing=torch.tensor([[False, True, False, False]]),
    )

    torch.testing.assert_close(
        targets[:, : M1_CFG.half_cycle, 1],
        current_foot[:, None, 1].expand(1, M1_CFG.half_cycle, 3),
        atol=2.0e-3,
        rtol=0.0,
    )


def test_m1_selected_small_obstacle_triggers_only_that_leg_swing():
    from extension.parallelism import ParallelismState, ParallelismTerrain
    from extension.parallelism.m1_kinematics import M1_CFG, M1_DEFAULT_JOINT_POS, M1_ROOT_Z_M
    from extension.parallelism.planner import plan_trajectory
    from extension.parallelism.robot_backend import get_robot_backend

    backend = get_robot_backend("m1")
    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.tensor([M1_DEFAULT_JOINT_POS]),
    )
    current_foot = backend.fk(state.root_pos_w, state.root_rpy_w, state.joint_pos).foot_pos_w
    terrain = ParallelismTerrain(
        height_w=torch.zeros(1, 121, 121),
        semantic_id=torch.zeros(1, 121, 121, dtype=torch.long),
        valid_mask=torch.ones(1, 121, 121, dtype=torch.bool),
        origin_w=torch.tensor([[-6.0, -6.0, 0.0]]),
        yaw_w=torch.zeros(1),
        resolution=0.1,
    )
    fl_xy = current_foot[0, 0, :2] + torch.tensor([0.25, 0.0])
    row = int(torch.round(fl_xy[1] / terrain.resolution + 60.0).item())
    col = int(torch.round(fl_xy[0] / terrain.resolution + 60.0).item())
    terrain.semantic_id[0, row, col] = 1

    result = plan_trajectory(
        state,
        torch.tensor([[0.5, 0.0, 0.0]]),
        terrain,
        M1_CFG,
        robot_backend=backend,
    )

    assert result.valid.tolist() == [True]
    assert result.diagnostics.selected_needs_swing.tolist() == [[True, False, False, False]]
    selected = result.selected_foothold_w
    tau = torch.linspace(0.0, 1.0, M1_CFG.half_cycle)
    expected_second_pair = current_foot[0, None, (1, 2)] * (1.0 - tau[:, None, None])
    expected_second_pair = expected_second_pair + selected[0, None, (1, 2)] * tau[:, None, None]
    torch.testing.assert_close(
        result.foot_pos_w[0, : M1_CFG.half_cycle, 1:3],
        current_foot[0, None, 1:3].expand(M1_CFG.half_cycle, 2, 3),
        atol=2.0e-3,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.foot_pos_w[0, M1_CFG.half_cycle :, 1:3],
        expected_second_pair,
        atol=2.0e-3,
        rtol=0.0,
    )
    expected_first_pair = current_foot[0, None, 3] * (1.0 - tau[:, None])
    expected_first_pair = expected_first_pair + selected[0, None, 3] * tau[:, None]
    torch.testing.assert_close(
        result.foot_pos_w[0, : M1_CFG.half_cycle, 3],
        expected_first_pair,
        atol=2.0e-3,
        rtol=0.0,
    )
    torch.testing.assert_close(
        result.foot_pos_w[0, M1_CFG.half_cycle :, 3],
        selected[0, 3].expand(M1_CFG.half_cycle, 3),
        atol=2.0e-3,
        rtol=0.0,
    )
    assert bool(result.contact_state[0, :, 1:].all())
    assert float((result.joint_pos[0, :, :3] - state.joint_pos[0, :3]).abs().amax()) > 0.1


def test_m1_second_pair_swing_starts_from_fixed_world_stance_pose():
    from extension.parallelism import ParallelismState, ParallelismTerrain
    from extension.parallelism.m1_kinematics import M1_CFG, M1_DEFAULT_JOINT_POS, M1_ROOT_Z_M
    from extension.parallelism.planner import plan_trajectory
    from extension.parallelism.robot_backend import get_robot_backend

    backend = get_robot_backend("m1")
    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.tensor([M1_DEFAULT_JOINT_POS]),
    )
    terrain = _terrain()
    current_foot = backend.fk(state.root_pos_w, state.root_rpy_w, state.joint_pos).foot_pos_w
    fr_xy = current_foot[0, 1, :2] + torch.tensor([0.10, 0.0])
    row = int(torch.round(fr_xy[1] / terrain.resolution + 60.0).item())
    col = int(torch.round(fr_xy[0] / terrain.resolution + 60.0).item())
    terrain.semantic_id[:, row, col] = 1

    result = plan_trajectory(
        state,
        torch.tensor([[0.5, 0.0, 0.0]]),
        terrain,
        M1_CFG,
        robot_backend=backend,
    )

    assert result.diagnostics.selected_needs_swing.tolist() == [[False, True, False, False]]
    torch.testing.assert_close(
        result.foot_pos_w[0, : M1_CFG.half_cycle, 1],
        current_foot[0, 1].expand(M1_CFG.half_cycle, 3),
        atol=2.0e-3,
        rtol=0.0,
    )


def test_m1_fixed_world_stance_remains_ik_reachable_during_root_rollout():
    from extension.parallelism import ParallelismState
    from extension.parallelism.m1_kinematics import M1_CFG, M1_DEFAULT_JOINT_POS, M1_ROOT_Z_M
    from extension.parallelism.root import rollout_root
    from extension.parallelism.robot_backend import get_robot_backend

    backend = get_robot_backend("m1")
    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.tensor([M1_DEFAULT_JOINT_POS]),
    )
    root = rollout_root(state, torch.tensor([[0.5, 0.0, 0.0]]), _terrain(), M1_CFG)
    current_foot = backend.fk(state.root_pos_w, state.root_rpy_w, state.joint_pos).foot_pos_w
    fixed_stance = current_foot[:, None].expand(-1, M1_CFG.horizon, -1, -1)
    _, reachable = backend.ik(
        root.root_pos_w.reshape(-1, 3),
        root.root_rpy_w.reshape(-1, 3),
        fixed_stance.reshape(-1, 4, 3),
    )

    assert bool(reachable.all())


def test_m1_planner_holds_pose_without_command(monkeypatch):
    from extension.parallelism import ParallelismState
    from extension.parallelism.m1_kinematics import M1_CFG, M1_ROOT_Z_M
    from extension.parallelism.planner import plan_trajectory
    from extension.parallelism.robot_backend import get_robot_backend

    state = ParallelismState(
        root_pos_w=torch.tensor([[1.0, -0.5, M1_ROOT_Z_M]]),
        root_rpy_w=torch.tensor([[0.05, -0.04, 0.2]]),
        joint_pos=torch.tensor([[0.1, 0.4, -0.8] * 4]),
        foot_pos_w=torch.tensor(
            [[[1.3, -0.3, 0.1], [1.3, -0.7, 0.1], [0.7, -0.3, 0.1], [0.7, -0.7, 0.1]]]
        ),
    )

    def fail_if_planning(*_args, **_kwargs):
        raise AssertionError("zero command must not generate a swing plan")

    monkeypatch.setattr("extension.parallelism.planner.build_candidates", fail_if_planning)
    result = plan_trajectory(
        state,
        torch.zeros(1, 3),
        _terrain(),
        M1_CFG,
        robot_backend=get_robot_backend("m1"),
    )

    torch.testing.assert_close(result.root_pos_w, state.root_pos_w[:, None].expand_as(result.root_pos_w))
    torch.testing.assert_close(result.root_rpy_w, state.root_rpy_w[:, None].expand_as(result.root_rpy_w))
    torch.testing.assert_close(result.joint_pos, state.joint_pos[:, None].expand_as(result.joint_pos))
    torch.testing.assert_close(result.foot_pos_w, state.foot_pos_w[:, None].expand_as(result.foot_pos_w))
    assert result.contact_state.all()
    assert result.valid.tolist() == [True]


def test_m1_small_obstacle_allows_wheel_liftoff_from_existing_contact():
    from extension.parallelism import ParallelismState, ParallelismTerrain
    from extension.parallelism.m1_kinematics import M1_CFG, M1_DEFAULT_JOINT_POS, M1_ROOT_Z_M
    from extension.parallelism.planner import plan_trajectory
    from extension.parallelism.robot_backend import get_robot_backend

    backend = get_robot_backend("m1")
    state = ParallelismState(
        root_pos_w=torch.tensor([[0.0, 0.0, M1_ROOT_Z_M]]),
        root_rpy_w=torch.zeros(1, 3),
        joint_pos=torch.tensor([M1_DEFAULT_JOINT_POS]),
    )
    terrain = ParallelismTerrain(
        height_w=torch.zeros(1, 401, 401),
        semantic_id=torch.zeros(1, 401, 401, dtype=torch.long),
        valid_mask=torch.ones(1, 401, 401, dtype=torch.bool),
        origin_w=torch.tensor([[-1.0, -1.0, 0.0]]),
        yaw_w=torch.zeros(1),
        resolution=0.01,
    )
    obstacle_center = torch.tensor([0.10, -0.30])
    row = int(torch.round((obstacle_center[1] - terrain.origin_w[0, 1]) / terrain.resolution).item())
    col = int(torch.round((obstacle_center[0] - terrain.origin_w[0, 0]) / terrain.resolution).item())
    terrain.height_w[:, row - 4 : row + 5, col - 4 : col + 5] = 0.08
    terrain.semantic_id[:, row - 4 : row + 5, col - 4 : col + 5] = 1

    result = plan_trajectory(
        state,
        torch.tensor([[0.3, 0.0, 0.0]]),
        terrain,
        M1_CFG,
        robot_backend=backend,
    )

    assert result.valid.tolist() == [True]
    assert result.diagnostics.selected_needs_swing.tolist() == [[False, True, False, False]]
