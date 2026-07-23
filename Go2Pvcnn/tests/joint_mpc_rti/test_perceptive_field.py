from __future__ import annotations

import pytest
import torch

from extension.joint_mpc_rti.config import JointMpcRtiCfg


def _inputs(batch: int = 2, size: int = 31):
    from extension.joint_mpc_rti.types import JointMpcFieldFrame

    height = torch.zeros(batch, size, size)
    semantic = torch.zeros(batch, size, size, dtype=torch.long)
    valid = torch.ones(batch, size, size, dtype=torch.bool)
    height[:, 15, 15] = 0.08
    semantic[:, 15, 15] = 1
    height[:, 20:23, 15:18] = 0.20
    semantic[:, 20:23, 15:18] = 2
    valid[:, 4, 4] = False
    frame = JointMpcFieldFrame(
        origin_w=torch.zeros(batch, 3),
        yaw_w=torch.zeros(batch),
        timestamp=torch.full((batch,), 0.14),
        refresh_id=torch.full((batch,), 7, dtype=torch.long),
    )
    return height, semantic, valid, frame


def test_field_uses_max_pool_layers_and_preserves_small_vs_large_semantics() -> None:
    from extension.joint_mpc_rti.terrain.perceptive_field import (
        GEOMETRY_NAMES,
        build_perceptive_field,
    )

    height, semantic, valid, frame = _inputs()
    field = build_perceptive_field(height, semantic, valid, frame, JointMpcRtiCfg())

    assert field.height_w.shape == (2, 31, 31)
    assert field.inflated_height_w.shape == (2, len(GEOMETRY_NAMES), 31, 31)
    assert field.landing_safe.shape == (2, 31, 31)
    assert field.small_mask.any() and field.large_mask.any() and field.unknown_mask.any()
    assert not field.landing_safe[field.small_mask].any()
    assert not field.landing_safe[field.large_mask].any()
    assert not field.landing_safe[field.unknown_mask].any()
    assert field.inflated_height_w[:, 0, 15, 15].min() >= 0.08
    assert field.inflated_height_w[:, 0, 21, 16].min() >= field.height_w.new_tensor(0.35)


def test_landing_safe_inflates_obstacles_unknown_and_map_boundary() -> None:
    from extension.joint_mpc_rti.terrain.perceptive_field import build_perceptive_field

    height, semantic, valid, frame = _inputs(batch=1)
    field = build_perceptive_field(height, semantic, valid, frame, JointMpcRtiCfg())

    assert not field.landing_safe[0, 15, 16]
    assert not field.landing_safe[0, 4, 5]
    assert not field.landing_safe[0, 0].any()
    assert field.landing_safe[0, 10, 10]


def test_field_reports_slope_roughness_and_exact_frame_metadata() -> None:
    from extension.joint_mpc_rti.terrain.perceptive_field import build_perceptive_field

    height, semantic, valid, frame = _inputs(batch=1)
    height[:, 10:, :] += 0.04
    field = build_perceptive_field(height, semantic, valid, frame, JointMpcRtiCfg())

    assert field.slope_xy.shape == (1, 31, 31, 2)
    assert field.roughness.shape == (1, 31, 31)
    assert field.slope_xy[0, 10].abs().max() > 0.0
    assert field.roughness[0, 10].max() > 0.0
    assert torch.equal(field.refresh_id, frame.refresh_id)
    assert torch.equal(field.timestamp, frame.timestamp)
    assert torch.equal(field.origin_w, frame.origin_w)
    assert torch.equal(field.yaw_w, frame.yaw_w)


def test_stale_or_mismatched_frame_is_invalid_not_previous_field_fallback() -> None:
    from extension.joint_mpc_rti.terrain.perceptive_field import validate_frame_freshness

    assert not validate_frame_freshness(
        field_refresh_id=torch.tensor([7]),
        state_refresh_id=torch.tensor([8]),
    ).all()
    assert validate_frame_freshness(
        field_refresh_id=torch.tensor([8]),
        state_refresh_id=torch.tensor([8]),
    ).all()
    assert not validate_frame_freshness(
        field_refresh_id=torch.tensor([8]),
        state_refresh_id=torch.tensor([8]),
        field_timestamp=torch.tensor([0.14]),
        state_timestamp=torch.tensor([0.16]),
    ).all()


def test_packed_world_query_returns_all_channels_and_rejects_out_of_map() -> None:
    from extension.joint_mpc_rti.terrain.perceptive_field import build_perceptive_field
    from extension.joint_mpc_rti.terrain.query import query_perceptive_world

    height, semantic, valid, frame = _inputs(batch=1)
    field = build_perceptive_field(height, semantic, valid, frame, JointMpcRtiCfg())
    points = torch.tensor([[[0.0, 0.0], [0.05, 0.05], [1.0, 1.0]]])

    query = query_perceptive_world(field, points)

    assert query.inflated_height_w.shape == (1, 3, 5)
    assert query.slope_xy.shape == (1, 3, 2)
    assert query.valid[0, 0]
    assert query.small_mask[0, 0]
    assert not query.valid[0, 2]
    assert query.boundary_distance_m[0, 2] < 0.0


def test_compact_region_query_matches_the_corresponding_packed_channels() -> None:
    from extension.joint_mpc_rti.terrain.perceptive_field import build_perceptive_field
    from extension.joint_mpc_rti.terrain.query import (
        query_landing_region_world,
        query_perceptive_world,
    )

    height, semantic, valid, frame = _inputs(batch=2)
    field = build_perceptive_field(height, semantic, valid, frame, JointMpcRtiCfg())
    points = torch.tensor(
        [[[0.0, 0.0], [0.04, -0.03], [0.2, 0.1]], [[0.01, 0.02], [-0.1, 0.0], [1.0, 1.0]]]
    )

    compact = query_landing_region_world(field, points)
    packed = query_perceptive_world(field, points)

    assert torch.equal(compact.valid, packed.valid)
    assert torch.equal(compact.landing_safe, packed.landing_safe)
    assert torch.equal(compact.semantic_edge_mask, packed.semantic_edge_mask)
    torch.testing.assert_close(compact.height_w, packed.height_w)
    torch.testing.assert_close(compact.slope_rad, packed.slope_rad)
    torch.testing.assert_close(compact.roughness, packed.roughness)


def test_perceptive_cache_overwrites_selected_rows_and_preserves_refresh_id_on_read() -> None:
    from extension.joint_mpc_rti.terrain.field_cache import JointMpcPerceptiveFieldCache
    from extension.joint_mpc_rti.types import JointMpcFieldFrame

    cache = JointMpcPerceptiveFieldCache(
        num_envs=3,
        grid_size=31,
        device="cpu",
        cfg=JointMpcRtiCfg(),
    )
    height, semantic, valid, frame = _inputs(batch=2)
    frame = JointMpcFieldFrame(
        origin_w=frame.origin_w,
        yaw_w=frame.yaw_w,
        timestamp=frame.timestamp,
        refresh_id=torch.tensor([4, 9]),
    )

    cache.update_rows(
        env_ids=torch.tensor([0, 2]),
        height_w=height,
        semantic_id=semantic,
        valid_mask=valid,
        frame=frame,
    )
    first = cache.as_field()
    second = cache.as_field()

    assert torch.equal(cache.ready, torch.tensor([True, False, True]))
    assert torch.equal(first.refresh_id, torch.tensor([4, -1, 9]))
    assert torch.equal(second.refresh_id, first.refresh_id)
    assert first.small_mask[[0, 2]].any(dim=(1, 2)).all()
    assert first.large_mask[[0, 2]].any(dim=(1, 2)).all()
    assert first.inflated_height_w[[0, 2]].amax() >= 0.35


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA parity requires a GPU")
def test_perceptive_field_cuda_matches_cpu_channels() -> None:
    from extension.joint_mpc_rti.terrain.perceptive_field import build_perceptive_field
    from extension.joint_mpc_rti.types import JointMpcFieldFrame

    height, semantic, valid, frame = _inputs(batch=2)
    cpu = build_perceptive_field(height, semantic, valid, frame, JointMpcRtiCfg())
    cuda_frame = JointMpcFieldFrame(
        origin_w=frame.origin_w.cuda(),
        yaw_w=frame.yaw_w.cuda(),
        timestamp=frame.timestamp.cuda(),
        refresh_id=frame.refresh_id.cuda(),
    )
    cuda = build_perceptive_field(
        height.cuda(), semantic.cuda(), valid.cuda(), cuda_frame, JointMpcRtiCfg()
    )

    for name in (
        "small_mask",
        "large_mask",
        "unknown_mask",
        "landing_safe",
        "semantic_edge_mask",
    ):
        assert torch.equal(getattr(cuda, name).cpu(), getattr(cpu, name))
    torch.testing.assert_close(cuda.inflated_height_w.cpu(), cpu.inflated_height_w)
    torch.testing.assert_close(cuda.slope_xy.cpu(), cpu.slope_xy, atol=1.0e-6, rtol=0.0)
    torch.testing.assert_close(cuda.roughness.cpu(), cpu.roughness, atol=1.0e-6, rtol=0.0)
