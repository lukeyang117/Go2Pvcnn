from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_small_is_real_height_for_swing_and_virtual_wall_for_touchdown() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.terrain.cost_map import effective_surface

    cfg = JointMpcRtiCfg()
    query = SimpleNamespace(
        height_w=torch.tensor([0.0]),
        small_occupancy=torch.tensor([1.0]),
        large_occupancy=torch.tensor([0.0]),
        small_propagated_height=torch.tensor([0.08]),
        large_propagated_height=torch.tensor([0.0]),
    )

    swing = effective_surface(query, body_part="foot", stance=False, cfg=cfg)
    stance = effective_surface(query, body_part="foot", stance=True, cfg=cfg)

    assert swing.height_w.item() == pytest.approx(0.08)
    assert stance.height_w.item() == pytest.approx(cfg.terrain.h_wall)


def test_large_is_virtual_wall_for_all_parts_and_phases() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.terrain.cost_map import effective_surface

    cfg = JointMpcRtiCfg()
    query = SimpleNamespace(
        height_w=torch.tensor([0.0]),
        small_occupancy=torch.tensor([0.0]),
        large_occupancy=torch.tensor([1.0]),
        small_propagated_height=torch.tensor([0.0]),
        large_propagated_height=torch.tensor([0.20]),
    )

    for part in ("foot", "knee", "calf", "thigh", "base"):
        for stance in (False, True):
            surface = effective_surface(query, body_part=part, stance=stance, cfg=cfg)
            assert surface.height_w.item() == pytest.approx(cfg.terrain.h_wall)


def test_convolution_propagates_small_height_and_nonzero_boundary_gradient() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.terrain.cost_map import build_soft_semantic_fields

    cfg = JointMpcRtiCfg()
    size = 51
    center = size // 2
    height = torch.zeros(1, size, size)
    height[:, center - 1 : center + 2, center - 1 : center + 2] = 0.08
    semantic = torch.zeros(1, size, size, dtype=torch.long)
    semantic[:, center - 1 : center + 2, center - 1 : center + 2] = 1

    fields = build_soft_semantic_fields(height, semantic, cfg.terrain, resolution=0.01)
    offset = cfg.terrain.kernel_radius_cells - 1
    boundary = fields.small_occupancy[:, :, center, center + offset]
    gradient = fields.small_gradient_xy[:, :, center, center + offset]

    assert torch.all(boundary > 0.0)
    assert torch.all(torch.linalg.vector_norm(gradient, dim=1) > 0.0)
    assert fields.small_height[:, :, center, center + 1].item() == pytest.approx(0.08)


def test_soft_semantic_query_is_differentiable_with_respect_to_xy() -> None:
    from extension.joint_mpc_rti.config import JointMpcRtiCfg
    from extension.joint_mpc_rti.terrain.field_builder import build_field_batch
    from extension.joint_mpc_rti.terrain.query import query_world

    cfg = JointMpcRtiCfg()
    size = 51
    semantic = torch.zeros(1, size, size, dtype=torch.long)
    semantic[:, 24:27, 24:27] = 1
    height = torch.zeros(1, size, size)
    height[:, 24:27, 24:27] = 0.08
    field = build_field_batch(
        height_w=height,
        semantic_id=semantic,
        origin_w=torch.zeros(1, 3),
        yaw_w=torch.zeros(1),
        timestamp=torch.zeros(1),
        version=torch.ones(1, dtype=torch.long),
        resolution=0.01,
        small_ids=(1,),
        large_ids=(2,),
        terrain_cfg=cfg.terrain,
    )
    point_xy = torch.tensor([[[0.055, 0.0]]], requires_grad=True)

    risk = query_world(field, point_xy).small_occupancy.sum()
    risk.backward()

    assert torch.isfinite(point_xy.grad).all()
    assert point_xy.grad.abs().sum() > 0.0


def _build_semantic_field_for_signed_test(semantic: torch.Tensor):
    from extension.joint_mpc_rti.terrain.field_builder import build_field_batch

    batch = int(semantic.shape[0])
    return build_field_batch(
        height_w=torch.zeros_like(semantic, dtype=torch.float32),
        semantic_id=semantic,
        origin_w=torch.zeros(batch, 3),
        yaw_w=torch.zeros(batch),
        timestamp=torch.zeros(batch),
        version=torch.zeros(batch, dtype=torch.long),
        resolution=0.01,
        small_ids=(1,),
        large_ids=(2,),
    )


def test_cpu_semantic_distance_is_signed_and_half_cell_corrected() -> None:
    semantic = torch.zeros(1, 151, 151, dtype=torch.long)
    semantic[:, 70:81, 70:81] = 1

    field = _build_semantic_field_for_signed_test(semantic)

    assert field.small_distance_m[0, 75, 75] < 0.0
    torch.testing.assert_close(
        field.small_distance_m[0, 75, 69],
        torch.tensor(0.005),
        atol=1.0e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        field.small_distance_m[0, 75, 70],
        torch.tensor(-0.005),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_cpu_signed_distance_degenerate_channels_are_finite() -> None:
    empty = _build_semantic_field_for_signed_test(torch.zeros(1, 151, 151, dtype=torch.long))
    full = _build_semantic_field_for_signed_test(torch.ones(1, 151, 151, dtype=torch.long))

    assert torch.isfinite(empty.small_distance_m).all()
    assert torch.all(empty.small_distance_m > 0.0)
    assert torch.isfinite(full.small_distance_m).all()
    assert torch.all(full.small_distance_m < 0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA signed EDT requires a GPU")
def test_cuda_semantic_distance_matches_cpu_signed_reference() -> None:
    semantic = torch.zeros(2, 151, 151, dtype=torch.long)
    semantic[0, 70:81, 70:81] = 1
    semantic[0, 30:41, 100:111] = 2
    semantic[1, 50:66, 85:101] = 1
    semantic[1, 90:106, 40:56] = 2
    cpu_field = _build_semantic_field_for_signed_test(semantic)
    cuda_field = _build_semantic_field_for_signed_test(semantic.cuda())

    torch.testing.assert_close(cuda_field.small_distance_m.cpu(), cpu_field.small_distance_m, atol=1.0e-5, rtol=0.0)
    torch.testing.assert_close(cuda_field.large_distance_m.cpu(), cpu_field.large_distance_m, atol=1.0e-5, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA signed EDT requires a GPU")
def test_cuda_signed_distance_degenerate_channels_are_finite() -> None:
    semantic = torch.stack(
        (
            torch.zeros(151, 151, dtype=torch.long),
            torch.ones(151, 151, dtype=torch.long),
        ),
        dim=0,
    ).cuda()
    field = _build_semantic_field_for_signed_test(semantic)

    assert torch.isfinite(field.small_distance_m).all()
    assert torch.all(field.small_distance_m[0] > 0.0)
    assert torch.all(field.small_distance_m[1] < 0.0)


def test_world_query_uses_bound_field_pose_and_returns_invalid_outside() -> None:
    from extension.joint_mpc_rti.terrain.field_builder import build_field_batch
    from extension.joint_mpc_rti.terrain.query import query_world

    height = torch.zeros(1, 151, 151)
    semantic = torch.zeros(1, 151, 151, dtype=torch.long)
    semantic[:, 75, 85] = 1
    field = build_field_batch(
        height_w=height,
        semantic_id=semantic,
        origin_w=torch.tensor([[2.0, 3.0, 0.0]]),
        yaw_w=torch.tensor([torch.pi / 2]),
        timestamp=torch.tensor([5.0]),
        version=torch.tensor([7]),
        resolution=0.01,
        small_ids=(1,),
        large_ids=(2,),
    )

    inside = query_world(field, torch.tensor([[[1.90, 3.00, 0.0]]]))
    outside = query_world(field, torch.tensor([[[5.0, 5.0, 0.0]]]))

    assert inside.valid.item()
    assert inside.small_distance_m.item() <= 0.02
    assert not outside.valid.item()


def test_world_query_returns_height_gradient_for_linear_terrain() -> None:
    from extension.joint_mpc_rti.terrain.field_builder import build_field_batch
    from extension.joint_mpc_rti.terrain.query import query_world

    grid_x = torch.arange(151, dtype=torch.float32).view(1, 151, 1)
    height = 0.01 * grid_x.expand(1, 151, 151)
    field = build_field_batch(
        height_w=height,
        semantic_id=torch.zeros(1, 151, 151, dtype=torch.long),
        origin_w=torch.zeros(1, 3),
        yaw_w=torch.zeros(1),
        timestamp=torch.zeros(1),
        version=torch.zeros(1, dtype=torch.long),
        resolution=0.01,
        small_ids=(1,),
        large_ids=(2,),
    )

    queried = query_world(field, torch.tensor([[[0.0, 0.0, 0.0]]]))

    torch.testing.assert_close(
        queried.height_gradient_w,
        torch.tensor([[[1.0, 0.0]]]),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_field_cache_updates_only_selected_env_rows_atomically() -> None:
    from extension.joint_mpc_rti.terrain.field_cache import JointMpcTerrainFieldCache

    cache = JointMpcTerrainFieldCache(num_envs=4, grid_size=151, device="cpu")
    before = cache.version.clone()
    update = dict(
        height_w=torch.zeros(2, 151, 151),
        semantic_id=torch.zeros(2, 151, 151, dtype=torch.long),
        origin_w=torch.zeros(2, 3),
        yaw_w=torch.zeros(2),
        timestamp=torch.ones(2),
    )

    cache.update_rows(env_ids=torch.tensor([1, 3]), **update)

    assert torch.equal(cache.version[[0, 2]], before[[0, 2]])
    assert torch.all(cache.version[[1, 3]] > before[[1, 3]])
    assert torch.equal(cache.ready, torch.tensor([False, True, False, True]))


def test_world_gradient_rotates_with_field_yaw() -> None:
    from extension.joint_mpc_rti.terrain.field_builder import build_field_batch
    from extension.joint_mpc_rti.terrain.query import query_world

    height = torch.zeros(1, 151, 151)
    semantic = torch.zeros(1, 151, 151, dtype=torch.long)
    semantic[:, 75, 75] = 2
    field = build_field_batch(
        height_w=height,
        semantic_id=semantic,
        origin_w=torch.zeros(1, 3),
        yaw_w=torch.tensor([torch.pi / 2]),
        timestamp=torch.zeros(1),
        version=torch.ones(1, dtype=torch.long),
        resolution=0.01,
        small_ids=(1,),
        large_ids=(2,),
    )

    query = query_world(field, torch.tensor([[[0.0, 0.10, 0.0]]]))

    assert query.valid.item()
    assert query.large_gradient_w[0, 0, 1] > 0.5
    assert query.large_gradient_w[0, 0, 0].abs() < 0.2


def test_world_query_derives_distance_gradient_at_query_time() -> None:
    from extension.joint_mpc_rti.terrain.field_builder import build_field_batch
    from extension.joint_mpc_rti.terrain.query import query_world

    semantic = torch.zeros(1, 151, 151, dtype=torch.long)
    semantic[:, 75, 75] = 2
    field = build_field_batch(
        height_w=torch.zeros(1, 151, 151),
        semantic_id=semantic,
        origin_w=torch.zeros(1, 3),
        yaw_w=torch.zeros(1),
        timestamp=torch.zeros(1),
        version=torch.ones(1, dtype=torch.long),
        resolution=0.01,
        small_ids=(1,),
        large_ids=(2,),
    )
    poisoned = replace(
        field,
        small_gradient_xy=torch.full_like(field.small_gradient_xy, float("nan")),
        large_gradient_xy=torch.full_like(field.large_gradient_xy, float("nan")),
    )

    query = query_world(poisoned, torch.tensor([[[0.10, 0.0, 0.0]]]))

    assert torch.isfinite(query.large_gradient_w).all()
    assert query.large_gradient_w[0, 0, 0] > 0.5


def test_world_query_maps_repeated_candidate_rows_to_original_field_rows() -> None:
    from extension.joint_mpc_rti.terrain.field_builder import build_field_batch
    from extension.joint_mpc_rti.terrain.query import query_world

    semantic = torch.zeros(2, 151, 151, dtype=torch.long)
    semantic[0, 85, 75] = 2
    semantic[1, 65, 75] = 2
    field = build_field_batch(
        height_w=torch.zeros(2, 151, 151),
        semantic_id=semantic,
        origin_w=torch.zeros(2, 3),
        yaw_w=torch.zeros(2),
        timestamp=torch.zeros(2),
        version=torch.ones(2, dtype=torch.long),
        resolution=0.01,
        small_ids=(1,),
        large_ids=(2,),
    )
    points = torch.tensor(
        [
            [[0.10, 0.0, 0.0]],
            [[0.10, 0.0, 0.0]],
            [[-0.10, 0.0, 0.0]],
            [[-0.10, 0.0, 0.0]],
        ]
    )

    query = query_world(field, points)

    assert torch.all(query.large_distance_m < 0.02)


def test_raycaster_field_sync_updates_the_same_env_ids_pose_and_timestamp() -> None:
    from extension.joint_mpc_rti.integration.field_sync import JointMpcRayCasterFieldSync

    batch = 4
    grid_size = 5
    root_quat = torch.zeros(batch, 4)
    root_quat[:, 0] = 1.0
    ray_hits = torch.zeros(batch, grid_size * grid_size, 3)
    ray_hits[0, :, 2] = 0.1
    ray_hits[2, :, 2] = 0.3
    scanner = SimpleNamespace(
        data=SimpleNamespace(
            ray_hits_w=ray_hits,
            semantic_map=torch.zeros(batch, grid_size, grid_size, dtype=torch.long),
            pos_w=torch.tensor(
                [[1.0, 2.0, 0.5], [3.0, 4.0, 0.5], [5.0, 6.0, 0.5], [7.0, 8.0, 0.5]]
            ),
            quat_w=root_quat,
        ),
        _timestamp=torch.tensor([0.02, 0.04, 0.06, 0.08]),
    )
    sync = JointMpcRayCasterFieldSync(
        num_envs=batch,
        grid_size=grid_size,
        device="cpu",
        resolution=0.01,
    )

    sync.on_raycaster_update(scanner, torch.tensor([0, 2]))
    assert torch.equal(sync.ready, torch.tensor([False, False, False, False]))
    field = sync.latest_field()

    assert torch.equal(field.version, torch.tensor([0, -1, 0, -1]))
    assert torch.equal(sync.ready, torch.tensor([True, False, True, False]))
    torch.testing.assert_close(field.height_w[[0, 2], 0, 0], torch.tensor([0.1, 0.3]))
    torch.testing.assert_close(field.origin_w[[0, 2]], scanner.data.pos_w[[0, 2]])
    torch.testing.assert_close(field.timestamp[[0, 2]], scanner._timestamp[[0, 2]])
    torch.testing.assert_close(field.origin_w[[1, 3]], torch.zeros(2, 3))


def test_latest_field_rebuilds_once_per_call_without_a_new_callback() -> None:
    from extension.joint_mpc_rti.integration.field_sync import JointMpcRayCasterFieldSync

    batch = 2
    grid_size = 5
    scanner = SimpleNamespace(
        data=SimpleNamespace(
            ray_hits_w=torch.zeros(batch, grid_size * grid_size, 3),
            semantic_map=torch.zeros(batch, grid_size, grid_size, dtype=torch.long),
            pos_w=torch.zeros(batch, 3),
            quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(batch, -1).clone(),
        ),
        _timestamp=torch.zeros(batch),
    )
    sync = JointMpcRayCasterFieldSync(
        num_envs=batch,
        grid_size=grid_size,
        device="cpu",
        resolution=0.01,
    )
    sync.attach(scanner)
    sync.on_raycaster_update(scanner, torch.arange(batch))

    first_version = sync.latest_field().version.clone()
    second_version = sync.latest_field().version.clone()

    assert torch.equal(first_version, torch.zeros(batch, dtype=torch.long))
    assert torch.equal(second_version, torch.ones(batch, dtype=torch.long))


def test_semantic_raycaster_notifies_optional_field_observer_after_map_writes() -> None:
    source = Path(
        "Go2Pvcnn/go2_pvcnn/sensor/semantic_raycaster/semantic_ray_caster.py"
    ).read_text()

    assert "def set_joint_mpc_field_observer" in source
    map_write = source.index("self._data.semantic_map[env_ids] =")
    observer_call = source.index("observer.on_raycaster_update(self, env_ids)")
    assert observer_call > map_write


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA exact EDT requires a GPU")
def test_cuda_exact_edt_matches_scipy_for_batched_semantic_masks() -> None:
    from scipy.ndimage import distance_transform_edt

    from extension.joint_mpc_rti.terrain.cuda_edt import exact_squared_edt_cuda

    mask = torch.zeros(4, 2, 151, 151, dtype=torch.bool, device="cuda")
    mask[0, 0, 75, 75] = True
    mask[0, 1, 0, 0] = True
    mask[1, 0, 25:40, 70:90] = True
    mask[1, 1, 110:130, 10:20] = True
    mask[2, 0, 10, 140] = True
    mask[2, 1, 140, 10] = True
    generator = torch.Generator(device="cuda").manual_seed(17)
    mask[3] = torch.rand(2, 151, 151, generator=generator, device="cuda") < 0.015

    actual = exact_squared_edt_cuda(mask).cpu().numpy()

    expected = np.empty_like(actual)
    mask_cpu = mask.cpu().numpy()
    for batch_index in range(4):
        for channel_index in range(2):
            distance = distance_transform_edt(~mask_cpu[batch_index, channel_index])
            expected[batch_index, channel_index] = distance * distance
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA exact EDT requires a GPU")
def test_cuda_exact_edt_empty_map_uses_finite_grid_diagonal() -> None:
    from extension.joint_mpc_rti.terrain.cuda_edt import exact_squared_edt_cuda

    actual = exact_squared_edt_cuda(torch.zeros(1, 2, 151, 151, dtype=torch.bool, device="cuda"))

    torch.testing.assert_close(actual, torch.full_like(actual, 2.0 * 150.0**2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA exact EDT requires a GPU")
def test_cuda_edt_batch_supports_1024_independent_small_large_maps() -> None:
    from extension.joint_mpc_rti.terrain.cuda_edt import exact_squared_edt_cuda

    mask = torch.zeros(1024, 2, 151, 151, dtype=torch.bool, device="cuda")
    mask[:, 0, 75, 75] = True
    mask[:, 1, 25, 125] = True
    actual = exact_squared_edt_cuda(mask)

    assert actual.shape == mask.shape
    assert actual.dtype == torch.float32
    assert actual.is_contiguous()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual[:, 0, 75, 75], torch.zeros(1024, device="cuda"))
    torch.testing.assert_close(actual[:, 1, 25, 125], torch.zeros(1024, device="cuda"))


@pytest.mark.parametrize("batch", (1, 40, 512, 1024))
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA exact EDT requires a GPU")
def test_cuda_exact_field_supports_each_configured_environment_batch(batch: int) -> None:
    from extension.joint_mpc_rti.terrain.field_cache import JointMpcTerrainFieldCache

    cache = JointMpcTerrainFieldCache(num_envs=batch, grid_size=151, device="cuda")
    semantic = torch.zeros(batch, 151, 151, dtype=torch.long, device="cuda")
    semantic[:, 75, 75] = 1
    semantic[:, 25, 125] = 2
    cache.update_rows(
        env_ids=torch.arange(batch, device="cuda"),
        height_w=torch.zeros(batch, 151, 151, device="cuda"),
        semantic_id=semantic,
        origin_w=torch.zeros(batch, 3, device="cuda"),
        yaw_w=torch.zeros(batch, device="cuda"),
        timestamp=torch.zeros(batch, device="cuda"),
        ordered_full_batch=True,
    )

    field = cache.as_field()
    assert field.height_w.shape == (batch, 151, 151)
    assert torch.all(field.version == 0)
    assert torch.isfinite(field.small_distance_m).all()
    assert torch.isfinite(field.large_distance_m).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA exact EDT requires a GPU")
def test_cuda_field_builder_uses_exact_edt_without_jump_flood(monkeypatch) -> None:
    from extension.joint_mpc_rti.terrain import field_builder
    from extension.joint_mpc_rti.terrain import cuda_edt

    def fail_cpu_signed_distance(*args, **kwargs):
        raise AssertionError("CUDA field construction must not call the CPU signed distance builder")

    def fail_unfused_edt(*args, **kwargs):
        raise AssertionError("CUDA field construction must use the fused semantic-field kernel")

    monkeypatch.setattr(field_builder, "signed_boundary_distance", fail_cpu_signed_distance)
    monkeypatch.setattr(cuda_edt, "exact_squared_edt_cuda", fail_unfused_edt)
    semantic = torch.zeros(2, 151, 151, dtype=torch.long, device="cuda")
    semantic[0, 75, 75] = 1
    semantic[1, 25, 125] = 2
    field = field_builder.build_field_batch(
        height_w=torch.zeros(2, 151, 151, device="cuda"),
        semantic_id=semantic,
        origin_w=torch.zeros(2, 3, device="cuda"),
        yaw_w=torch.zeros(2, device="cuda"),
        timestamp=torch.zeros(2, device="cuda"),
        version=torch.ones(2, dtype=torch.long, device="cuda"),
        resolution=0.01,
        small_ids=(1,),
        large_ids=(2,),
    )

    torch.testing.assert_close(field.small_distance_m[0, 75, 75], torch.tensor(-0.005, device="cuda"))
    torch.testing.assert_close(field.large_distance_m[1, 25, 125], torch.tensor(-0.005, device="cuda"))
    assert torch.isfinite(field.small_gradient_xy).all()
    assert torch.isfinite(field.large_gradient_xy).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA exact EDT requires a GPU")
def test_cuda_full_cache_refresh_writes_exact_edt_in_place(monkeypatch) -> None:
    from extension.joint_mpc_rti.terrain import field_cache

    def fail_temporary_builder(*args, **kwargs):
        raise AssertionError("full CUDA cache refresh must use fixed in-place EDT workspace")

    monkeypatch.setattr(field_cache, "build_field_batch", fail_temporary_builder)
    cache = field_cache.JointMpcTerrainFieldCache(
        num_envs=4,
        grid_size=151,
        device="cuda",
        resolution=0.01,
    )
    semantic = torch.zeros(4, 151, 151, dtype=torch.long, device="cuda")
    semantic[:, 75, 75] = 1
    semantic[:, 25, 125] = 2
    cache.update_rows(
        env_ids=torch.arange(4, device="cuda"),
        height_w=torch.zeros(4, 151, 151, device="cuda"),
        semantic_id=semantic,
        origin_w=torch.zeros(4, 3, device="cuda"),
        yaw_w=torch.zeros(4, device="cuda"),
        timestamp=torch.zeros(4, device="cuda"),
        ordered_full_batch=True,
    )

    assert torch.all(cache.version == 0)
    torch.testing.assert_close(cache.small_distance_m[:, 75, 75], torch.full((4,), -0.005, device="cuda"))
    torch.testing.assert_close(cache.large_distance_m[:, 25, 125], torch.full((4,), -0.005, device="cuda"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA exact EDT requires a GPU")
def test_cuda_partial_refresh_after_full_refresh_does_not_mutate_scanner_semantics() -> None:
    from extension.joint_mpc_rti.terrain.field_cache import JointMpcTerrainFieldCache

    cache = JointMpcTerrainFieldCache(num_envs=2, grid_size=151, device="cuda")
    scanner_semantic = torch.zeros(2, 151, 151, dtype=torch.long, device="cuda")
    cache.update_rows(
        env_ids=torch.arange(2, device="cuda"),
        height_w=torch.zeros(2, 151, 151, device="cuda"),
        semantic_id=scanner_semantic,
        origin_w=torch.zeros(2, 3, device="cuda"),
        yaw_w=torch.zeros(2, device="cuda"),
        timestamp=torch.zeros(2, device="cuda"),
        ordered_full_batch=True,
    )
    partial_semantic = torch.full((1, 151, 151), 2, dtype=torch.long, device="cuda")
    cache.update_rows(
        env_ids=torch.tensor([1], device="cuda"),
        height_w=torch.zeros(1, 151, 151, device="cuda"),
        semantic_id=partial_semantic,
        origin_w=torch.zeros(1, 3, device="cuda"),
        yaw_w=torch.zeros(1, device="cuda"),
        timestamp=torch.ones(1, device="cuda"),
    )

    assert torch.count_nonzero(scanner_semantic) == 0
    assert torch.all(cache.semantic_id[1] == 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA exact EDT requires a GPU")
def test_cuda_full_but_permuted_env_ids_preserve_row_mapping() -> None:
    from extension.joint_mpc_rti.terrain.field_cache import JointMpcTerrainFieldCache

    cache = JointMpcTerrainFieldCache(num_envs=2, grid_size=151, device="cuda")
    semantic = torch.zeros(2, 151, 151, dtype=torch.long, device="cuda")
    semantic[0, 25, 25] = 1
    semantic[1, 125, 125] = 1
    cache.update_rows(
        env_ids=torch.tensor([1, 0], device="cuda"),
        height_w=torch.zeros(2, 151, 151, device="cuda"),
        semantic_id=semantic,
        origin_w=torch.zeros(2, 3, device="cuda"),
        yaw_w=torch.zeros(2, device="cuda"),
        timestamp=torch.zeros(2, device="cuda"),
    )

    torch.testing.assert_close(cache.small_distance_m[1, 25, 25], torch.tensor(-0.005, device="cuda"))
    torch.testing.assert_close(cache.small_distance_m[0, 125, 125], torch.tensor(-0.005, device="cuda"))
