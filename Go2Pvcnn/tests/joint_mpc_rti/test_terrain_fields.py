from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch


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
    field = sync.latest_field()

    assert torch.equal(field.version, torch.tensor([0, -1, 0, -1]))
    assert torch.equal(sync.ready, torch.tensor([True, False, True, False]))
    torch.testing.assert_close(field.height_w[[0, 2], 0, 0], torch.tensor([0.1, 0.3]))
    torch.testing.assert_close(field.origin_w[[0, 2]], scanner.data.pos_w[[0, 2]])
    torch.testing.assert_close(field.timestamp[[0, 2]], scanner._timestamp[[0, 2]])
    torch.testing.assert_close(field.origin_w[[1, 3]], torch.zeros(2, 3))


def test_semantic_raycaster_notifies_optional_field_observer_after_map_writes() -> None:
    source = Path(
        "Go2Pvcnn/go2_pvcnn/sensor/semantic_raycaster/semantic_ray_caster.py"
    ).read_text()

    assert "def set_joint_mpc_field_observer" in source
    map_write = source.index("self._data.semantic_map[env_ids] =")
    observer_call = source.index("observer.on_raycaster_update(self, env_ids)")
    assert observer_call > map_write
