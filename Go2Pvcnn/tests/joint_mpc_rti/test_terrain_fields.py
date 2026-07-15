from __future__ import annotations

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
