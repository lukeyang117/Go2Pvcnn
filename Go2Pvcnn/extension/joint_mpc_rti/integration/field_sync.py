"""RayCaster-aligned terrain-field updates for joint MPC RTI."""

from __future__ import annotations

import math

import torch

from extension.convention import extract_yaw_batch
from extension.joint_mpc_rti.terrain.field_cache import JointMpcTerrainFieldCache


class JointMpcRayCasterFieldSync:
    """Build and publish only the scanner rows refreshed by RayCaster."""

    def __init__(
        self,
        *,
        num_envs: int,
        grid_size: int,
        device: torch.device | str,
        resolution: float = 0.01,
        small_ids: tuple[int, ...] = (1,),
        large_ids: tuple[int, ...] = (2,),
    ) -> None:
        self._device = torch.device(device)
        self._num_envs = int(num_envs)
        self._cache = JointMpcTerrainFieldCache(
            num_envs=num_envs,
            grid_size=grid_size,
            device=self._device,
            resolution=resolution,
            small_ids=small_ids,
            large_ids=large_ids,
        )
        self._scanner = None
        self._pending_env_ids: list[object] = []

    @property
    def ready(self) -> torch.Tensor:
        return self._cache.ready

    def on_raycaster_update(self, scanner, env_ids) -> None:
        """Record completed scanner rows without launching PyTorch kernels in the sensor callback."""
        self._scanner = scanner
        self._pending_env_ids.append(env_ids)

    def _flush_pending(self) -> None:
        scanner = self._scanner
        if scanner is None:
            if self._pending_env_ids:
                raise RuntimeError("RayCaster field sync has pending rows without an attached scanner")
            return
        pending = self._pending_env_ids or [slice(0, self._num_envs)]
        self._pending_env_ids = []
        id_tensors = []
        for env_ids in pending:
            if isinstance(env_ids, slice):
                start, stop, step = env_ids.indices(self._num_envs)
                ids = torch.arange(start, stop, step, dtype=torch.long, device=self._device)
            else:
                ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._device).reshape(-1)
            if int(ids.numel()) > 0:
                id_tensors.append(ids)
        if not id_tensors:
            return
        ids = torch.unique(torch.cat(id_tensors), sorted=True)
        data = scanner.data
        ray_hits = torch.as_tensor(data.ray_hits_w, dtype=torch.float32, device=self._device)
        side = int(round(math.sqrt(int(ray_hits.shape[1]))))
        if ray_hits.ndim != 3 or int(ray_hits.shape[-1]) != 3 or side * side != int(ray_hits.shape[1]):
            raise ValueError("scanner ray_hits_w must have shape [B,N*N,3]")
        semantic = torch.as_tensor(data.semantic_map, dtype=torch.long, device=self._device)
        pos_w = torch.as_tensor(data.pos_w, dtype=torch.float32, device=self._device)
        quat_w = torch.as_tensor(data.quat_w, dtype=torch.float32, device=self._device)
        timestamp_source = getattr(scanner, "_timestamp", getattr(data, "timestamp", None))
        if timestamp_source is None:
            timestamp = torch.zeros(ray_hits.shape[0], dtype=torch.float32, device=self._device)
        else:
            timestamp = torch.as_tensor(timestamp_source, dtype=torch.float32, device=self._device).reshape(-1)

        full_refresh = int(ids.numel()) == self._num_envs
        self._cache.update_rows(
            env_ids=ids,
            height_w=(ray_hits if full_refresh else ray_hits.index_select(0, ids))[..., 2].reshape(-1, side, side),
            semantic_id=(semantic if full_refresh else semantic.index_select(0, ids)).reshape(-1, side, side),
            origin_w=pos_w if full_refresh else pos_w.index_select(0, ids),
            yaw_w=extract_yaw_batch(quat_w if full_refresh else quat_w.index_select(0, ids)),
            timestamp=timestamp if full_refresh else timestamp.index_select(0, ids),
            ordered_full_batch=full_refresh,
        )

    def latest_field(self):
        self._flush_pending()
        return self._cache.as_field()

    def attach(self, scanner) -> None:
        self._scanner = scanner
        if hasattr(scanner, "set_joint_mpc_field_observer"):
            scanner.set_joint_mpc_field_observer(self)
        else:
            scanner._joint_mpc_field_observer = self


__all__ = ["JointMpcRayCasterFieldSync"]
