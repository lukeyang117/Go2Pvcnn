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
        self._cache = JointMpcTerrainFieldCache(
            num_envs=num_envs,
            grid_size=grid_size,
            device=self._device,
            resolution=resolution,
            small_ids=small_ids,
            large_ids=large_ids,
        )
        if self._device.type == "cuda":
            self._build_stream: torch.cuda.Stream | None = torch.cuda.Stream(device=self._device)
            self._source_ready: torch.cuda.Event | None = torch.cuda.Event()
            self._field_ready: torch.cuda.Event | None = torch.cuda.Event()
        else:
            self._build_stream = None
            self._source_ready = None
            self._field_ready = None

    @property
    def ready(self) -> torch.Tensor:
        return self._cache.ready

    def on_raycaster_update(self, scanner, env_ids) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._device).reshape(-1)
        if int(ids.numel()) == 0:
            return
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

        def update() -> None:
            self._cache.update_rows(
                env_ids=ids,
                height_w=ray_hits.index_select(0, ids)[..., 2].reshape(-1, side, side),
                semantic_id=semantic.index_select(0, ids).reshape(-1, side, side),
                origin_w=pos_w.index_select(0, ids),
                yaw_w=extract_yaw_batch(quat_w.index_select(0, ids)),
                timestamp=timestamp.index_select(0, ids),
            )

        if self._build_stream is None:
            update()
            return
        current_stream = torch.cuda.current_stream(device=self._device)
        assert self._source_ready is not None and self._field_ready is not None
        self._source_ready.record(current_stream)
        self._build_stream.wait_event(self._source_ready)
        with torch.cuda.stream(self._build_stream):
            update()
            self._field_ready.record(self._build_stream)

    def latest_field(self):
        if self._field_ready is not None:
            torch.cuda.current_stream(device=self._device).wait_event(self._field_ready)
        return self._cache.as_field()

    def attach(self, scanner) -> None:
        if hasattr(scanner, "set_joint_mpc_field_observer"):
            scanner.set_joint_mpc_field_observer(self)
        else:
            scanner._joint_mpc_field_observer = self


__all__ = ["JointMpcRayCasterFieldSync"]
