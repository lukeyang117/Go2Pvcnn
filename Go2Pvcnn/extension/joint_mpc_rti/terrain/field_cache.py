"""Mutable per-environment cache that atomically publishes completed field rows."""

from __future__ import annotations

import torch
from torch import Tensor

from extension.joint_mpc_rti.terrain.field_builder import build_field_batch
from extension.joint_mpc_rti.types import JointMpcTerrainField


class JointMpcTerrainFieldCache:
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
        self.resolution = float(resolution)
        self.small_ids = tuple(small_ids)
        self.large_ids = tuple(large_ids)
        shape = (int(num_envs), int(grid_size), int(grid_size))
        maximum_distance = self.resolution * ((2.0 * float(grid_size - 1) ** 2) ** 0.5)
        self.height_w = torch.zeros(shape, dtype=torch.float32, device=device)
        self.semantic_id = torch.zeros(shape, dtype=torch.long, device=device)
        self._semantic_storage = self.semantic_id
        self._distance_cb = torch.full((2, *shape), maximum_distance, dtype=torch.float32, device=device)
        self.small_distance_m = self._distance_cb[0]
        self.large_distance_m = self._distance_cb[1]
        self._edt_vertical_workspace = torch.empty((4, *shape), dtype=torch.int16, device=device)
        zero_gradient = torch.zeros(1, 1, 1, 2, dtype=torch.float32, device=device)
        self.small_gradient_xy = zero_gradient.expand(*shape, 2)
        self.large_gradient_xy = zero_gradient.expand(*shape, 2)
        self.valid_mask = torch.zeros(shape, dtype=torch.bool, device=device)
        self.origin_w = torch.zeros(int(num_envs), 3, dtype=torch.float32, device=device)
        self.yaw_w = torch.zeros(int(num_envs), dtype=torch.float32, device=device)
        self.timestamp = torch.zeros(int(num_envs), dtype=torch.float32, device=device)
        self.version = torch.full((int(num_envs),), -1, dtype=torch.long, device=device)
        self.ready = torch.zeros(int(num_envs), dtype=torch.bool, device=device)
        self._height_stream = torch.cuda.Stream(device=device) if self.height_w.is_cuda else None

    def update_rows(
        self,
        *,
        env_ids: Tensor,
        height_w: Tensor,
        semantic_id: Tensor,
        origin_w: Tensor,
        yaw_w: Tensor,
        timestamp: Tensor,
        ordered_full_batch: bool = False,
    ) -> None:
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.version.device)
        if (
            bool(ordered_full_batch)
            and self.height_w.is_cuda
            and int(ids.numel()) == int(self.version.numel())
            and self.height_w.shape[1:] == (151, 151)
        ):
            from extension.joint_mpc_rti.terrain.cuda_edt import copy_height_valid_cuda, semantic_distance_fields_out_cuda

            height = torch.as_tensor(height_w, dtype=torch.float32, device=self.height_w.device)
            semantic = torch.as_tensor(semantic_id, dtype=torch.long, device=self.semantic_id.device).contiguous()
            origin = torch.as_tensor(origin_w, dtype=torch.float32, device=self.origin_w.device).contiguous()
            yaw = torch.as_tensor(yaw_w, dtype=torch.float32, device=self.yaw_w.device).contiguous()
            timestamp_value = torch.as_tensor(timestamp, dtype=torch.float32, device=self.timestamp.device).contiguous()
            current_stream = torch.cuda.current_stream(device=self.height_w.device)
            if self._height_stream is None:
                copy_height_valid_cuda(
                    height, self.height_w, self.valid_mask, origin, self.origin_w, yaw,
                    self.yaw_w, timestamp_value, self.timestamp, self.version, self.ready
                )
            else:
                self._height_stream.wait_stream(current_stream)
                with torch.cuda.stream(self._height_stream):
                    copy_height_valid_cuda(
                        height, self.height_w, self.valid_mask, origin, self.origin_w, yaw,
                        self.yaw_w, timestamp_value, self.timestamp, self.version, self.ready
                    )
            self.semantic_id = semantic
            semantic_distance_fields_out_cuda(
                semantic,
                self._distance_cb,
                self._edt_vertical_workspace,
                small_ids=self.small_ids,
                large_ids=self.large_ids,
                resolution=self.resolution,
            )
            if self._height_stream is not None:
                current_stream.wait_stream(self._height_stream)
            return
        if self.semantic_id.data_ptr() != self._semantic_storage.data_ptr():
            self._semantic_storage.copy_(self.semantic_id)
            self.semantic_id = self._semantic_storage
        next_version = self.version.index_select(0, ids) + 1
        built = build_field_batch(
            height_w=height_w,
            semantic_id=semantic_id,
            origin_w=origin_w,
            yaw_w=yaw_w,
            timestamp=timestamp,
            version=next_version,
            resolution=self.resolution,
            small_ids=self.small_ids,
            large_ids=self.large_ids,
        )
        for name in (
            "height_w",
            "semantic_id",
            "small_distance_m",
            "large_distance_m",
            "valid_mask",
            "origin_w",
            "yaw_w",
            "timestamp",
            "version",
        ):
            destination = getattr(self, name)
            destination.index_copy_(0, ids, getattr(built, name).to(device=destination.device))
        self.ready.index_fill_(0, ids, True)

    def as_field(self) -> JointMpcTerrainField:
        return JointMpcTerrainField(
            height_w=self.height_w,
            semantic_id=self.semantic_id,
            small_distance_m=self.small_distance_m,
            large_distance_m=self.large_distance_m,
            small_gradient_xy=self.small_gradient_xy,
            large_gradient_xy=self.large_gradient_xy,
            valid_mask=self.valid_mask,
            origin_w=self.origin_w,
            yaw_w=self.yaw_w,
            timestamp=self.timestamp,
            version=self.version,
            resolution=self.resolution,
        )


__all__ = ["JointMpcTerrainFieldCache"]
