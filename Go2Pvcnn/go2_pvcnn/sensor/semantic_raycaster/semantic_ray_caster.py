# Copyright (c) 2026, Go2Pvcnn contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Multi-source semantic grid ray caster extending Isaac Lab :class:`~isaaclab.sensors.ray_caster.ray_caster.RayCaster`.

All ``mesh_prim_paths`` geometries are merged into **one** ``wp.Mesh`` (concatenated vertices, offset triangle
indices). A dense table maps triangle index → semantic id. Each step uses a **single**
``raycast_mesh(..., return_face_id=True)`` (Isaac Lab warp) instead of one raycast per submesh.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import omni
import omni.physics.tensors.impl.api as physx
import torch
from pxr import Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sensors.ray_caster.ray_caster import RayCaster
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.math import convert_quat, quat_apply, quat_apply_yaw
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

from go2_pvcnn.sensor.semantic_raycaster.semantic_ray_caster_data import SemanticGridRayCasterData

if TYPE_CHECKING:
    from go2_pvcnn.sensor.semantic_raycaster.semantic_ray_caster_cfg import SemanticGridRayCasterCfg

logger = logging.getLogger(__name__)

_SUPPORTED_GEOMETRY_TYPES = ("Mesh", "Plane", "Cube", "Sphere", "Cylinder", "Capsule", "Cone")

# 设置环境变量 SEMANTIC_RAYCASTER_DEBUG=N（N 为正整数）：打印合并 mesh 摘要，并在前 N 次 _update_buffers_impl 打印 face_id 统计。
# 例：export SEMANTIC_RAYCASTER_DEBUG=5


def _cfg_use_yaw_only_rays(cfg) -> bool:
    """Match both legacy ``attach_yaw_only`` and newer ``ray_alignment='yaw'`` RayCasterCfg."""
    if getattr(cfg, "attach_yaw_only", False):
        return True
    return getattr(cfg, "ray_alignment", None) == "yaw"


def _grid_nx_ny_from_pattern(cfg: "SemanticGridRayCasterCfg", device: str) -> tuple[int, int]:
    """Match ``grid_pattern`` flatten order: ``grid_x`` has shape (len(x), len(y)), ``flatten`` is C-order."""
    pc = cfg.pattern_cfg
    if not hasattr(pc, "size") or not hasattr(pc, "resolution"):
        raise TypeError("SemanticGridRayCaster requires a GridPatternCfg-style pattern_cfg (size, resolution).")
    x = torch.arange(
        start=-pc.size[0] / 2,
        end=pc.size[0] / 2 + 1.0e-9,
        step=pc.resolution,
        device=device,
    )
    y = torch.arange(
        start=-pc.size[1] / 2,
        end=pc.size[1] / 2 + 1.0e-9,
        step=pc.resolution,
        device=device,
    )
    return len(x), len(y)


def _world_transform_matrix_T(usd_geom) -> np.ndarray:
    """4×4 row-vector transform (p_row @ R.T + t) as numpy, matching existing Mesh path."""
    return np.array(omni.usd.get_world_transform_matrix(usd_geom)).T


def _apply_world_transform(points_local: np.ndarray, transform_T: np.ndarray) -> np.ndarray:
    r = transform_T[:3, :3].astype(np.float64)
    t = transform_T[:3, 3].astype(np.float64)
    return (points_local @ r.T + t).astype(np.float32)


def _usd_axis_token(usd_geom, default: str = "Z") -> str:
    axis_attr = usd_geom.GetAxisAttr() if hasattr(usd_geom, "GetAxisAttr") else None
    if axis_attr is None:
        return default
    axis_value = axis_attr.Get()
    if axis_value is None:
        return default
    return str(axis_value).upper()


def _orient_points_from_z_axis(points_local: np.ndarray, axis: str) -> np.ndarray:
    axis = axis.upper()
    if axis == "Z":
        return points_local
    if axis == "X":
        return np.stack((points_local[:, 2], points_local[:, 1], -points_local[:, 0]), axis=1)
    if axis == "Y":
        return np.stack((points_local[:, 0], points_local[:, 2], -points_local[:, 1]), axis=1)
    raise RuntimeError(f"Unsupported axis token {axis!r}; expected one of 'X', 'Y', 'Z'.")


def _collect_supported_geometry_prims(root_prim: Usd.Prim) -> list[tuple[Usd.Prim, str]]:
    """Depth-first collect every supported geometry prim under ``root_prim``."""
    collected: list[tuple[Usd.Prim, str]] = []
    prim_type = root_prim.GetTypeName()
    if prim_type in _SUPPORTED_GEOMETRY_TYPES:
        collected.append((root_prim, prim_type))
    for child in root_prim.GetChildren():
        collected.extend(_collect_supported_geometry_prims(child))
    return collected


def _mesh_prim_to_world_trimesh(prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray]:
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    transform_T = _world_transform_matrix_T(mesh)
    points = points @ transform_T[:3, :3].T + transform_T[:3, 3]
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
    return points.astype(np.float32), indices


def _cube_prim_to_world_trimesh(prim: Usd.Prim) -> tuple[np.ndarray, np.ndarray]:
    """Isaac ``CuboidCfg`` typically spawns a USD ``Cube`` (size = full edge length, default 2)."""
    try:
        import trimesh

        cube = UsdGeom.Cube(prim)
        sz_attr = cube.GetSizeAttr()
        size = float(sz_attr.Get()) if sz_attr and sz_attr.Get() is not None else 2.0
        tm = trimesh.creation.box(extents=[size, size, size])
        pts = np.asarray(tm.vertices, dtype=np.float64)
        tri = np.asarray(tm.faces, dtype=np.int32)
    except Exception:
        size = 2.0
        ca = prim.GetAttribute("size")
        if ca and ca.Get() is not None:
            size = float(ca.Get())
        h = size * 0.5
        pts = np.array(
            [
                [-h, -h, -h],
                [h, -h, -h],
                [h, h, -h],
                [-h, h, -h],
                [-h, -h, h],
                [h, -h, h],
                [h, h, h],
                [-h, h, h],
            ],
            dtype=np.float64,
        )
        tri = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [2, 3, 7],
                [2, 7, 6],
                [0, 3, 7],
                [0, 7, 4],
                [1, 2, 6],
                [1, 6, 5],
            ],
            dtype=np.int32,
        )
    transform_T = _world_transform_matrix_T(UsdGeom.Cube(prim))
    return _apply_world_transform(pts, transform_T), tri


def _tessellate_native_shape(geom_prim: Usd.Prim, geom_type: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        import trimesh
    except Exception:
        raise RuntimeError(f"{geom_type} at {geom_prim.GetPath()} requires trimesh for tessellation.") from None

    if geom_type == "Sphere":
        geom = UsdGeom.Sphere(geom_prim)
        radius_attr = geom.GetRadiusAttr()
        radius = float(radius_attr.Get()) if radius_attr and radius_attr.Get() is not None else 1.0
        tm = trimesh.creation.uv_sphere(radius=radius, count=(16, 16))
    elif geom_type == "Cylinder":
        geom = UsdGeom.Cylinder(geom_prim)
        radius_attr = geom.GetRadiusAttr()
        height_attr = geom.GetHeightAttr()
        radius = float(radius_attr.Get()) if radius_attr and radius_attr.Get() is not None else 1.0
        height = float(height_attr.Get()) if height_attr and height_attr.Get() is not None else 2.0
        tm = trimesh.creation.cylinder(radius=radius, height=height)
    elif geom_type == "Capsule":
        geom = UsdGeom.Capsule(geom_prim)
        radius_attr = geom.GetRadiusAttr()
        height_attr = geom.GetHeightAttr()
        radius = float(radius_attr.Get()) if radius_attr and radius_attr.Get() is not None else 1.0
        height = float(height_attr.Get()) if height_attr and height_attr.Get() is not None else 2.0
        tm = trimesh.creation.capsule(radius=radius, height=height)
    elif geom_type == "Cone":
        geom = UsdGeom.Cone(geom_prim)
        radius_attr = geom.GetRadiusAttr()
        height_attr = geom.GetHeightAttr()
        radius = float(radius_attr.Get()) if radius_attr and radius_attr.Get() is not None else 1.0
        height = float(height_attr.Get()) if height_attr and height_attr.Get() is not None else 2.0
        tm = trimesh.creation.cone(radius=radius, height=height)
    else:
        raise RuntimeError(f"Tessellation helper does not handle geometry type {geom_type!r}.")

    points = np.asarray(tm.vertices, dtype=np.float64)
    triangles = np.asarray(tm.faces, dtype=np.int32)
    if geom_type in {"Cylinder", "Capsule", "Cone"}:
        points = _orient_points_from_z_axis(points, _usd_axis_token(geom, default="Z"))
    return points, triangles


def _geometry_prim_to_world_trimesh(geom_prim: Usd.Prim, geom_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert one supported USD geometry prim into a world-space triangle mesh."""
    if geom_type == "Mesh":
        return _mesh_prim_to_world_trimesh(geom_prim)
    if geom_type == "Plane":
        mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
        pts = mesh.vertices.astype(np.float64)
        transform_T = _world_transform_matrix_T(UsdGeom.Plane(geom_prim))
        return _apply_world_transform(pts, transform_T), mesh.faces.astype(np.int32)
    if geom_type == "Cube":
        return _cube_prim_to_world_trimesh(geom_prim)
    if geom_type in {"Sphere", "Cylinder", "Capsule", "Cone"}:
        pts, tri = _tessellate_native_shape(geom_prim, geom_type)
        usd_geom_ctor = getattr(UsdGeom, geom_type)
        transform_T = _world_transform_matrix_T(usd_geom_ctor(geom_prim))
        return _apply_world_transform(pts, transform_T), tri

    raise RuntimeError(f"Unsupported geometry type {geom_type!r} at {geom_prim.GetPath()}.")


def _usd_prim_to_world_trimeshes(mesh_prim_path: str) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """Resolve ``mesh_prim_path`` and extract every supported world-space triangle mesh in its subtree."""
    root = sim_utils.find_first_matching_prim(mesh_prim_path)
    if root is None or not root.IsValid():
        raise RuntimeError(f"No prim matched for ray-cast path: {mesh_prim_path!r}")

    meshes: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for geom_prim, geom_type in _collect_supported_geometry_prims(root):
        points, triangles = _geometry_prim_to_world_trimesh(geom_prim, geom_type)
        meshes.append((str(geom_prim.GetPath()), geom_type, points, triangles))
    return meshes


def _semantic_ids_from_face_ids(
    face_ids: torch.Tensor,
    face_semantic_ids: torch.Tensor,
    device: str | torch.device,
) -> torch.Tensor:
    """Map ray-cast face ids to semantic ids without crashing on invalid indices."""
    fid_flat = face_ids.reshape(-1).to(device=device, dtype=torch.long)
    if face_semantic_ids.numel() == 0:
        return torch.zeros(fid_flat.shape[0], device=device, dtype=torch.float32)

    table = face_semantic_ids.to(device=device, dtype=torch.long)
    safe_idx = torch.clamp(fid_flat, min=0, max=table.shape[0] - 1)
    gathered = table[safe_idx].float()
    valid = (fid_flat >= 0) & (fid_flat < table.shape[0])
    return torch.where(valid, gathered, torch.zeros_like(gathered))


class SemanticGridRayCaster(RayCaster):
    """Isaac Lab ``RayCaster`` with one merged mesh, face-id semantics, and elevation + semantic rasters."""

    cfg: SemanticGridRayCasterCfg

    def __init__(self, cfg: SemanticGridRayCasterCfg):
        super().__init__(cfg)
        self._data = SemanticGridRayCasterData(pos_w=self._data.pos_w, quat_w=self._data.quat_w, ray_hits_w=self._data.ray_hits_w)
        self._grid_nx: int = 0
        self._grid_ny: int = 0
        self._combined_wp_mesh = None
        self._face_semantic_ids: torch.Tensor | None = None
        self._late_semantic_mesh_refresh_done = False
        try:
            self._semantic_dbg_remaining = max(0, int(os.environ.get("SEMANTIC_RAYCASTER_DEBUG", "0")))
        except ValueError:
            self._semantic_dbg_remaining = 0

    def _initialize_warp_meshes(self):
        """Merge all ``mesh_prim_paths`` into one warp mesh and build per-face semantic ids."""
        vert_blocks: list[np.ndarray] = []
        tri_blocks: list[np.ndarray] = []
        face_semantic: list[int] = []

        vertex_offset = 0
        for mesh_prim_path in self.cfg.mesh_prim_paths:
            semantic_id = int(self.cfg.mesh_semantic_ids[mesh_prim_path])
            try:
                submeshes = _usd_prim_to_world_trimeshes(mesh_prim_path)
            except RuntimeError as exc:
                if semantic_id == 0:
                    raise
                logger.info(
                    "SemanticGridRayCaster: late semantic root %r is not available yet; skipping semantic=%d. Error: %s",
                    mesh_prim_path,
                    semantic_id,
                    exc,
                )
                continue
            if not submeshes:
                if semantic_id == 0:
                    raise RuntimeError(f"No supported geometry descendants under required root: {mesh_prim_path!r}")
                logger.info(
                    "SemanticGridRayCaster: late semantic root %r is present but empty; skipping semantic=%d.",
                    mesh_prim_path,
                    semantic_id,
                )
                continue

            for geom_path, geom_type, points, triangles in submeshes:
                nt = triangles.shape[0]
                if nt == 0:
                    logger.info(
                        "SemanticGridRayCaster: skipping zero-triangle geometry %r under root %r.",
                        geom_path,
                        mesh_prim_path,
                    )
                    continue
                face_semantic.extend([semantic_id] * nt)
                vert_blocks.append(points)
                tri_blocks.append(triangles.astype(np.int32) + vertex_offset)
                vertex_offset += points.shape[0]
                logger.info(
                    "SemanticGridRayCaster: merged %s %r from root %r — %d verts, %d tris, semantic=%d.",
                    geom_type,
                    geom_path,
                    mesh_prim_path,
                    points.shape[0],
                    nt,
                    semantic_id,
                )

        if not vert_blocks or not tri_blocks:
            raise RuntimeError("SemanticGridRayCaster: no supported geometry collected from configured roots.")

        all_points = np.concatenate(vert_blocks, axis=0)
        all_triangles = np.concatenate(tri_blocks, axis=0)
        self._combined_wp_mesh = convert_to_warp_mesh(all_points, all_triangles, self.device)
        self._face_semantic_ids = torch.tensor(face_semantic, device=self.device, dtype=torch.long)
        logger.info(
            "SemanticGridRayCaster: combined mesh — %d verts, %d faces.",
            all_points.shape[0],
            all_triangles.shape[0],
        )
        if self._semantic_dbg_remaining > 0:
            n_tab = int(self._face_semantic_ids.shape[0])
            uniq_sem = torch.unique(self._face_semantic_ids).cpu().tolist()
            print(
                "[SemanticGridRayCaster][DEBUG init] mesh_prim_paths=%s | verts=%d tris_stored=%d "
                "face_semantic_table_len=%d unique_semantic_ids=%s device=%s"
                % (
                    list(self.cfg.mesh_prim_paths),
                    int(all_points.shape[0]),
                    int(all_triangles.shape[0]),
                    n_tab,
                    uniq_sem,
                    self.device,
                ),
                flush=True,
            )

    def _has_all_configured_semantic_ids(self) -> bool:
        if self._face_semantic_ids is None:
            return False
        present = torch.unique(self._face_semantic_ids).detach().cpu().tolist()
        required = {int(v) for v in self.cfg.mesh_semantic_ids.values() if int(v) != 0}
        return required.issubset({int(v) for v in present})

    def _semantic_roots_have_geometry(self) -> bool:
        for mesh_prim_path in self.cfg.mesh_prim_paths:
            semantic_id = int(self.cfg.mesh_semantic_ids[mesh_prim_path])
            if semantic_id == 0:
                continue
            if _usd_prim_to_world_trimeshes(mesh_prim_path):
                return True
        return False

    def _refresh_late_semantic_mesh_if_needed(self) -> None:
        """Rebuild the warp mesh if startup-generated semantic ids 1/2 appeared after sensor init."""
        if self._late_semantic_mesh_refresh_done or self._has_all_configured_semantic_ids():
            return
        if not self._semantic_roots_have_geometry():
            return
        self._initialize_warp_meshes()
        self._late_semantic_mesh_refresh_done = self._has_all_configured_semantic_ids()

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        nx, ny = _grid_nx_ny_from_pattern(self.cfg, self._device)
        self._grid_nx, self._grid_ny = nx, ny
        n = self._view.count
        if nx * ny != self.num_rays:
            raise RuntimeError(
                f"Grid shape {nx}x{ny}={nx * ny} does not match num_rays={self.num_rays} "
                "(check pattern_cfg size/resolution)."
            )
        self._data.elevation_map = torch.zeros(n, nx, ny, device=self._device)
        self._data.semantic_map = torch.zeros(n, nx, ny, device=self._device)
        if self._semantic_dbg_remaining > 0:
            print(
                "[SemanticGridRayCaster][DEBUG rays] num_envs=%d num_rays=%d grid_nx=%d grid_ny=%d"
                % (int(n), int(self.num_rays), int(nx), int(ny)),
                flush=True,
            )

    def update_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        """Refresh only selected env rows without triggering SensorBase's full outdated pass."""
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._device).reshape(-1)
        if int(ids.numel()) == 0:
            return self._data
        if int(ids.numel()) > 0:
            valid = torch.logical_and(ids >= 0, ids < self._num_envs)
            if not bool(torch.all(valid)):
                bad = ids[torch.logical_not(valid)]
                raise IndexError(
                    f"SemanticGridRayCaster env_ids out of bounds for num_envs={self._num_envs}; "
                    f"first bad ids={bad[:8].tolist()}"
                )
        chunk_size = max(1, int(getattr(self.cfg, "max_update_envs_per_call", 64)))
        for chunk_ids in ids.split(chunk_size):
            self._update_buffers_impl(chunk_ids)
            self._timestamp_last_update[chunk_ids] = self._timestamp[chunk_ids]
            self._is_outdated[chunk_ids] = False
        return self._data

    def _update_outdated_buffers(self):
        outdated_env_ids = self._is_outdated.nonzero().squeeze(-1)
        if len(outdated_env_ids) > 0:
            self.update_env_ids(outdated_env_ids)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        if self._combined_wp_mesh is None or self._face_semantic_ids is None:
            raise RuntimeError("SemanticGridRayCaster: combined mesh not initialized.")
        self._refresh_late_semantic_mesh_if_needed()

        # Inline pose + world rays (matches pre-``_update_ray_infos`` RayCaster; works with XFormPrim / XformPrimView).
        if isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        elif hasattr(self._view, "get_world_poses"):
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")

        pos_w = pos_w.clone()
        quat_w = quat_w.clone()
        pos_w += self.drift[env_ids]
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        if _cfg_use_yaw_only_rays(self.cfg):
            ray_starts_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        else:
            ray_starts_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])

        ray_hits, _, _, face_ids = raycast_mesh(
            ray_starts_w,
            ray_directions_w,
            mesh=self._combined_wp_mesh,
            max_dist=self.cfg.max_distance,
            return_distance=False,
            return_normal=False,
            return_face_id=True,
        )

        self._data.ray_hits_w[env_ids] = ray_hits
        if hasattr(self, "ray_cast_drift"):
            self._data.ray_hits_w[env_ids, :, 2] += self.ray_cast_drift[env_ids, 2].unsqueeze(-1)

        ne, nr = face_ids.shape
        fid_flat = face_ids.reshape(-1).to(device=self.device, dtype=torch.long)
        n_faces = int(self._face_semantic_ids.shape[0])
        sem_flat = _semantic_ids_from_face_ids(face_ids, self._face_semantic_ids, self.device)
        mask = (fid_flat >= 0) & (fid_flat < n_faces)
        sem_ray = sem_flat.view(ne, nr)

        pos_z = self._data.pos_w[env_ids, 2].unsqueeze(1)
        elev_ray = pos_z - ray_hits[..., 2] - self.cfg.height_scan_offset

        if self._semantic_dbg_remaining > 0:
            n_miss = int((fid_flat < 0).sum().item())
            n_oob = int((fid_flat >= n_faces).sum().item())
            n_ok = int(mask.sum().item()) if n_faces > 0 else 0
            total = int(fid_flat.numel())
            fmin = int(fid_flat.min().item()) if total else 0
            fmax = int(fid_flat.max().item()) if total else 0
            n_inf_hit = int(torch.isinf(ray_hits).any(dim=-1).sum().item())
            elev_min = float(elev_ray.min().item()) if elev_ray.numel() else 0.0
            elev_max = float(elev_ray.max().item()) if elev_ray.numel() else 0.0
            sem_u = torch.unique(sem_ray).cpu().tolist()
            try:
                env_ids_repr = f"len={len(env_ids)}"
            except TypeError:
                env_ids_repr = repr(env_ids)
            print(
                "[SemanticGridRayCaster][DEBUG update] env_ids_subset=%s ne=%d nr=%d | "
                "face_id: min=%d max=%d miss(-1)=%d oob(>=n_faces)=%d ok=%d / total=%d | "
                "n_faces_table=%d | ray_hits any_inf=%d | elev[min,max]=[%.4f,%.4f] sem_unique=%s"
                % (
                    env_ids_repr,
                    ne,
                    nr,
                    fmin,
                    fmax,
                    n_miss,
                    n_oob,
                    n_ok,
                    total,
                    n_faces,
                    n_inf_hit,
                    elev_min,
                    elev_max,
                    sem_u,
                ),
                flush=True,
            )
            self._semantic_dbg_remaining -= 1

        nx, ny = self._grid_nx, self._grid_ny
        self._data.elevation_map[env_ids] = elev_ray.view(ne, nx, ny)
        self._data.semantic_map[env_ids] = sem_ray.view(ne, nx, ny)
