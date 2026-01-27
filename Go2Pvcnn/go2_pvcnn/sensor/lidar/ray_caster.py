"""Ray caster implementation for LiDAR sensor with dynamic mesh support.

This is adapted from grouped_ray_caster to support dynamic mesh updating for
efficient multi-environment ray casting with moving objects.

The RayCaster handles:
- Dynamic mesh tracking with moving rigid bodies
- Multi-environment collision group management  
- Efficient ray casting using warp-accelerated grouped ray-mesh intersection
"""

from __future__ import annotations

import numpy as np
import re
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaacsim.core.utils.prims as prim_utils
import omni.log
import omni.physics.tensors.impl.api as physx
import warp as wp
from isaacsim.core.prims import XFormPrim
from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.sensors.ray_caster import RayCaster as IsaacRayCaster
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

# Import our custom grouped raycast function
from go2_pvcnn.utlis.warp.raycast import raycast_mesh_grouped

if TYPE_CHECKING:
    from .lidar_cfg import LidarCfg


class LidarRayCaster(IsaacRayCaster):
    """LiDAR ray caster with support for dynamic mesh updating.
    
    This extends the base Isaac Lab RayCaster to support multiple meshes with dynamic
    transforms, allowing for efficient ray casting against moving objects
    in multi-environment simulations.
    
    Key Features:
        - Dynamic mesh tracking: meshes can have moving rigid bodies
        - Collision groups: track which environment each mesh belongs to
        - Grouped ray casting: efficient warp-accelerated ray-mesh intersection
        - Multi-environment support: handles batched ray casting across environments
    """

    cfg: LidarCfg
    """The configuration parameters."""

    def __init__(self, cfg: LidarCfg):
        """Initialize the LiDAR ray caster.
        
        Args:
            cfg: The LiDAR configuration containing mesh paths and ray casting parameters.
        """
        # Note: DO NOT modify cfg.mesh_prim_paths here
        # The env regex namespace will be resolved dynamically in _initialize_warp_meshes()
        # using the pattern from OmniPerception reference code
        super().__init__(cfg)
        
        # Dynamic mesh tracking structures
        self.meshes: dict[str, list[wp.Mesh]] = dict()
        """Dictionary mapping mesh prim path patterns to lists of warp meshes."""
        
        self.mesh_transforms: torch.Tensor | None = None
        """World transforms for each mesh (N, 4, 4)."""
        
        self.mesh_inv_transforms: torch.Tensor | None = None
        """Inverse transforms for each mesh (N, 4, 4)."""
        #self.mesh_prototype_ids的长度等于地形mesh+静态物体数量+num_envs*动态物体数
        self.mesh_prototype_ids: torch.Tensor | list = []
        """Warp mesh prototype IDs (int64, shape N)."""
        
        self.mesh_collision_groups: torch.Tensor | list = []
        """Collision group ID for each mesh (int32, shape N)."""
        
        self.rigid_body_mesh_transform_segments: dict[str, slice] = dict()
        """Slice indices mapping mesh paths to positions in transform tensors."""
        
        self.rigid_body_views: dict[str, physx.RigidBodyView] = dict()
        """PhysX rigid body views for tracking mesh transforms."""

    def _resolve_env_regex_ns(self) -> str:
        """Resolve environment regex namespace from sensor prim path.
        
        Following the pattern from OmniPerception reference code:
        Extracts the environment regex namespace from the sensor's prim path.
        
        Returns:
            The substring ending with '/env_.*' if present, otherwise '/World/envs/env_.*'.
        """
        try:
            result = re.search(r"(.*/envs/env_\\\.\*)", self.cfg.prim_path)
            if result:
                return result.group(1)
        except Exception:
            pass
        return "/World/envs/env_.*"

    def _get_rigid_body_view(
        self, env_prim_path_expr: str, matched_leaf_names: list[str]
    ) -> physx.RigidBodyView | None:
        """Get the rigid body view for a given prim path.
        
        This finds the rigid bodies corresponding to the matched mesh names within
        the given environment prim path expression. Used for tracking dynamic mesh
        transforms across environments.
        
        Args:
            env_prim_path_expr: Environment prim path regex expression (e.g., "/World/envs/env_.*").
            matched_leaf_names: List of mesh leaf names to search for.
            
        Returns:
            PhysX RigidBodyView if rigid bodies found, None otherwise.
        """
        parent_prim_paths = sim_utils.find_matching_prim_paths(env_prim_path_expr)
        
        body_names = list()
        for potential_body_name in matched_leaf_names:
            prim = prim_utils.get_prim_at_path(parent_prim_paths[0] + "/" + potential_body_name)
            if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim_path = prim.GetPath().pathString
                # print(f"[RayCaster]   Found rigid body prim for mesh : {prim_path}")
                body_names.append(prim_path.rsplit("/", 1)[-1])

        if body_names:
            body_names_regex = r"(" + "|".join(body_names) + r")"
            body_names_regex = f"{env_prim_path_expr}/{body_names_regex}"
            body_names_glob = body_names_regex.replace(".*", "*")
            rigid_body_view = self._physics_sim_view.create_rigid_body_view(body_names_glob)
            return rigid_body_view
        else:
            prim = prim_utils.get_prim_at_path(parent_prim_paths[0])
            if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                body_name_glob = env_prim_path_expr.replace(".*", "*")
                return self._physics_sim_view.create_rigid_body_view(body_name_glob)
            else:
                return None

    def _get_collision_groups_by_env_str(
        self,
        rigid_body_view: physx.RigidBodyView,
        env_name_pattern: str = "/World/envs/env_\\d+",
    ) -> list[int]:
        """Get the collision groups for each rigid body in the rigid body view.
        
        Extracts environment IDs from prim paths using a regex pattern and assigns
        each rigid body to its corresponding collision group (environment ID).
        
        Args:
            rigid_body_view: PhysX rigid body view to process.
            env_name_pattern: Regex pattern to extract environment name (default: "/World/envs/env_\\d+").
            
        Returns:
            List of collision group IDs (environment IDs) for each rigid body.
        """
        collision_groups = []
        #遍历同一个prim_path下的所有刚体
        for prim_path in rigid_body_view.prim_paths:
            match = re.search(env_name_pattern, prim_path)
            if match:
                env_id = int(match.group(0).split("_")[-1])
                collision_groups.append(env_id)
            else:
                omni.log.warn(f"Could not match environment name pattern in {prim_path}.")
                collision_groups.append(-1)
        return collision_groups

    def _get_merged_mesh_from_xform_prim(self, xform_prim_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Search through the xform prim and its children to find all meshes and merge them.
        
        This method searches for mesh prims under an Xform prim (including in /visuals subdirectory)
        and merges them into a single mesh in the Xform's local frame. This is useful for
        complex robot models where each link may have multiple visual meshes.
        
        Args:
            xform_prim_path: Full path to the Xform prim.
            
        Returns:
            Tuple of (merged_vertices, merged_faces) as numpy arrays, or (None, None) if no meshes found.
        """
        points, indices = [], []
        indices_offset = 0
        xform_leaf = xform_prim_path.split("/")[-1]
        aux_mesh_name_search_list = [xform_leaf] + list(self.cfg.aux_mesh_and_link_names.keys())

        for prim_leaf_name in aux_mesh_name_search_list:
            mesh_prim_path = "/".join([xform_prim_path, "visuals", prim_leaf_name, "mesh"])
            mesh_prim = prim_utils.get_prim_at_path(mesh_prim_path)
            if not mesh_prim.IsValid():
                continue
            
            mesh = UsdGeom.Mesh(mesh_prim)
            local_points = np.asarray(mesh.GetPointsAttr().Get())
            if len(local_points) == 0:
                continue
            local_indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())

            local_transform = Gf.Matrix4d(1.0)
            if self.cfg.aux_mesh_and_link_names.get(prim_leaf_name, None) is not None:
                fixed_link_prim_path = "/".join([xform_prim_path, self.cfg.aux_mesh_and_link_names[prim_leaf_name]])
                fixed_link_prim = prim_utils.get_prim_at_path(fixed_link_prim_path)
                if fixed_link_prim.IsValid():
                    xformable = UsdGeom.Xformable(fixed_link_prim)
                    xform_ops = xformable.GetOrderedXformOps()
                    for op in xform_ops:
                        local_transform = local_transform * op.GetOpTransform(Usd.TimeCode.Default())

            transformed_points = np.array(
                [local_transform.Transform(Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])))[:3] for p in local_points]
            )

            points.append(transformed_points)
            indices.append(local_indices + indices_offset)
            indices_offset += len(transformed_points)
        
        if len(points) == 0:
            return None, None
        else:
            merged_points = np.concatenate(points, axis=0)
            merged_indices = np.concatenate(indices, axis=0)
            return merged_points, merged_indices

    def _initialize_warp_meshes(self):
        """Initialize warp meshes for ray casting.
        
        This method:
        1. Finds all mesh prims matching configured patterns
        2. Converts them to warp-accelerated mesh format
        3. Creates rigid body views to track moving meshes
        4. Sets up collision groups for environment isolation
        5. Pre-allocates tensors for mesh transforms
        """
        # _initialize_warp_meshes: initialization started (debug prints removed)
        
        # Resolve the environment regex namespace dynamically (following OmniPerception pattern)
        env_regex_ns = self._resolve_env_regex_ns()
        
        for mesh_prim_path_regex_orig in self.cfg.mesh_prim_paths:
            # mesh_prim_path_regex_ori是不同序列号的物体，未用{ENV_REGEX_NS}则是当前对应mesh_prim_paths的一个元素；
            # 如果用了{ENV_REGEX_NS}，这是所有环境的相同语义且序号一样的物体
            try:
                mesh_prim_path_regex = mesh_prim_path_regex_orig.format(ENV_REGEX_NS=env_regex_ns)
            except Exception:
                mesh_prim_path_regex = mesh_prim_path_regex_orig.replace("{ENV_REGEX_NS}", env_regex_ns)
            
            # print(f"[RayCaster] Processing mesh pattern: {mesh_prim_path_regex}")
            env_prim_path_expr, leaf_pattern = mesh_prim_path_regex.rsplit("/", 1)
            # print(f"[RayCaster]   env_prim_path_expr: {env_prim_path_expr}")
            # print(f"[RayCaster]   leaf_pattern: {leaf_pattern}")
            
            matched_mesh_prim_paths = sim_utils.find_matching_prim_paths(env_prim_path_expr)
            assert len(matched_mesh_prim_paths) >= 1, f"No matching mesh prim paths found for {env_prim_path_expr}."
            
            mesh_prims = sim_utils.get_all_matching_child_prims(
                matched_mesh_prim_paths[0],
                lambda prim: (
                    (prim.GetTypeName() == "Plane" or prim.GetTypeName() == "Mesh" or prim.GetTypeName() == "Xform")
                    and re.search("/".join([matched_mesh_prim_paths[0], leaf_pattern]), prim.GetPath().pathString)
                    is not None
                ),
            )
            # print(f"[RayCaster]   Found {len(mesh_prims)} matching mesh prims.")
            wp_meshes = []
            wp_mesh_names = []
            wp_mesh_ids = []
            
            for mesh_prim in mesh_prims:
                mesh_name = mesh_prim.GetPath().pathString.rsplit("/", 1)[-1]
                mesh_type = mesh_prim.GetTypeName()
                if mesh_type == "Xform":
                    # Get the mesh prim if it is proxy referenced from a Xform prim.
                    xform_prim_path = mesh_prim.GetPath().pathString
                    points, indices = self._get_merged_mesh_from_xform_prim(xform_prim_path)
                    if points is None:
                        continue
                    wp_mesh = convert_to_warp_mesh(points, indices, device=self.device)
                    
                elif mesh_type == "Mesh":
                    mesh_prim_usd = UsdGeom.Mesh(mesh_prim)
                    # read the vertices and faces
                    points = np.asarray(mesh_prim_usd.GetPointsAttr().Get())
                    indices = np.asarray(mesh_prim_usd.GetFaceVertexIndicesAttr().Get())
                    wp_mesh = convert_to_warp_mesh(points, indices, device=self.device)
                    
                elif mesh_type == "Plane":
                    mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
                    wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
                    
                else:
                    omni.log.warn(
                        f"Unsupported mesh type: {mesh_type} in {mesh_prim.GetPath().pathString}."
                        " Skipping."
                    )
                    continue
                
                wp_mesh_names.append(mesh_name)#存储对应mesh_prim_path_regex_orig每个mesh的名字
                wp_meshes.append(wp_mesh)#存储对应mesh_prim_path_regex_orig每个mesh的warp mesh对象
                wp_mesh_ids.append(wp_mesh.id)#存储对应mesh_prim_path_regex_orig每个mesh的warp mesh ID
            
            # mesh_prim_path_regex是对应mesh_prim_path_regex_orig的把{ENV_REGEX_NS}替换后的结果
            self.meshes[mesh_prim_path_regex] = wp_meshes
            
            
            # collect match rigid body names to mesh ids for transform updates
            rigid_body_view = self._get_rigid_body_view(env_prim_path_expr, wp_mesh_names)
            if rigid_body_view is None:
                # no rigid body view found, the mesh transform will not be updated at rollouts.
                # the meshes transforms will be set to identity.

                # self.mesh_prototype_ids的长度等于地形mesh+静态物体数量+num_envs*动态物体数
                self.mesh_prototype_ids.extend(wp_mesh_ids)
                # self.mesh_collision_groups的长度等于1+静态物体数量+num_envs
                self.mesh_collision_groups.extend([-1] * len(wp_mesh_ids))
            else:
                #rigid_body_view.count等于num_envs
                # NOTE: This mesh_prototype_ids.extend is assuming the rigid_body_view has the order of
                # iterating over the mesh names and again and again across envs.
                self.mesh_prototype_ids.extend(wp_mesh_ids * (rigid_body_view.count // len(wp_mesh_ids)))
                #self.rigid_body_mesh_transform_segments记录的是每个mesh_prim_path_regex对应的动态物体在
                # self.mesh_prototype_ids中的slice位置，包前不包后，self.rigid_body_mesh_transform_segments不包括地形和静态物体
                self.rigid_body_mesh_transform_segments[mesh_prim_path_regex] = slice(
                    len(self.mesh_prototype_ids) - rigid_body_view.count, len(self.mesh_prototype_ids)
                )
                self.rigid_body_views[mesh_prim_path_regex] = rigid_body_view
                collision_groups = self._get_collision_groups_by_env_str(rigid_body_view)#返回每个刚体对应的环境编号
                self.mesh_collision_groups.extend(collision_groups)#self.mesh_collision_groups内容为（1+静态物体数量）个-1+num_envs的环境编号*动态物体数量
        
        # build buffers for all the meshes
        # print(f"\n[RayCaster] Building mesh transform buffers:")
        # print(f"[RayCaster]   total_mesh_count: {len(self.mesh_prototype_ids)}")
        # print(f"[RayCaster]   mesh_collision_groups count: {len(self.mesh_collision_groups)}")
        
        self.mesh_transforms_pyt = torch.zeros(
            len(self.mesh_prototype_ids), 7, dtype=torch.float32, device=self.device
        )
        # print(f"[RayCaster]   mesh_transforms_pyt shape: {self.mesh_transforms_pyt.shape}")
        #用于把mesh从局部坐标转世界坐标
        self.mesh_transforms_pyt[:, -1] = 1.0#shape (N, 7)的四元数最后一维初始化为1.0,N为地形+静态物体数量+num_envs*动态物体数
        #用于把点从世界坐标转局部坐标
        self.mesh_inv_transforms_pyt = torch.zeros_like(self.mesh_transforms_pyt)
        self.mesh_inv_transforms_pyt[:, -1] = 1.0#shape (N, 7)的四元数最后一维初始化为1.0
        self.mesh_collision_groups_pyt = torch.tensor(self.mesh_collision_groups, dtype=torch.int32, device=self.device)
        # print(f"[RayCaster]   mesh_collision_groups_pyt: {self.mesh_collision_groups_pyt}")
        #warp的id
        self.mesh_prototype_ids_pyt = torch.tensor(self.mesh_prototype_ids, dtype=torch.int64, device=self.device)
        #对warp的id的索引
        self.all_mesh_indices = torch.arange(len(self.mesh_prototype_ids), dtype=torch.int32, device=self.device)
        
        # print(f"[RayCaster] _initialize_warp_meshes DONE\n")

    def _initialize_rays_impl(self):
        """Initialize ray patterns and collision group buffers.
        """
        # Call parent class initialization for rays, drift, etc.
        super()._initialize_rays_impl()
        
        # Create buffer to store ray collision groups for grouped ray casting
        self._create_ray_collision_groups()

    def _create_ray_collision_groups(self):
        """Create buffer to store ray collision groups and mesh ids for group ids.
        
        Maps each ray to its corresponding environment (collision group) and creates
        lookup tables for efficient grouped ray casting against environment-specific meshes.
        """
        # print(f"\n[RayCaster] _create_ray_collision_groups START")
        # print(f"[RayCaster] _view.count: {self._view.count}, num_rays: {self.num_rays}")
        
        self._ray_collision_groups = (
            torch.arange(self._view.count, dtype=torch.int32, device=self.device).unsqueeze(1).repeat(1, self.num_rays)
        )
        # print(f"[RayCaster] _ray_collision_groups shape: {self._ray_collision_groups.shape}")
        # print(f"[RayCaster] _ray_collision_groups unique values: {torch.unique(self._ray_collision_groups).tolist()}")
        
        unique_groups = torch.unique(self._ray_collision_groups)
        # print(f"[RayCaster] unique collision groups: {unique_groups.tolist()}")
        
        self._mesh_ids_for_group = []
        self._mesh_ids_slice_for_group = []
        for group_id in unique_groups:
            negative_one_indices = torch.where(self.mesh_collision_groups_pyt == -1)[0]
            group_indices = torch.where(self.mesh_collision_groups_pyt == group_id)[0]
            ray_group = torch.cat([negative_one_indices, group_indices]).tolist()
            # print(f"[RayCaster]   group_id={group_id.item()}: static_meshes={len(negative_one_indices)}, dynamic_meshes={len(group_indices)}, total={len(ray_group)}")
            self._mesh_ids_for_group.append(ray_group)
            self._mesh_ids_slice_for_group.append(len(ray_group))
        
        self._mesh_ids_for_group = torch.tensor(self._mesh_ids_for_group, dtype=torch.int32, device=self._device)
        self._mesh_ids_for_group = self._mesh_ids_for_group.view(-1)
        self._mesh_ids_slice_for_group = [0] + [
            sum(self._mesh_ids_slice_for_group[: i + 1]) for i in range(len(self._mesh_ids_slice_for_group))
        ]
        self._mesh_ids_slice_for_group = torch.tensor(
            self._mesh_ids_slice_for_group, dtype=torch.int32, device=self._device
        )
        
        # print(f"[RayCaster] _mesh_ids_for_group shape: {self._mesh_ids_for_group.shape}")
        # print(f"[RayCaster] _mesh_ids_slice_for_group: {self._mesh_ids_slice_for_group.tolist()}")
        # print(f"[RayCaster] _create_ray_collision_groups DONE")

    def _update_mesh_transforms(self, env_ids: torch.Tensor | None = None):
        """Update the mesh transforms for the given environment IDs.
        
        Queries the rigid body views to get current mesh positions and orientations,
        then updates the mesh transform buffers used in ray casting.
        
        Args:
            env_ids: Environment IDs to update. If None, updates all meshes.
        """
        if not self.meshes:
            omni.log.warn("No meshes found for ray casting.")
            return

        for mesh_prim_path_regex, rigid_body_view in self.rigid_body_views.items():
            segment: slice = self.rigid_body_mesh_transform_segments[mesh_prim_path_regex]
            rigid_body_transforms = rigid_body_view.get_transforms().view(-1, 7)  # shape (N, 7)

            rigid_body_env_ids = self.mesh_collision_groups_pyt[segment]
            rigid_body_view_mask = None if env_ids is None else torch.isin(rigid_body_env_ids, env_ids)
             # get the selected transforms in px, py, pz, qx, qy, qz, qw format
            selected_transforms = rigid_body_transforms[rigid_body_view_mask]
            mesh_tf_indices = self.all_mesh_indices[segment][rigid_body_view_mask]

            self.mesh_transforms_pyt[mesh_tf_indices] = selected_transforms
            pos_inv, quat_wxyz_inv = math_utils.subtract_frame_transforms(
                selected_transforms[:, :3],
                math_utils.convert_quat(selected_transforms[:, 3:], to="wxyz"),
            )
            quat_xyzw_inv = math_utils.convert_quat(quat_wxyz_inv, to="xyzw")
            self.mesh_inv_transforms_pyt[mesh_tf_indices, 3:] = quat_xyzw_inv
            self.mesh_inv_transforms_pyt[mesh_tf_indices, :3] = pos_inv

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Update the ray caster buffers with current mesh positions and orientations.
        
        This method:
        1. Updates mesh transforms for moving rigid bodies
        2. Gets sensor poses and applies orientation-based ray transformation
        3. Performs grouped ray casting against all meshes
        4. Stores results in sensor data
        
        Args:
            env_ids: Environment IDs to update.
        """
        # _update_buffers_impl started (debug prints removed)
        
        # Update mesh transforms
        self._update_mesh_transforms(env_ids)
        
        # Debug prints for scene positions removed
        # print(f"[RayCaster] Mesh transforms updated")
        # print(f"[RayCaster] mesh_transforms_pyt shape: {self.mesh_transforms_pyt.shape}, dtype: {self.mesh_transforms_pyt.dtype}")
        # print(f"[RayCaster] mesh_prototype_ids_pyt shape: {self.mesh_prototype_ids_pyt.shape}")
        # print(f"[RayCaster] mesh_collision_groups_pyt shape: {self.mesh_collision_groups_pyt.shape}")

        # Obtain the poses of the sensors
        if isinstance(self._view, XFormPrim):
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = math_utils.convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = math_utils.convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")
        
        # print(f"[RayCaster] pos_w (sensor position): shape={pos_w.shape}, dtype={pos_w.dtype}")
        # print(f"[RayCaster] pos_w range: [{pos_w.min().item():.4f}, {pos_w.max().item():.4f}]")
        # print(f"[RayCaster] quat_w (sensor quaternion): shape={quat_w.shape}, dtype={quat_w.dtype}")
        
        # Clone for read-only operations
        pos_w = pos_w.clone()
        quat_w = quat_w.clone()
        
        # Scene layout debug output removed
        
        # Apply drift
        # print(f"[RayCaster] drift shape: {self.drift.shape}, applying to env_ids subset")
        # print(f"[RayCaster] drift[env_ids] shape: {self.drift[env_ids].shape}")
        # print(f"[RayCaster] drift[env_ids] range: [{self.drift[env_ids].min().item():.6f}, {self.drift[env_ids].max().item():.6f}]")
        pos_w += self.drift[env_ids]
        # print(f"[RayCaster] pos_w after drift: [{pos_w.min().item():.4f}, {pos_w.max().item():.4f}]")
        
        # Store the poses
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w
        # print(f"[RayCaster] Stored poses in _data")

        # Ray cast based on the sensor poses and alignment mode
        # print(f"\n[RayCaster] Ray Transformation: alignment_mode={self.cfg.ray_alignment}")
        # print(f"[RayCaster] num_rays: {self.num_rays}")
        # print(f"[RayCaster] ray_starts[env_ids] shape: {self.ray_starts[env_ids].shape}")
        # print(f"[RayCaster] ray_directions[env_ids] shape: {self.ray_directions[env_ids].shape}")
        # print(f"[RayCaster] quat_w.repeat(1, num_rays) shape: {quat_w.repeat(1, self.num_rays).shape}")
        # print(f"[RayCaster] pos_w.unsqueeze(1) shape: {pos_w.unsqueeze(1).shape}")
        
        if self.cfg.ray_alignment == "yaw":
            # Only yaw orientation is considered - both start positions AND directions need yaw rotation
            quat_repeated = quat_w.repeat(1, self.num_rays)
            ray_starts_w = math_utils.quat_apply_yaw(quat_repeated, self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
            
        elif self.cfg.ray_alignment == "base":
            # Full orientation is considered
            quat_repeated = quat_w.repeat(1, self.num_rays)
            ray_starts_w = math_utils.quat_apply(quat_repeated, self.ray_starts[env_ids])
            #print(f"[RayCaster] ray_starts_w after full quat_apply: shape={ray_starts_w.shape}")
            ray_starts_w += pos_w.unsqueeze(1)
            #print(f"[RayCaster] ray_starts_w after adding pos_w offset: [{ray_starts_w.min().item():.4f}, {ray_starts_w.max().item():.4f}]")
            ray_directions_w = math_utils.quat_apply(quat_repeated, self.ray_directions[env_ids])
            # print(f"[RayCaster] ray_directions_w (base mode, full rotation): shape={ray_directions_w.shape}")
        else:
            # World alignment - no rotation applied
            ray_starts_w = self.ray_starts[env_ids].clone()
            ray_starts_w += pos_w.unsqueeze(1)
            #print(f"[RayCaster] ray_starts_w (world mode, no rotation): shape={ray_starts_w.shape}")
            ray_directions_w = self.ray_directions[env_ids]
            #print(f"[RayCaster] ray_directions_w (world mode): shape={ray_directions_w.shape}")
        
        ray_group_ids = self._ray_collision_groups[env_ids]
        # print(f"\n[RayCaster] ray_group_ids shape: {ray_group_ids.shape}, unique groups: {torch.unique(ray_group_ids).tolist()}")
        # print(f"[RayCaster] _mesh_ids_for_group shape: {self._mesh_ids_for_group.shape}")
        # print(f"[RayCaster] _mesh_ids_slice_for_group: {self._mesh_ids_slice_for_group}")

        # Perform grouped ray casting against all meshes with collision groups
        # print(f"\n[RayCaster] Starting raycast_mesh_grouped:")
        # print(f"[RayCaster]   - num_meshes: {len(self.mesh_prototype_ids)}")
        # print(f"[RayCaster]   - ray_starts_w: shape={ray_starts_w.shape}, dtype={ray_starts_w.dtype}, device={ray_starts_w.device}")
        # print(f"[RayCaster]   - ray_directions_w: shape={ray_directions_w.shape}, dtype={ray_directions_w.dtype}")
        # Calling raycast_mesh_grouped
        
        ray_hits, _, _, ray_mesh_ids = raycast_mesh_grouped(
            mesh_prototypes=self.meshes,
            mesh_prototype_ids=self.mesh_prototype_ids_pyt,
            mesh_transforms=self.mesh_transforms_pyt,
            mesh_inv_transforms=self.mesh_inv_transforms_pyt,
            ray_group_ids=ray_group_ids,
            mesh_ids_for_group=self._mesh_ids_for_group,
            mesh_ids_slice_for_group=self._mesh_ids_slice_for_group,
            ray_starts=ray_starts_w,
            ray_directions=ray_directions_w,
            max_dist=self.cfg.max_distance,
            min_dist=self.cfg.min_distance,
            return_face_id=True,
        )
        
        self._data.ray_hits_w[env_ids] = ray_hits
        
        # Store mesh IDs for semantic classification (if subclass needs it)
        if hasattr(self._data, 'ray_mesh_ids'):
            self._data.ray_mesh_ids[env_ids] = ray_mesh_ids
        
        # Removed detailed debug outputs about ray hits and mesh ids
        
        # _update_buffers_impl done
        
