"""Ray casting utilities using Warp for GPU-accelerated mesh intersection.

This module provides functions for ray-mesh intersection testing with support
for collision groups, enabling efficient multi-environment simulations.
"""

import numpy as np
import torch

import warp as wp

# Disable warp module initialization messages
wp.config.quiet = True
# Initialize the warp module
wp.init()

from . import kernels


def raycast_mesh_grouped(
    mesh_prototypes: dict[str, wp.Mesh],
    mesh_prototype_ids: torch.Tensor,
    mesh_transforms: torch.Tensor,
    mesh_inv_transforms: torch.Tensor,
    ray_group_ids: torch.Tensor,
    mesh_ids_for_group: torch.Tensor,
    mesh_ids_slice_for_group: torch.Tensor,
    ray_starts: torch.Tensor,
    ray_directions: torch.Tensor,
    max_dist: float = 1e6,
    min_dist: float = 0.0,
    return_distance: bool = False,
    return_normal: bool = False,
    return_face_id: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Performs ray-casting against meshes with different collision groups.

    Each ray and mesh has its own collision group ID. Rays only test against meshes
    in the same collision group (group ID -1 is special and collides with all rays).
    This enables efficient multi-environment simulations where objects in different
    environments don't interfere with each other.

    This is an extended implementation of the `raycast_mesh` in `isaaclab.utils.warp.ops`.

    Args:
        mesh_prototypes: Dictionary mapping mesh names to Warp mesh objects
        mesh_prototype_ids: Tensor of mesh prototype IDs for each mesh instance (int64)
        mesh_transforms: Tensor of mesh transforms in world space, shape (N, 7) as
            (pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z)
        mesh_inv_transforms: Tensor of inverse mesh transforms, shape (N, 7)
        ray_group_ids: Collision group ID for each ray, shape (num_rays,)
        mesh_ids_for_group: Flattened array of mesh IDs for each collision group
        mesh_ids_slice_for_group: Slice indices into mesh_ids_for_group, shape (num_groups + 1,)
        ray_starts: Starting points of rays, shape (..., 3)
        ray_directions: Direction vectors of rays (should be normalized), shape (..., 3)
        max_dist: Maximum ray casting distance. Default is 1e6.
        min_dist: Minimum ray casting distance (hits closer are ignored). Default is 0.0.
        return_distance: Whether to return hit distances. Default is False.
        return_normal: Whether to return surface normals at hit points. Default is False.
        return_face_id: Whether to return face IDs of hits. Default is False.

    Returns:
        A tuple containing:
        - ray_hits: Hit positions in world space, shape (..., 3). Points at max_dist if no hit.
        - ray_distance: Hit distances (if return_distance=True), otherwise None
        - ray_normal: Surface normals at hit points (if return_normal=True), otherwise None
        - ray_face_id: Face IDs of hits (if return_face_id=True), otherwise None

    Note:
        The return_face_id feature might not work correctly with the current Warp implementation.
    """
    # Extract device and shape information
    shape = ray_starts.shape
    device = ray_starts.device
    
    # Get device of the mesh (all meshes should be on the same device)
    # mesh_prototypes is dict[str, list[wp.Mesh]], so extract the first mesh from the first list
    meshes_list = list(mesh_prototypes.values())
    if len(meshes_list) == 0:
        raise RuntimeError(f"mesh_prototypes is empty! Cannot extract device information")
    
    first_mesh_list = meshes_list[0]
    if isinstance(first_mesh_list, list) and len(first_mesh_list) > 0:
        torch_device = wp.device_to_torch(first_mesh_list[0].device)
    elif isinstance(first_mesh_list, list):
        raise RuntimeError("First mesh list is empty!")
    else:
        # Single mesh object, not list
        torch_device = wp.device_to_torch(first_mesh_list.device)
    
    # Reshape the tensors for processing
    ray_starts = ray_starts.to(torch_device).view(-1, 3).contiguous()
    ray_directions = ray_directions.to(torch_device).view(-1, 3).contiguous()
    ray_group_ids = ray_group_ids.to(torch_device).view(-1).contiguous()
    mesh_ids_for_group = mesh_ids_for_group.to(torch_device).contiguous()
    mesh_ids_slice_for_group = mesh_ids_slice_for_group.to(torch_device).view(-1).contiguous()
    num_rays = ray_starts.shape[0]
    
    # Create output tensor for the ray hits
    # Initialize to inf for no-hit rays (changed from max_dist to enable torch.isinf() detection)
    ray_hits = torch.full((num_rays, 3), float('inf'), dtype=ray_starts.dtype, device=torch_device).contiguous()
    
    # Map the memory to warp arrays
    mesh_prototype_ids_wp = wp.from_torch(mesh_prototype_ids, dtype=wp.uint64)
    mesh_transforms_wp = wp.from_torch(mesh_transforms, dtype=wp.transform)
    mesh_inv_transforms_wp = wp.from_torch(mesh_inv_transforms, dtype=wp.transform)
    ray_group_ids_wp = wp.from_torch(ray_group_ids, dtype=wp.int32)
    mesh_ids_for_group_wp = wp.from_torch(mesh_ids_for_group, dtype=wp.int32)
    mesh_ids_slice_for_group_wp = wp.from_torch(mesh_ids_slice_for_group, dtype=wp.int32)
    ray_starts_wp = wp.from_torch(ray_starts, dtype=wp.vec3)
    ray_directions_wp = wp.from_torch(ray_directions, dtype=wp.vec3)
    ray_hits_wp = wp.from_torch(ray_hits, dtype=wp.vec3)

    # Prepare optional output buffers
    if return_distance:
        ray_distance = torch.full((num_rays,), float("inf"), device=torch_device).contiguous()
        ray_distance_wp = wp.from_torch(ray_distance, dtype=wp.float32)
    else:
        ray_distance = None
        ray_distance_wp = wp.empty((1,), dtype=wp.float32, device=torch_device)

    if return_normal:
        ray_normal = torch.full((num_rays, 3), float("inf"), device=torch_device).contiguous()
        ray_normal_wp = wp.from_torch(ray_normal, dtype=wp.vec3)
    else:
        ray_normal = None
        ray_normal_wp = wp.empty((1,), dtype=wp.vec3, device=torch_device)

    # Always return face_id (will be used to store mesh_idx for semantic classification)
    ray_face_id = torch.ones((num_rays,), dtype=torch.int32, device=torch_device).contiguous() * (-1)
    ray_face_id_wp = wp.from_torch(ray_face_id, dtype=wp.int32)

    # Launch the warp kernel
    # Get the warp device from the first mesh
    warp_device_for_kernel = first_mesh_list[0].device if isinstance(first_mesh_list, list) else first_mesh_list.device
    
    # Launching kernel (debug prints removed)
    
    wp.launch(
        kernel=kernels.raycast_mesh_kernel_grouped_transformed_v2,  # Use new kernel name
        dim=num_rays,
        inputs=[
            mesh_prototype_ids_wp,
            mesh_transforms_wp,
            mesh_inv_transforms_wp,
            ray_group_ids_wp,
            mesh_ids_for_group_wp,
            mesh_ids_slice_for_group_wp,
            ray_starts_wp,
            ray_directions_wp,
            ray_hits_wp,
            ray_distance_wp,
            ray_normal_wp,
            ray_face_id_wp,
            max_dist,
            min_dist,
            int(return_distance),
            int(return_normal),
            int(return_face_id),
        ],
        device=warp_device_for_kernel,
    )
    
    # Synchronize to ensure kernel completion
    # NOTE: Synchronization may not be needed in newer versions, but kept for compatibility
    wp.synchronize()

    # Reshape output tensors back to original shape
    if return_distance:
        ray_distance = ray_distance.to(device).view(shape[0], shape[1])
    if return_normal:
        ray_normal = ray_normal.to(device).view(shape)
    if return_face_id:
        ray_face_id = ray_face_id.to(device).view(shape[0], shape[1])

    ray_hits_output = ray_hits.to(device).view(shape)
    return ray_hits_output, ray_distance, ray_normal, ray_face_id
