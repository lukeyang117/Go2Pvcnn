"""Warp kernels for grouped ray casting against meshes.

This module provides CUDA kernels implemented in Warp for efficient ray-mesh
intersection testing with collision groups.
"""

from typing import Any

import warp as wp


@wp.kernel(enable_backward=False)
def raycast_mesh_kernel_grouped_transformed_v2(  # Changed name to force recompilation
    mesh_prototype_ids: wp.array(dtype=wp.uint64),  # all meshes in the scene
    mesh_transforms: wp.array(dtype=wp.transform),  # transforms of the meshes
    mesh_inv_transforms: wp.array(dtype=wp.transform),  # inverse transforms of the meshes
    ray_collision_groups: wp.array(dtype=wp.int32),
    mesh_ids_for_group: wp.array(dtype=wp.int32),
    mesh_ids_slice_for_group: wp.array(
        dtype=wp.int32
    ),  # Given the ray collision group (i), mesh_ids_for_group[mesh_ids_slice_for_group[i]:mesh_ids_slice_for_group[i+1]] are the mesh ids within this group.
    ray_starts: wp.array(dtype=wp.vec3),
    ray_directions: wp.array(dtype=wp.vec3),
    ray_hits: wp.array(dtype=wp.vec3),
    ray_distance: wp.array(dtype=wp.float32),
    ray_normal: wp.array(dtype=wp.vec3),
    ray_face_id: wp.array(dtype=wp.int32),
    max_dist: float = 1e6,
    min_dist: float = 0.0,
    return_distance: int = False,
    return_normal: int = False,
    return_face_id: int = False,
):
    """Warp kernel for ray casting against grouped meshes with transforms.
    
    This kernel performs ray-mesh intersection tests where each ray belongs to a
    collision group and only tests against meshes in the same group. This enables
    efficient multi-environment simulations where objects in different environments
    don't interfere with each other.
    
    Args:
        mesh_prototype_ids: Array of mesh prototype IDs (one per mesh instance)
        mesh_transforms: Array of mesh transforms in world space
        mesh_inv_transforms: Array of inverse mesh transforms
        ray_collision_groups: Collision group ID for each ray
        mesh_ids_for_group: Flattened array of mesh IDs for each collision group
        mesh_ids_slice_for_group: Slice indices into mesh_ids_for_group for each group
        ray_starts: Starting points of rays in world space
        ray_directions: Direction vectors of rays in world space
        ray_hits: Output array for hit positions
        ray_distance: Output array for hit distances
        ray_normal: Output array for surface normals at hit points
        ray_face_id: Output array for face IDs of hits
        max_dist: Maximum ray casting distance
        min_dist: Minimum ray casting distance (hits closer are ignored)
        return_distance: Whether to compute and return distances
        return_normal: Whether to compute and return normals
        return_face_id: Whether to return face IDs
    """
    tid = wp.tid()
    t = float(0.0)  # hit distance along ray
    u = float(0.0)  # hit face barycentric u
    v = float(0.0)  # hit face barycentric v
    sign = float(0.0)  # hit face sign
    n = wp.vec3()  # hit face normal
    f = int(0)  # hit face index

    ray_distance_buf = float(max_dist)
    ray_collision_group = int(ray_collision_groups[tid])
    start = ray_starts[tid]
    direction = ray_directions[tid]
    
    # Track the closest valid hit (within max_dist)
    closest_t = float(max_dist)
    closest_mesh_idx = int(-1)
    closest_n = wp.vec3()

    # Iterate through all meshes in this collision group
    for idx in range(mesh_ids_slice_for_group[ray_collision_group], mesh_ids_slice_for_group[ray_collision_group + 1]):
        mesh_idx = int(mesh_ids_for_group[idx])
        mesh_prototype = mesh_prototype_ids[mesh_idx]

        # Transform the ray start and direction to the mesh's local space
        mesh_transform = mesh_transforms[mesh_idx]
        mesh_inv_transform = mesh_inv_transforms[mesh_idx]
        start_local = wp.transform_point(mesh_inv_transform, start)
        direction_local = wp.transform_vector(mesh_inv_transform, direction)

        # Ray cast against the mesh with a large max distance
        # We'll filter by max_dist ourselves
        hit_success = wp.mesh_query_ray(
            mesh_prototype, start_local, direction_local, 1e10, t, u, v, sign, n, f
        )

        # Accept hit only if within valid distance range [min_dist, max_dist]
        if hit_success and t > min_dist and t <= max_dist and t < closest_t:
            closest_t = t
            closest_mesh_idx = mesh_idx
            closest_n = n
    
    # Write the closest valid hit (only if we found something within max_dist)
    if closest_mesh_idx >= 0:  # Check if we found a valid hit
        ray_hits[tid] = start + direction * closest_t
        if return_distance == 1:
            ray_distance[tid] = closest_t
        if return_normal == 1:
            # Transform the normal back to world space
            mesh_transform = mesh_transforms[closest_mesh_idx]
            closest_n = wp.transform_vector(mesh_transform, closest_n)
            ray_normal[tid] = closest_n
        # Always write mesh_idx to ray_face_id for semantic classification
        ray_face_id[tid] = closest_mesh_idx
