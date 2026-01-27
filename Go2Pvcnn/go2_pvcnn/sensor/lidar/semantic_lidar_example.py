"""Example usage of SemanticLidarSensor.

This script demonstrates how to use the Semantic LiDAR sensor to classify
objects into terrain, dynamic obstacles, and valuable items.
"""

from go2_pvcnn.sensor.lidar import SemanticLidarCfg, LivoxPatternCfg

# Example configuration with semantic classification
semantic_lidar_cfg = SemanticLidarCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=SemanticLidarCfg.OffsetCfg(
        pos=[0.3, 0.0, 0.2],
        rot=[1.0, 0.0, 0.0, 0.0],
    ),
    
    # Ray pattern configuration
    pattern_cfg=LivoxPatternCfg(
        sensor_type="mid360",
        samples=10000,
        use_simple_grid=True,
        horizontal_line_num=100,
        vertical_line_num=50,
    ),
    
    # Mesh paths to ray cast against
    mesh_prim_paths=[
        "/World/ground",  # Terrain
        "/world/SM_sofa_1",  # Valuable
        "/world/SM_armchair_1",  # Valuable
        "/world/SM_table_1",  # Valuable
        "/world/SM_sofa_2",
        "/world/SM_armchair_2",
        "/world/SM_table_2",
        "/world/SM_sofa_3",
        "/world/SM_armchair_3",
        "/world/SM_table_3",
        # Dynamic obstacles
        "{ENV_REGEX_NS}/cracker_box_0/_03_cracker_box",
        "{ENV_REGEX_NS}/cracker_box_1/_03_cracker_box",
        "{ENV_REGEX_NS}/cracker_box_2/_03_cracker_box",
        "{ENV_REGEX_NS}/sugar_box_0/_04_sugar_box",
        "{ENV_REGEX_NS}/sugar_box_1/_04_sugar_box",
        "{ENV_REGEX_NS}/sugar_box_2/_04_sugar_box",
        "{ENV_REGEX_NS}/tomato_soup_can_0/_05_tomato_soup_can",
        "{ENV_REGEX_NS}/tomato_soup_can_1/_05_tomato_soup_can",
        "{ENV_REGEX_NS}/tomato_soup_can_2/_05_tomato_soup_can",
    ],
    
    # Semantic classification mapping
    semantic_class_mapping={
        "terrain": ["ground", "wall", "floor", "plane"],
        "dynamic_obstacle": [
            "cracker_box", "sugar_box", "tomato_soup_can",
            "_03_cracker_box", "_04_sugar_box", "_05_tomato_soup_can"
        ],
        "valuable": [
            "sofa", "armchair", "table", "chair",
            "SM_Sofa", "SM_Armchair", "SM_Table"
        ],
    },
    
    # Standard LiDAR settings
    update_frequency=10.0,
    max_distance=40.0,
    min_range=0.1,
    ray_alignment="yaw",
    
    # Output settings
    return_pointcloud=True,
    pointcloud_in_world_frame=False,
    return_semantic_labels=True,  # Enable semantic classification
    return_mesh_ids=False,  # Optional: track specific mesh IDs
    
    # Noise settings
    enable_sensor_noise=False,
    debug_vis=True,
)

# Usage in environment:
# lidar = SemanticLidarSensor(semantic_lidar_cfg)
# 
# # Get standard LiDAR data
# distances = lidar.get_distances()  # Shape: (num_envs, num_rays)
# pointcloud = lidar.get_pointcloud()  # Shape: (num_envs, num_rays, 3)
# 
# # Get semantic classification
# semantic_labels = lidar.get_semantic_labels()  # Shape: (num_envs, num_rays)
# # semantic_labels values:
# #   0 = No hit / Unknown
# #   1 = Terrain (ground, walls)
# #   2 = Dynamic obstacle (boxes, cans)
# #   3 = Valuable item (furniture)
# 
# # Filter point cloud by semantic class
# terrain_mask = (semantic_labels == 1)
# obstacle_mask = (semantic_labels == 2)
# valuable_mask = (semantic_labels == 3)
# 
# terrain_points = pointcloud[terrain_mask]
# obstacle_points = pointcloud[obstacle_mask]
# valuable_points = pointcloud[valuable_mask]
