"""Cost map generation from ground truth semantic labels (Teacher Mode).

This module generates cost maps directly from LiDAR semantic segmentation without PVCNN inference.
It's designed for teacher training where we have access to perfect semantic labels.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


class TeacherCostMapGenerator:
    """
    生成2D代价地图（使用真实语义标签，无PVCNN推理）
    
    Teacher模式特点：
    1. 直接使用LiDAR语义标签（semantic_labels），无需PVCNN
    2. 结合高程图(height_map)计算代价
    3. 支持command方向减少代价（鼓励沿目标方向移动）
    
    代价计算：
    1. Obstacle cost: 障碍物位置代价（基于语义标签）
    2. Distance cost: 离机器人距离代价
    3. Gradient cost: 地形陡峭度代价
    4. Height cost: 绝对高度代价
    5. Command alignment bonus: 沿command方向的奖励（减少代价）
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (32, 32),
        grid_resolution: float = 0.1,
        x_range: Tuple[float, float] = (-3.2, 3.2),
        y_range: Tuple[float, float] = (-3.2, 3.2),
        obstacle_class_ids: list = [2, 3],  # Semantic classes: 2=dynamic_obstacle, 3=valuable
        device: str = "cuda",
        # 代价权重参数
        weight_obstacle: float = 1.0,
        weight_distance: float = 0.5,
        weight_gradient: float = 1.0,
        weight_abs_height: float = 0.3,
        weight_command_bonus: float = 0.5,  # Command方向奖励权重
        command_bonus_radius: float = 1.0,  # Command方向奖励作用半径(米)
    ):
        """
        初始化Teacher代价地图生成器
        
        Args:
            grid_size: 网格尺寸 (height, width)
            grid_resolution: 每个格子的实际尺寸(米)
            x_range: X轴范围(米)
            y_range: Y轴范围(米)
            obstacle_class_ids: 障碍物语义类别ID列表
                - Class 0: No hit / Unknown
                - Class 1: Terrain (地形，可通行)
                - Class 2: Dynamic obstacle (动态障碍物)
                - Class 3: Valuable (重要物体如家具)
            device: 计算设备
            weight_command_bonus: Command方向奖励权重（减少代价）
            command_bonus_radius: Command方向奖励的作用半径(米)
        """
        self.grid_size = grid_size
        self.grid_resolution = grid_resolution
        self.x_range = x_range
        self.y_range = y_range
        self.obstacle_class_ids = obstacle_class_ids
        self.device = device
        
        # 代价权重
        self.weight_obstacle = weight_obstacle
        self.weight_distance = weight_distance
        self.weight_gradient = weight_gradient
        self.weight_abs_height = weight_abs_height
        self.weight_command_bonus = weight_command_bonus
        self.command_bonus_radius = command_bonus_radius
        
        # 预计算网格坐标
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        
        # Debug counters
        self._call_count = 0
    
    def generate_cost_map(
        self,
        point_xyz: torch.Tensor,
        semantic_labels: torch.Tensor,
        height_map: torch.Tensor,
        command_velocity: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        从真实语义标签和高程图生成代价地图（Teacher模式）
        
        Args:
            point_xyz: (batch, num_rays, 3) - 机器人坐标系下的点云坐标
            semantic_labels: (batch, num_rays) - 真实语义标签 (0=unknown, 1=terrain, 2=obstacle, 3=valuable)
            height_map: (batch, H, W) - 高程图
            command_velocity: (batch, 3) - 目标速度指令 [vx, vy, wz]，可选
        
        Returns:
            cost_map: (batch, H, W) - 单通道代价地图
        """
        self._call_count += 1
        batch_size = point_xyz.shape[0]
        num_rays = point_xyz.shape[1]
        H, W = self.grid_size
        
        # 验证输入形状
        assert semantic_labels.shape[0] == batch_size
        assert semantic_labels.shape[1] == num_rays
        assert height_map.shape == (batch_size, H, W)
        
        # 1. 将XY坐标映射到网格索引
        x_coords = point_xyz[:, :, 0]  # (batch, num_rays)
        y_coords = point_xyz[:, :, 1]
        
        x_indices = ((x_coords - self.x_min) / self.grid_resolution).long()
        y_indices = ((y_coords - self.y_min) / self.grid_resolution).long()
        
        # 创建有效点mask
        valid_mask = (
            (x_indices >= 0) & (x_indices < W) &
            (y_indices >= 0) & (y_indices < H) &
            torch.isfinite(x_coords) & torch.isfinite(y_coords)
        )
        
        # 2. 初始化障碍物地图
        obstacle_map = torch.zeros((batch_size, H, W), device=self.device)
        
        # 3. 投影语义标签到2D网格
        # 创建obstacle mask: 是否为障碍物类别且在范围内
        is_obstacle = torch.zeros((batch_size, num_rays), dtype=torch.bool, device=self.device)
        for obs_id in self.obstacle_class_ids:
            is_obstacle |= (semantic_labels == obs_id)
        
        is_obstacle = is_obstacle & valid_mask
        
        # 使用scatter操作进行向量化更新
        for b in range(batch_size):
            batch_valid = valid_mask[b]
            
            if batch_valid.sum() == 0:
                continue
            
            valid_x_indices = x_indices[b][batch_valid]
            valid_y_indices = y_indices[b][batch_valid]
            valid_obstacle = is_obstacle[b][batch_valid]
            
            # 创建线性索引: y*W + x
            linear_indices = valid_y_indices * W + valid_x_indices
            
            # 展平grid用于scatter操作
            obstacle_flat = obstacle_map[b].view(-1)
            
            # 取最大值（只要有一个障碍物就标记）
            obstacle_flat.scatter_reduce_(
                0, linear_indices, valid_obstacle.float(),
                reduce='amax', include_self=False
            )
            
            # 重新reshape
            obstacle_map[b] = obstacle_flat.view(H, W)
        
        # 4. 计算各项代价
        obstacle_cost = obstacle_map  # 障碍物代价（0或1）
        distance_cost = self._compute_distance_from_robot(H, W)  # 距离代价
        gradient_cost = self._compute_gradient_cost(height_map)  # 梯度代价
        abs_height_cost = torch.abs(height_map)  # 绝对高度代价
        
        # 5. 计算command方向奖励（减少代价）
        command_bonus = torch.zeros((batch_size, H, W), device=self.device)
        if command_velocity is not None:
            command_bonus = self._compute_command_bonus(
                command_velocity, H, W
            )
        
        # 6. 加权求和得到最终代价
        cost_map = (
            self.weight_obstacle * obstacle_cost +
            self.weight_distance * distance_cost +
            self.weight_gradient * gradient_cost +
            self.weight_abs_height * abs_height_cost -
            self.weight_command_bonus * command_bonus  # 减少代价
        )
        
        # 确保代价非负
        cost_map = torch.clamp(cost_map, min=0.0)
        
        return cost_map
    
    def _compute_distance_from_robot(self, H: int, W: int) -> torch.Tensor:
        """
        计算离机器人的距离代价（机器人在网格中心）
        
        Returns:
            distance_cost: (H, W) - 归一化距离代价
        """
        # 机器人位置在网格中心
        center_y = H / 2.0
        center_x = W / 2.0
        
        # 创建坐标网格
        y_coords = torch.arange(H, dtype=torch.float32, device=self.device)
        x_coords = torch.arange(W, dtype=torch.float32, device=self.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # 计算每个格子到中心的欧氏距离
        distance = torch.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
        
        # 归一化到[0, 1]
        max_distance = torch.sqrt(torch.tensor(center_x ** 2 + center_y ** 2, device=self.device))
        distance_cost = distance / max_distance
        
        return distance_cost
    
    def _compute_gradient_cost(self, height_map: torch.Tensor) -> torch.Tensor:
        """
        计算地形梯度代价
        
        Args:
            height_map: (batch, H, W) - 高度图
        
        Returns:
            gradient_cost: (batch, H, W) - 梯度代价
        """
        batch_size, H, W = height_map.shape
        
        # 使用padding避免边界效应
        height_padded = F.pad(
            height_map.unsqueeze(1), (1, 1, 1, 1), mode='replicate'
        )  # (batch, 1, H+2, W+2)
        
        # 计算与8个邻居的高度差绝对值之和
        neighbor_diff = torch.zeros((batch_size, H, W), device=self.device)
        
        # 8个方向的偏移
        offsets = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # 上下左右
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # 四个对角
        ]
        
        for dy, dx in offsets:
            # 提取邻居格子的高度
            neighbor_height = height_padded[:, 0, 1+dy:H+1+dy, 1+dx:W+1+dx]
            # 计算高度差绝对值
            diff = torch.abs(height_map - neighbor_height)
            neighbor_diff += diff
        
        # 归一化梯度代价（8个邻居的平均高度差）
        gradient_cost = neighbor_diff / 8.0
        
        # 归一化到[0, 1]范围
        gradient_cost = torch.clamp(gradient_cost, min=0.0, max=1.0)
        
        return gradient_cost
    
    def _compute_command_bonus(
        self,
        command_velocity: torch.Tensor,
        H: int,
        W: int
    ) -> torch.Tensor:
        """
        计算沿command方向的奖励（减少那些方向上的代价）
        
        逻辑：
        - 从机器人位置开始，沿着command_velocity方向
        - 在一定半径内的格子获得奖励（代价减少）
        - 距离command方向越近，奖励越大
        
        Args:
            command_velocity: (batch, 3) - [vx, vy, wz]
            H: 网格高度
            W: 网格宽度
        
        Returns:
            command_bonus: (batch, H, W) - Command方向奖励
        """
        batch_size = command_velocity.shape[0]
        
        # 提取XY方向的速度指令
        cmd_vx = command_velocity[:, 0]  # (batch,)
        cmd_vy = command_velocity[:, 1]  # (batch,)
        
        # 计算command的方向向量和大小
        cmd_magnitude = torch.sqrt(cmd_vx**2 + cmd_vy**2)  # (batch,)
        
        # 归一化command方向（处理零速度的情况）
        cmd_dir_x = torch.where(
            cmd_magnitude > 0.1,
            cmd_vx / cmd_magnitude,
            torch.zeros_like(cmd_vx)
        )  # (batch,)
        cmd_dir_y = torch.where(
            cmd_magnitude > 0.1,
            cmd_vy / cmd_magnitude,
            torch.zeros_like(cmd_vy)
        )
        
        # 机器人位置在网格中心
        center_y = H / 2.0
        center_x = W / 2.0
        
        # 创建坐标网格（相对机器人的物理坐标）
        y_coords = torch.arange(H, dtype=torch.float32, device=self.device)
        x_coords = torch.arange(W, dtype=torch.float32, device=self.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')  # (H, W)
        
        # 转换为物理坐标（米）
        xx_physical = (xx - center_x) * self.grid_resolution
        yy_physical = (yy - center_y) * self.grid_resolution
        
        # 初始化奖励
        command_bonus = torch.zeros((batch_size, H, W), device=self.device)
        
        for b in range(batch_size):
            # 计算每个格子相对机器人的向量
            grid_x = xx_physical  # (H, W)
            grid_y = yy_physical
            
            # 计算每个格子到机器人的距离
            grid_distance = torch.sqrt(grid_x**2 + grid_y**2)
            
            # 计算每个格子相对机器人的方向向量
            grid_dir_x = torch.where(
                grid_distance > 1e-3,
                grid_x / grid_distance,
                torch.zeros_like(grid_x)
            )
            grid_dir_y = torch.where(
                grid_distance > 1e-3,
                grid_y / grid_distance,
                torch.zeros_like(grid_y)
            )
            
            # 计算格子方向与command方向的点积（余弦相似度）
            direction_alignment = (
                grid_dir_x * cmd_dir_x[b] +
                grid_dir_y * cmd_dir_y[b]
            )  # (H, W)
            
            # 只对正向方向（前方）给予奖励
            direction_alignment = torch.clamp(direction_alignment, min=0.0, max=1.0)
            
            # 距离衰减：离机器人越远，奖励越小
            distance_decay = torch.exp(-grid_distance / self.command_bonus_radius)
            
            # 综合奖励：方向对齐 × 距离衰减
            command_bonus[b] = direction_alignment * distance_decay
        
        return command_bonus
