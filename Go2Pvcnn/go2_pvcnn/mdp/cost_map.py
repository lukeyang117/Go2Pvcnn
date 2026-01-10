"""Cost map generation from point cloud semantic segmentation."""

import torch
import torch.nn.functional as F
from typing import Tuple


class CostMapGenerator:
    """
    生成2D代价地图，用于2D CNN策略网络输入
    
    从3D点云语义分割结果投影到2D网格，计算三种代价:
    1. Distance cost: 到最近障碍物的距离
    2. Height gradient cost: 地形陡峭度
    3. Semantic confidence cost: 分割不确定性
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (32, 32),  # Reduced from (64, 64) for memory
        grid_resolution: float = 0.2,  # Increased from 0.1 to match larger cells
        x_range: Tuple[float, float] = (-3.2, 3.2),
        y_range: Tuple[float, float] = (-3.2, 3.2),
        obstacle_class_ids: list = [1, 2, 3],  # Go2 objects: 1=CrackerBox, 2=SugarBox, 3=TomatoSoupCan (class 0=Terrain)
        device: str = "cuda",
        # 代价权重参数（可人为调节）
        weight_obstacle: float = 1.0,  # 障碍物置信度权重
        weight_distance: float = 0.5,  # 距离机器人远近权重
        weight_gradient: float = 1.0,  # 地形梯度权重
        weight_abs_height: float = 0.3,  # 绝对高度权重
    ):
        """
        初始化代价地图生成器
        
        Args:
            grid_size: 网格尺寸 (height, width)，默认32×32 (优化显存)
            grid_resolution: 每个格子的实际尺寸(米)，默认0.2m
            x_range: X轴范围(米)，默认(-3.2, 3.2) = 6.4m覆盖
            y_range: Y轴范围(米)，默认(-3.2, 3.2) = 6.4m覆盖
            obstacle_class_ids: 障碍物类别ID列表
                - Class 0: Terrain（地形，可通行）
                - Class 1: CrackerBox（Object_0，障碍物）
                - Class 2: SugarBox（Object_1，障碍物）
                - Class 3: TomatoSoupCan（Object_2，障碍物）
            device: 计算设备
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
        
        # 预计算网格坐标
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        
        # Debug counters
        self._call_count = 0
    
    def generate_cost_map(
        self,
        point_xyz: torch.Tensor,
        pred_classes: torch.Tensor,
        semantic_confidence: torch.Tensor,
        height_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        从点云语义分割和高程图生成3通道代价地图
        
        Args:
            point_xyz: (batch, num_points, 3) - 机器人坐标系下的点云坐标（用于xy投影）
            pred_classes: (batch, num_points) - 预测的类别索引（避免重复计算argmax）
            semantic_confidence: (batch, num_points) - 分类置信度
            height_map: (batch, H, W) - 高程图（来自height_scanner传感器）
        
        Returns:
            cost_map: (batch, H, W) - 单通道代价地图（加权组合）
                - 综合了障碍物、距离、梯度、高度的加权代价
                - 所有代价均为正值（0-1范围）
        """
        self._call_count += 1
        batch_size = point_xyz.shape[0]
        num_points = point_xyz.shape[1]
        H, W = self.grid_size  # H对应y方向，W对应x方向
        
        # CRITICAL: Validate input shapes to prevent index errors
        assert pred_classes.shape[0] == batch_size, f"Batch size mismatch: {pred_classes.shape[0]} != {batch_size}"
        assert pred_classes.shape[1] == num_points, f"Point count mismatch: {pred_classes.shape[1]} != {num_points}"
        assert semantic_confidence.shape[0] == batch_size, f"Batch size mismatch in confidence: {semantic_confidence.shape[0]} != {batch_size}"
        assert semantic_confidence.shape[1] == num_points, f"Point count mismatch in confidence: {semantic_confidence.shape[1]} != {num_points}"
        
        # Debug info (每100次调用打印一次)
        if self._call_count % 100 == 1:
            print(f"\n[CostMapGenerator] Call {self._call_count}")
            print(f"  - Input: point_xyz {point_xyz.shape}, pred_classes {pred_classes.shape}, confidence {semantic_confidence.shape}")
            print(f"  - Grid: {H}×{W}, resolution {self.grid_resolution}m")
            print(f"  - Coordinate range: X=[{self.x_min}, {self.x_max}]m, Y=[{self.y_min}, {self.y_max}]m")
            print(f"  - Validated: batch_size={batch_size}, num_points={num_points}")
        
        # 1. 将XY坐标映射到网格索引（使用与height_scanner相同的物理坐标系）
        # point_xyz: (batch, num_points, 3) -> x, y, z
        x_coords = point_xyz[:, :, 0]  # (batch, num_points) - 前方距离
        y_coords = point_xyz[:, :, 1]  # (batch, num_points) - 左右距离
        
        # 将物理坐标映射到网格索引（不归一化，直接使用物理尺寸）
        # height_scanner的配置: size=[1.5, 1.5], resolution=0.1
        # x_range: [-0.75, 0.75]m, W=16; y_range: [-0.75, 0.75]m, H=16 (GridPatternCfg包含边界点)
        x_indices = ((x_coords - self.x_min) / self.grid_resolution).long()
        y_indices = ((y_coords - self.y_min) / self.grid_resolution).long()
        
        # 创建有效点mask：只处理在高程图范围内的点
        valid_mask = (x_indices >= 0) & (x_indices < W) & (y_indices >= 0) & (y_indices < H)
        
        if self._call_count % 100 == 1:
            valid_count = valid_mask.sum()
            total_count = valid_mask.numel()
            print(f"  - Points in height map range: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)")
        
        # 2. 初始化代价地图（高度图直接使用传入的height_map）
        confidence_map = torch.ones((batch_size, H, W), device=self.device)
        obstacle_map = torch.zeros((batch_size, H, W), device=self.device)
        
        # 验证高度图尺寸
        assert height_map.shape == (batch_size, H, W), f"Height map shape mismatch: {height_map.shape} != ({batch_size}, {H}, {W})"
        
        # 3. 投影到2D网格 - 只处理有效范围内的障碍物点
        # 创建obstacle mask: 是否为障碍物类别且在范围内
        is_obstacle = torch.zeros((batch_size, num_points), dtype=torch.bool, device=self.device)
        for obs_id in self.obstacle_class_ids:
            is_obstacle |= (pred_classes == obs_id)
        
        # 只保留在有效范围内的障碍物点
        is_obstacle = is_obstacle & valid_mask
        
        if self._call_count % 100 == 1:
            obstacle_count = is_obstacle.sum()
            print(f"  - Obstacle points in range: {obstacle_count}")
        
        # 使用scatter操作进行向量化更新（只更新有效范围内的点）
        for b in range(batch_size):
            # 获取该batch中的有效点mask
            batch_valid = valid_mask[b]  # (num_points,)
            
            if batch_valid.sum() == 0:
                continue  # 跳过没有有效点的batch
            
            # 只处理有效范围内的点
            valid_x_indices = x_indices[b][batch_valid] #size(num_valid_points,)
            valid_y_indices = y_indices[b][batch_valid] #size(num_valid_points,)
            valid_confidence = semantic_confidence[b][batch_valid] #size(num_valid_points,)
            valid_obstacle = is_obstacle[b][batch_valid]    #size(num_valid_points,)
            
            # 创建线性索引: y*W + x
            linear_indices = valid_y_indices * W + valid_x_indices  # (num_valid_points,)
            
            # 展平grid用于scatter操作
            confidence_flat = confidence_map[b].view(-1)
            obstacle_flat = obstacle_map[b].view(-1)
            
            # 先更新obstacle: 取最大值（只要有一个障碍物就标记为障碍物）- 安全优先
            obstacle_flat.scatter_reduce_(0, linear_indices, valid_obstacle.float(), reduce='amax', include_self=False)
            
            # 只对障碍物点更新confidence（取障碍物点中的最大置信度）
            # 过滤出障碍物点的索引和置信度
            obstacle_mask = valid_obstacle  # (num_valid_points,)
            if obstacle_mask.sum() > 0:  # 如果有障碍物点
                obstacle_indices = linear_indices[obstacle_mask]
                obstacle_confidence = valid_confidence[obstacle_mask]
                
                # 对障碍物点的置信度取最大值（最确定的障碍物）
                confidence_flat.scatter_reduce_(0, obstacle_indices, obstacle_confidence, reduce='amax', include_self=False)
            
            # 重新reshape
            confidence_map[b] = confidence_flat.view(H, W)
            obstacle_map[b] = obstacle_flat.view(H, W)
        
        # 4. 计算障碍物代价（基于语义置信度，符号由高度决定）
        obstacle_cost_abs = obstacle_map * confidence_map  # 障碍物位置的置信度（绝对值）
        
        # 5. 计算距离代价（离机器人越远代价越高，符号由高度决定）
        distance_cost_abs = self._compute_distance_from_robot(H, W)
        # 6. 计算地形梯度代价（陡峭度，符号由高度决定）
        gradient_cost_abs = self._compute_gradient_cost(height_map)
        
        # 7. 计算绝对高度代价（高度越大代价越高）
        abs_height_cost = torch.abs(height_map)
        
        # 8. 加权求和得到最终代价（全部为正值，范围0-1）
        cost_map = (
            self.weight_obstacle * obstacle_cost_abs +
            self.weight_distance * distance_cost_abs +
            self.weight_gradient * gradient_cost_abs +
            self.weight_abs_height * abs_height_cost
        )  # (batch, H, W)
        if self._call_count % 100 == 1:
            print(f"  - Output: cost_map {cost_map.shape}")
            print(f"  - Obstacle cost range: [{obstacle_cost_abs.min():.3f}, {obstacle_cost_abs.max():.3f}]")
            print(f"  - Distance cost range: [{distance_cost_abs.min():.3f}, {distance_cost_abs.max():.3f}]")
            print(f"  - Gradient cost range: [{gradient_cost_abs.min():.3f}, {gradient_cost_abs.max():.3f}]")
            print(f"  - Abs height cost range: [{abs_height_cost.min():.3f}, {abs_height_cost.max():.3f}]")
            print(f"  - Final cost range: [{cost_map.min():.3f}, {cost_map.max():.3f}]")
            print(f"  - Weights: obstacle={self.weight_obstacle}, distance={self.weight_distance}, gradient={self.weight_gradient}, abs_height={self.weight_abs_height}")
        
        return cost_map
    
    def _compute_distance_from_robot(self, H: int, W: int) -> torch.Tensor:
        """
        计算离机器人的距离代价（机器人在网格中心）
        
        Args:
            H: 网格高度
            W: 网格宽度
        
        Returns:
            distance_cost: (H, W) - 归一化距离代价（离中心越远代价越高）
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
        
        # 归一化到[0, 1]（最远角落为1）
        max_distance = torch.sqrt(torch.tensor(center_x ** 2 + center_y ** 2, device=self.device))
        distance_cost = distance / max_distance
        
        return distance_cost
    
    def _compute_gradient_cost(self, height_map: torch.Tensor) -> torch.Tensor:
        """
        计算地形梯度代价（仅计算陡峭度，不考虑符号）
        
        Args:
            height_map: (batch, H, W) - 高度图
        
        Returns:
            gradient_cost: (batch, H, W) - 梯度代价（绝对值，符号由外部根据height_map决定）
        """
        batch_size, H, W = height_map.shape
        
        # 相邻格子高度差代价（计算与周围8个格子的高度差之和）
        # 使用padding='replicate'避免边界效应
        height_padded = F.pad(height_map.unsqueeze(1), (1, 1, 1, 1), mode='replicate')  # (batch, 1, H+2, W+2)
        
        # 计算与8个邻居的高度差绝对值之和
        neighbor_diff = torch.zeros((batch_size, H, W), device=self.device)
        
        # 8个方向的偏移: 上、下、左、右、左上、右上、左下、右下
        offsets = [
            (-1, 0), (1, 0), (0, -1), (0, 1),  # 上下左右
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # 四个对角
        ]
        
        for dy, dx in offsets:
            # 提取邻居格子的高度
            neighbor_height = height_padded[:, 0, 1+dy:H+1+dy, 1+dx:W+1+dx]
            # 累加高度差的绝对值
            neighbor_diff += torch.abs(height_map - neighbor_height)
        
        # 归一化（假设8个邻居总和>4m为最高陡峭度）
        gradient_cost = torch.clamp(neighbor_diff / 4.0, 0, 1)
        
        return gradient_cost
