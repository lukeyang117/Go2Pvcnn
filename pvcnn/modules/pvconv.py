import torch.nn as nn

import modules.functional as F
from modules.voxelization import Voxelization
from modules.shared_mlp import SharedMLP
from modules.se import SE3d

__all__ = ['PVConv']


class PVConv(nn.Module):
    """
    Point-Voxel Convolution (PVConv) 模块
    
    核心思想：融合点云特征和体素化 3D 卷积特征
    - 点云分支: 使用 SharedMLP (1x1 卷积) 提取逐点特征
    - 体素分支: 将点云体素化后使用 3D 卷积提取局部几何特征
    - 融合: 将两个分支的特征相加，结合全局和局部信息
    
    优势:
    1. 3D 卷积能捕获局部几何结构（如表面法线、曲率）
    2. 点云 MLP 保持对不规则点分布的灵活性
    3. 特征融合结合了两种表示的优点
    """
    def __init__(self, in_channels, out_channels, kernel_size, resolution, with_se=False, normalize=True, eps=0):
        """
        参数:
            in_channels (int): 输入特征通道数
            out_channels (int): 输出特征通道数
            kernel_size (int): 3D 卷积核大小 (通常为 3)
            resolution (int): 体素网格分辨率 (如 32 表示 32x32x32 网格)
            with_se (bool): 是否使用 SE (Squeeze-and-Excitation) 注意力模块
            normalize (bool): 是否归一化点云坐标到 [-1, 1]
            eps (float): 体素化时避免除零的小常数
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution

        # 体素化模块: 将不规则点云转换为规则的 3D 网格
        # normalize=True 会将点云坐标归一化到 [-1, 1] 范围
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        #体素化输出尺度 : [B, C_in, R, R, R] (R = resolution)
        # 体素分支: 3D 卷积网络
        voxel_layers = [
            # 第一层 3D 卷积: in_channels -> out_channels
            # padding=kernel_size//2 保证输出尺寸不变 (same padding)
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            # 什么是batchnorm3d? 对3D数据进行归一化处理
            #均值和方差怎么来的? 是在batch维度上计算的
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),  # 负斜率 0.1 的 LeakyReLU
            
            # 第二层 3D 卷积: out_channels -> out_channels (残差结构思想)
            nn.Conv3d(out_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2),
            nn.BatchNorm3d(out_channels, eps=1e-4),
            nn.LeakyReLU(0.1, True),
         ]
        
        # 可选: 添加 SE 注意力模块，自适应调整通道权重
        if with_se:
            voxel_layers.append(SE3d(out_channels))
        
        self.voxel_layers = nn.Sequential(*voxel_layers)
        
        # 点云分支: 使用 1x1 卷积 (SharedMLP) 直接处理点特征
        # 这个分支保留了点云的灵活性，不受体素化的离散化误差影响
        self.point_features = SharedMLP(in_channels, out_channels)

    def forward(self, inputs):
        """
        前向传播
        
        参数:
            inputs: (features, coords)
                features (Tensor): 点云特征 [B, C_in, N]
                    B: batch size
                    C_in: 输入特征通道数
                    N: 点数量
                coords (Tensor): 点云坐标 [B, 3, N]
                    3: (x, y, z) 坐标
        
        返回:
            (fused_features, coords): 融合后的特征和原始坐标
                fused_features (Tensor): [B, C_out, N]
                coords (Tensor): [B, 3, N] (原样返回，用于后续层)
        """
        features, coords = inputs
        
        # === 体素分支 ===
        # 步骤 1: 体素化 - 将点云转换为规则的 3D 网格
        # voxel_features: [B, C_in, R, R, R] (R = resolution)
        # voxel_coords: 体素化后的坐标信息（用于反体素化）
        voxel_features, voxel_coords = self.voxelization(features, coords)
        
        # 步骤 2: 3D 卷积 - 在体素网格上提取局部几何特征
        # [B, C_in, R, R, R] -> [B, C_out, R, R, R]
        voxel_features = self.voxel_layers(voxel_features)
        
        # 步骤 3: 反体素化 - 将体素特征插值回原始点云
        # 使用三线性插值将规则网格特征映射回不规则点云
        # [B, C_out, R, R, R] -> [B, C_out, N]
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        
        # === 点云分支 ===
        # 直接在原始点云上使用 1x1 卷积提取特征
        # [B, C_in, N] -> [B, C_out, N]
        point_features = self.point_features(features)
        
        # === 特征融合 ===
        # 将体素特征和点云特征相加（element-wise addition）
        # 这种融合方式简单有效，结合了局部几何和全局语义信息
        # [B, C_out, N] + [B, C_out, N] = [B, C_out, N]
        fused_features = voxel_features + point_features
        
        # 返回融合特征和原始坐标（坐标用于下一层的体素化）
        return fused_features, coords
