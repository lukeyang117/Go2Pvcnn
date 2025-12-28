import torch
import torch.nn as nn

from models.utils import create_mlp_components, create_pointnet_components

__all__ = ['PVCNN']


class PVCNN(nn.Module):
    # blocks 定义了网络的层次结构，每个元组是 (输出通道数, 块数量, 体素分辨率)
    # (64, 1, 32): 64通道, 1个块, 32x32x32体素网格 - 第一层特征提取
    # (64, 2, 16): 64通道, 2个块, 16x16x16体素网格 - 中层特征聚合
    # (128, 1, 16): 128通道, 1个块, 16x16x16体素网格 - 深层特征
    # (1024, 1, None): 1024通道, 1个块, 无体素化(使用纯PointNet) - 最高级语义特征
    blocks = ((64, 1, 32), (64, 2, 16), (128, 1, 16), (1024, 1, None))

    def __init__(self, num_classes, extra_feature_channels=6, width_multiplier=1, voxel_resolution_multiplier=1):
        """
        初始化 PVCNN 模型（Point-Voxel CNN）
        
        参数说明:
            num_classes (int): 
                分类类别数量
                - 对于 S3DIS 数据集 = 13 (天花板、地板、墙壁、梁、柱、窗户、门、桌子、椅子、沙发、书柜、板子、杂物)
                - 对于语义分割任务，表示每个点可以属于的类别数
            
            extra_feature_channels (int, 默认=6): 
                除了 XYZ 坐标外的额外特征通道数
                - 6 = RGB(3) + 法向量(3) - S3DIS 标准配置
                - 3 = 只有 RGB 颜色
                - 0 = 只使用 XYZ 几何信息
                - 总输入通道数 = 3(XYZ) + extra_feature_channels
            
            width_multiplier (float, 默认=1): 
                网络宽度倍增因子，控制每层的通道数
                - 1.0: 使用默认通道数 (64, 128, 1024等)
                - 0.5: 通道数减半，网络更轻量 (32, 64, 512等)
                - 2.0: 通道数翻倍，网络更强大 (128, 256, 2048等)
                - 影响参数量和计算量: params ∝ width_multiplier²
            
            voxel_resolution_multiplier (float, 默认=1): 
                体素分辨率倍增因子，控制体素网格的精细程度
                - 1.0: 使用默认分辨率 (32x32x32, 16x16x16等)
                - 0.5: 分辨率减半，计算更快但精度降低 (16x16x16, 8x8x8等)
                - 2.0: 分辨率翻倍，精度更高但计算更慢 (64x64x64, 32x32x32等)
                - 影响: 体素数量 ∝ voxel_resolution_multiplier³
        
        网络结构:
            1. point_features: 逐点特征提取器 (PVConv层)
               - 4个阶段，逐渐提取从低级到高级的特征
               - 输出: 每个阶段的特征 (64, 64, 128, 1024维)
            
            2. cloud_features: 全局云特征提取器
               - 通过max pooling聚合所有点 -> 全局特征
               - 1024维 -> 256维 -> 128维
            
            3. classifier: 分类器
               - 拼接所有阶段特征 + 全局特征
               - 输出: 每个点的类别概率 (num_points, num_classes)
        """
        super().__init__()
        self.in_channels = extra_feature_channels + 3  # 总输入通道数: XYZ(3) + 额外特征

        # 创建逐点特征提取层 (Point-Voxel Convolution)
        # 返回: layers (PVConv层列表), channels_point (最后一层输出通道数), concat_channels_point (所有层输出通道数之和)
        layers, channels_point, concat_channels_point = create_pointnet_components(
            blocks=self.blocks,                          # 网络结构定义
            in_channels=self.in_channels,                # 输入通道数 (3+extra_feature_channels)
            with_se=False,                               # 是否使用 Squeeze-and-Excitation 注意力机制
            width_multiplier=width_multiplier,           # 宽度倍增因子
            voxel_resolution_multiplier=voxel_resolution_multiplier  # 体素分辨率倍增因子
        )
        self.point_features = nn.ModuleList(layers)      # 保存为模块列表，forward时逐层提取特征

        # 创建全局云特征提取层 (Global Feature Aggregation)
        # 作用: 将逐点特征 (batch, 1024, num_points) 聚合为全局特征 (batch, 128)，
        # [256, 128]: 先降维到256，再降维到128
        layers, channels_cloud = create_mlp_components(
            in_channels=channels_point,                  # 输入: 最后一层point特征的通道数 (1024)
            out_channels=[256, 128],                     # 输出: 两层MLP，降维到128维全局特征
            classifier=False,                            # 不是最终分类器，需要BN+ReLU激活
            dim=1,                                       # 1D卷积(实际是Linear层)，处理全局特征
            width_multiplier=width_multiplier            # 宽度倍增因子
        )
        self.cloud_features = nn.Sequential(*layers)     # 全局特征提取器

        # 创建最终分类器 (Semantic Segmentation Head)
        # 作用: 将拼接的所有特征映射到每个点的类别概率
        # [512, 0.3, 256, 0.3, num_classes]: 
        #   - 512通道 -> Dropout(0.3) -> 256通道 -> Dropout(0.3) -> num_classes
        #   - 小于1的数字表示Dropout比例
        layers, _ = create_mlp_components(
            in_channels=(concat_channels_point + channels_cloud),  # 输入: 所有point特征 + 全局云特征
            out_channels=[512, 0.3, 256, 0.3, num_classes],        # 输出: 逐层降维，加Dropout防止过拟合
            classifier=True,                                       # 最终分类器，最后一层无激活函数
            dim=2,                                                 # 2D卷积(Conv1d)，处理逐点特征
            width_multiplier=width_multiplier                      # 宽度倍增因子
        )
        self.classifier = nn.Sequential(*layers)         # 分类器

    def forward(self, inputs):
        """
        前向传播：从点云输入到语义分割输出
        
        参数:
            inputs: 点云输入，支持两种格式
                - Tensor: (batch_size, channels, num_points)
                  - channels = 3(XYZ) + extra_feature_channels (RGB, 法向量等)
                  - 例如: (16, 9, 10000) = 16个batch, 9通道(XYZ+RGB+法向量), 10000个点
                - Dict: {'features': Tensor} - 字典格式，从中提取 'features' 键
        
        返回:
            output: (batch_size, num_classes, num_points)
                - 每个点对于每个类别的logit (未归一化的分数)
                - 例如: (16, 13, 10000) = 16个batch, 13个类别, 10000个点
                - 使用 softmax(output, dim=1) 可得到概率分布
        
        计算流程:
            1. 提取 XYZ 坐标
            2. 逐层提取point特征 (4个阶段)
            3. 全局max pooling + MLP -> 128维云特征
            4. 拼接所有特征并复制云特征到每个点
            5. 通过分类器输出每个点的类别logit
        """
        # 支持字典格式输入 (某些数据加载器使用)
        if isinstance(inputs, dict):
            inputs = inputs['features']

        # 提取 XYZ 坐标 (前3个通道)
        # coords: (batch_size, 3, num_points)
        coords = inputs[:, :3, :]
        
        # 存储每一层的输出特征，用于后续拼接 (类似U-Net的skip connection)
        out_features_list = []
        
        # 逐层提取point特征 (Point-Voxel Convolution)
        # 每层提取不同尺度的特征: 64维(局部) -> 64维(中层) -> 128维(深层) -> 1024维(高级语义)
        for i in range(len(self.point_features)):
            # PVConv 需要 (features, coords) 作为输入
            # inputs: 逐渐变化的特征 (batch, channels, num_points)
            # coords: 固定的坐标 (batch, 3, num_points)
            inputs, _ = self.point_features[i]((inputs, coords))
            out_features_list.append(inputs)  # 保存当前层特征
        
        # 全局特征聚合 (Global Feature Aggregation)
        # inputs: (batch, 1024, num_points) - 最后一层的逐点特征
        # max pooling: 对每个通道，取所有点的最大值 -> (batch, 1024)
        # MLP降维: (batch, 1024) -> (batch, 256) -> (batch, 128)
        inputs = self.cloud_features(inputs.max(dim=-1, keepdim=False).values)
        
        # 将全局云特征扩展到每个点
        # (batch, 128) -> (batch, 128, 1) -> (batch, 128, num_points)
        # 作用: 让每个点都能"感知"到全局场景信息
        out_features_list.append(inputs.unsqueeze(-1).repeat([1, 1, coords.size(-1)]))
        
        # 拼接所有特征并通过分类器
        # torch.cat(out_features_list, dim=1): 在通道维度拼接
        #   - 第1层: (batch, 64, num_points)
        #   - 第2层: (batch, 64, num_points)
        #   - 第3层: (batch, 128, num_points)
        #   - 第4层: (batch, 1024, num_points)
        #   - 全局: (batch, 128, num_points)
        #   = (batch, 64+64+128+1024+128, num_points) = (batch, 1408, num_points)
        # 
        # classifier: (batch, 1408, num_points) -> (batch, num_classes, num_points)
        logits = self.classifier(torch.cat(out_features_list, dim=1))
        
        # 计算语义置信度 (用于代价地图生成)
        # softmax后取最大概率作为置信度
        confidence = torch.softmax(logits, dim=1).max(dim=1)[0]  # (batch, num_points)
        
        # 返回字典格式，包含logits、置信度和全局特征
        return {
            'logits': logits,           # (batch, num_classes, num_points) - 语义分割logits
            'confidence': confidence,   # (batch, num_points) - 每个点的分类置信度
            'global_features': inputs   # (batch, 128) - 全局云特征，用于RL策略
        }
