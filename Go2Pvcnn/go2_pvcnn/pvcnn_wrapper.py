"""PVCNN Model Wrapper for Inference."""

import os
import sys
import torch
import torch.nn as nn
import numpy as np

# Add pvcnn directory to path
PVCNN_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "pvcnn")
if PVCNN_ROOT not in sys.path:
    sys.path.insert(0, PVCNN_ROOT)


from models.s3dis.pvcnn import PVCNN


class PVCNNWrapper:
    """
    Wrapper for PVCNN model to extract features from point cloud data.
    
    This wrapper loads a pre-trained PVCNN model and provides an interface
    to extract features from LiDAR point cloud data for reinforcement learning.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        num_classes: int = 13,
        extra_feature_channels: int = 6,
        num_points: int = 2046,
        device: str = "cuda",
        feature_layer: str = "cloud",  # "cloud" 或 "point"
        max_batch_size: int = 64,  # PVCNN推理的最大批量大小
    ):
        """
        初始化 PVCNN 包装器。
        
        参数:
            checkpoint_path: 训练好的PVCNN checkpoint路径 (.pth.tar文件)
            num_classes: 输出类别数量 (S3DIS数据集有13类)
            extra_feature_channels: 额外特征通道数 (RGB + 法向量 = 6)
            num_points: 从点云中采样的点数量
            device: 推理运行的设备 (字符串或torch.device)
            feature_layer: 提取哪层特征 ("cloud"表示全局特征, "point"表示逐点特征)
            max_batch_size: PVCNN的最大批量大小，用于节省显存 (默认: 64)
        """
        # 将设备字符串转换为torch.device对象，确保统一处理
        if isinstance(device, str):
            self.device = torch.device(device)  # 转换为torch.device对象
        else:
            self.device = device  # 已经是torch.device对象，直接使用
        
        # 保存配置参数到实例变量
        self.num_points = num_points  # 点云采样点数
        self.num_classes = num_classes  # 分类类别数
        self.extra_feature_channels = extra_feature_channels  # 额外特征通道数
        self.feature_layer = feature_layer  # 特征提取层类型
        self.max_batch_size = max_batch_size  # 最大批处理大小
        
        # 创建PVCNN模型实例
        self.model = PVCNN(
            num_classes=num_classes,  # 输出类别数
            extra_feature_channels=extra_feature_channels,  # 额外特征通道数
            width_multiplier=1,  # 网络宽度倍数（1表示使用默认通道数）
            voxel_resolution_multiplier=1  # 体素分辨率倍数（1表示使用默认分辨率）
        )
        
        # 加载预训练权重
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # 处理DataParallel格式的checkpoint
        state_dict = checkpoint['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k[7:] if k.startswith('module.') else k] = v
        
        # 调整输入通道数：9通道(XYZ+RGB+法向量) -> 3通道(XYZ)
        if extra_feature_channels == 0:
            for key in list(new_state_dict.keys()):
                if 'point_features.0' in key and 'weight' in key:
                    original_weight = new_state_dict[key]
                    if len(original_weight.shape) >= 2 and original_weight.shape[1] == 9:
                        new_state_dict[key] = original_weight[:, :3, ...]
        
        # 调整输出类别数：13类 -> 4类
        if num_classes == 4:
            for key in list(new_state_dict.keys()):
                if 'classifier' in key and ('weight' in key or 'bias' in key):
                    original_param = new_state_dict[key]
                    if original_param.shape[0] == 13:
                        new_state_dict[key] = original_param[:4, ...] if 'weight' in key else original_param[:4]
        
        # 加载权重
        self.model.load_state_dict(new_state_dict, strict=False)
        
        # 将模型移动到指定设备并设置为评估模式
        self.model = self.model.to(device)
        self.model.eval()
        
        # 注册前向钩子以提取中间层特征
        self.features = {}
        if feature_layer == "cloud":
            self.model.cloud_features.register_forward_hook(self._get_activation('cloud'))
        elif feature_layer != "point":
            raise ValueError(f"Unsupported feature_layer: {feature_layer}. Use 'cloud' or 'point'.")
    
    def _get_activation(self, name):
        """Hook to capture layer activations."""
        def hook(model, input, output):
            self.features[name] = output.detach()
        return hook
    
    @torch.no_grad()
    def extract_features(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Extract features from point cloud data using batched processing.
        
        Args:
            point_cloud: Tensor of shape (batch_size, num_points, 3+C)
        
        Returns:
            features: Tensor based on feature_layer:
                - "cloud": (batch_size, 128)
                - "point": (batch_size, num_points, num_classes)
        """
        original_device = point_cloud.device
        if original_device != self.device:
            point_cloud = point_cloud.to(self.device, non_blocking=True)
        
        total_batch_size = point_cloud.shape[0]
        
        # Process in batches if needed
        if total_batch_size <= self.max_batch_size:
            features = self._extract_features_single_batch(point_cloud)
        else:
            all_features = []
            num_batches = (total_batch_size + self.max_batch_size - 1) // self.max_batch_size
            
            for i in range(num_batches):
                start_idx = i * self.max_batch_size
                end_idx = min((i + 1) * self.max_batch_size, total_batch_size)
                batch_features = self._extract_features_single_batch(point_cloud[start_idx:end_idx])
                all_features.append(batch_features)
            
            features = torch.cat(all_features, dim=0)
        
        # Transfer back to original device if needed
        if features.device != original_device:
            features = features.to(original_device, non_blocking=True)
        
        return features
    
    def _extract_features_single_batch(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Extract features from a single batch of point cloud data.
        
        Args:
            point_cloud: Tensor of shape (batch_size, num_points, 3+C)
        
        Returns:
            features: Tensor based on feature_layer
        """
        batch_size = point_cloud.shape[0]
        
        
        
        # Transpose to (batch_size, channels, num_points)
        inputs = point_cloud.transpose(1, 2).contiguous()
        
        # Forward pass
        output = self.model(inputs)
        
        # Extract features
        if isinstance(output, dict):
            if self.feature_layer == "cloud":
                features = output['global_features']
            else:  # point
                features = output['logits'].transpose(1, 2).contiguous()
        else:
            if self.feature_layer == "cloud":
                features = self.features['cloud']
            else:  # point
                features = output.transpose(1, 2).contiguous()
        
        return features
    
    
    def get_feature_dim(self) -> int:
        """Get the dimension of extracted features."""
        if self.feature_layer == "cloud":
            return 128
        else:  # point
            return (self.num_points, self.num_classes)


def create_pvcnn_wrapper(
    checkpoint_path: str, 
    device: str = "cuda", 
    num_points: int = 2046,
    max_batch_size: int = 32
) -> PVCNNWrapper:
    """Factory function to create PVCNN wrapper."""
    return PVCNNWrapper(
        checkpoint_path=checkpoint_path,
        num_classes=4,
        extra_feature_channels=0,
        num_points=num_points,
        device=device,
        feature_layer="point",
        max_batch_size=max_batch_size
    )
