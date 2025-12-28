#!/usr/bin/env python
"""
PVCNN 语义分割推理脚本
用于从 Isaac Lab 的 LiDAR 传感器获取点云数据，并使用训练好的 PVCNN 模型进行语义分割
"""

import os
import sys
import torch
import numpy as np
from typing import Optional, Tuple
import argparse

# 添加 PVCNN 模块路径
PVCNN_ROOT = os.path.join(os.path.dirname(__file__), 'pvcnn')
sys.path.insert(0, PVCNN_ROOT)

from models.s3dis import PVCNN


class PVCNNInference:
    """PVCNN 推理类，用于语义分割"""
    
    # S3DIS 数据集的类别标签
    CLASS_LABELS = [
        'ceiling', 'floor', 'wall', 'beam', 'column',
        'window', 'door', 'table', 'chair', 'sofa',
        'bookcase', 'board', 'clutter'
    ]
    
    # 为可视化定义的颜色映射 (RGB)
    CLASS_COLORS = np.array([
        [0, 255, 0],      # ceiling - 绿色
        [0, 0, 255],      # floor - 蓝色
        [0, 255, 255],    # wall - 青色
        [255, 255, 0],    # beam - 黄色
        [255, 0, 255],    # column - 品红
        [100, 100, 255],  # window - 浅蓝
        [200, 200, 100],  # door - 淡黄
        [170, 120, 200],  # table - 紫色
        [255, 0, 0],      # chair - 红色
        [200, 100, 100],  # sofa - 褐红
        [10, 200, 100],   # bookcase - 青绿
        [200, 200, 200],  # board - 浅灰
        [50, 50, 50],     # clutter - 深灰
    ])
    
    def __init__(
        self,
        checkpoint_path: str,
        num_points: int = 4096,
        device: str = 'cuda',
        width_multiplier: float = 1.0,
        voxel_resolution_multiplier: int = 1
    ):
        """
        初始化 PVCNN 推理器
        
        Args:
            checkpoint_path: 模型权重文件路径 (.pth.tar)
            num_points: 输入点云的点数 (默认: 4096)
            device: 计算设备 ('cuda' 或 'cpu')
            width_multiplier: 模型宽度倍数 (默认: 1.0)
            voxel_resolution_multiplier: 体素分辨率倍数 (默认: 1)
        """
        self.num_points = num_points
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 先加载权重以检测特征通道数
        self.extra_feature_channels = self._detect_feature_channels(checkpoint_path)
        
        # 创建模型
        self.model = PVCNN(
            num_classes=13,  # S3DIS 有 13 个类别
            extra_feature_channels=self.extra_feature_channels,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier
        ).to(self.device)
        
        # 加载模型权重
        self._load_checkpoint(checkpoint_path)
        
        # 设置为评估模式
        self.model.eval()
        
        print(f"[PVCNN Inference] 模型加载成功!")
        print(f"[PVCNN Inference] 设备: {self.device}")
        print(f"[PVCNN Inference] 输入点数: {num_points}")
        print(f"[PVCNN Inference] 类别数: 13")
    
    def _detect_feature_channels(self, checkpoint_path: str) -> int:
        """从权重文件检测特征通道数"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 查找第一层的权重来确定输入通道数
        for key in state_dict.keys():
            if 'point_features.0.voxel_layers.0.weight' in key or 'module.point_features.0.voxel_layers.0.weight' in key:
                weight_shape = state_dict[key].shape
                # weight_shape[1] 是输入通道数
                # 输入通道 = 3 (xyz) + extra_feature_channels
                in_channels = weight_shape[1]
                extra_channels = in_channels - 3
                print(f"[PVCNN Inference] 检测到模型输入通道数: {in_channels} (3 xyz + {extra_channels} 额外特征)")
                
                # S3DIS 标准配置: 
                # extra_channels = 6 表示 RGB(3) + 归一化坐标(3)
                # extra_channels = 3 表示只有归一化坐标(3)
                if extra_channels == 6:
                    print(f"[PVCNN Inference] 配置: RGB + 归一化坐标")
                elif extra_channels == 3:
                    print(f"[PVCNN Inference] 配置: 仅归一化坐标")
                
                return extra_channels
        
        # 默认返回6（标准S3DIS配置）
        print(f"[PVCNN Inference] 使用默认特征通道数: 6")
        return 6
    
    def _load_checkpoint(self, checkpoint_path: str):
        """加载模型权重"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"找不到模型权重文件: {checkpoint_path}")
        
        print(f"[PVCNN Inference] 正在加载权重: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # 提取模型状态字典
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # 处理 DataParallel 保存的模型（移除 "module." 前缀）
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                new_key = key[7:]  # 移除 "module." 前缀
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        # 加载权重
        self.model.load_state_dict(new_state_dict)
        
        # 打印一些信息
        if 'epoch' in checkpoint:
            print(f"[PVCNN Inference] 训练轮次: {checkpoint['epoch']}")
        if 'best_metric' in checkpoint:
            print(f"[PVCNN Inference] 最佳指标: {checkpoint['best_metric']:.4f}")
    
    def preprocess_pointcloud(
        self,
        pointcloud: np.ndarray,
        normalize_coords: bool = True
    ) -> torch.Tensor:
        """
        预处理点云数据
        
        Args:
            pointcloud: 输入点云 (N, 3) 或 (N, 3+C)，其中 C 是额外特征
            normalize_coords: 是否归一化坐标
            
        Returns:
            处理后的点云张量 (1, in_channels, num_points)
        """
        # 确保点云是 numpy 数组
        if isinstance(pointcloud, torch.Tensor):
            pointcloud = pointcloud.cpu().numpy()
        
        # 提取 XYZ 坐标
        coords = pointcloud[:, :3].astype(np.float32)
        
        # 检查是否有颜色信息
        has_color = pointcloud.shape[1] >= 6
        if has_color:
            colors = pointcloud[:, 3:6].astype(np.float32)
            # 归一化颜色到 [0, 1]
            if colors.max() > 1.0:
                colors = colors / 255.0
        
        # 随机采样或重复采样到指定点数
        num_available_points = coords.shape[0]
        if num_available_points >= self.num_points:
            # 随机采样
            indices = np.random.choice(num_available_points, self.num_points, replace=False)
        else:
            # 重复采样
            indices = np.random.choice(num_available_points, self.num_points, replace=True)
        
        coords = coords[indices]
        if has_color:
            colors = colors[indices]
        
        # 计算归一化坐标 (相对于点云中心和范围)
        if normalize_coords:
            # 中心化
            center = coords.mean(axis=0, keepdims=True)
            coords_normalized = coords - center
            
            # 归一化到 [-1, 1]
            max_dist = np.max(np.linalg.norm(coords_normalized, axis=1))
            if max_dist > 0:
                coords_normalized = coords_normalized / max_dist
            else:
                coords_normalized = coords.copy()
        else:
            coords_normalized = coords.copy()
        
        # 根据模型配置组合特征
        # extra_feature_channels = 3: [x, y, z, nx, ny, nz]
        # extra_feature_channels = 6: [x, y, z, r, g, b, nx, ny, nz]
        if self.extra_feature_channels == 3:
            # 格式: [x, y, z, nx, ny, nz] (仅归一化坐标)
            features = np.concatenate([coords, coords_normalized], axis=1)
        elif self.extra_feature_channels == 6:
            # 格式: [x, y, z, r, g, b, nx, ny, nz] (RGB + 归一化坐标)
            if has_color:
                features = np.concatenate([coords, colors, coords_normalized], axis=1)
            else:
                # 如果没有颜色，用灰色填充
                gray_colors = np.ones((coords.shape[0], 3), dtype=np.float32) * 0.5
                features = np.concatenate([coords, gray_colors, coords_normalized], axis=1)
        else:
            # 其他情况，使用归一化坐标
            features = np.concatenate([coords, coords_normalized[:, :self.extra_feature_channels]], axis=1)
        
        # 转换为 PyTorch 张量: (feature_dim, num_points)
        features = torch.from_numpy(features.T).float()
        
        # 添加批次维度: (1, feature_dim, num_points)
        features = features.unsqueeze(0)
        
        return features.to(self.device)
    
    def predict(
        self,
        pointcloud: np.ndarray,
        return_probabilities: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        对点云进行语义分割预测
        
        Args:
            pointcloud: 输入点云 (N, 3) 或 (N, 3+C)
            return_probabilities: 是否返回类别概率
            
        Returns:
            predictions: 预测的类别标签 (num_points,)
            probabilities: 类别概率 (num_points, 13)，如果 return_probabilities=True
        """
        # 预处理点云
        features = self.preprocess_pointcloud(pointcloud)
        
        # 推理
        with torch.no_grad():
            logits = self.model(features)  # (1, 13, num_points)
        
        # 获取预测结果
        probabilities = torch.softmax(logits, dim=1)  # (1, 13, num_points)
        predictions = torch.argmax(probabilities, dim=1)  # (1, num_points)
        
        # 转换为 numpy 数组
        predictions = predictions.squeeze(0).cpu().numpy()  # (num_points,)
        
        if return_probabilities:
            probabilities = probabilities.squeeze(0).permute(1, 0).cpu().numpy()  # (num_points, 13)
            return predictions, probabilities
        else:
            return predictions, None
    
    def predict_batch(
        self,
        pointclouds: list,
        return_probabilities: bool = False
    ) -> Tuple[list, Optional[list]]:
        """
        批量预测多个点云
        
        Args:
            pointclouds: 点云列表，每个点云为 (N, 3) 或 (N, 3+C)
            return_probabilities: 是否返回类别概率
            
        Returns:
            predictions_list: 预测结果列表
            probabilities_list: 概率列表 (如果 return_probabilities=True)
        """
        predictions_list = []
        probabilities_list = [] if return_probabilities else None
        
        for pc in pointclouds:
            pred, prob = self.predict(pc, return_probabilities)
            predictions_list.append(pred)
            if return_probabilities:
                probabilities_list.append(prob)
        
        return predictions_list, probabilities_list
    
    def colorize_predictions(
        self,
        predictions: np.ndarray
    ) -> np.ndarray:
        """
        将预测标签转换为颜色
        
        Args:
            predictions: 预测的类别标签 (num_points,)
            
        Returns:
            colors: RGB 颜色数组 (num_points, 3)
        """
        return self.CLASS_COLORS[predictions]
    
    def get_label_name(self, label_id: int) -> str:
        """获取类别名称"""
        if 0 <= label_id < len(self.CLASS_LABELS):
            return self.CLASS_LABELS[label_id]
        return f"unknown_{label_id}"


def main():
    """主函数：演示如何使用 PVCNN 推理器"""
    parser = argparse.ArgumentParser(description='PVCNN 语义分割推理')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/home/lhy/testPvcnnWithIsaacsim/pvcnn/runs/s3dis.pvcnn.area5.c1/best.pth.tar',
        help='模型权重文件路径'
    )
    parser.add_argument('--num_points', type=int, default=4096, help='输入点数')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    parser.add_argument('--test', action='store_true', help='运行测试')
    
    args = parser.parse_args()
    
    # 创建推理器
    inferencer = PVCNNInference(
        checkpoint_path=args.checkpoint,
        num_points=args.num_points,
        device=args.device
    )
    
    if args.test:
        print("\n[测试模式] 生成随机点云进行测试...")
        
        # 生成测试点云 (模拟一个房间)
        num_test_points = 8000
        test_pointcloud = np.random.randn(num_test_points, 3).astype(np.float32)
        test_pointcloud[:, 0] *= 5  # x 方向
        test_pointcloud[:, 1] *= 5  # y 方向
        test_pointcloud[:, 2] *= 3  # z 方向 (房间高度)
        
        print(f"测试点云形状: {test_pointcloud.shape}")
        
        # 进行预测
        predictions, probabilities = inferencer.predict(
            test_pointcloud,
            return_probabilities=True
        )
        
        print(f"\n预测结果形状: {predictions.shape}")
        print(f"概率形状: {probabilities.shape}")
        
        # 统计各类别的点数
        unique, counts = np.unique(predictions, return_counts=True)
        print("\n类别分布:")
        for label_id, count in zip(unique, counts):
            label_name = inferencer.get_label_name(label_id)
            percentage = count / len(predictions) * 100
            print(f"  {label_name:12s}: {count:4d} 点 ({percentage:5.2f}%)")
        
        # 获取颜色
        colors = inferencer.colorize_predictions(predictions)
        print(f"\n颜色数组形状: {colors.shape}")
        
        print("\n✓ 测试成功!")
    else:
        print("\n使用方法:")
        print("1. 在 Isaac Lab 环境中，从 LiDAR 传感器获取点云数据")
        print("2. 调用 inferencer.predict(pointcloud) 进行预测")
        print("3. 使用 inferencer.colorize_predictions() 为点云着色")
        print("\n运行 --test 参数查看测试示例")


if __name__ == '__main__':
    main()
