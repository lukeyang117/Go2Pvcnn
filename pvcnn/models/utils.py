import functools

import torch.nn as nn

from modules import SharedMLP, PVConv, PointNetSAModule, PointNetAModule, PointNetFPModule

__all__ = ['create_mlp_components', 'create_pointnet_components',
           'create_pointnet2_sa_components', 'create_pointnet2_fp_modules']


def _linear_bn_relu(in_channels, out_channels):
    return nn.Sequential(nn.Linear(in_channels, out_channels), nn.BatchNorm1d(out_channels), nn.ReLU(True))


def create_mlp_components(in_channels, out_channels, classifier=False, dim=2, width_multiplier=1):
    r = width_multiplier

    if dim == 1:
        block = _linear_bn_relu
    else:
        block = SharedMLP
    if not isinstance(out_channels, (list, tuple)):
        out_channels = [out_channels]
    if len(out_channels) == 0 or (len(out_channels) == 1 and out_channels[0] is None):
        return nn.Sequential(), in_channels, in_channels

    layers = []
    for oc in out_channels[:-1]:
        if oc < 1:
            layers.append(nn.Dropout(oc))
        else:
            oc = int(r * oc)
            layers.append(block(in_channels, oc))
            in_channels = oc
    if dim == 1:
        if classifier:
            layers.append(nn.Linear(in_channels, out_channels[-1]))
        else:
            layers.append(_linear_bn_relu(in_channels, int(r * out_channels[-1])))
    else:
        if classifier:
            #kenel_size=1的卷积相当于线性变换
            layers.append(nn.Conv1d(in_channels, out_channels[-1], 1))
        else:
            layers.append(SharedMLP(in_channels, int(r * out_channels[-1])))
    # if不包括layers
    return layers, out_channels[-1] if classifier else int(r * out_channels[-1])


def create_pointnet_components(blocks, in_channels, with_se=False, normalize=True, eps=0,
                               width_multiplier=1, voxel_resolution_multiplier=1):
    """
    创建 PointNet 风格的网络组件（用于分类/分割任务）
    
    参数:
        blocks: 网络层次结构定义，格式为 [(out_channels, num_blocks, voxel_resolution), ...]
               - out_channels: 输出特征通道数
               - num_blocks: 该层重复的块数量
               - voxel_resolution: 体素分辨率（None 表示纯 MLP，否则使用 PVConv）
        in_channels: 输入特征通道数
        with_se: 是否使用 Squeeze-and-Excitation 模块
        normalize: 是否在体素化时归一化坐标
        eps: 体素化时的数值稳定性参数
        width_multiplier: 宽度乘数（用于缩放通道数）
        voxel_resolution_multiplier: 体素分辨率乘数
    
    返回:
        layers: 网络层列表
        in_channels: 最后一层的输出通道数
        concat_channels: 所有层输出通道数的总和（用于特征融合/跳跃连接）
    """
    # r: 宽度缩放因子, vr: 体素分辨率缩放因子
    r, vr = width_multiplier, voxel_resolution_multiplier
    
    # layers: 存储所有网络层
    # concat_channels: 累计所有层的输出通道数（用于后续的特征拼接）
    layers, concat_channels = [], 0
    
    # 遍历每个块配置
    for out_channels, num_blocks, voxel_resolution in blocks:
        # 应用宽度乘数缩放输出通道数
        out_channels = int(r * out_channels)
        
        # 根据 voxel_resolution 决定使用哪种块类型
        if voxel_resolution is None:
            # voxel_resolution=None: 使用纯 MLP (1x1 卷积)
            # 适用于高层特征，不需要体素化的 3D 卷积
            block = SharedMLP
        else:
            # voxel_resolution 有值: 使用 PVConv (Point-Voxel Convolution)
            # 结合点云特征和体素化 3D 卷积，捕获局部几何结构
            # functools.partial 预设参数，创建可复用的块工厂函数
            block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution),
                                      with_se=with_se, normalize=normalize, eps=eps)
        
        # 重复添加 num_blocks 个相同类型的块
        for _ in range(num_blocks):
            # 创建块: 输入通道 -> 输出通道
            layers.append(block(in_channels, out_channels))
            
            # 更新输入通道数为当前输出通道数（为下一层做准备）
            in_channels = out_channels
            
            # 累加输出通道数（用于后续的多尺度特征拼接）
            # 例如: 如果有 [64, 128, 256] 三层，concat_channels = 64+128+256 = 448
            concat_channels += out_channels
    
    # 返回三个值:
    # 1. layers: 所有网络层的列表
    # 2. in_channels: 最后一层的输出通道数（用于后续层的输入）
    # 3. concat_channels: 所有层输出通道数的总和（用于 skip connection/特征金字塔）
    return layers, in_channels, concat_channels


def create_pointnet2_sa_components(sa_blocks, extra_feature_channels, with_se=False, normalize=True, eps=0,
                                   width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier
    in_channels = extra_feature_channels + 3

    sa_layers, sa_in_channels = [], []
    for conv_configs, sa_configs in sa_blocks:
        sa_in_channels.append(in_channels)
        sa_blocks = []
        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            if voxel_resolution is None:
                block = SharedMLP
            else:
                block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution),
                                          with_se=with_se, normalize=normalize, eps=eps)
            for _ in range(num_blocks):
                sa_blocks.append(block(in_channels, out_channels))
                in_channels = out_channels
            extra_feature_channels = in_channels
        num_centers, radius, num_neighbors, out_channels = sa_configs
        _out_channels = []
        for oc in out_channels:
            if isinstance(oc, (list, tuple)):
                _out_channels.append([int(r * _oc) for _oc in oc])
            else:
                _out_channels.append(int(r * oc))
        out_channels = _out_channels
        if num_centers is None:
            block = PointNetAModule
        else:
            block = functools.partial(PointNetSAModule, num_centers=num_centers, radius=radius,
                                      num_neighbors=num_neighbors)
        sa_blocks.append(block(in_channels=extra_feature_channels, out_channels=out_channels,
                               include_coordinates=True))
        in_channels = extra_feature_channels = sa_blocks[-1].out_channels
        if len(sa_blocks) == 1:
            sa_layers.append(sa_blocks[0])
        else:
            sa_layers.append(nn.Sequential(*sa_blocks))

    return sa_layers, sa_in_channels, in_channels, 1 if num_centers is None else num_centers


def create_pointnet2_fp_modules(fp_blocks, in_channels, sa_in_channels, with_se=False, normalize=True, eps=0,
                                width_multiplier=1, voxel_resolution_multiplier=1):
    r, vr = width_multiplier, voxel_resolution_multiplier

    fp_layers = []
    for fp_idx, (fp_configs, conv_configs) in enumerate(fp_blocks):
        fp_blocks = []
        out_channels = tuple(int(r * oc) for oc in fp_configs)
        fp_blocks.append(
            PointNetFPModule(in_channels=in_channels + sa_in_channels[-1 - fp_idx], out_channels=out_channels)
        )
        in_channels = out_channels[-1]
        if conv_configs is not None:
            out_channels, num_blocks, voxel_resolution = conv_configs
            out_channels = int(r * out_channels)
            if voxel_resolution is None:
                block = SharedMLP
            else:
                block = functools.partial(PVConv, kernel_size=3, resolution=int(vr * voxel_resolution),
                                          with_se=with_se, normalize=normalize, eps=eps)
            for _ in range(num_blocks):
                fp_blocks.append(block(in_channels, out_channels))
                in_channels = out_channels
        if len(fp_blocks) == 1:
            fp_layers.append(fp_blocks[0])
        else:
            fp_layers.append(nn.Sequential(*fp_blocks))

    return fp_layers, in_channels
