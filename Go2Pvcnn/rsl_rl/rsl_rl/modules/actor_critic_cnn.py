#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

"""Actor-Critic network with 2D CNN encoder for cost map input."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


def get_activation(act_name):
    """Get activation function by name."""
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.CReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("Invalid activation function!")
        return None


class ActorCriticCNN(nn.Module):
    """
    Actor-Critic网络，整合2D CNN编码器处理代价地图
    
    输入:
        - observations: (batch, num_obs) - 扁平化观测 [proprio..., cost_map_flat...]
    
    输出:
        - actions: (batch, num_actions) - 动作输出
        - values: (batch, 1) - 价值估计
    """
    
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        cost_map_channels=2,  # 双通道: cost_map + height_map
        cost_map_size=16,
        cnn_channels=[32, 64, 128],
        cnn_feature_dim=256,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        use_cost_map=True,
        **kwargs,
    ):
        """
        初始化Actor-Critic CNN网络
        
        Args:
            num_actor_obs: 总观测维度 (包含双通道map的扁平化维度)
            num_critic_obs: Critic观测维度
            num_actions: 动作维度
            cost_map_channels: 代价地图通道数 (默认2: cost_map + height_map)
            cost_map_size: 代价地图尺寸 (默认16×16)
            cnn_channels: CNN编码器通道数列表
            cnn_feature_dim: CNN特征降维后的维度
            actor_hidden_dims: Actor MLP隐藏层维度
            critic_hidden_dims: Critic MLP隐藏层维度
            activation: 激活函数
            init_noise_std: 初始噪声标准差
            use_cost_map: 是否使用代价地图 (如果False，退化为纯MLP)
        """
        actor_cnn_cfg = kwargs.pop("actor_cnn_cfg", None)
        critic_cnn_cfg = kwargs.pop("critic_cnn_cfg", None)
        kwargs.pop("noise_std_type", None)
        kwargs.pop("state_dependent_std", None)
        if kwargs:
            print(
                "ActorCriticCNN.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()
        
        self.use_cost_map = use_cost_map
        self.cost_map_size = cost_map_size
        self.cost_map_channels = cost_map_channels
        self.cost_map_dim = cost_map_channels * cost_map_size * cost_map_size  
        activation_fn = get_activation(activation)
        self.group_cnn_mode = actor_cnn_cfg is not None or critic_cnn_cfg is not None

        if self.group_cnn_mode:
            self.actor_cnns = nn.ModuleDict(
                {
                    "policy_elevation_semantic_map": self._build_group_cnn(
                        actor_cnn_cfg or {},
                        in_channels=cost_map_channels,
                        activation_name=activation,
                    )
                }
            )
            self.critic_cnns = nn.ModuleDict(
                {
                    "critic_elevation_semantic_map": self._build_group_cnn(
                        critic_cnn_cfg or {},
                        in_channels=cost_map_channels,
                        activation_name=activation,
                    )
                }
            )
            cnn_feature_dim = self._group_cnn_flatten_dim(actor_cnn_cfg or {}, cost_map_size=cost_map_size)
            critic_cnn_feature_dim = self._group_cnn_flatten_dim(critic_cnn_cfg or {}, cost_map_size=cost_map_size)
            proprio_dim = num_actor_obs - self.cost_map_dim
            critic_proprio_dim = num_critic_obs - self.cost_map_dim
            actor_input_dim = proprio_dim + cnn_feature_dim
            critic_input_dim = critic_proprio_dim + critic_cnn_feature_dim
            self.cnn_encoder = None
        else:
            self.actor_cnns = nn.ModuleDict()
            self.critic_cnns = nn.ModuleDict()
        
        # ========================================
        # CNN编码器: 处理代价地图
        # ========================================
        if use_cost_map and not self.group_cnn_mode:
            cnn_layers = []
            in_channels = cost_map_channels
            current_size = cost_map_size
            
            for out_channels in cnn_channels:
                cnn_layers.append(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
                )
                cnn_layers.append(activation_fn.__class__())
                in_channels = out_channels
                current_size = current_size // 2
            
            # Flatten层
            cnn_layers.append(nn.Flatten())
            
            # 计算flatten后的维度，从特征图变成一维tensor
            flatten_dim = cnn_channels[-1] * (current_size ** 2)
            
            # 降维到cnn_feature_dim
            cnn_layers.append(nn.Linear(flatten_dim, cnn_feature_dim))
            cnn_layers.append(activation_fn.__class__())
            
            self.cnn_encoder = nn.Sequential(*cnn_layers)
            
            print(f"[ActorCriticCNN] CNN Encoder initialized:")
            print(f"  - Input: ({cost_map_channels}, {cost_map_size}, {cost_map_size})")
            print(f"  - CNN channels: {cnn_channels}")
            print(f"  - Flatten dim: {flatten_dim}")
            print(f"  - Output feature dim: {cnn_feature_dim}")
        elif not self.group_cnn_mode:
            self.cnn_encoder = None
            cnn_feature_dim = 0
            print(f"[ActorCriticCNN] CNN Encoder disabled (use_cost_map=False)")
        
        # ========================================
        # Actor网络: CNN特征 + 本体感知 → 动作
        # ========================================
        # 计算本体感知维度 (总观测维度 - cost_map维度)
        if self.group_cnn_mode:
            pass
        elif use_cost_map:
            proprio_dim = num_actor_obs - self.cost_map_dim
        else:
            proprio_dim = num_actor_obs
        
        if not self.group_cnn_mode:
            actor_input_dim = proprio_dim + cnn_feature_dim
        actor_layers = []
        actor_layers.append(nn.Linear(actor_input_dim, actor_hidden_dims[0]))
        actor_layers.append(activation_fn.__class__())
        
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[layer_index], num_actions))
            else:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[layer_index], actor_hidden_dims[layer_index + 1])
                )
                actor_layers.append(activation_fn.__class__())
        
        self.actor = nn.Sequential(*actor_layers)
        
        # ========================================
        # Critic网络: CNN特征 + 本体感知 → 价值
        # ========================================
        if self.group_cnn_mode:
            pass
        elif use_cost_map:
            critic_proprio_dim = num_critic_obs - self.cost_map_dim
        else:
            critic_proprio_dim = num_critic_obs
        
        if not self.group_cnn_mode:
            critic_input_dim = critic_proprio_dim + cnn_feature_dim
        critic_layers = []
        critic_layers.append(nn.Linear(critic_input_dim, critic_hidden_dims[0]))
        critic_layers.append(activation_fn.__class__())
        
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(
                    nn.Linear(critic_hidden_dims[layer_index], critic_hidden_dims[layer_index + 1])
                )
                critic_layers.append(activation_fn.__class__())
        
        self.critic = nn.Sequential(*critic_layers)
        
        print(f"[ActorCriticCNN] Network initialized:")
        print(f"  - Actor input dim: {actor_input_dim} (proprio {proprio_dim} + CNN {cnn_feature_dim})")
        print(f"  - Critic input dim: {critic_input_dim}")
        print(f"  - Actions: {num_actions}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions)) #shape: (num_actions,)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    def _activation_from_name(name: str) -> nn.Module:
        activation = get_activation(str(name))
        if activation is None:
            raise ValueError(f"Unsupported activation: {name}")
        return activation

    @classmethod
    def _build_group_cnn(cls, cfg: dict, *, in_channels: int, activation_name: str) -> nn.Sequential:
        channels = list(cfg.get("output_channels", [32, 64]))
        kernels = list(cfg.get("kernel_size", [3] * len(channels)))
        max_pool = list(cfg.get("max_pool", [True] * len(channels)))
        activation = str(cfg.get("activation", activation_name))
        layers: list[nn.Module] = []
        current_channels = int(in_channels)
        for idx, out_channels in enumerate(channels):
            kernel_size = int(kernels[idx]) if idx < len(kernels) else 3
            layers.append(nn.Conv2d(current_channels, int(out_channels), kernel_size=kernel_size, stride=1, padding=1))
            layers.append(cls._activation_from_name(activation).__class__())
            if idx < len(max_pool) and bool(max_pool[idx]):
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_channels = int(out_channels)
        return nn.Sequential(*layers)

    @staticmethod
    def _group_cnn_flatten_dim(cfg: dict, *, cost_map_size: int) -> int:
        channels = list(cfg.get("output_channels", [32, 64]))
        max_pool = list(cfg.get("max_pool", [True] * len(channels)))
        current_size = int(cost_map_size)
        for idx, _ in enumerate(channels):
            if idx < len(max_pool) and bool(max_pool[idx]):
                current_size = current_size // 2
        return int(channels[-1]) * current_size * current_size

    @staticmethod
    def init_weights(sequential, scales):
        """Initialize weights using orthogonal initialization."""
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        """Reset for recurrent policies (not used here)."""
        pass

    def forward(self):
        """Forward pass not implemented - use act() or evaluate()."""
        raise NotImplementedError

    def _extract_features(self, observations):
        """
        从扁平化观测中提取CNN特征并与本体感知拼接
        
        Args:
            observations: (batch, num_obs) - 扁平化观测
                格式: [dual_channel_map_flat..., proprio_obs...]
                其中dual_channel_map_flat占前512维 (2*16*16)
        
        Returns:
            combined_features: (batch, proprio_dim + cnn_feature_dim)
        """
        batch_size = observations.shape[0]
        
        if self.group_cnn_mode:
            dual_channel_flat = observations[:, :self.cost_map_dim]
            proprio_obs = observations[:, self.cost_map_dim:]
            dual_channel_map = dual_channel_flat.reshape(
                batch_size, self.cost_map_channels, self.cost_map_size, self.cost_map_size
            )
            cnn_features = self.actor_cnns["policy_elevation_semantic_map"](dual_channel_map).reshape(batch_size, -1)
            combined = torch.cat([proprio_obs, cnn_features], dim=-1)
        elif self.use_cost_map:
            # 分离dual_channel_map和proprio观测
            # observations = [dual_channel_map_flat..., proprio_obs...]
            # dual_channel_map_flat在前面，因为在ObservationsCfg中pvcnn_with_costmap是第一个
            dual_channel_flat = observations[:, :self.cost_map_dim]
            proprio_obs = observations[:, self.cost_map_dim:] 
            
            # Reshape dual_channel_map: (batch, 512) -> (batch, 2, 16, 16)
            dual_channel_map = dual_channel_flat.reshape(
                batch_size, self.cost_map_channels, self.cost_map_size, self.cost_map_size
            )
            
            # 提取CNN特征
            cnn_features = self.cnn_encoder(dual_channel_map)  # (batch, cnn_feature_dim)
            
            # 拼接
            combined = torch.cat([proprio_obs, cnn_features], dim=-1)
        else:
            combined = observations
        
        return combined

    def _extract_critic_features(self, observations):
        if not self.group_cnn_mode:
            return self._extract_features(observations)
        batch_size = observations.shape[0]
        dual_channel_flat = observations[:, :self.cost_map_dim]
        proprio_obs = observations[:, self.cost_map_dim:]
        dual_channel_map = dual_channel_flat.reshape(
            batch_size, self.cost_map_channels, self.cost_map_size, self.cost_map_size
        )
        cnn_features = self.critic_cnns["critic_elevation_semantic_map"](dual_channel_map).reshape(batch_size, -1)
        return torch.cat([proprio_obs, cnn_features], dim=-1)

    def act(self, observations, masks=None, hidden_states=None):
        """
        根据观测选择动作
        
        Args:
            observations: (batch, num_obs) - 扁平化观测
            masks: Optional masks for recurrent policies (not used)
            hidden_states: Optional hidden states for recurrent policies (not used)
        
        Returns:
            actions: (batch, num_actions)
        """
        features = self._extract_features(observations)
        action_mean = self.actor(features)
        
        self.distribution = Normal(action_mean, action_mean * 0.0 + self.std)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        """Get log probability of actions under current distribution."""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        """推理模式 (确定性动作)"""
        features = self._extract_features(observations)
        action_mean = self.actor(features)
        return action_mean

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        """
        评估价值函数
        
        Args:
            critic_observations: (batch, num_critic_obs) - Critic观测
            masks: Optional masks for recurrent policies (not used)
            hidden_states: Optional hidden states for recurrent policies (not used)
        
        Returns:
            value: (batch, 1) - 价值估计
        """
        features = self._extract_critic_features(critic_observations)
        value = self.critic(features)
        return value

    @property
    def action_mean(self):
        """Mean of the action distribution."""
        return self.distribution.mean

    @property
    def action_std(self):
        """Standard deviation of the action distribution."""
        return self.distribution.stddev

    @property
    def entropy(self):
        """Entropy of the action distribution."""
        return self.distribution.entropy().sum(dim=-1)

    def clip_std(self, min=None, max=None):
        """Clip the standard deviation of the policy."""
        if min is not None:
            self.std.data = torch.clamp(self.std.data, min=min)
        if max is not None:
            self.std.data = torch.clamp(self.std.data, max=max)
