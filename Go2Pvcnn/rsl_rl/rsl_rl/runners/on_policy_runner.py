#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import statistics
import time
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter

import rsl_rl
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCritic, ActorCriticCNN, ActorCriticRecurrent, EmpiricalNormalization
from rsl_rl.utils import store_code_state


class OnPolicyRunner:
    """On-policy runner for training and evaluation."""

    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        obs, extras = self.env.get_observations()
        num_obs = obs.shape[1]
        if "critic" in extras["observations"]:
            num_critic_obs = extras["observations"]["critic"].shape[1]
        else:
            num_critic_obs = num_obs
        actor_critic_class = eval(self.policy_cfg.pop("class_name"))  # ActorCritic
        actor_critic: ActorCritic | ActorCriticRecurrent = actor_critic_class(
            num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        alg_class = eval(self.alg_cfg.pop("class_name"))  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"] #这个参数决定什么时候更新一次策略
        self.save_interval = self.cfg["save_interval"] #保存间隔的权重
        self.empirical_normalization = self.cfg["empirical_normalization"] #什么是经验归一化 ？维护Running Mean和Variance
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[num_obs], until=1.0e8).to(self.device)
            self.critic_obs_normalizer = EmpiricalNormalization(shape=[num_critic_obs], until=1.0e8).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity()  # no normalization
            self.critic_obs_normalizer = torch.nn.Identity()  # no normalization
        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [num_obs],
            [num_critic_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [rsl_rl.__file__]
        
        # Synchronous PVCNN training
        self.enable_pvcnn_sync_training = self.cfg.get("enable_pvcnn_sync_training", False)
        self.pvcnn_train_interval = self.cfg.get("pvcnn_train_interval", 10)  # Train PVCNN every N iterations
        self.pvcnn_train_epochs = self.cfg.get("pvcnn_train_epochs", 5)  # PVCNN training epochs per update

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """
        主训练循环函数。
        
        参数:
            num_learning_iterations: 训练迭代次数（每次迭代包含数据收集→策略更新→可选的PVCNN训练）
            init_at_random_ep_len: 是否在随机episode长度处开始训练（用于curriculum learning）
        """
        # ========================================
        # 初始化日志记录器
        # ========================================
        if self.log_dir is not None and self.writer is None:  # 如果有日志目录且writer未初始化
            # 启动Tensorboard、Neptune或WandB日志记录器，默认使用Tensorboard
            self.logger_type = self.cfg.get("logger", "tensorboard")  # 获取日志类型配置
            self.logger_type = self.logger_type.lower()  # 转换为小写

            if self.logger_type == "neptune":  # Neptune云端实验追踪
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "wandb":  # Weights & Biases云端实验追踪
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                self.writer.log_config(self.env.cfg, self.cfg, self.alg_cfg, self.policy_cfg)
            elif self.logger_type == "tensorboard":  # TensorBoard本地日志记录
                self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise AssertionError("logger type not found")

        # ========================================
        # 初始化训练环境
        # ========================================
        if init_at_random_ep_len:  # 如果启用随机episode长度初始化
            # 将每个环境的episode长度设置为随机值（用于课程学习，避免所有环境同时reset）
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        
        # 获取初始观测值
        obs, extras = self.env.get_observations()  # obs: (num_envs, obs_dim)
        critic_obs = extras["observations"].get("critic", obs)  # critic观测（可能包含额外信息）
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)  # 移动到GPU/CPU
        self.train_mode()  # 切换到训练模式（启用dropout、batchnorm训练行为等）
        

        # ========================================
        # 初始化训练统计变量
        # ========================================
        ep_infos = []  # 存储episode信息（每个episode结束时的统计数据）
        rewbuffer = deque(maxlen=100)  # 最近100个episode的奖励，用于计算平均奖励
        lenbuffer = deque(maxlen=100)  # 最近100个episode的长度，用于计算平均长度
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 当前episode累积奖励
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)  # 当前episode长度

        # ========================================
        # 主训练循环
        # ========================================
        start_iter = self.current_learning_iteration  # 起始迭代次数（用于resume训练）
        tot_iter = start_iter + num_learning_iterations  # 总迭代次数 = 起始 + 新增迭代
        for it in range(start_iter, tot_iter):  # 迭代训练
            start = time.time()  # 记录数据收集开始时间
            
            # ========================================
            # 阶段1: Rollout（数据收集）
            # ========================================
            with torch.inference_mode():  # 推理模式，不计算梯度（节省内存和计算）
                for i in range(self.num_steps_per_env):  # 每个环境收集num_steps_per_env步数据
                    # 使用当前策略选择动作
                    actions = self.alg.act(obs, critic_obs, env=self.env)  # (num_envs, action_dim)
                    
                    # 执行动作，获取下一步观测、奖励、结束标志和额外信息
                    obs, rewards, dones, infos = self.env.step(actions)
                    
                    # 归一化观测值（使用running mean/std）
                    obs = self.obs_normalizer(obs)
                    
                    # 获取critic观测值（可能包含特权信息，如真实状态）
                    if "critic" in infos["observations"]:
                        critic_obs = self.critic_obs_normalizer(infos["observations"]["critic"])
                    else:
                        critic_obs = obs  # 如果没有特权信息，使用普通观测
                    
                    # 将所有数据移动到训练设备（GPU/CPU）
                    obs, critic_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    
                    # 将数据存储到rollout buffer（用于后续训练）
                    self.alg.process_env_step(rewards, dones, infos)

                    # ========================================
                    # 日志记录（episode统计）
                    # ========================================
                    if self.log_dir is not None:
                        # 收集episode信息（当episode结束时会有"episode"或"log"字段）
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        elif "log" in infos:
                            ep_infos.append(infos["log"])
                        
                        # 累积奖励和episode长度
                        cur_reward_sum += rewards  # 每步累加奖励 shape: (num_envs,)
                        cur_episode_length += 1  # 每步增加长度
                        
                        # 找到所有完成的episode（dones > 0）
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        
                        # 将完成的episode的统计数据添加到buffer
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        
                        # 重置已完成episode的统计变量
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start  # 计算数据收集耗时

                # ========================================
                # 阶段2: 计算returns（优势函数和回报）
                # ========================================
                start = stop
                self.alg.compute_returns(critic_obs)  # 使用GAE计算优势函数和回报值

            # ========================================
            # 阶段3: 策略更新（PPO update）
            # ========================================
            mean_value_loss, mean_surrogate_loss = self.alg.update()  # PPO策略更新
            stop = time.time()
            learn_time = stop - start  # 计算策略更新耗时
            
            # ========================================
            # 阶段4: PVCNN同步训练（可选）
            # ========================================
            pvcnn_loss = 0.0  # 初始化PVCNN loss
            if self.enable_pvcnn_sync_training and (it + 1) % self.pvcnn_train_interval == 0:
                # 每pvcnn_train_interval次PPO迭代后训练一次PVCNN
                pvcnn_start = time.time()
                # 使用收集到的点云数据训练PVCNN语义分割模型
                pvcnn_loss = self.alg.train_pvcnn_on_collected_data(num_epochs=self.pvcnn_train_epochs)
                pvcnn_time = time.time() - pvcnn_start
                print(f"[PVCNN Training] Iter {it}: Loss={pvcnn_loss:.4f}, Time={pvcnn_time:.2f}s")
                
                # 记录PVCNN loss到TensorBoard
                if self.log_dir is not None:
                    self.writer.add_scalar("Loss/pvcnn_semantic", pvcnn_loss, it)
            
            # ========================================
            # 阶段5: 日志记录和模型保存
            # ========================================
            self.current_learning_iteration = it  # 更新当前迭代次数
            
            if self.log_dir is not None:
                self.log(locals())  # 记录所有统计数据到TensorBoard
            
            if it % self.save_interval == 0:  # 每save_interval次迭代保存一次模型
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            
            ep_infos.clear()  # 清空episode信息列表，准备下一次迭代

        # ========================================
        # 训练结束，保存最终模型
        # ========================================
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if key not in ep_info:
                        continue
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                # log to logger and terminal
                if "/" in key:
                    self.writer.add_scalar(key, value, locs["it"])
                    ep_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
                else:
                    self.writer.add_scalar("Episode/" + key, value, locs["it"])
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            if self.logger_type != "wandb":  # wandb does not support non-integer x-axis logging
                self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
                self.writer.add_scalar(
                    "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time
                )

        str = f" \033[1m Learning iteration {locs['it']}/{locs['tot_iter']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                            locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["critic_obs_norm_state_dict"] = self.critic_obs_normalizer.state_dict()
        
        # 保存PVCNN模型和优化器 (如果存在)
        if hasattr(self.alg, 'pvcnn_model') and self.alg.pvcnn_model is not None:
            saved_dict["pvcnn_state_dict"] = self.alg.pvcnn_model.state_dict()
            if hasattr(self.alg, 'pvcnn_optimizer') and self.alg.pvcnn_optimizer is not None:
                saved_dict["pvcnn_optimizer_state_dict"] = self.alg.pvcnn_optimizer.state_dict()
        
        torch.save(saved_dict, path)

        # Upload model to external logging service
        if self.logger_type in ["neptune", "wandb"]:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        state_dict = loaded_dict["model_state_dict"]
        state_dict.pop("std", None)  # Remove the 'std' key if it exists
        self.alg.actor_critic.load_state_dict(state_dict, strict=False)
        if self.empirical_normalization:
            self.obs_normalizer.load_state_dict(loaded_dict["obs_norm_state_dict"])
            self.critic_obs_normalizer.load_state_dict(loaded_dict["critic_obs_norm_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        
        # 加载PVCNN模型和优化器 (如果存在)
        if "pvcnn_state_dict" in loaded_dict:
            if hasattr(self.alg, 'pvcnn_model') and self.alg.pvcnn_model is not None:
                self.alg.pvcnn_model.load_state_dict(loaded_dict["pvcnn_state_dict"])
                print("[Checkpoint] PVCNN model loaded successfully")
            else:
                print("[Checkpoint] WARNING: PVCNN state_dict found but no pvcnn_model in algorithm")
        
        if "pvcnn_optimizer_state_dict" in loaded_dict and load_optimizer:
            if hasattr(self.alg, 'pvcnn_optimizer') and self.alg.pvcnn_optimizer is not None:
                self.alg.pvcnn_optimizer.load_state_dict(loaded_dict["pvcnn_optimizer_state_dict"])
                print("[Checkpoint] PVCNN optimizer loaded successfully")
        
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.eval_mode()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        policy = self.alg.actor_critic.act_inference
        if self.cfg["empirical_normalization"]:
            if device is not None:
                self.obs_normalizer.to(device)
            policy = lambda x: self.alg.actor_critic.act_inference(self.obs_normalizer(x))  # noqa: E731
        return policy

    def train_mode(self):
        self.alg.actor_critic.train()
        if self.empirical_normalization:
            self.obs_normalizer.train()
            self.critic_obs_normalizer.train()

    def eval_mode(self):
        self.alg.actor_critic.eval()
        if self.empirical_normalization:
            self.obs_normalizer.eval()
            self.critic_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
