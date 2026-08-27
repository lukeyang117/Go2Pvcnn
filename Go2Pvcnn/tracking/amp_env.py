"""Isaac environment adapter that exposes AMP payloads without new observations."""

from __future__ import annotations

import torch

from tracking.env import ParallelismTrackingEnv
from tracking.managers.parallelism_amp_manager import ParallelismAmpManager
from tracking.managers.parallelism_reference_manager import get_parallelism_reference_manager


def _frame_from_robot(env) -> torch.Tensor:
    robot = env.scene["robot"]
    root_pos = robot.data.root_pos_w
    joint_pos = robot.data.joint_pos
    frame = torch.zeros((env.num_envs, 39), dtype=torch.float32, device=env.device)
    frame[:, : min(12, joint_pos.shape[-1])] = joint_pos[:, :12]
    frame[:, 24:27] = root_pos
    quat = robot.data.root_quat_w
    w, x, y, z = quat.unbind(-1)
    frame[:, 27:33] = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z + w * y), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 2 * (x * z - w * y)), dim=-1)
    frame[:, 33:36] = robot.data.root_lin_vel_w
    frame[:, 36:39] = robot.data.root_ang_vel_w
    return frame


class ParallelismAmpEnv(ParallelismTrackingEnv):
    """Attach a batched AMP manager and append payloads to step infos."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.cfg
        self.amp_manager = ParallelismAmpManager(self.num_envs, self.device, getattr(cfg, "amp_window_frames", 24), getattr(cfg, "amp_dt", 0.02))
        self._amp_previous_agent = _frame_from_robot(self)
        self._amp_previous_expert = self._amp_previous_agent.clone()

    def step(self, action):
        reference_manager = get_parallelism_reference_manager(self)
        reference_manager.prepare_step_reference()
        agent_start = _frame_from_robot(self)
        result = super().step(action)
        agent_target = _frame_from_robot(self)
        expert_start = self._amp_previous_expert
        expert_target = expert_start.clone()
        expert_target[:, :12] = reference_manager.step_joint_pos[:, :12]
        expert_target[:, 24:27] = reference_manager.step_root_pos_w
        valid = torch.as_tensor(reference_manager.step_plan_valid, device=self.device, dtype=torch.bool)
        payload = self.amp_manager.push_transition(agent_start, agent_target, expert_start, expert_target, valid)
        self._amp_previous_agent = agent_target
        self._amp_previous_expert = expert_target
        obs, rewards, dones, truncated, infos = result
        infos.setdefault("amp", {})
        infos["amp"].update({"expert_window": payload.expert_window, "agent_window": payload.agent_window, "amp_active": payload.amp_active, "history_ratio": payload.history_ratio})
        return obs, rewards, dones, truncated, infos

