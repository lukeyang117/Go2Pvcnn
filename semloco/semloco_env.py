"""IsaacLab adapter for the isolated SemLoco task."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv

from .semantic_raibert_planner import SemanticRaibertPlanner
from .semloco_lifecycle import SemlocoTargetCache


class SemlocoEnv(ManagerBasedRLEnv):
    """ManagerBasedRLEnv with planner targets owned by each environment."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._semloco_planner = SemanticRaibertPlanner()
        self._semloco_cache = SemlocoTargetCache(self.num_envs, self.device)
        self._semloco_target_foothold_w = self._semloco_cache.target
        self._semloco_target_valid = self._semloco_cache.valid
        self._semloco_swing_mask = self._semloco_cache.swing
        self._semloco_swing_phase = torch.zeros(self.num_envs, 4, device=self.device)
        self._semloco_foot_ids = torch.as_tensor(
            self.scene["robot"].find_bodies(".*_foot")[0], device=self.device, dtype=torch.long
        )

    def step(self, action):
        # The manager computes rewards inside super().step. Targets are prepared
        # from the latest observation and remain locked until the next swing.
        self._semloco_prepare_targets()
        return super().step(action)

    def _semloco_prepare_targets(self) -> None:
        robot = self.scene["robot"]
        try:
            scanner = self.scene["semantic_height_scanner"]
        except KeyError:
            return
        command_term = getattr(getattr(self, "command_manager", None), "get_command", lambda _: None)("base_velocity")
        if scanner is None or command_term is None:
            return
        try:
            data = scanner.data
            height = getattr(data, "elevation_map", None)
            semantic = getattr(data, "semantic_map", None)
            if height is None or semantic is None:
                return
            foot = robot.data.body_pos_w.index_select(1, self._semloco_foot_ids)
            cycle = 0.5
            base_phase = torch.remainder(self.episode_length_buf * float(self.step_dt) / cycle, 1.0)
            offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=self.device)
            leg_phase = torch.remainder(base_phase[:, None] + offsets[None], 1.0)
            swing = leg_phase < 0.5
            self._semloco_swing_phase = torch.where(swing, leg_phase * 2.0, torch.zeros_like(leg_phase))
            ray_hits = getattr(data, "ray_hits_w", None)
            resolution = float(getattr(getattr(scanner.cfg, "pattern_cfg", None), "resolution", 0.01))
            if ray_hits is not None:
                side = int(round(float(ray_hits.shape[1]) ** 0.5))
                origin = ray_hits.reshape(self.num_envs, side, side, 3)[:, 0, 0]
            else:
                origin = robot.data.root_pos_w.clone()
                half = 0.5 * (height.shape[-1] - 1) * resolution
                origin[:, :2] -= half
            output = self._semloco_planner.plan(
                robot.data.root_pos_w,
                robot.data.root_quat_w,
                robot.data.root_lin_vel_b,
                robot.data.root_ang_vel_b,
                foot,
                command_term,
                self._semloco_swing_phase,
                {"height": height, "semantic": semantic, "origin": origin, "resolution": resolution},
            )
            self._semloco_cache.update_swing_targets(output, swing)
        except (AttributeError, RuntimeError, ValueError):
            # Missing scanner fields are a disabled planner, not a failed episode.
            return

    def _reset_idx(self, env_ids):
        self._semloco_cache.reset(env_ids)
        return super()._reset_idx(env_ids)


__all__ = ["SemlocoEnv"]
