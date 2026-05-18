"""Preset builders for train/eval/viewer MPC profiles."""

from __future__ import annotations

from .config import MpcPlannerCfg


def make_train_cfg() -> MpcPlannerCfg:
    cfg = MpcPlannerCfg(profile_name="train_4096")
    cfg.runtime.optimize_steps = 10
    cfg.runtime.replan_interval_steps = 50
    cfg.runtime.parallel_plan_batch_size = 256
    cfg.runtime.heavy_loss_stride = 2
    cfg.diagnostics.enabled = False
    return cfg


def make_eval_cfg() -> MpcPlannerCfg:
    cfg = MpcPlannerCfg(profile_name="eval_high_quality")
    cfg.runtime.optimize_steps = 48
    cfg.runtime.replan_interval_steps = 20
    cfg.runtime.parallel_plan_batch_size = 512
    cfg.runtime.heavy_loss_stride = 1
    cfg.diagnostics.enabled = True
    return cfg


def make_viewer_cfg() -> MpcPlannerCfg:
    cfg = make_eval_cfg()
    cfg.profile_name = "viewer_debug"
    cfg.diagnostics.emit_viewer_fields = True
    return cfg


__all__ = ["make_eval_cfg", "make_train_cfg", "make_viewer_cfg"]
