"""Optimizer loop for dense MPC residual variables."""

from __future__ import annotations

import torch
from torch import Tensor

from .config import MpcPlannerCfg
from .losses.registry import compute_total_loss
from .profiling import MpcProfile
from .types import MpcPlannerTerrain, MpcRobotState
from .variables import DecodedMpcTrajectory, MpcOptimizationVariables, decode_trajectory


def optimize_variables(
    nominal: dict[str, Tensor],
    variables: MpcOptimizationVariables,
    state: MpcRobotState,
    command: Tensor,
    terrain: MpcPlannerTerrain,
    cfg: MpcPlannerCfg,
    *,
    profile: MpcProfile | None = None,
) -> tuple[DecodedMpcTrajectory, Tensor, dict[str, Tensor], Tensor]:
    """Run dense gradient optimization and return decoded trajectory."""
    # RSL-RL rollout calls env.step() under torch.inference_mode(); local MPC
    # optimization must explicitly re-enable autograd for backward().
    with torch.inference_mode(False):
        runtime = cfg.runtime
        params = variables.parameters()
        if runtime.optimizer != "adam":
            raise ValueError(f"Unsupported optimizer {runtime.optimizer!r}; only 'adam' is supported")
        optimizer = torch.optim.Adam(params, lr=float(runtime.lr))
        finite_ok = torch.ones(
            nominal["root_pos"].shape[0],
            dtype=torch.bool,
            device=nominal["root_pos"].device,
        )
        per_env_total = torch.zeros(
            nominal["root_pos"].shape[0],
            dtype=nominal["root_pos"].dtype,
            device=nominal["root_pos"].device,
        )
        breakdown: dict[str, Tensor] = {}

        opt_t0 = profile.now() if profile is not None else 0.0
        with torch.enable_grad():
            for _ in range(int(runtime.optimize_steps)):
                iter_t0 = profile.now() if profile is not None else 0.0
                optimizer.zero_grad(set_to_none=True)
                if profile is not None:
                    profile.add_stage("optimizer.zero_grad", (profile.now() - iter_t0) * 1000.0)
                decode_t0 = profile.now() if profile is not None else 0.0
                decoded = decode_trajectory(nominal, variables, runtime)
                if profile is not None:
                    profile.add_stage("optimizer.decode", (profile.now() - decode_t0) * 1000.0)
                loss_t0 = profile.now() if profile is not None else 0.0
                total_scalar, per_env_total, breakdown = compute_total_loss(
                    decoded,
                    nominal,
                    state,
                    command,
                    terrain,
                    cfg,
                    profile=profile,
                )
                if profile is not None:
                    profile.add_stage("optimizer.loss", (profile.now() - loss_t0) * 1000.0)
                finite_ok = torch.logical_and(finite_ok, torch.isfinite(per_env_total))
                backward_t0 = profile.now() if profile is not None else 0.0
                total_scalar.backward()
                if profile is not None:
                    profile.add_stage("optimizer.backward", (profile.now() - backward_t0) * 1000.0)
                if runtime.grad_clip_norm > 0.0:
                    clip_t0 = profile.now() if profile is not None else 0.0
                    torch.nn.utils.clip_grad_norm_(params, max_norm=float(runtime.grad_clip_norm))
                    if profile is not None:
                        profile.add_stage("optimizer.grad_clip", (profile.now() - clip_t0) * 1000.0)
                step_t0 = profile.now() if profile is not None else 0.0
                optimizer.step()
                if profile is not None:
                    profile.add_stage("optimizer.step", (profile.now() - step_t0) * 1000.0)
                    profile.optimize_iters += 1
        if profile is not None:
            profile.add_stage("optimizer.loop", (profile.now() - opt_t0) * 1000.0)

        decode_t0 = profile.now() if profile is not None else 0.0
        decoded = decode_trajectory(nominal, variables, runtime)
        if profile is not None:
            profile.add_stage("optimizer.final_decode", (profile.now() - decode_t0) * 1000.0)
        loss_t0 = profile.now() if profile is not None else 0.0
        _, per_env_total, breakdown = compute_total_loss(decoded, nominal, state, command, terrain, cfg, profile=profile)
        if profile is not None:
            profile.add_stage("optimizer.final_loss", (profile.now() - loss_t0) * 1000.0)
            profile.add_stage("optimizer.total", (profile.now() - opt_t0) * 1000.0)
        finite_ok = torch.logical_and(finite_ok, torch.isfinite(per_env_total))
        detached_breakdown = {name: value.detach() for name, value in breakdown.items()}
        return decoded, per_env_total.detach(), detached_breakdown, finite_ok


__all__ = ["optimize_variables"]
