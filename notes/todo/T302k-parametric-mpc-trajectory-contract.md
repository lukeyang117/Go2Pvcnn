# T302k Parametric MPC Trajectory Contract

## Current State

- T302k is the new active implementation front for `extension/batch_mpc_planner`.
- Parent context:
  - [T302h](T302h-semantic-obstacle-jitter-reproduction.md) proved that low-small crossing and high/large avoidance can pass task metrics, but visual swing quality and planned-vs-realized foot mismatch remain.
  - [T302i](T302i-viewer-realized-foot-mismatch.md) localized the visible mismatch to unreachable Cartesian foot/touchdown targets after clamped IK.
  - [T302j](T302j-touchdown-endpoint-consistency.md) tried endpoint/export repairs and structured low-small touchdown coupling, but the remaining failure is structural: dense per-frame foot residuals are the wrong optimization contract.
- Approved design:
  - [../../docs/superpowers/specs/2026-05-26-mpc-parametric-trajectory-contract-design.md](../../docs/superpowers/specs/2026-05-26-mpc-parametric-trajectory-contract-design.md)
- Goal:
  - Replace dense foot residual optimization with parametric root/foot trajectory optimization.
  - Optimize touchdown `xy` only; derive touchdown `z` from `height_at(terrain, touchdown_xy)`.
  - Decode root and foot cubic Bezier curves, sample 25 frames for losses, solve clamped IK, and export FK-realized feet.
- 2026-05-26 14:50 update:
  - T302k.1-T302k.4 are implemented in the working tree.
  - `MpcRuntimeCfg.use_parametric_trajectory` now defaults to `True`; the old dense optimizer path is only kept as an explicit `False` legacy fallback for existing tests/callers.
  - Default `plan_segment` now decodes parametric root/foot curves, solves clamped IK, and exports FK-realized `foot_pos`.
  - Current parametric diagnostics expose `parametric_target_fk_error`; full terrain/semantic/gait sampled-frame loss replacement remains T302k.5.
- 2026-05-26 15:24 update:
  - T302k.5 local implementation added sampled-frame parametric losses and an Adam loop over parametric variables.
  - T302k.6 added `parametric_v1` probe entrypoints for low-small and semantic obstacle probes.
  - T302k.7 IsaacLab smoke on GPU3 shows low-small translation foot-over now succeeds with FK export error near zero, but high-small/large non-regression fails and touchdown endpoints still lag the realized swing.
- 2026-05-26 15:54 update:
  - T302k.9 added sampled loss keys for semantic avoidance, touchdown endpoint consistency, and swing foot height guarding.
  - Parametric decode now uses existing high/large semantic command shaping for root geometry while preserving original command progress loss.
  - Local focused suite passes, and low-small IsaacLab improves to `3/3` foot-over with no small contact/penetration.
  - High-small/large IsaacLab still fails acceptance: `semantic_task_violation_count=6/6`, `large_avoid_success_count=0`, max semantic penetration about `0.0217`.
  - Next step remains structural parametric curve constraint work, not old dense residual tuning.
- 2026-05-26 16:49 update:
  - T302k.10 fixed the parametric foot trajectory gait contract: foot XY/Z curves now use per-leg local swing phase from `swing_center`/`swing_width`, so diagonal trot pairs alternate instead of all four feet moving across the full horizon.
  - Added a regression test that first reproduced all four feet moving together, then passed after local phase gating.
  - Local focused suite passes, and GPU3 low-small smoke remains `3/3` foot-over with no small stance/touchdown/penetration.
  - Remaining open issue: high-small/large rolling acceptance still needs root nominal acceleration/bias and semantic/touchdown constraints.
- 2026-05-26 17:13 update:
  - T302k.11 changed the parametric replan output contract for the initial frame: `result.foot_pos[:, 0]` now comes directly from current IsaacLab `state.foot_pos`, matching the already-current root/joint frame0 anchors.
  - Frame1+ remains FK-realized from the clamped IK joint sequence, so the planner still exports physically realized future feet while avoiding an initial discontinuity when IsaacLab returns feet that differ from FK of the clamped joint anchor.
  - Added regressions for parametric decode starting from current foot positions and `plan_segment` preserving frame0 foot state.
  - Local focused suite passes: `227 passed, 1 warning`.
- 2026-05-26 17:17 update:
  - IsaacLab GPU3 low-small smoke confirms T302k.11 on real rollout: `max_replan_initial_foot_error=0.0` across forward, mixed lateral+yaw, and pure-yaw commands.
  - The same run exposes the next issue: `max_replan_initial_touchdown_to_current_foot_error=0.4467m`, so planned touchdown markers still do not represent the current stance/current foot at replan boundaries.
  - FK future export remains tight (`max_terminal_planned_vs_fk_foot_error=3.8e-6m`), but touchdown IK/FK mismatch remains high (`max_touchdown_ik_fk_error=0.6610m`).
- 2026-05-26 17:57 update:
  - T302k.8 removed the obsolete dense residual MPC path from source: `nominal.py`, `optimizer.py`, `variables.py`, and `losses/registry.py` are deleted.
  - `plan_segment()` is now parametric-only; `MpcRuntimeCfg.use_parametric_trajectory` and task override `mpc_use_parametric_trajectory` are removed.
  - Tests/probes no longer import or monkeypatch the deleted dense modules. Focused suite passed with `209 passed, 1 warning`, and source scan found no old dense symbols.
  - IsaacLab GPU3 cleanup smoke executed successfully after cleanup; `max_replan_initial_foot_error=0.0`, but T302k.12 remains open with touchdown-current mismatch `0.4458m`.
- 2026-05-26 19:49 update:
  - Viewer MPC now treats teleop/scripted `vx vy yaw_rate` as root/body-frame command values and rotates linear XY into world frame at the viewer boundary before calling `plan_segment`.
  - The MPC planner internal command contract remains unchanged; the conversion is scoped to `extension/viz/go2_foostep_planner.py` for `backend="mpc"`.
  - TDD red reproduced the missing conversion helper, then focused viewer tests passed with `14 passed`; pycompile passed for the viewer file.
- 2026-05-26 20:21 update:
  - T302k.14 fixes the downstairs/sloped support attitude gap: parametric decode now fits a contact-weighted foot support plane in the yaw frame and ramps root roll/pitch toward that estimate after frame0.
  - Frame0 root roll/pitch still preserves the current IsaacLab state to avoid a replan discontinuity.
  - Focused red reproduced terminal pitch staying `0.0`; after the fix, the sloped/downstairs fixture reports terminal pitch `0.2354 rad` while frame0 remains `[0.0, 0.0]`.
  - Verification passed both local pytest and the requested `env_isaacsim` Python environment.
- 2026-05-26 20:40 update:
  - T302k.15 reproduces the user's long-step/mid-replan foot deformation using flat terrain and commands for forward, backward, lateral left/right, and yaw left/right.
  - The key metric is body-yaw relative foot position, not world foot position: `Rz(-yaw) * (foot_w - root_w)`.
  - Frame0 replan alignment is not the problem (`max_frame0_rel_mismatch ~= 0`), and playback readback matches the planned frame (`~1e-6m`).
  - The planned/current body-relative foot coordinates themselves drift over repeated replans. After 8 cycles with playback frame `24`, max relative drift reaches `0.278-0.314m`; lateral commands accumulate about `0.174m` in body-y, yaw commands accumulate coupled body-x/body-y drift.
  - Likely cause: `decode_parametric_trajectory()` restarts fixed `swing_center`/phase every replan and uses current `state.foot_pos` as `foot0`, so a mid-swing/current-offset foot becomes the next segment's nominal start. The fix should add phase/contact/stance-anchor continuity, not readback plumbing.
- 2026-05-26 21:33 update:
  - T302k.15 local fix changes the full-cycle terminal foot anchor: frame0 still uses current IsaacLab `state.foot_pos`, but touchdown/terminal feet now anchor to a canonical body-yaw footprint under the terminal root instead of `foot0 + delta`.
  - `touchdown_delta_raw` is de-meaned across legs, so it can reshape relative footholds but cannot translate the whole four-foot footprint independently from `root_goal_delta`.
  - Red regression reproduced two-cycle accumulation (`0.0546m -> 0.1093m`) and now passes.
  - `env_isaacsim` 8-cycle flat long-replan probe keeps `mpc_horizon_steps=25`, frame0/readback errors at zero or `~1e-6`, and reduces total relative drift: lateral `~0.278/0.288m -> ~0.088/0.087m`, yaw `~0.314/0.279m -> ~0.132/0.140m`, z drift `~0.26-0.27m -> ~0.009m`.
  - Remaining follow-up: yaw body-x still grows mildly over cycles; explicit manager-carried gait/contact phase can further reduce this after visual acceptance.

## Open Children

| Child | Status | Priority | Purpose | Primary Files |
| --- | --- | --- | --- | --- |
| T302k.1 | done | P0 | Add parametric command-frame geometry helpers and curve sampling tests | `Go2Pvcnn/extension/batch_mpc_planner/parametric.py`, `Go2Pvcnn/tests/test_batch_mpc_parametric.py` |
| T302k.2 | done | P0 | Add `MpcParametricVariables` and initialization from nominal/current IsaacLab state | `Go2Pvcnn/extension/batch_mpc_planner/parametric.py`, `Go2Pvcnn/tests/test_batch_mpc_parametric.py` |
| T302k.3 | done | P0 | Decode parametric variables into 25-frame root and target-foot curves with touchdown z grounded from height map | `Go2Pvcnn/extension/batch_mpc_planner/parametric.py`, `Go2Pvcnn/tests/test_batch_mpc_parametric.py` |
| T302k.4 | done | P0 | Integrate clamped IK and FK-realized default output contract | `Go2Pvcnn/extension/batch_mpc_planner/planner.py`, `Go2Pvcnn/extension/batch_mpc_planner/config.py`, `Go2Pvcnn/tests/test_batch_mpc_backend.py` |
| T302k.5 | verify | P0 | Port/replace losses so 25 sampled frames use target-vs-FK reachability, terrain/semantic collision, low-small crossing, gait, command progress, and curve regularization | `Go2Pvcnn/extension/batch_mpc_planner/planner.py`, `Go2Pvcnn/tests/test_batch_mpc_backend.py` |
| T302k.6 | verify | P0 | Add low-small parametric crossing probe and prove focused local/unit gates before IsaacLab | `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py`, `Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py` |
| T302k.7 | partial | P0 | Run real IsaacLab rolling25 low-small/high-small/large acceptance and record notes/logs | `tmp/t302k-parametric-mpc/`, `notes/log/` |
| T302k.9 | partial | P0 | Add semantic avoidance and endpoint/touchdown constraints for parametric path after IsaacLab failures | `Go2Pvcnn/extension/batch_mpc_planner/planner.py`, `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py` |
| T302k.10 | verify | P0 | Make parametric foot curves obey trot pair swing windows instead of moving all four feet together | `Go2Pvcnn/extension/batch_mpc_planner/parametric.py`, `Go2Pvcnn/tests/test_batch_mpc_parametric.py` |
| T302k.11 | verify | P0 | Make every parametric replan output start from current IsaacLab foot positions at frame0, with frame1+ FK-realized | `Go2Pvcnn/extension/batch_mpc_planner/planner.py`, `Go2Pvcnn/tests/test_batch_mpc_backend.py`, `Go2Pvcnn/tests/test_batch_mpc_parametric.py` |
| T302k.12 | todo | P0 | Make current stance/current-foot touchdowns consistent at replan boundaries instead of exporting only future touchdown targets | `Go2Pvcnn/extension/batch_mpc_planner/planner.py`, `Go2Pvcnn/extension/batch_mpc_planner/parametric.py`, `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py` |
| T302k.8 | done | P1 | Remove obsolete dense-foot residual planner path and tests so current MPC is parametric-only | `Go2Pvcnn/extension/batch_mpc_planner/planner.py`, `Go2Pvcnn/extension/batch_mpc_planner/config.py`, `Go2Pvcnn/tests/` |
| T302k.13 | verify | P1 | Convert viewer MPC root/body-frame linear commands to world-frame commands before planning | `Go2Pvcnn/extension/viz/go2_foostep_planner.py`, `Go2Pvcnn/tests/test_viewer_reset.py` |
| T302k.14 | verify | P1 | Make parametric root roll/pitch follow the foot support plane on sloped/downstairs terrain after frame0 | `Go2Pvcnn/extension/batch_mpc_planner/parametric.py`, `Go2Pvcnn/tests/test_batch_mpc_parametric.py` |
| T302k.15 | verify | P0 | Fix long-step mid-replan body-relative foot drift by anchoring full-cycle terminal feet to a stable body-yaw footprint; remaining yaw body-x drift may need manager-carried phase/contact anchors | `Go2Pvcnn/extension/batch_mpc_planner/parametric.py`, `Go2Pvcnn/tests/test_batch_mpc_parametric.py` |

## Closed Children Archive

- T302k.1: Added `bounded_unit_interval`, `command_frame_axes`, and `cubic_bezier`; helper tests cover command-frame fallback and Bezier endpoints.
- T302k.2: Added `MpcParametricVariables` and initialization tensors for touchdown xy deltas, foot/root Bezier controls, root goal/height, and diagonal timing parameters.
- T302k.3: Added parametric decode to 25-frame root and foot target curves; touchdown z is sampled from `height_at`.
- T302k.4: Integrated default parametric `plan_segment` output path; exported `foot_pos` is FK-realized from clamped IK, while old dense optimizer tests opt into `use_parametric_trajectory=False`.
- T302k.5 local slice: Added sampled losses `parametric_reachability`, `parametric_terrain_clearance`, `parametric_semantic_contact`, `parametric_low_small_crossing`, `parametric_gait_regularization`, `parametric_command_progress`, and `parametric_curve_regularization`; `cost_total` now sums them and `optimize_steps` updates parametric variables.
- T302k.6 local slice: Added `parametric_v1` as a probe/runtime variant that enables parametric planner output.
- T302k.9 local slice: Added `parametric_semantic_avoidance`, `parametric_touchdown_endpoint`, and `parametric_foot_height_guard`; high/large semantic command shaping now feeds parametric decode. Local tests pass, but IsaacLab high/large remains unaccepted.
- T302k.10 local slice: Replaced full-horizon foot Bezier phase with per-leg local swing phase. Contact/stanced legs hold their endpoint while the active diagonal pair follows the cubic foot curve.
- T302k.11 local slice: Kept parametric decode anchored at current `state.foot_pos`; changed `plan_segment` parametric export so frame0 `foot_pos` is current IsaacLab foot state, while frame1+ remains FK-realized. Updated FK export tests to check frame1+.
- T302k.8 cleanup: Deleted dense residual modules (`nominal.py`, `optimizer.py`, `variables.py`, `losses/registry.py`), removed the parametric feature switch, updated probes/tests away from old dense imports, and verified no old dense symbols remain in source.
- T302k.13 local slice: Added `_viewer_mpc_world_command_from_root_frame()` and applied it only for viewer `backend="mpc"` so body-frame `vx/vy` is rotated by `state.root_rpy[:, 2]`; yaw-rate is unchanged.
- T302k.14 local slice: Added contact-weighted support-plane roll/pitch estimation inside parametric decode, ramped roll/pitch after frame0, and added a downhill support-plane regression.
- T302k.15 reproduction: `tmp/t302k-replan-direction-repro/root_relative_long_replan_probe.jsonl` shows body-yaw relative foot drift despite frame0 and playback readback alignment.
- T302k.15 local slice: Added a two-cycle full-horizon regression and changed parametric touchdown generation so terminal feet anchor to terminal root plus a canonical body-yaw footprint; de-meaned four-leg touchdown deltas prevent global foot translation from accumulating separately from root progress. `env_isaacsim` confirms lateral/yaw drift reductions with `mpc_horizon_steps=25`.

## Related Logs

- [../log/2026-05-26-2040-t302k-long-step-root-relative-foot-drift-repro.md](../log/2026-05-26-2040-t302k-long-step-root-relative-foot-drift-repro.md)
- [../log/2026-05-26-2133-t302k-body-relative-foot-anchor-fix.md](../log/2026-05-26-2133-t302k-body-relative-foot-anchor-fix.md)
- [../log/2026-05-26-2021-t302k-support-plane-root-roll-pitch.md](../log/2026-05-26-2021-t302k-support-plane-root-roll-pitch.md)
- [../log/2026-05-26-1949-viewer-mpc-body-frame-command.md](../log/2026-05-26-1949-viewer-mpc-body-frame-command.md)
- [../log/2026-05-26-1336-t302j-structured-low-small-touchdown-runtime.md](../log/2026-05-26-1336-t302j-structured-low-small-touchdown-runtime.md)
- [../log/2026-05-26-1757-t302k-dense-path-retirement.md](../log/2026-05-26-1757-t302k-dense-path-retirement.md)
- [../log/2026-05-26-1717-t302k-isaaclab-current-foot-touchdown-check.md](../log/2026-05-26-1717-t302k-isaaclab-current-foot-touchdown-check.md)
- [../log/2026-05-26-1713-t302k-parametric-current-foot-replan-anchor.md](../log/2026-05-26-1713-t302k-parametric-current-foot-replan-anchor.md)
- [../log/2026-05-26-1649-t302k-parametric-trot-phase-foot-curves.md](../log/2026-05-26-1649-t302k-parametric-trot-phase-foot-curves.md)
- [../log/2026-05-26-1554-t302k-parametric-semantic-endpoint-losses.md](../log/2026-05-26-1554-t302k-parametric-semantic-endpoint-losses.md)
- [../log/2026-05-26-1450-t302k-parametric-default-fk-output.md](../log/2026-05-26-1450-t302k-parametric-default-fk-output.md)
- [../log/2026-05-26-1524-t302k-parametric-sampled-loss-and-isaaclab-smoke.md](../log/2026-05-26-1524-t302k-parametric-sampled-loss-and-isaaclab-smoke.md)
- [../log/2026-05-26-1259-t302j-low-small-crossing-acceptance-test-contract.md](../log/2026-05-26-1259-t302j-low-small-crossing-acceptance-test-contract.md)
- [../log/2026-05-25-1723-t302i-ik-clamp-foot-mismatch-trace.md](../log/2026-05-25-1723-t302i-ik-clamp-foot-mismatch-trace.md)
- [../log/2026-05-25-1904-t302i-reachable-crossing-probe-baseline.md](../log/2026-05-25-1904-t302i-reachable-crossing-probe-baseline.md)
- [../log/2026-05-25-1222-t302h-rolling25-low-small-foot-over-production.md](../log/2026-05-25-1222-t302h-rolling25-low-small-foot-over-production.md)

## Git Refs

- Last Feature Commit: `1b799cd` (parametric helper module)
- Last Verified Commit: `working tree @ 1b799cd`
- Current Work Ref: `working tree on top of 1b799cd (2026-05-26 15:24 +0800)`
- Key Files:
  - [../../docs/superpowers/specs/2026-05-26-mpc-parametric-trajectory-contract-design.md](../../docs/superpowers/specs/2026-05-26-mpc-parametric-trajectory-contract-design.md)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/parametric.py](../../Go2Pvcnn/extension/batch_mpc_planner/parametric.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/kinematics.py](../../Go2Pvcnn/extension/batch_mpc_planner/kinematics.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)

## Next Step

Continue high-small/large work next: use root nominal acceleration/bias and semantic/touchdown constraints, while preserving the new trot pair foot phase and frame0 current-foot replan anchor. The current parametric path can run under IsaacLab and crosses low-small with FK-realized future feet, but high-small/large avoidance still fails task acceptance and touchdown endpoints still lag realized swing. Do not continue V9/V10/V11/V12 scalar-loss tuning unless it is needed as a regression comparison. The active execution path is parametric trajectory optimization:

```text
current IsaacLab state + command + terrain/semantic
-> parametric touchdown/root/swing variables
-> cubic curves
-> 25 sampled frames
-> clamped IK
-> FK-realized feet
-> physical/semantic/task losses
-> FK-realized output
```

## Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved parametric MPC trajectory contract for `extension/batch_mpc_planner`.

**Architecture:** Add a focused `parametric.py` module for command-frame transforms, bounded Bezier parameters, root/foot curve sampling, and FK-realized decode support. Integrate it behind a config/debug switch first, then migrate losses and acceptance probes before making it default.

**Tech Stack:** Python, PyTorch tensors, IsaacLab runtime probes, existing `Go2Pvcnn/extension/batch_mpc_planner` APIs.

---

### Task 1: Parametric Geometry Helpers

**Files:**
- Create: `Go2Pvcnn/extension/batch_mpc_planner/parametric.py`
- Create: `Go2Pvcnn/tests/test_batch_mpc_parametric.py`

- [ ] **Step 1: Write failing tests for command-frame axes and Bezier sampling**

Add:

```python
import torch

from extension.batch_mpc_planner.parametric import (
    bounded_unit_interval,
    command_frame_axes,
    cubic_bezier,
)


def test_command_frame_axes_uses_translation_direction() -> None:
    command = torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float32)
    yaw = torch.tensor([0.0], dtype=torch.float32)

    forward, left, active = command_frame_axes(command, yaw, linear_eps=1.0e-4)

    assert active.tolist() == [True]
    torch.testing.assert_close(forward, torch.tensor([[0.0, 1.0]]))
    torch.testing.assert_close(left, torch.tensor([[-1.0, 0.0]]))


def test_command_frame_axes_falls_back_to_root_yaw_for_pure_yaw() -> None:
    command = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    yaw = torch.tensor([1.5707964], dtype=torch.float32)

    forward, left, active = command_frame_axes(command, yaw, linear_eps=1.0e-4)

    assert active.tolist() == [False]
    torch.testing.assert_close(forward, torch.tensor([[0.0, 1.0]]), atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(left, torch.tensor([[-1.0, 0.0]]), atol=1.0e-6, rtol=1.0e-6)


def test_cubic_bezier_starts_and_ends_at_control_points() -> None:
    p0 = torch.tensor([[[0.0, 0.0]]])
    p1 = torch.tensor([[[0.3, 0.1]]])
    p2 = torch.tensor([[[0.7, 0.1]]])
    p3 = torch.tensor([[[1.0, 0.0]]])
    phase = torch.tensor([0.0, 0.5, 1.0])

    curve = cubic_bezier(p0, p1, p2, p3, phase)

    torch.testing.assert_close(curve[:, 0], p0)
    torch.testing.assert_close(curve[:, -1], p3)


def test_bounded_unit_interval_maps_zero_raw_to_midpoint() -> None:
    raw = torch.zeros((2, 4), dtype=torch.float32)

    value = bounded_unit_interval(raw, low=0.15, high=0.85)

    torch.testing.assert_close(value, torch.full((2, 4), 0.5))
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
```

Expected:

```text
ModuleNotFoundError: No module named 'extension.batch_mpc_planner.parametric'
```

- [ ] **Step 3: Implement geometry helpers**

Add `Go2Pvcnn/extension/batch_mpc_planner/parametric.py`:

```python
"""Parametric trajectory helpers for the batch MPC backend."""

from __future__ import annotations

import torch
from torch import Tensor


def bounded_unit_interval(raw: Tensor, *, low: float, high: float) -> Tensor:
    return float(low) + (float(high) - float(low)) * torch.sigmoid(raw)


def command_frame_axes(command: Tensor, root_yaw: Tensor, *, linear_eps: float) -> tuple[Tensor, Tensor, Tensor]:
    cmd = torch.as_tensor(command)
    if int(cmd.shape[-1]) < 3:
        pad = torch.zeros((*cmd.shape[:-1], 3 - int(cmd.shape[-1])), dtype=cmd.dtype, device=cmd.device)
        cmd = torch.cat((cmd, pad), dim=-1)
    xy = cmd[:, :2]
    speed = torch.linalg.vector_norm(xy, dim=-1)
    active = speed > float(linear_eps)
    cmd_forward = xy / speed.clamp_min(1.0e-6).unsqueeze(-1)
    yaw = torch.as_tensor(root_yaw, dtype=cmd.dtype, device=cmd.device).reshape(-1)
    yaw_forward = torch.stack((torch.cos(yaw), torch.sin(yaw)), dim=-1)
    forward = torch.where(active.unsqueeze(-1), cmd_forward, yaw_forward)
    left = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
    return forward, left, active


def cubic_bezier(p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor, phase: Tensor) -> Tensor:
    t = torch.as_tensor(phase, dtype=p0.dtype, device=p0.device)
    view_shape = (1, int(t.numel())) + (1,) * (p0.ndim - 1)
    t = t.reshape(view_shape)
    one = 1.0 - t
    return one.pow(3) * p0.unsqueeze(1) + 3.0 * one.pow(2) * t * p1.unsqueeze(1) + 3.0 * one * t.pow(2) * p2.unsqueeze(1) + t.pow(3) * p3.unsqueeze(1)


__all__ = ["bounded_unit_interval", "command_frame_axes", "cubic_bezier"]
```

- [ ] **Step 4: Run tests and confirm they pass**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
```

Expected:

```text
4 passed
```

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/batch_mpc_planner/parametric.py Go2Pvcnn/tests/test_batch_mpc_parametric.py
git commit -m "feat: add mpc parametric geometry helpers"
```

### Task 2: Parametric Variables And Initialization

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/parametric.py`
- Modify: `Go2Pvcnn/tests/test_batch_mpc_parametric.py`

- [ ] **Step 1: Write failing tests for variable shapes and grounded touchdown z**

Append:

```python
from extension.batch_mpc_planner.parametric import (
    MpcParametricVariables,
    init_parametric_variables,
)
from extension.batch_mpc_planner.types import MpcPlannerTerrain, MpcRobotState


def _flat_terrain(batch: int = 1, height: float = 0.2) -> MpcPlannerTerrain:
    return MpcPlannerTerrain(
        height_map=torch.full((batch, 5, 5), height, dtype=torch.float32),
        world_x_range=(-1.0, 1.0),
        world_y_range=(-1.0, 1.0),
    )


def _state(batch: int = 1) -> MpcRobotState:
    return MpcRobotState(
        root_pos=torch.tensor([[0.0, 0.0, 0.35]], dtype=torch.float32).expand(batch, 3).clone(),
        root_rpy=torch.zeros((batch, 3), dtype=torch.float32),
        foot_pos=torch.tensor(
            [[
                [0.25, 0.12, 0.2],
                [0.25, -0.12, 0.2],
                [-0.25, 0.12, 0.2],
                [-0.25, -0.12, 0.2],
            ]],
            dtype=torch.float32,
        ).expand(batch, 4, 3).clone(),
        joint_angles=torch.zeros((batch, 12), dtype=torch.float32),
    )


def test_init_parametric_variables_has_expected_shapes() -> None:
    variables = init_parametric_variables(_state(), torch.tensor([[0.5, 0.0, 0.0]]), horizon=25)

    assert isinstance(variables, MpcParametricVariables)
    assert variables.touchdown_delta_raw.shape == (1, 4, 2)
    assert variables.swing_clearance_raw.shape == (1, 4)
    assert variables.bezier_ab_raw.shape == (1, 4, 2)
    assert variables.root_goal_delta_raw.shape == (1, 2)
    assert variables.root_bezier_raw.shape == (1, 2)
    assert variables.diagonal_phase_raw.shape == (1,)


def test_parametric_variables_parameters_are_optimizable() -> None:
    variables = init_parametric_variables(_state(), torch.tensor([[0.5, 0.0, 0.0]]), horizon=25)

    params = variables.parameters()

    assert len(params) >= 8
    assert all(param.requires_grad for param in params)
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
```

Expected: import failure for `MpcParametricVariables` or `init_parametric_variables`.

- [ ] **Step 3: Implement variable dataclass and initializer**

Add to `parametric.py`:

```python
from dataclasses import dataclass


@dataclass
class MpcParametricVariables:
    touchdown_delta_raw: Tensor
    swing_clearance_raw: Tensor
    bezier_ab_raw: Tensor
    lateral_bias_raw: Tensor
    root_goal_delta_raw: Tensor
    root_bezier_raw: Tensor
    root_lateral_bias_raw: Tensor
    root_height_offset_raw: Tensor
    swing_center_raw: Tensor
    swing_width_raw: Tensor
    diagonal_phase_raw: Tensor

    def parameters(self) -> list[Tensor]:
        return [
            self.touchdown_delta_raw,
            self.swing_clearance_raw,
            self.bezier_ab_raw,
            self.lateral_bias_raw,
            self.root_goal_delta_raw,
            self.root_bezier_raw,
            self.root_lateral_bias_raw,
            self.root_height_offset_raw,
            self.swing_center_raw,
            self.swing_width_raw,
            self.diagonal_phase_raw,
        ]


def _optim_zeros(shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device) -> Tensor:
    return torch.zeros(shape, dtype=dtype, device=device).requires_grad_(True)


def init_parametric_variables(state, command: Tensor, *, horizon: int) -> MpcParametricVariables:
    root = torch.as_tensor(state.root_pos)
    batch = int(root.shape[0])
    dtype = root.dtype
    device = root.device
    return MpcParametricVariables(
        touchdown_delta_raw=_optim_zeros((batch, 4, 2), dtype=dtype, device=device),
        swing_clearance_raw=_optim_zeros((batch, 4), dtype=dtype, device=device),
        bezier_ab_raw=_optim_zeros((batch, 4, 2), dtype=dtype, device=device),
        lateral_bias_raw=_optim_zeros((batch, 4, 2), dtype=dtype, device=device),
        root_goal_delta_raw=_optim_zeros((batch, 2), dtype=dtype, device=device),
        root_bezier_raw=_optim_zeros((batch, 2), dtype=dtype, device=device),
        root_lateral_bias_raw=_optim_zeros((batch, 2), dtype=dtype, device=device),
        root_height_offset_raw=_optim_zeros((batch,), dtype=dtype, device=device),
        swing_center_raw=_optim_zeros((batch, 4), dtype=dtype, device=device),
        swing_width_raw=_optim_zeros((batch, 4), dtype=dtype, device=device),
        diagonal_phase_raw=_optim_zeros((batch,), dtype=dtype, device=device),
    )
```

Update `__all__`.

- [ ] **Step 4: Run tests and confirm they pass**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
```

Expected:

```text
6 passed
```

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/batch_mpc_planner/parametric.py Go2Pvcnn/tests/test_batch_mpc_parametric.py
git commit -m "feat: add mpc parametric variables"
```

### Task 3: Decode Root And Foot Curves

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/parametric.py`
- Modify: `Go2Pvcnn/tests/test_batch_mpc_parametric.py`

- [ ] **Step 1: Write failing tests for decoded curve contract**

Append:

```python
from extension.batch_mpc_planner.parametric import decode_parametric_trajectory


def test_decode_parametric_trajectory_starts_from_current_state_and_has_25_frames() -> None:
    state = _state()
    terrain = _flat_terrain()
    command = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    variables = init_parametric_variables(state, command, horizon=25)

    decoded = decode_parametric_trajectory(state, terrain, command, variables, horizon=25)

    assert decoded.root_pos.shape == (1, 25, 3)
    assert decoded.target_foot_pos.shape == (1, 25, 4, 3)
    torch.testing.assert_close(decoded.root_pos[:, 0], state.root_pos)
    torch.testing.assert_close(decoded.target_foot_pos[:, 0], state.foot_pos)


def test_decode_parametric_touchdown_z_comes_from_height_map() -> None:
    state = _state()
    terrain = _flat_terrain(height=0.42)
    command = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32)
    variables = init_parametric_variables(state, command, horizon=25)

    decoded = decode_parametric_trajectory(state, terrain, command, variables, horizon=25)

    torch.testing.assert_close(decoded.touchdown_w[..., 2], torch.full((1, 4), 0.42))
```

- [ ] **Step 2: Run tests and confirm they fail**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
```

Expected: import failure for `decode_parametric_trajectory`.

- [ ] **Step 3: Implement decoded trajectory dataclass and decoder**

Add:

```python
@dataclass(frozen=True)
class DecodedParametricTrajectory:
    root_pos: Tensor
    root_rpy: Tensor
    target_foot_pos: Tensor
    touchdown_w: Tensor
    swing_center: Tensor
    swing_width: Tensor
    contact_prob: Tensor
    swing_prob: Tensor


def _phase(horizon: int, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    return torch.linspace(0.0, 1.0, int(horizon), dtype=dtype, device=device)


def decode_parametric_trajectory(state, terrain, command: Tensor, variables: MpcParametricVariables, *, horizon: int) -> DecodedParametricTrajectory:
    from .terrain import height_at

    root0 = torch.as_tensor(state.root_pos)
    rpy0 = torch.as_tensor(state.root_rpy, dtype=root0.dtype, device=root0.device)
    foot0 = torch.as_tensor(state.foot_pos, dtype=root0.dtype, device=root0.device)
    batch = int(root0.shape[0])
    dtype = root0.dtype
    device = root0.device
    phase = _phase(horizon, dtype=dtype, device=device)
    forward, left, linear_active = command_frame_axes(command, rpy0[:, 2], linear_eps=1.0e-4)

    td_delta = 0.35 * torch.tanh(variables.touchdown_delta_raw)
    touchdown_xy = foot0[..., :2] + forward[:, None, :] * td_delta[..., 0:1] + left[:, None, :] * td_delta[..., 1:2]
    touchdown_z = height_at(terrain, touchdown_xy).to(dtype=dtype, device=device)
    touchdown_w = torch.cat((touchdown_xy, touchdown_z.unsqueeze(-1)), dim=-1)

    ab = bounded_unit_interval(variables.bezier_ab_raw, low=0.15, high=0.85)
    a = ab[..., 0:1]
    b = ab[..., 1:2]
    lateral = 0.20 * torch.tanh(variables.lateral_bias_raw)
    step = touchdown_xy - foot0[..., :2]
    length = torch.linalg.vector_norm(step, dim=-1, keepdim=True).clamp_min(1.0e-6)
    p0 = foot0[..., :2]
    p3 = touchdown_xy
    p1 = p0 + forward[:, None, :] * (a * length) + left[:, None, :] * lateral[..., 0:1]
    p2 = p3 - forward[:, None, :] * (b * length) + left[:, None, :] * lateral[..., 1:2]
    foot_xy = cubic_bezier(p0, p1, p2, p3, phase)
    terrain_z = height_at(terrain, foot_xy.reshape(batch, int(horizon) * 4, 2)).reshape(batch, int(horizon), 4)
    clearance = 0.04 + 0.16 * torch.sigmoid(variables.swing_clearance_raw)
    base_z = torch.lerp(foot0[:, None, :, 2], touchdown_z[:, None, :], phase.view(1, int(horizon), 1))
    arc = 4.0 * phase.view(1, int(horizon), 1) * (1.0 - phase.view(1, int(horizon), 1)) * clearance[:, None, :]
    foot_z = torch.maximum(base_z + arc, terrain_z + 0.025)
    target_foot_pos = torch.cat((foot_xy, foot_z.unsqueeze(-1)), dim=-1)

    root_delta = torch.stack((0.30 + 0.50 * torch.sigmoid(variables.root_goal_delta_raw[:, 0]), 0.25 * torch.tanh(variables.root_goal_delta_raw[:, 1])), dim=-1)
    root_goal_xy = root0[:, :2] + forward * root_delta[:, 0:1] + left * root_delta[:, 1:2]
    root_c = bounded_unit_interval(variables.root_bezier_raw, low=0.15, high=0.85)
    root_len = torch.linalg.vector_norm(root_goal_xy - root0[:, :2], dim=-1, keepdim=True).clamp_min(1.0e-6)
    root_lat = 0.20 * torch.tanh(variables.root_lateral_bias_raw)
    r0 = root0[:, :2]
    r3 = root_goal_xy
    r1 = r0 + forward * (root_c[:, 0:1] * root_len) + left * root_lat[:, 0:1]
    r2 = r3 - forward * (root_c[:, 1:2] * root_len) + left * root_lat[:, 1:2]
    root_xy = cubic_bezier(r0, r1, r2, r3, phase)
    root_ground = height_at(terrain, root_xy).to(dtype=dtype, device=device)
    root_z = root_ground + 0.32 + 0.06 * torch.tanh(variables.root_height_offset_raw).view(batch, 1)
    root_pos = torch.cat((root_xy, root_z.unsqueeze(-1)), dim=-1)
    root_rpy = rpy0[:, None, :].expand(batch, int(horizon), 3).clone()
    root_rpy[..., 2] = torch.lerp(rpy0[:, None, 2], rpy0[:, None, 2] + command[:, None, 2] * 0.5, phase.view(1, int(horizon)))

    swing_center = torch.remainder(torch.tensor((0.75, 0.25, 0.25, 0.75), dtype=dtype, device=device).view(1, 4) + 0.20 * torch.tanh(variables.swing_center_raw), 1.0)
    swing_width = bounded_unit_interval(variables.swing_width_raw, low=0.30, high=0.70)
    frame_phase = phase.view(1, int(horizon), 1)
    dist = torch.abs(torch.remainder(frame_phase - swing_center[:, None, :] + 0.5, 1.0) - 0.5)
    swing_prob = torch.sigmoid(40.0 * (0.5 * swing_width[:, None, :] - dist))
    contact_prob = 1.0 - swing_prob
    return DecodedParametricTrajectory(root_pos, root_rpy, target_foot_pos, touchdown_w, swing_center, swing_width, contact_prob, swing_prob)
```

Update `__all__`.

- [ ] **Step 4: Run tests and confirm they pass**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
```

Expected: all parametric tests pass.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/batch_mpc_planner/parametric.py Go2Pvcnn/tests/test_batch_mpc_parametric.py
git commit -m "feat: decode mpc parametric curves"
```

### Task 4: FK-Realized Output Contract Behind Switch

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/config.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
- Modify: `Go2Pvcnn/tests/test_batch_mpc_backend.py`

- [ ] **Step 1: Add failing backend test**

Add a test that calls `plan_segment()` with `cfg.runtime.use_parametric_trajectory = True` and asserts:

```python
from extension.batch_mpc_planner.kinematics import fk_feet_from_joint_angles


def test_parametric_plan_exports_fk_realized_feet() -> None:
    terrain, state, command, cfg = make_flat_mpc_case()  # use existing helper in this test module
    cfg.runtime.use_parametric_trajectory = True

    result = plan_segment(terrain, state, command, cfg=cfg)
    fk = fk_feet_from_joint_angles(result.root_pos, result.root_rpy, result.joint_angles)

    torch.testing.assert_close(result.foot_pos, fk, atol=1.0e-5, rtol=1.0e-5)
```

If the existing helper is named differently, use the local flat-case helper already present in `test_batch_mpc_backend.py`; do not create a second fixture with divergent semantics.

- [ ] **Step 2: Run the single test and confirm it fails**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k parametric_plan_exports_fk_realized_feet
```

Expected: config attribute missing or result mismatch.

- [ ] **Step 3: Add config switch**

Add to `MpcRuntimeCfg`:

```python
use_parametric_trajectory: bool = False
```

Add task override support in `config_from_task()` using the existing `_set_if_has` pattern:

```python
_set_if_has(task_cfg, "mpc_use_parametric_trajectory", bool, runtime, "use_parametric_trajectory")
```

- [ ] **Step 4: Integrate parametric decode in `plan_segment()`**

In the first integration, keep optimizer/loss path conservative:

```python
if bool(cfg.runtime.use_parametric_trajectory):
    param_vars = init_parametric_variables(state, planning_command, horizon=int(cfg.runtime.horizon_steps))
    param_decoded = decode_parametric_trajectory(state, terrain, planning_command, param_vars, horizon=int(cfg.runtime.horizon_steps))
    root_pos = param_decoded.root_pos
    root_rpy = param_decoded.root_rpy
    foot_target = param_decoded.target_foot_pos
    joint_seq = solve_joint_angles_from_trajectory(root_pos, root_rpy, foot_target)
    foot_pos = fk_feet_from_joint_angles(root_pos, root_rpy, joint_seq)
    contact_state = param_decoded.contact_prob >= float(cfg.runtime.contact_threshold)
    touchdown_w = param_decoded.touchdown_w
```

Wire this as a guarded path before the legacy postprocess output block. Keep legacy path unchanged when the switch is false.

- [ ] **Step 5: Run focused backend tests**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k "parametric or zero_command or output"
```

Expected: focused subset passes.

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "feat: export fk-realized parametric mpc output"
```

### Task 5: Parametric Loss Set

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/parametric.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py`
- Modify: `Go2Pvcnn/tests/test_batch_mpc_parametric.py`

- [ ] **Step 1: Add failing tests for curve regularization and target-vs-FK reachability**

Append tests that construct target feet and FK feet with known error:

```python
from extension.batch_mpc_planner.parametric import curve_regularization_loss, target_fk_reachability_loss


def test_target_fk_reachability_loss_is_zero_when_matching() -> None:
    target = torch.zeros((1, 25, 4, 3))
    fk = target.clone()

    loss = target_fk_reachability_loss(target, fk)

    torch.testing.assert_close(loss, torch.zeros((1,)))


def test_target_fk_reachability_loss_penalizes_unreachable_target() -> None:
    target = torch.zeros((1, 25, 4, 3))
    fk = target.clone()
    fk[:, :, 0, 2] -= 0.2

    loss = target_fk_reachability_loss(target, fk)

    assert loss.item() > 0.0
```

- [ ] **Step 2: Implement minimal loss helpers**

Add:

```python
def target_fk_reachability_loss(target_foot_pos: Tensor, fk_foot_pos: Tensor) -> Tensor:
    err = torch.linalg.vector_norm(target_foot_pos - fk_foot_pos, dim=-1)
    return err.square().mean(dim=(1, 2))


def curve_regularization_loss(points: Tensor) -> Tensor:
    if int(points.shape[1]) < 3:
        return torch.zeros((int(points.shape[0]),), dtype=points.dtype, device=points.device)
    vel = points[:, 1:] - points[:, :-1]
    accel = vel[:, 1:] - vel[:, :-1]
    return accel.square().mean(dim=tuple(range(1, accel.ndim)))
```

- [ ] **Step 3: Run local parametric tests**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
```

Expected: pass.

- [ ] **Step 4: Wire losses into guarded planner path**

In the parametric branch, include at least:

```python
reachability = target_fk_reachability_loss(foot_target, foot_pos)
root_curve = curve_regularization_loss(root_pos)
foot_curve = curve_regularization_loss(foot_target)
```

Then add existing collision/semantic losses against FK-realized `foot_pos`, knees/shanks, root, and grounded touchdown. Keep initial weights conservative and behind `use_parametric_trajectory`.

- [ ] **Step 5: Run backend subset**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_batch_mpc_parametric.py -q -k "parametric or collision or touchdown"
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add Go2Pvcnn/extension/batch_mpc_planner/parametric.py Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py Go2Pvcnn/tests/test_batch_mpc_parametric.py Go2Pvcnn/tests/test_batch_mpc_backend.py
git commit -m "feat: add parametric mpc losses"
```

### Task 6: Low-Small Parametric Probe

**Files:**
- Modify: `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py`
- Modify: `Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py`

- [ ] **Step 1: Add probe variant/config for parametric path**

Add a variant name:

```python
PARAMETRIC_VARIANTS = {"parametric_v1"}
```

When selected:

```python
cfg.runtime.use_parametric_trajectory = True
```

- [ ] **Step 2: Add unit test that variant enables parametric runtime**

Add:

```python
def test_parametric_v1_enables_parametric_trajectory_cfg() -> None:
    cfg = reachable_cfg_for_variant(MpcPlannerCfg(), "parametric_v1")

    assert cfg.runtime.use_parametric_trajectory is True
```

- [ ] **Step 3: Run probe unit tests**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py -q -k parametric
```

Expected: pass.

- [ ] **Step 4: Run focused local/backend tests**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_parametric.py Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py
git commit -m "test: add parametric low-small mpc probe"
```

### Task 7: Real IsaacLab Acceptance

**Files:**
- Create: `notes/log/YYYY-MM-DD-HHMM-t302k-parametric-mpc-acceptance.md`
- Modify: `notes/log/index.md`
- Modify: `notes/todo/T302k-parametric-mpc-trajectory-contract.md`
- Modify: `notes/todo.md`

- [ ] **Step 1: Run low-small rolling25 parametric probe**

Run:

```bash
mkdir -p tmp/t302k-parametric-mpc
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py --device cuda:0 --variants parametric_v1 --requested-n-frames 300 --warmup-steps 6 --commands 'forward_v050:0.50 0.00 0.00,forward_yaw_v050_vy025_yaw100:0.50 0.25 1.00,pure_yaw:0.00 0.00 1.00' > tmp/t302k-parametric-mpc/low_small_parametric_v1.jsonl 2>&1
```

Expected:

```text
process exit code 0
JSONL rows include parametric_v1
small contact/penetration metrics are 0
FK foot-over passes for translation commands
pure yaw has no required foot-over failure
```

- [ ] **Step 2: Run high-small/large non-regression probe**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py --device cuda:0 --variants parametric_v1 --cases small,large --semantic-small-height-m 0.46 --requested-n-frames 300 --warmup-steps 6 > tmp/t302k-parametric-mpc/high_large_parametric_v1.jsonl 2>&1
```

Expected:

```text
process exit code 0
high-small and large semantic_task violations do not regress from current production baseline
stance/root semantic contact remains 0
continuity gates remain bounded
```

- [ ] **Step 3: Update notes/log evidence**

Create a log with:

```markdown
# T302k Parametric MPC Acceptance

## Purpose

Verify the parametric MPC trajectory contract under real IsaacLab probes.

## Stage

`extension/batch_mpc_planner`

## Related Todo

[T302k](../todo/T302k-parametric-mpc-trajectory-contract.md)

## Commands

<paste exact commands and output files>

## Key Metrics

<record low-small, pure-yaw, high-small, large metrics>

## Result

pass/partial/fail

## Follow-up

<next child node>

## Git Refs

- Baseline Ref: `d922eef`
- Candidate Ref: `<commit>`
```

- [ ] **Step 4: Update dashboard and branch page**

Move passing children to `verify` or `done`, keep failing children open with concrete next steps, and put the new log at the top of `notes/log/index.md`.

- [ ] **Step 5: Commit notes**

```bash
git add notes/todo.md notes/todo/T302k-parametric-mpc-trajectory-contract.md notes/log/index.md notes/log/YYYY-MM-DD-HHMM-t302k-parametric-mpc-acceptance.md
git commit -m "docs: record t302k parametric mpc acceptance"
```

### Task 8: Retire Obsolete Dense-Foot Path

**Files:**
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/variables.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/planner.py`
- Modify: `Go2Pvcnn/extension/batch_mpc_planner/losses/`
- Modify: `Go2Pvcnn/tests/`
- Modify: `notes/todo/T302k-parametric-mpc-trajectory-contract.md`

- [ ] **Step 1: List dense-foot residual usages**

Run:

```bash
rg -n "foot_pos_residual|low_small_stepcap|farthest_touchdown|_command_farthest_touchdown_positions|_align_low_small_swing_to_touchdown" Go2Pvcnn/extension/batch_mpc_planner Go2Pvcnn/tests
```

Expected: produce the exact usage list to classify as keep, delete, or compatibility.

- [ ] **Step 2: Delete only after parametric acceptance is green**

Remove dense-foot residual losses that are replaced by parametric curve regularization and target-FK reachability. Keep collision, semantic, touchdown support, gait, and command-direction losses.

- [ ] **Step 3: Run regression suite**

Run:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_batch_mpc_parametric.py Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py -q
```

Expected: pass.

- [ ] **Step 4: Commit cleanup**

```bash
git add Go2Pvcnn/extension/batch_mpc_planner Go2Pvcnn/tests notes/todo/T302k-parametric-mpc-trajectory-contract.md
git commit -m "refactor: retire dense-foot mpc residual path"
```

## Node Details

### T302k.1 Parametric Geometry Helpers

- why-created: The dense residual representation lets the optimizer move individual frames independently, making continuity hard to tune.
- acceptance: command-frame axes, pure-yaw fallback, Bezier endpoints, and bounded shape parameters are unit-tested.

### T302k.2 Parametric Variables

- why-created: The optimizer should tune touchdown/root/swing parameters, not every foot frame.
- acceptance: all parameters are fixed-shape, differentiable tensors with no CPU sync and no per-env Python loop requirement.

### T302k.3 Curve Decode

- why-created: The 25-frame trajectory should be sampled from continuous root/foot curves.
- acceptance: frame 0 starts from current IsaacLab state; touchdown z always comes from `height_at`.

### T302k.4 FK-Realized Output

- why-created: T302i proved viewer/Isaac readback matches FK of exported joints, while planned Cartesian feet can be unreachable.
- acceptance: `MpcPlannerResult.foot_pos == FK(result.joint_angles)` for the parametric path.

### T302k.5 Parametric Losses

- why-created: Old stepcap/smoothness losses repaired dense residual artifacts; new losses should evaluate curve samples and physical feasibility.
- acceptance: target-vs-FK reachability, terrain/semantic collision, gait, command progress, and curve regularization are active.

### T302k.6 Low-Small Probe

- why-created: The user wants a reusable crossing action across terrain heights, not another one-off scalar loss tune.
- acceptance: low-small translation commands cross with FK-realized feet; pure yaw stays stable without foot-over pressure.

### T302k.7 IsaacLab Acceptance

- why-created: Existing failures are visible only in real rolling viewer/runtime conditions.
- acceptance: real rolling25 probes pass or create concrete child nodes with metrics.

### T302k.8 Dense Path Retirement

- why-created: Keeping dense residual repair losses as defaults would keep future agents tuning the old architecture.
- acceptance: obsolete dense-foot residual losses are removed or demoted after parametric acceptance, with tests/logs updated.
- status: closed locally on 2026-05-26 17:57; source cleanup, focused tests, pycompile, and IsaacLab cleanup smoke ran. Runtime smoke still shows the known T302k.12 touchdown-current mismatch.
