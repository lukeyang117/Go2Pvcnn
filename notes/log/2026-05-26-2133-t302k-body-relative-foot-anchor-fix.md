# 2026-05-26 21:33 T302k Body-Relative Foot Anchor Fix

## Purpose

Fix the long-step/mid-replan foot deformation where body-yaw-relative foot coordinates accumulated across repeated replans, especially during lateral motion and yaw.

## Stage

`extension/batch_mpc_planner` parametric MPC decode and viewer MPC runtime verification.

## Related Todo

- [../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md)

## Command / Procedure

Red regression:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_parametric.py -q -k full_cycle_replan
```

Initial failure:

```text
first_drift=0.0546m
second_drift=0.1093m
```

Local verification:

```bash
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_parametric.py -q
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_viewer_reset.py -q -k 'mpc_cfg_keeps_fixed_cycle_horizon or rotates_mpc_body_frame'
PYTHONPATH=Go2Pvcnn pytest --noconftest Go2Pvcnn/tests/test_batch_mpc_backend.py -q -k 'parametric or touchdown_endpoint or swing_center'
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/parametric.py Go2Pvcnn/extension/viz/go2_foostep_planner.py
```

IsaacLab runtime verification:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python /tmp/t302k_root_relative_probe.py
```

Output:

- `tmp/t302k-replan-direction-repro/root_relative_long_replan_after_body_anchor.jsonl`

## Input Conditions

- Terrain: flat.
- Commands: forward/backward/lateral left/lateral right/yaw left/yaw right.
- Cycles: `8`.
- Playback frame before next replan: `24`.
- Requested viewer frames: `300`.
- MPC horizon: `25`.
- Metric: `Rz(-root_yaw) * (foot_w - root_w)`.

## Key Metrics

| Command | Before Total Drift | After Total Drift | Before Body-Y Drift | After Body-Y Drift | Before Z Drift | After Z Drift |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `forward_v050` | `0.2916` | `0.1069` | `0.0144` | `0.0053` | `0.2713` | `0.0093` |
| `backward_v050` | `0.2771` | `0.1079` | `0.0138` | `0.0055` | `0.2678` | `0.0093` |
| `lateral_left_v050` | `0.2780` | `0.0881` | `0.1738` | `0.0751` | `0.2621` | `0.0093` |
| `lateral_right_v050` | `0.2883` | `0.0874` | `0.1738` | `0.0753` | `0.2713` | `0.0093` |
| `yaw_left_100` | `0.3136` | `0.1317` | `0.1388` | `0.0546` | `0.2675` | `0.0093` |
| `yaw_right_100` | `0.2786` | `0.1400` | `0.0790` | `0.0502` | `0.2622` | `0.0093` |

Frame/readback checks:

- `max_frame0_rel_mismatch`: `0.0` to `1.39e-16`.
- `max_plan_vs_actual_rel_error`: `3.9e-7` to `7.5e-7`.
- `mpc_horizon_steps`: `25`.

## Result

Pass for the user-reported major accumulated foot deformation. The fix keeps replan frame0 feet equal to current IsaacLab feet, but anchors full-cycle touchdown targets to a canonical body-yaw footprint under the terminal root instead of `current_foot + delta`.

The touchdown delta is also de-meaned across the four legs so root translation stays represented by `root_goal_delta`, while foot variables only express relative foothold adjustment.

## Conclusion

The root cause was confirmed: full-cycle replans were treating the previous segment's current or terminal foot offset as the next segment's fresh nominal foot start, so body-relative foot coordinates accumulated. A stable terminal body-footprint anchor removes the dominant accumulation while preserving the 25-frame complete-gait-cycle contract.

Yaw still shows a smaller body-x drift trend (`~0.13-0.14m` over 8 cycles), but the large lateral/yaw foot flying and z ratcheting are removed.

## Follow-Up

- Keep T302k.15 in `verify` state rather than `done` until a viewer visual pass confirms the subjective foot shape is acceptable.
- A later refinement can carry explicit gait phase/contact anchors through the manager to reduce the remaining yaw body-x drift.

## Git Refs

- Baseline Ref: working tree on top of `1b799cd`
- Candidate Ref: working tree after body-relative terminal footprint anchor
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/parametric.py](../../Go2Pvcnn/extension/batch_mpc_planner/parametric.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_parametric.py](../../Go2Pvcnn/tests/test_batch_mpc_parametric.py)
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
