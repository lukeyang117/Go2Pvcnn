# 2026-05-26 20:40 T302k Long-Step Root-Relative Foot Drift Reproduction

## Purpose

Reproduce the user-reported long-step/mid-replan deformation where lateral motion and turning make feet appear to fly or distort, and inspect foot coordinates relative to root rather than only world-frame foot paths.

## Stage

`extension/batch_mpc_planner` parametric MPC runtime under IsaacLab.

## Related Todo

- [../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md)

## Command / Procedure

First pass:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/mpc_swing_trajectory_quality_probe.py \
  --device cuda:0 --terrain flat --cycles 8 --requested-n-frames 300 --playback-frame 24 \
  --warmup-steps 6 --variants baseline \
  --commands 'forward_v050:0.50 0.00 0.00,backward_v050:-0.50 0.00 0.00,lateral_left_v050:0.00 0.50 0.00,lateral_right_v050:0.00 -0.50 0.00,yaw_left_100:0.00 0.00 1.00,yaw_right_100:0.00 0.00 -1.00'
```

Second pass:

- One-off `env_isaacsim` script written in the session.
- It records body-yaw relative foot coordinates:

```text
Rz(-root_yaw) * (foot_w - root_w)
```

Outputs:

- `tmp/t302k-replan-direction-repro/flat_long_replan_direction_probe.jsonl`
- `tmp/t302k-replan-direction-repro/root_relative_long_replan_probe.jsonl`

## Input Conditions

- Terrain: flat.
- Commands:
  - `forward_v050`
  - `backward_v050`
  - `lateral_left_v050`
  - `lateral_right_v050`
  - `yaw_left_100`
  - `yaw_right_100`
- Cycles: `8`.
- Requested frames: `300`.
- Playback frame before next replan: `24`.

## Key Metrics

Summary from body-yaw relative coordinates:

| Command | Total Relative Drift Max | Body-X Drift Max | Body-Y Drift Max | Body-Z Drift Max | Per-Replan Max | Frame0 Rel Mismatch | Plan-vs-Actual Rel Error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `forward_v050` | `0.2916` | `0.1727` | `0.0144` | `0.2713` | `0.0610` | `2.00e-16` | `6.31e-07` |
| `backward_v050` | `0.2771` | `0.1751` | `0.0138` | `0.2678` | `0.0588` | `2.02e-16` | `7.91e-07` |
| `lateral_left_v050` | `0.2780` | `0.0968` | `0.1738` | `0.2621` | `0.0758` | `0.00e+00` | `6.88e-07` |
| `lateral_right_v050` | `0.2883` | `0.0945` | `0.1738` | `0.2713` | `0.0760` | `1.62e-16` | `5.63e-07` |
| `yaw_left_100` | `0.3136` | `0.1879` | `0.1388` | `0.2675` | `0.0629` | `1.11e-16` | `8.98e-07` |
| `yaw_right_100` | `0.2786` | `0.2066` | `0.0790` | `0.2622` | `0.0647` | `1.39e-16` | `4.73e-07` |

Cycle trends show monotonic accumulation. Example:

```text
lateral_left_v050:
c0 total/y/x/z 0.076/0.022/0.044/0.060
c7 total/y/x/z 0.278/0.174/0.097/0.262

yaw_left_100:
c0 total/y/x/z 0.063/0.012/0.028/0.059
c7 total/y/x/z 0.314/0.139/0.177/0.267
```

## Result

Reproduced. The important finding is negative evidence:

- Replan frame0 body-relative foot coordinates match the current state: `max_frame0_rel_mismatch ~= 0`.
- Playback readback matches the planned playback frame: `~1e-6m`.

So the visible deformation is not caused by Isaac readback, joint writeback, or a frame0 export mismatch. The planner is producing accumulated body-relative foot offsets across repeated mid-step replans.

## Conclusion

The most likely root cause is in the parametric replan contract:

- `decode_parametric_trajectory()` uses current `state.foot_pos` as `foot0` every replan.
- Current `MpcRobotState` does not carry contact state, gait phase, stance anchors, or whether a foot is mid-swing.
- `swing_center` restarts from a fixed base phase every replan.

Therefore, when a long step is interrupted at frame 24, mid-swing/current-offset foot positions become the next segment's fresh nominal start. Repeating this creates body-relative drift. Lateral and yaw commands reveal it more strongly because the accumulated error has body-y/rotational components.

## Follow-Up

Open T302k.15: add phase/contact/stance-anchor continuity to parametric MPC replans. Candidate directions:

- carry planner-owned phase through `MpcTrajectoryManager` into parametric decode;
- identify stance legs and anchor them instead of using current swing feet as fresh nominal starts;
- constrain body-yaw relative foot coordinates toward stable nominal envelopes across replans.

## Git Refs

- Baseline Ref: working tree on top of `1b799cd`
- Candidate Ref: working tree on top of `1b799cd`
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/parametric.py](../../Go2Pvcnn/extension/batch_mpc_planner/parametric.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
  - [../../Go2Pvcnn/tests/mpc_swing_trajectory_quality_probe.py](../../Go2Pvcnn/tests/mpc_swing_trajectory_quality_probe.py)
