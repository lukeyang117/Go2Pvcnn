# 2026-07-07 MPC QP Idle Jitter MPC Baseline A/B

## Purpose

Respond to the user's correction that existing `mpc` and earlier `mpc_qp` did not show the same fast no-key jitter. The diagnostic must compare current `mpc_qp` against `mpc` under the same stop-after-motion playback sequence and use quantitative metrics before drawing a conclusion.

## Stage

MPC-QP viewer/runtime diagnostics / planner replan continuity.

## Related Todo

[T302v MPC QP backend](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Procedure

Extended the diagnostic-only probe:

```bash
Go2Pvcnn/tests/mpc_qp_idle_jitter_probe.py
```

The probe now supports both backends and motion-to-idle reproduction:

```bash
--planner-backend mpc|mpc_qp
--pre-command 0.45,0,0
--pre-cycles 2
--cycles 4
```

The important A/B condition is two moving replans followed by two zero-command replans. This is closer to the viewer report than pure zero-command startup.

## Metrics

Two metric families are used:

- **Per-frame playback motion**: `max_actual_root_step_m`, `max_actual_foot_step_m`, `max_actual_joint_step_rad`. This measures how much the played robot state changes frame-to-frame inside a planned segment.
- **Replan trajectory change**: `planned_*_trajectory_replan_delta_*`. This compares the whole newly planned horizon against the previous planned horizon. This is different from first-frame boundary continuity and catches the future trajectory/marker shape changing sharply across replans.

The first-frame boundary metric is still recorded, but it is not the main conclusion metric here because current `mpc_qp` keeps the first frame nearly continuous.

## Key Metrics

Stop-after-motion playback, `--pre-command 0.45,0,0 --pre-cycles 2 --cycles 4`.

`mpc` baseline:

- moving-cycle `max_actual_root_step_m≈0.01136`
- moving-cycle `max_actual_foot_step_m≈0.04012`
- moving-cycle `max_actual_joint_step_rad≈0.33939`
- max planned root full-trajectory replan delta `≈0.17988m`
- max planned foot full-trajectory replan delta `≈0.21539m`
- max planned joint full-trajectory replan delta `≈1.07066rad`
- first idle cycle actual step metrics are `0`
- second idle cycle trajectory replan deltas are `0`

Current `mpc_qp`:

- moving-cycle `max_actual_root_step_m≈0.10657`
- moving-cycle `max_actual_foot_step_m≈0.09633`
- moving-cycle `max_actual_joint_step_rad≈1.26483`
- max planned root full-trajectory replan delta `≈0.71195m`
- max planned foot full-trajectory replan delta `≈0.48323m`
- max planned joint full-trajectory replan delta `≈3.45675rad`
- first-frame planned foot replan-boundary delta is only `≈7.57e-7m`
- first idle cycle actual step metrics are `0`

## Result

The user-visible complaint is supported by the A/B metrics, but the failure is not a simple first-frame discontinuity:

- current `mpc_qp` moving root step is about `9.4x` the `mpc` baseline
- current `mpc_qp` moving foot step is about `2.4x` the `mpc` baseline
- current `mpc_qp` moving joint step is about `3.7x` the `mpc` baseline
- current `mpc_qp` full-horizon root replan delta is about `4.0x` the `mpc` baseline
- current `mpc_qp` full-horizon foot replan delta is about `2.2x` the `mpc` baseline
- current `mpc_qp` full-horizon joint replan delta is about `3.2x` the `mpc` baseline

## Conclusion

The current `mpc_qp` regression is excessive whole-horizon trajectory reshaping and overly large moving-frame root/foot/joint changes compared with `mpc`, especially after motion-to-idle replans.

The first frame at the replan boundary is already almost continuous, so fixing only the segment boundary is not enough. The next planner fix should target horizon-level replan consistency and moving-segment smoothness inside the isolated `mpc_qp` path.

## Follow-Up

- Keep this A/B as the main idle/stop-after-motion jitter metric.
- Do not blame the viewer/env as the primary conclusion for this symptom unless the same A/B metric is also bad for `mpc`.
- Candidate fix direction, if implementation resumes: warm-start/trust-region or smoothness over the whole horizon relative to the previous `mpc_qp` plan, without candidate endpoints, touchdown lookup, hard repair, or changes to the default `mpc` backend.

## Git Refs

- Baseline Ref: dirty worktree with current `mpc_qp` implementation and `mpc` baseline available through the same diagnostic probe.
- Candidate Ref: dirty worktree after diagnostic probe extension.
- Key Files:
  - `Go2Pvcnn/tests/mpc_qp_idle_jitter_probe.py`
  - `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
