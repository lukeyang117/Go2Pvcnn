# 2026-07-07 MPC QP Idle Jitter Root/Foot Displacement Repro

## Purpose

Reproduce the user's report that `mpc_qp` still jitters badly when no velocity command is given, and record root/foot displacement rather than relying on visual impression.

## Stage

MPC-QP backend / viewer runtime diagnostics / idle playback versus real env stepping.

## Related Todo

[T302v MPC QP backend](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Procedure

Added a diagnostic-only probe:

```bash
Go2Pvcnn/tests/mpc_qp_idle_jitter_probe.py
```

The probe has two modes:

- `playback`: plan zero-command `mpc_qp` and directly write each planned frame into the displayed robot, matching viewer direct playback.
- `env-step`: do not play a reference; instead step IsaacLab with zero actions and record actual root/foot motion.

Commands run on visible GPU0:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_idle_jitter_probe.py --device cuda:0 --cycles 3 --requested-n-frames 25 --playback-frames 25 --qp-iterations 1 --terrain task
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_idle_jitter_probe.py --device cuda:0 --cycles 3 --requested-n-frames 25 --playback-frames 25 --qp-iterations 1 --terrain task --terrain-row 8 --terrain-col 12
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_idle_jitter_probe.py --device cuda:0 --cycles 2 --requested-n-frames 25 --playback-frames 50 --qp-iterations 1 --terrain task --mode env-step
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_idle_jitter_probe.py --device cuda:0 --cycles 2 --requested-n-frames 25 --playback-frames 50 --qp-iterations 1 --terrain task --terrain-row 8 --terrain-col 12 --mode env-step
```

## Key Metrics

Direct playback, task terrain:

- `max_planned_root_step_m=0`
- `max_planned_foot_step_m=0`
- `max_actual_root_step_m=0`
- `max_actual_foot_step_m=0`
- `max_foot_planned_vs_actual_m≈1.05e-6`

Direct playback, row `8`, col `12`:

- `max_planned_root_step_m=0`
- `max_planned_foot_step_m=0`
- `max_actual_root_step_m=0`
- `max_actual_foot_step_m=0`
- `max_foot_planned_vs_actual_m≈2.14e-6`

Zero-action env step, task terrain:

- `max_actual_root_step_m≈0.02339`
- `max_actual_root_total_delta_m≈0.13160`
- `max_actual_foot_step_m≈0.01487`
- `max_actual_foot_total_delta_m≈0.06240`

Zero-action env step, row `8`, col `12`:

- `max_actual_root_step_m≈8.68562`
- `max_actual_root_total_delta_m≈24.49031`
- `max_actual_foot_step_m≈9.01197`
- `max_actual_foot_total_delta_m≈24.85799`

## Result

Reproduced real idle motion in the actual Isaac env-step path, but not in `mpc_qp` direct playback/reference output.

The zero-command `mpc_qp` reference itself is static in both default task terrain and row `8`, col `12`. The large motion appears when the robot is allowed to run under zero actions / physics, especially after moving to row `8`, col `12`.

## Conclusion

This evidence points away from "the zero-command `mpc_qp` trajectory is jittering" and toward a runtime/viewer/env-step stability issue: the robot is physically moving or falling when zero actions are stepped. The next debugging pass should inspect the live viewer path the user is watching and confirm whether it is direct playback, paused render, env step, policy action, or a reset/grounding path.

## Follow-Up

- If the user is using the planner viewer, instrument the actual main loop to print `playback_path`, `need_replan`, `playback_frame`, command values, and actual root/foot deltas.
- If the user is using policy/env playback, debug zero-action standstill stability separately from `mpc_qp` reference generation.
- Do not tune QP losses for this idle symptom unless the main-loop instrumentation shows nonzero planned reference motion.

## Git Refs

- Baseline Ref: dirty worktree with `mpc_qp` local suite green
- Candidate Ref: dirty worktree after diagnostic probe
- Key Files:
  - `Go2Pvcnn/tests/mpc_qp_idle_jitter_probe.py`
  - `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
