# 2026-07-06 MPC QP Row8 Col12 Reachability

## Purpose

Respond to the visual complaint that `mpc_qp` did not look like walking on the required hard terrain. The goal was to test directly, diagnose root cause, and tune QP/loss behavior without adding candidate search or hard repair.

## Stage

MPC-QP backend / continuous Bezier trajectory / hard-terrain reachability.

## Related Todo

[../todo/T302v-mpc-qp-safety-constrained-backend-plan.md](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Commands

Focused static:

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
python -m py_compile Go2Pvcnn/extension/batch_mpc_qp_planner/*.py Go2Pvcnn/tests/test_mpc_qp_backend.py Go2Pvcnn/tests/mpc_qp_hard_terrain_probe.py
```

Required hard tile, single QP:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_hard_terrain_probe.py \
  --device cuda:0 \
  --tiles 8:12 \
  --commands 'forward:0.35,0.0,0.0;diag_left:0.30,0.12,0.0' \
  --cycles 1 \
  --requested-n-frames 25 \
  --playback-frames 25 \
  --qp-iterations 1 \
  --warmup-steps 4
```

Required hard tile, two QP iterations:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_qp_hard_terrain_probe.py \
  --device cuda:0 \
  --tiles 8:12 \
  --commands 'forward:0.35,0.0,0.0;diag_left:0.30,0.12,0.0' \
  --cycles 1 \
  --requested-n-frames 25 \
  --playback-frames 25 \
  --qp-iterations 2 \
  --warmup-steps 4
```

## Input Conditions

- Terrain tile: `row=8`, `col=12`.
- Commands: `forward:0.35,0.0,0.0` and `diag_left:0.30,0.12,0.0`.
- No `CUDA_VISIBLE_DEVICES` hard-code; command used `--device cuda:0`.
- Isaac logs showed four RTX 4090 GPUs visible.

## Root Cause

The bad-looking gait was not primarily the alternating gait phase anymore. On row `8`, col `12`, single-iteration continuous output terrain-bound a swing foot/touchdown near a lower terrain region while root height stayed high. The target was outside the Go2 reachable leg workspace and clamped at the calf lower joint limit.

Before the fix, the hard probe reported max playback/FK readback around `0.43-0.44m`. Diagnostic detail showed worst cases with root z about `2.349m`, target foot z about `1.643m`, hip-relative reach about `0.75m`, and saturated calf joint, while Go2 nominal two-link leg reach is about `0.426m`.

## Changes

- Added readback detail diagnostics to `Go2Pvcnn/tests/mpc_qp_hard_terrain_probe.py`.
- Added fixed-shape continuous reachability update in `Go2Pvcnn/extension/batch_mpc_qp_planner/solver.py`.
- Added gait-segment FK readback update so residuals from the diagonal swing phases update matching Bezier controls.
- Kept z governed by terrain binding and swing clearance instead of arbitrary readback repair.
- Stabilized low-small swing clearance and diagnostics to ignore contact-transition boundary frames.
- Increased continuous body-leg root-lift margin/max to keep FK knee/shank semantic clearance solved in the QP path.
- Added regression coverage in `Go2Pvcnn/tests/test_mpc_qp_backend.py`.

## Metrics

Static:

- `Go2Pvcnn/tests/test_mpc_qp_backend.py`: `54 passed`.
- Pycompile: pass.

Hard tile with `qp_iterations=1`:

- max readback: about `0.0583m`.
- max foot jump: about `0.04277m`.
- max joint jump: about `0.16809rad`.
- FK semantic collision: `0`.
- touchdown-on-small: `0`.
- result: not accepted only because strict readback gate is `0.05m`.

Hard tile with `qp_iterations=2`:

- `viewer_hard_terrain_acceptance_passed=true`.
- max readback: about `0.04941m`.
- max foot jump: about `0.04277m`.
- max joint jump: about `0.16241rad`.
- FK semantic collision: `0`.
- touchdown-on-small: `0`.

## Conclusion

The required hard terrain now passes numerically with `qp_iterations=2`. Single-iteration QP is much better than before but still misses the strict readback gate by about `8mm` on the diagonal-left command.

For visual testing on `row=8`, `col=12`, use `--qp-iterations 2` for now. This follows the current design rule: tune loss/iteration count rather than adding hard repair or candidate search.

## Follow-Up

- Decide whether to keep default `qp_iterations=1` for speed and document `2` for hard-terrain viewer tests, or tune single-iteration loss further.
- Run livestream viewer on row `8`, col `12` with `--qp-iterations 2` for visual confirmation.

## Git Refs

- Baseline Ref: current dirty workspace before this debugging pass.
- Candidate Ref: current dirty workspace after reachability/readback tuning.
- Key Files:
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/solver.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/config.py`
  - `Go2Pvcnn/extension/batch_mpc_qp_planner/losses.py`
  - `Go2Pvcnn/tests/test_mpc_qp_backend.py`
  - `Go2Pvcnn/tests/mpc_qp_hard_terrain_probe.py`
