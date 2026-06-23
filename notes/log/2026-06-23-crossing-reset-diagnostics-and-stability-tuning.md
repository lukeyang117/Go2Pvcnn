# Crossing Reset Diagnostics And Stability Tuning

## Purpose

Address the observed behavior where the policy visually starts to step over small obstacles but still falls and resets. The goal is to expose where resets happen relative to the crossing sequence and make the flat-small training signal less "jump over" dominated and more stability preserving.

## Stage

- `mpc_policy_eval.py` controlled crossing diagnostics
- Flat-small RL reward weighting
- Related todos:
  - [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)
  - [../todo/T302o-mpc-policy-eval-plan.md](../todo/T302o-mpc-policy-eval-plan.md)

## Changes

`Go2Pvcnn/scripts/mpc_policy_eval.py`:

- `ControlledCrossingAccumulator` now records reset diagnostics:
  - `reset_reason_counts`
  - `reset_stage_counts`
  - `first_reset_step_by_env`
  - `first_reset_reason_by_env`
  - `first_reset_stage_by_env`
  - `reset_after_foot_over_count`
  - `reset_after_root_crossed_count`
- Runtime controlled-crossing loop reads `bad_orientation`, `base_contact`, and `time_out` from IsaacLab `termination_manager`.

`Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`:

- Reduced `semantic_foot_over_clearance` weight from `1.0` to `0.12`.
- Strengthened flat-small-only stability terms:
  - `flat_orientation_l2`: `-2.5 -> -3.5`
  - `base_angular_velocity`: `-0.05 -> -0.12`
  - `feet_slide`: `-0.1 -> -0.18`
- Kept `action_rate` unchanged so the policy can still make fast swing corrections.

## TDD Evidence

RED tests:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q \
  Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py::test_controlled_crossing_accumulator_records_reset_stage_after_foot_over
```

Failed before implementation with missing `update_reset_diagnostics`.

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract
```

Failed before implementation because `semantic_foot_over_clearance.weight` was `1.0`.

GREEN tests:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q \
  Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py::test_controlled_crossing_accumulator_records_reset_stage_after_foot_over
```

Result: `1 passed`.

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract
```

Result: `1 passed`.

## Verification

Focused:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q \
  Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py \
  Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py
```

Result: `24 passed`.

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_flat_small_avoidance_cfg_static_contract \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_policy_eval_cfgs_enable_reference_without_changing_play
```

Result: `2 passed`.

Compile / whitespace:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/scripts/mpc_policy_eval.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
git diff --check
```

Both exit `0`.

Real smoke:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 600s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --headless --device cuda:0 \
  --mode controlled_crossing \
  --run-dir /mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-17_12-01-10 \
  --checkpoint model_14700.pt \
  --num-envs 4 \
  --num-rounds 1 \
  --max-steps 200 \
  --terrain-rows 0,1,2,3 \
  --terrain-cols 0 \
  --crossing-speeds 0.6,0.8 \
  --crossing-lateral-offsets=-0.08,0.08 \
  --crossing-obstacles-per-env 16 \
  --output-dir logs/mpc_policy_eval/flat_small_20260617_120110_model14700_crossing_reset_diag_smoke
```

Output:

[../../logs/mpc_policy_eval/flat_small_20260617_120110_model14700_crossing_reset_diag_smoke/2026-06-23_21-24-23-657215/summary.json](../../logs/mpc_policy_eval/flat_small_20260617_120110_model14700_crossing_reset_diag_smoke/2026-06-23_21-24-23-657215/summary.json)

Key smoke metrics:

- opportunity envs: `2/4`
- root crossed: `1`
- foot-over: `0`
- small contact: `1`
- success: `0`
- reset envs: `2`
- reset reasons: `bad_orientation=2`, `base_contact=0`, `time_out=0`
- reset stages: `before_opportunity=1`, `before_foot_over=1`

The short smoke is not a behavior acceptance test. Its purpose was to verify that reset diagnostics are emitted and that the tuned reward weights are active in the real env.

## Conclusion

The code now lets the next controlled crossing eval answer whether falls happen before obstacle engagement, after foot-over, or after root crossing. The first short smoke already confirms `bad_orientation` is visible in the new diagnostics.

The reward tuning intentionally favors stability after/before crossing rather than making foot-over stronger. This should be evaluated with a fresh resumed training run rather than by expecting old `model_14700.pt` to improve immediately.

## Follow-up

- Run a short resumed training with the new weights.
- Re-run full controlled crossing and compare:
  - `reset_reason_counts`
  - `reset_stage_counts`
  - `foot_over_count`
  - `root_crossed_count`
  - `small_overpass_success_count`
- If resets concentrate in `after_foot_over_before_root_cross` or `after_root_cross`, add a crossing-recovery curriculum criterion instead of further increasing foot-over reward.

## Git Refs

- Baseline Ref: working tree
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py](../../Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
