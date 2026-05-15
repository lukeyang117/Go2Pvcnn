# T300e MPC IK/FK And Grounding Runtime Tuning

## Purpose

Record the follow-up T300e runtime pass after the support-plane fix: align IK/FK feasibility loss with the clamped joint output contract, strengthen terrain grounding loss scale, add root-height regularization, and capture `env_isaacsim` evidence for backward-fast/mixed residual commands.

## Stage

Production `extension/batch_mpc_planner` loss correctness and IsaacLab runtime acceptance.

## Related Todo

- [T300e continuous swing-window plan](../todo/T300e-mpc-continuous-swing-window-plan.md)
- [T300 unified dense MPC backend](../todo/T300-unified-dense-mpc-backend.md)

## Input Conditions

- Baseline ref: `65f0d991c201ed50df8be1fa7ad62b3525dae20a`
- Candidate ref: working tree on top of `65f0d991c201ed50df8be1fa7ad62b3525dae20a`
- Python environment: `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`
- Runtime device: `cuda:2`
- Active code path: `Go2Pvcnn/extension/batch_mpc_planner`

## Changes

- `ik_fk_residual_loss` now solves IK with `clamp_to_limits=True`, matching the joint sequence emitted by `plan_segment`; this catches targets that raw IK can solve only by leaving hardware limits.
- Default `ik_fk_residual.weight` is raised to `8.0`, matching the runtime sweep where `ikfk8` cleaned mixed-yaw residuals better than `ikfk1/4`.
- `stance_ground_loss` and `swing_clearance_terrain_loss` now normalize by active contact/swing probability mass instead of the full `[T, legs]` grid.
- Stance/touchdown ground errors now use a small-beta smooth L1 so 5-15 cm foot-height errors are not numerically tiny.
- Added `root_height` loss to keep root z near nominal/current height and prevent the optimizer from lowering the body to trade against IK/velocity objectives.
- Added backend regression tests for clamped IK/FK contract, non-diluted stance grounding, default IK/FK weight, and root-height loss exposure.

## Commands And Results

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Result: `39 passed in 3.28s`.

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/extension/batch_mpc_planner/config.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/gait_coupling.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py \
  Go2Pvcnn/tests/mpc_root_cause_probe.py
```

Result: exit code `0`.

```bash
git diff --check
```

Result: exit code `0`.

```bash
MPC_TEST_DEVICE=cuda:2 MPC_ROOT_CAUSE_CYCLES=4 \
MPC_ROOT_CAUSE_SEQUENCES='backward_speeds:backward_slow,backward,backward_fast;mixed_yaw:forward_yaw_right,lateral_left_yaw_right,lateral_right_yaw_left' \
MPC_ROOT_CAUSE_VARIANTS='default:8.0' \
MPC_ROOT_CAUSE_OUTPUT=/tmp/mpc_continuous_window_root_cause_root_height_default8.jsonl \
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_root_cause_probe.py
```

Result: exit code `0`, `24` cycle rows.

## Key Metrics

Compared with the prior `ikfk8` sweep before stance/root-height tuning:

- Mixed-yaw commands:
  - `forward_yaw_right`, `lateral_left_yaw_right`, and `lateral_right_yaw_left` ended with `actual_last_stance_airborne_ratio_mean = 0.0`.
  - Mixed-yaw contact IK/FK residuals stayed at `0.0` mean in the final probe.
- Backward normal:
  - `backward` ended with `actual_last_stance_airborne_ratio_mean = 0.0`.
- Backward fast:
  - before stance/root-height tuning: `actual_last_stance_airborne_ratio_mean = 1.0`, mean max gap `0.3852m`, plan air ratio `1.0`.
  - after stance/root-height tuning: `actual_last_stance_airborne_ratio_mean = 0.75`, mean max gap `0.2497m`, plan air ratio `0.75`.
  - residual remains: max actual gap still reached `0.3517m` in one cycle.

## Result

The IK/FK loss now matches the final clamped joint playback contract, and the terrain/root-height losses are no longer numerically diluted. Mixed-yaw and normal backward cases are much cleaner in the targeted `env_isaacsim` probe.

## Conclusion

This pass fixes two real loss-contract bugs and one missing support geometry term. It does not fully solve `backward_fast`: the remaining failure is now narrower and appears tied to high-speed backward reachability/velocity trade-offs, not the old yaw/contact-collapse path.

## Follow-Up

- Treat `backward_fast` as the next T300e residual issue: likely needs command-speed-aware stride/root tracking feasibility, not more blind stance-ground weight tuning.
- Re-run a clean command-matrix artifact once the pytest/Isaac output ambiguity is resolved.
- Keep acceptance focused on unmonkeypatched manager/viewer MPC paths.

## Git Refs

- Baseline Ref: `65f0d991c201ed50df8be1fa7ad62b3525dae20a`
- Candidate Ref: working tree on top of `65f0d991c201ed50df8be1fa7ad62b3525dae20a`
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/gait_coupling.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/gait_coupling.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
