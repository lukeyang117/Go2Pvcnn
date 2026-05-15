# T300e MPC Contact Support And Touchdown Anchor Acceptance

## Purpose

Record the follow-up T300e runtime pass that closed the remaining `backward_fast` stance-airborne residual and produced a clean command-matrix pytest artifact.

## Stage

Production `extension/batch_mpc_planner` nominal/contact/IK loss correctness and IsaacLab runtime acceptance.

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

- `support_stability_loss` now uses a top-k hinge against `runtime.contact_threshold`, with default `min_support_legs=2`, so diffuse sub-threshold contact probabilities cannot export as zero/one stance legs without penalty.
- `ik_fk_residual_loss` now keeps a base mean plus a contact-mass-normalized residual, so sparse contact/touchdown reachability errors are not diluted by all non-contact foot frames.
- `build_nominal_trajectory` now locks post-touchdown stance frames to the computed touchdown target for swings that land inside the horizon; stance no longer snaps back to stale replan-start foot positions after touchdown.
- Review follow-up: finite-horizon touchdown sampling now treats wrap-around touchdowns at the current horizon end as phase `1.0` and samples the last frame, so touchdown losses/exports no longer wrap those events back to the stale replan-start foot point.
- Removed an unused `torch.nn.functional` import from `terrain_clearance.py`.
- Added backend regression coverage for contact-threshold support stability, contact-mass IK/FK residual, post-touchdown nominal anchoring, wrap-around touchdown endpoint sampling, and `root_height` task-config overrides.

## Commands And Results

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Result: `42 passed in 4.13s`.

After the wrap-around touchdown review fix:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Result: `43 passed in 4.60s`.

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/extension/batch_mpc_planner/config.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/contact.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/gait_coupling.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py \
  Go2Pvcnn/extension/batch_mpc_planner/nominal.py \
  Go2Pvcnn/extension/batch_mpc_planner/planner.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py \
  Go2Pvcnn/tests/mpc_root_cause_probe.py
```

Result: exit code `0`.

```bash
MPC_TEST_DEVICE=cuda:2 MPC_ROOT_CAUSE_CYCLES=4 \
MPC_ROOT_CAUSE_SEQUENCES='backward_speeds:backward_slow,backward,backward_fast;mixed_yaw:forward_yaw_right,lateral_left_yaw_right,lateral_right_yaw_left' \
MPC_ROOT_CAUSE_VARIANTS='default:8.0' \
MPC_ROOT_CAUSE_OUTPUT=/tmp/mpc_touchdown_anchor_contact_ikfk_probe.jsonl \
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_root_cause_probe.py
```

Result: exit code `0`, `24` cycle rows.

After the wrap-around touchdown review fix:

```bash
MPC_TEST_DEVICE=cuda:2 MPC_ROOT_CAUSE_CYCLES=4 \
MPC_ROOT_CAUSE_SEQUENCES='backward_speeds:backward_slow,backward,backward_fast;mixed_yaw:forward_yaw_right,lateral_left_yaw_right,lateral_right_yaw_left' \
MPC_ROOT_CAUSE_VARIANTS='default:8.0' \
MPC_ROOT_CAUSE_OUTPUT=/tmp/mpc_touchdown_endpoint_final_probe.jsonl \
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_root_cause_probe.py
```

Result: exit code `0`, `24` cycle rows.

```bash
MPC_TEST_DEVICE=cuda:2 \
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_mpc_runtime_headless.py::test_mpc_runtime_command_matrix_tracks_motion_and_limits_drift -q
```

Result: clean pytest progress `.` and exit code `0`.

```bash
git diff --check
```

Result: exit code `0`.

## Key Metrics

Compared with the prior tuning log where `backward_fast` still had `actual_last_stance_airborne_ratio_mean=0.75` and mean max gap `0.2497m`:

- `backward_fast`:
  - actual last stance airborne ratio mean: `0.0`
  - actual last stance airborne max gap mean: `0.00043m`
  - actual last stance airborne max gap max: `0.00171m`
  - plan last stance airborne ratio mean: `0.0`
  - IK/FK contact residual mean: `1.88e-08m`
- `backward`:
  - actual last stance airborne ratio mean: `0.0`
  - actual last stance airborne max gap mean: `0.00393m`
- `forward_yaw_right`, `lateral_left_yaw_right`, `lateral_right_yaw_left`:
  - actual last stance airborne ratio mean: `0.0`
  - max observed actual stance gap stayed at millimeter scale to low-centimeter scale; largest targeted mixed-yaw max gap was `0.01338m`.

## Result

The remaining `backward_fast` runtime residual was traced to three coupled contract issues: support probability did not match boolean contact export, contact IK/FK residual was diluted by non-contact frames, and nominal post-touchdown stance frames reused stale foot anchors. Fixing those together removed the targeted backward-fast and mixed-yaw stance-airborne failures in the `env_isaacsim` probe.

The subagent review then found a P1 edge case where wrap-around touchdowns sampled phase `0` instead of the finite-horizon endpoint. That has a dedicated red-green regression and is now fixed by using phase `1.0` / non-cyclic sampling for touchdown losses and export while keeping cyclic `swing_end` for contact-window decoding.

## Conclusion

T300e has a clean targeted runtime acceptance artifact for the commands that were previously failing, plus a clean command-matrix pytest selector. Remaining risks are longer-horizon viewer behavior, broader yaw gait probes, and large-scale `4096` runtime counter/throughput stability.

## Follow-Up

- Run longer unmonkeypatched yaw/command-switch probes before treating viewer behavior as visually final.
- Keep `4096` runtime counter extraction as a separate scale-stability issue.

## Git Refs

- Baseline Ref: `65f0d991c201ed50df8be1fa7ad62b3525dae20a`
- Candidate Ref: working tree on top of `65f0d991c201ed50df8be1fa7ad62b3525dae20a`
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/contact.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/contact.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/gait_coupling.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/gait_coupling.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py](../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
