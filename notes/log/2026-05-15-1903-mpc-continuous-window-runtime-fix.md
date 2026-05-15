# T300e MPC Continuous Window Runtime Fix

## Purpose

Record the follow-up T300e runtime/debug pass after the continuous swing-window implementation: fix NaN gradients in touchdown/support losses, harden zero-command behavior, repair support-plane roll/pitch yaw-frame semantics, and capture IsaacLab runtime probe evidence from `env_isaacsim`.

## Stage

Production `extension/batch_mpc_planner` runtime verification and loss correctness.

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

- Added safe differentiable norms in terrain/support sampling and touchdown surface loss so flat terrain/support-distance zero no longer produces NaN x/y gradients.
- Added zero-command standstill output contract in `plan_segment`: root/rpy/feet/joints hold current IsaacLab state, contact is all stance, and touchdown exports use current foot positions.
- Fixed `support_plane_roll_pitch_loss` to fit support points in the root yaw frame before comparing estimated roll/pitch to `root_rpy[..., :2]`.
- Updated yaw playback test tolerance for T300e: roll/pitch are now planned support-plane values, while playback still must match planned RPY.
- Added focused backend regression tests for flat-ground touchdown gradients, finite optimized trajectories, zero-command standstill, and yaw-frame support-plane roll/pitch.

## Commands And Results

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_touchdown_surface_loss_has_finite_flat_ground_gradients -q
```

Result: passed.

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_support_plane_roll_pitch_uses_root_yaw_frame -q
```

Red/green result: failed before the yaw-frame fix, then passed after the fix.

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Result: `35 passed in 3.38s`.

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile \
  Go2Pvcnn/extension/batch_mpc_planner/losses/gait_coupling.py \
  Go2Pvcnn/extension/batch_mpc_planner/terrain.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py \
  Go2Pvcnn/extension/batch_mpc_planner/planner.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py \
  Go2Pvcnn/tests/test_mpc_runtime_headless.py
```

Result: exit code `0`.

```bash
MPC_TEST_DEVICE=cuda:2 MPC_ROOT_CAUSE_CYCLES=4 \
MPC_ROOT_CAUSE_OUTPUT=/tmp/mpc_continuous_window_root_cause_support_plane_fix.jsonl \
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_root_cause_probe.py
```

Result: exit code `0`, JSONL rows `253` (`168` cycle rows, `42` runtime segment summaries).

## Key Metrics

- Backend suite: `35 passed`.
- Flat-ground touchdown/support gradients: finite after safe norm fix.
- Root-cause runtime probe:
  - baseline runtime segment summaries: `21`
  - `ikfk` variant segment summaries: `21`
  - baseline mean `ik_fk_foot_err_mean_mean`: `0.1080m`
  - `ikfk` mean `ik_fk_foot_err_mean_mean`: `0.0470m`
  - baseline mean actual last-frame stance airborne ratio: `0.1429`
  - `ikfk` mean actual last-frame stance airborne ratio: `0.0714`
  - yaw-left segment after fix showed last-frame actual/plan stance airborne ratio `0.0` and IK/FK/contact errors near `1e-7`.

## Runtime Output Caveat

Selected `test_mpc_runtime_headless.py` cases and probes can run under `env_isaacsim`, but pytest console output remains noisy/ambiguous for one command-matrix selector: one rerun returned exit code `0` while stdout only contained `F`, and a JUnit XML attempt produced no XML. A previous `-vv -s` log also had Isaac/manager `FAILED[[...]]` text without a pytest failure summary. Treat the JSONL probes and focused backend suite as the reliable evidence from this pass, not the single-character pytest progress output.

## Result

The NaN/contact-collapse regression is fixed, zero-command standstill is explicit, and support-plane roll/pitch now respects root yaw. The runtime probe confirms the planner no longer collapses into all-NaN/all-false contact on the tested real IsaacLab path.

## Conclusion

T300e is locally and partially runtime verified, but broad runtime behavior is not fully accepted yet. The new design is substantially healthier than the first continuous-window implementation, while backward/fast/mixed commands still show residual IK/FK and stance-airborne risk in the root-cause probe.

## Follow-Up

- Tune or redesign the IK/FK/contact feasibility weights for backward-fast and mixed-yaw residuals.
- Keep acceptance focused on unmonkeypatched manager/viewer MPC paths; probe variants are diagnostic only.
- Re-run a cleaner command-matrix acceptance once pytest/Isaac output can produce an unambiguous failure/pass artifact.
- Decide whether swing-phase foot XY over semantic obstacles should be fully forbidden or only penalized near obstacle contact/low clearance.

## Git Refs

- Baseline Ref: `65f0d991c201ed50df8be1fa7ad62b3525dae20a`
- Candidate Ref: working tree on top of `65f0d991c201ed50df8be1fa7ad62b3525dae20a`
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/terrain.py](../../Go2Pvcnn/extension/batch_mpc_planner/terrain.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/gait_coupling.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/gait_coupling.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
  - [../../Go2Pvcnn/tests/test_mpc_runtime_headless.py](../../Go2Pvcnn/tests/test_mpc_runtime_headless.py)
