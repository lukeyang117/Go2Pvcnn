# T302o Foot Trajectory Lag Reproduction

## Purpose

Reproduce and quantify the user-observed lag between MPC planned foot trajectory markers and IsaacLab actual robot foot positions in `mpc_policy_eval.py` visualization.

## Stage

MPC semantic policy evaluation / policy-vs-MPC foot tracking diagnostics.

## Related Todo

- [../todo/T302o-mpc-policy-eval-plan.md](../todo/T302o-mpc-policy-eval-plan.md)

## Baseline Ref

- Working tree after [2026-06-06-1452-t302o-follow-camera-reproduction-fix.md](2026-06-06-1452-t302o-follow-camera-reproduction-fix.md).

## Candidate Ref

- No runtime code change for this investigation. Diagnostics were run as one-off Python snippets against the existing `mpc_policy_eval.py` helpers.

## Current Work Ref

- Branch: `costmap-teacher-ablation`

## Key Files

- [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
- [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
- [../../Go2Pvcnn/extension/mdp/rewards_reference.py](../../Go2Pvcnn/extension/mdp/rewards_reference.py)

## Command / Procedure

Two headless real IsaacLab probes were run with:

- `CUDA_VISIBLE_DEVICES=0`
- `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`
- `--num-envs 1`
- fixed command `"1.0 0.0 0.0"`
- checkpoint `2026-05-31_20-03-27/model_14000.pt`

Probe 1 compared actual feet against MPC cache frames `current_frame_id + shift` for `shift=-8..+8`.

Probe 2 widened the search to `shift=-24..+8` and additionally recorded the best absolute cache frame.

## Input Conditions

- Headless, no livestream rendering.
- Evaluation cfg with MPC reference/cache enabled.
- Policy actions were produced by the loaded RSL-RL checkpoint.
- No changes were made to `mpc_policy_eval.py` runtime code.

## Key Metrics

Probe 1 output:

- `logs/mpc_policy_eval/lag_probe/2026-06-06_15-10-16-225323`
- Steps: `80`
- Warmed samples (`frame_id > 2`): `68`
- `shift=0` average foot error: `0.172880m`
- `shift=-8` average foot error: `0.111690m`
- Best-shift counts: `-8` appeared `51/68` times, hitting the negative search boundary.
- Mean improvement best-vs-current: `0.086713m`

Probe 2 output:

- `logs/mpc_policy_eval/lag_probe_wide/2026-06-06_15-12-03-405289`
- Steps: `60`
- Warmed samples (`frame_id > 2`): `51`
- `shift=0` average foot error: `0.248018m`
- Best average shift: `-3` with average error `0.222049m`
- Best absolute cache-frame counts:
  - frame `0`: `28`
  - frame `2`: `14`
  - frame `5`: `5`
- Mean improvement best-vs-current: `0.082798m`
- Mean improvement frame0-vs-current: `0.077132m`

Follow-up along-command projection output:

- `logs/mpc_policy_eval/along_probe/2026-06-06_15-44-48-104399`
- Steps: `60`
- Warmed samples (`frame_id > 2`): `51`
- Average `(actual_foot - current_mpc_ref_foot)` projected onto current root-forward / command direction: `+0.111534m`
- Per-leg average along-current projection: `[+0.095843, +0.112572, +0.157359, +0.080362]m`
- Per-leg ahead counts with `> 0.02m`: `[30, 42, 44, 30]` out of `51`
- Per-leg behind counts with `< -0.02m`: `[11, 6, 6, 8]` out of `51`
- Average L2 to current reference: `0.206360m`
- Average L2 to frame-0 reference: `0.154213m`

## Result

Reproduced, with direction-specific correction.

## Conclusion

The observed mismatch is not just a one-frame marker/display delay. Cross-frame matching shows actual feet often match earlier MPC cache frames better than `current_frame_ids()`, and many samples match cache frame `0` or another early frame better than the current frame.

However, the user's visual observation is correct in the command direction: when projecting `(actual - current_ref)` onto the current root-forward command direction, actual feet are on average ahead of the current MPC reference by about `0.112m`. Therefore the issue should not be described simply as "actual feet lag behind MPC feet." More precise wording is:

- Along the commanded forward direction, actual feet are usually ahead of the current MPC reference feet.
- In full 3D/L2 shape matching, actual feet often look closer to earlier MPC cache frames than to `current_frame_ids()`.

This suggests a phase/shape mismatch between the learned policy gait and the MPC reference, not a pure visualization delay and not a simple scalar time lag.

There is still a possible timing component: `MpcTrajectoryManager.refresh_from_env()` advances `_phase_counter` at the end of refresh, and rewards/metrics consume `current_reference()` after refresh. But the wide probe shows a larger and variable offset, so a simple one-frame visualization shift is not enough to explain the screenshot.

## Follow-Up

- Do not change runtime code from this log alone.
- Next useful diagnostic: record per-step `phase_counter`, command, contact state, policy action, actual foot velocity, MPC contact state, and command-direction projection to separate forward placement mismatch from gait-phase/shape mismatch.
- Consider comparing actual feet to MPC cache in root frame as well as world frame to separate base-motion drift from leg-phase lag.
