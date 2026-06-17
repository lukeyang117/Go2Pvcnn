# MPC Policy Eval Foot Phase Reproduction

## Purpose

Reproduce the user-observed mismatch between MPC planned foot trajectories and the trained policy's realized foot swing/landing trajectory in tracking eval.

## Stage

`mpc_policy_eval.py` tracking mode / MPC reference cache / policy-vs-reference foot tracking.

## Related Todo

- [../todo/T302o-mpc-policy-eval-plan.md](../todo/T302o-mpc-policy-eval-plan.md)

## Commands

User command reproduced:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 300s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode tracking \
  --headless \
  --device cuda:0 \
  --num-envs 4 \
  --num-rounds 1 \
  --max-steps 20 \
  --run-dir 2026-05-31_20-03-27 \
  --checkpoint model_14000.pt \
  --terrain-rows 0 \
  --terrain-cols 0 \
  --command-mode fixed \
  --command "0.4 0.0 0.0" \
  --output-dir logs/mpc_policy_eval/tracking_smoke
```

Follow-up one-off phase probe compared actual feet against every horizon frame in the MPC reference cache.

## Output

- Tracking eval output:
  - [../../logs/mpc_policy_eval/tracking_smoke/2026-06-16_18-03-35-772458/summary.json](../../logs/mpc_policy_eval/tracking_smoke/2026-06-16_18-03-35-772458/summary.json)
  - [../../logs/mpc_policy_eval/tracking_smoke/2026-06-16_18-03-35-772458/metrics.jsonl](../../logs/mpc_policy_eval/tracking_smoke/2026-06-16_18-03-35-772458/metrics.jsonl)
- Phase probe output:
  - [../../logs/mpc_policy_eval/tracking_phase_probe/2026-06-16_18-03-command04-repro/summary.json](../../logs/mpc_policy_eval/tracking_phase_probe/2026-06-16_18-03-command04-repro/summary.json)
  - [../../logs/mpc_policy_eval/tracking_phase_probe/2026-06-16_18-03-command04-repro/phase_rows.jsonl](../../logs/mpc_policy_eval/tracking_phase_probe/2026-06-16_18-03-command04-repro/phase_rows.jsonl)

## Key Metrics

Tracking eval:

- Exit code `0`.
- `reference_valid_ratio=1.0`.
- `foot_tracking_error_mean_m=0.0934109`.
- `foot_tracking_error_p95_m=0.2085816`.
- Per-leg mean foot errors:
  - FL `0.05949m`
  - FR `0.14832m`
  - RL `0.12223m`
  - RR `0.04360m`
- Command source matches exactly:
  - `command_body_match_max_abs_error=0.0`
- Planned root/foot direction is aligned with requested forward command:
  - `planned_root_direction_cosine=0.9999994`
  - lateral ratio `0.00104`

Phase probe:

- Warm-step mean current-frame error: `0.09945m`.
- Warm-step mean best-horizon-frame error: `0.03149m`.
- Warm-step mean frame-0 error: `0.03158m`.
- Current frame minus best frame error: `0.06796m`.
- Best-frame histogram over warm env-step samples:
  - frame `0`: `70`
  - frame `1`: `2`
  - frame `2`: `4`
  - all later frames: `0`
- Mean contact mismatch against current reference: `0.4901`.
- Mean actual foot z minus current reference foot z: `-0.04781m`.

## Result

Reproduced. The user's visual impression is backed by numeric evidence.

The policy's realized foot positions are much closer to the earliest MPC cache frames than to `manager.current_frame_ids()`. Actual feet are also lower than the current reference feet on average, which matches the observation that the policy appears to land earlier than the MPC swing reference.

## Conclusion

This reproduction does not point to command mismatch: policy and MPC both receive `[0.4, 0.0, 0.0]`, and the planned root/leg direction is forward-aligned.

It also matches the earlier T302o timebase finding: eval reads a synchronous post-step cache, not an async planner thread. The remaining issue is a gait/phase/reference-consumption mismatch: after the first step, current reference frame advances while the trained policy's realized feet remain closer to frame 0/early reference frames.

## Follow-Up

Do not tune MPC losses from this reproduction alone. Next investigation should compare:

- reward-consumed frame vs post-step `current_frame_ids()`;
- policy contact schedule vs MPC `contact_state`;
- whether tracking eval should report best-frame/phase-lag diagnostics by default;
- whether the trained policy was learned against a different reference phase convention.

## Git Refs

- Baseline Ref: current working tree on `costmap-teacher-ablation`
- Candidate Ref: no production code change
- Key Files:
  - [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
  - [../../Go2Pvcnn/extension/mdp/rewards_reference.py](../../Go2Pvcnn/extension/mdp/rewards_reference.py)
