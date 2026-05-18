# T302 MPC Optimize-Steps Small Sweep

## Purpose

Compare T302 strict metric sensitivity for `mpc_optimize_steps` values `5`, `15`, `20`, and `25` while avoiding heavy server usage. The user requested only about `10` environments and flexible GPU selection because the server GPUs are shared.

## Stage

Production `extension/batch_mpc_planner` T302 strict metric sampling.

## Related Todo

[T302](../todo/T302-mpc-body-leg-height-field-collision-safety.md) and [T302g](../todo/T302g-mpc-semantic-rl-training-config.md).

## Command / Procedure

- Checked GPU memory with `nvidia-smi`; selected `cuda:1`, which had about `24081 MB` free.
- Ran single-process IsaacLab headless probes with `num_envs=10`.
- Completed:
  - low-small obstacle, commands `forward`, `backward`, `lateral_left`, `lateral_right`
  - high-small obstacle, command `forward`
- Attempted to continue to large/cobblestone in the same process, but IsaacLab fixture close/reopen hung twice. The hung processes were stopped to avoid occupying the shared GPU.

Output files:

- [../../tmp/t302_optimize_steps_sweep/steps_5_15_20_25.jsonl](../../tmp/t302_optimize_steps_sweep/steps_5_15_20_25.jsonl)
- [../../tmp/t302_optimize_steps_sweep/high_large_steps_5_15_20_25.jsonl](../../tmp/t302_optimize_steps_sweep/high_large_steps_5_15_20_25.jsonl)

## Input Conditions

- Device: `cuda:1`
- Envs: `10`
- Steps swept: `5`, `15`, `20`, `25`
- Heavy-loss scheduling unchanged.
- T302 loss weights unchanged.
- Acceptance subset rows: `20` total JSONL rows.

## Key Metrics

| optimize steps | rows | strict failures | acceptance failures | max swing foot collision | max shank collision | max stance semantic count | min swing foot clearance | min shank clearance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 5 | 1 | 4 | 0.0104167 | 0.0 | 0 | -0.004436 | 0.218333 |
| 15 | 5 | 1 | 1 | 0.0 | 0.005 | 0 | 0.012398 | -0.048930 |
| 20 | 5 | 0 | 0 | 0.0 | 0.0 | 0 | 0.030844 | 0.056421 |
| 25 | 5 | 1 | 1 | 0.0 | 0.0 | 1 | 0.026880 | 0.085936 |

Observed failures:

- `5` steps:
  - low-small forward: swing-foot collision ratio `0.0104167`, min swing clearance `-0.004436m`
  - low-small lateral-left/right: did not cross
  - high-small forward: `min_dist=0.137636m`, required `0.14m`
- `15` steps:
  - high-small forward: shank collision ratio `0.005`, min shank clearance `-0.048930m`
- `20` steps:
  - no failures in the completed low-small + high-small subset
- `25` steps:
  - low-small backward: stance semantic count `1`

## Result

Partial pass. Under the completed representative subset, `20` optimize steps was the only swept value with zero strict and zero acceptance failures. `5` clearly degrades T302 behavior. `15` fixes low-small but leaves a high-small shank collision. `25` mostly works but produced one low-small backward stance-semantic anomaly in this run.

## Conclusion

For the sampled subset, `20` steps is the best candidate to investigate further. It may reduce MPC runtime compared with the current `24/25`-like quality regime while preserving the completed T302 metrics. This is not enough to replace the T302 strict baseline yet because large-obstacle and cobblestone rows were not completed in this resource-limited run.

## Follow-Up

- Run large forward/yaw and cobblestone command rows for `20` and current baseline only, preferably as separate short-lived commands to avoid fixture close/reopen hangs.
- If `20` passes the full T302 strict matrix, compare MPC planning time against `24`.
- Do not adopt `5` or `15` for production T302/T302g based on current evidence.

## Git Refs

- Baseline Ref: working tree on top of `946811f`
- Candidate Ref: working tree, 2026-05-18 15:21 CST
- Key Files:
  - [../../Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py](../../Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py)
  - [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
