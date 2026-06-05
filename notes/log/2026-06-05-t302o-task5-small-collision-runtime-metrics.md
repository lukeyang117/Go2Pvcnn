# T302o Task 5 Small Collision Runtime Metrics

## Purpose

Implement and locally verify `small_collision` runtime metrics for `Go2Pvcnn/scripts/mpc_policy_eval.py`.

## Stage

MPC semantic policy evaluation / small obstacle collision runtime metrics.

## Related Todo

- [../todo/T302o-mpc-policy-eval-plan.md](../todo/T302o-mpc-policy-eval-plan.md)

## Baseline Ref

- `2f38ad9` after Task 4 tracking runtime metrics and verification notes.

## Candidate Ref

- `7f1e4cd` Task 5 small collision runtime metrics.

## Current Work Ref

- Branch: `costmap-teacher-ablation`

## Key Files

- [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
- [../../Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py](../../Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py)
- [../../Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py](../../Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py)

## Command / Procedure

The implementer inherited an existing Task 5 draft in the shared worktree, so no fresh RED was replayed without rewinding other work.

GREEN:

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py -q
pytest Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py -q
python -m py_compile Go2Pvcnn/scripts/mpc_policy_eval.py
git diff --check -- Go2Pvcnn/scripts/mpc_policy_eval.py Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py notes/todo.md notes/todo/T302o-mpc-policy-eval-plan.md notes/log/index.md notes/log/2026-06-05-t302o-task5-small-collision-runtime-metrics.md
```

## Input Conditions

- Implementer subagent scope only.
- No long IsaacLab run.
- Existing unrelated dirty files preserved:
  - `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
  - deleted legacy `.sh` files under `Go2Pvcnn/scripts/`

## Key Metrics

- Metric helper tests: `7 passed in 1.61s`.
- Static script tests: `7 passed in 0.03s`.
- Pycompile: exit `0`.
- Diff check: exit `0`.
- Aggregate denominator test covers `collided_env_count` sum `3` over `total_env_rounds` `8`, yielding `aggregate_small_collision_env_rate=3/8`.

## Result

Pass locally.

Implemented or confirmed:

- Runtime small collision force source reads `semantic_contact_small.data.force_matrix_w`.
- Per-round accumulator counts each environment at most once after any small-obstacle collision.
- `rounds.jsonl` round summary includes `collided_env_count`, `small_collision_env_rate_per_round`, `first_collision_step_by_env`, `collision_body_names_by_env`, and `round_small_force_max`.
- `summary.json` aggregates `aggregate_small_collision_env_rate`, `total_collided_envs`, `total_env_rounds`, and `round_count`.
- Static contract test now explicitly checks for `force_matrix_w` to guard against height-map approximation regressions.

## Conclusion

Task 5 is locally implemented for the user-defined collision metric:

```text
aggregate_small_collision_env_rate = total_collided_envs / total_env_rounds
```

where `total_env_rounds` is the sum of `num_envs` over rounds.

## Follow-Up

- Main agent should run a real card0/env_isaacsim `small_collision` smoke.
- Task 3 terrain row/col review note remains open before claiming real multi-terrain comparison semantics.
- Continue Task 6 livestream command sync and MPC foot markers after Task 5 review/smoke.
