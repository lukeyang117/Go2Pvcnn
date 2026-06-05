# T302o Task 4 Tracking Runtime Metrics

## Purpose

Implement and locally verify tracking mode runtime metrics for `Go2Pvcnn/scripts/mpc_policy_eval.py`.

## Stage

MPC semantic policy evaluation / tracking runtime metrics.

## Related Todo

- [../todo/T302o-mpc-policy-eval-plan.md](../todo/T302o-mpc-policy-eval-plan.md)

## Baseline Ref

- `52ff128` Task 3 rollout skeleton verification notes.

## Candidate Ref

- `b737977` Task 4 tracking runtime metrics.

## Key Files

- [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
- [../../Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py](../../Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py)
- [../../Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py](../../Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py)

## Command / Procedure

RED:

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py -q
```

GREEN:

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py -q
pytest Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py -q
python -m py_compile Go2Pvcnn/scripts/mpc_policy_eval.py
```

## Input Conditions

- Implementer subagent scope only.
- No long IsaacLab run.
- Existing unrelated dirty files preserved:
  - `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
  - deleted legacy `.sh` files under `Go2Pvcnn/scripts/`

## Key Metrics

- RED: `AttributeError: module 'mpc_policy_eval_under_test' has no attribute 'TrackingRoundAccumulator'`.
- GREEN metric tests: `6 passed in 1.60s`.
- GREEN static tests: `6 passed in 0.03s`.
- Pycompile: exit `0`.

## Result

Pass locally.

Implemented:

- `TrackingRoundAccumulator`.
- Runtime actual feet from `env.scene["robot"].data.body_pos_w` and `.*_foot` body ids.
- Runtime reference feet from `_trajectory_manager.current_reference()["foot_pos_w"]`.
- Fallback reference feet from `_trajectory_reference_cache.foot_pos_w` indexed by `current_frame_ids()`.
- Per-step `metrics.jsonl` tracking object.
- Per-round and top-level summary tracking aggregates.
- Missing/non-finite reference handling with `reference_valid_ratio=0.0`.
- Unique run output directories using microsecond timestamp plus suffix fallback.

## Conclusion

Task 4 local implementation is ready for main-agent real IsaacLab tracking smoke on card 0 with `env_isaacsim`.

## Follow-Up

- Run real tracking smoke with `CUDA_VISIBLE_DEVICES=0` and `/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python`.
- Implement Task 5 small_collision runtime metrics.
- Task 3 terrain row/col review note remains open: `--terrain-rows/--terrain-cols` currently control generated grid dimensions, not semantic selection of original row/col ids. Resolve before relying on multi-terrain comparison metrics.
