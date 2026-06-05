# T302o Task 1 Static Contracts

## Purpose

Verify T302o Task 1 static contracts for eval cfgs and CLI skeleton.

## Stage

MPC semantic policy evaluation / static contracts.

## Related Todo

- [T302o](../todo/T302o-mpc-policy-eval-plan.md)

## Git Refs

- Baseline Ref: `a217f02`
- Candidate Ref: `d6a0d45`
- Current Work Ref: `costmap-teacher-ablation`
- Key Files:
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
  - [../../Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py](../../Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py)

## Commands And Results

Initial RED:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_policy_eval_cfgs_enable_reference_without_changing_play Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py -q
```

Result: `4 failed`.

Final GREEN:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_policy_eval_cfgs_enable_reference_without_changing_play Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py -q
```

Result: `4 passed in 2.10s`.

Pycompile:

```bash
python -m py_compile Go2Pvcnn/scripts/mpc_policy_eval.py Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
```

Result: exit `0`.

Staged diff check:

```bash
git diff --cached --check
```

Result: exit `0`.

## Result

Pass. Task 1 static eval cfg and CLI skeleton contracts are implemented and verified at `d6a0d45`.

## Conclusion

Task 1 preserves `scripts/play.py` no-MPC behavior while adding eval-specific cfg classes and a Python-only `mpc_policy_eval.py` CLI skeleton. Runtime rollout metrics are not implemented in Task 1.

## Follow-up

Continue Task 2.
