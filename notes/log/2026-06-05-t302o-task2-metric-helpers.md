# T302o Task 2 Metric Helpers

## Purpose

Verify Task 2 pure metric helpers for MPC policy evaluation: tracking foot metrics, fixed/sweep/random command helpers, and small-collision per-round env counting.

## Stage

MPC semantic policy evaluation / pure metric helpers.

## Related Todo

- [T302o](../todo/T302o-mpc-policy-eval-plan.md)

## Git Refs

- Baseline Ref: `d6a0d45`
- Candidate Ref: `e84a78c`
- Current Work Ref: `costmap-teacher-ablation`
- Key Files:
  - [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
  - [../../Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py](../../Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py)

## Commands And Results

Original RED, now invalid:

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py -q
```

Result before import isolation: `1 error in 1.62s`, `ModuleNotFoundError: No module named 'Go2Pvcnn'`. This did not prove missing Task 2 helpers; it only proved the test import path was wrong for the exact pytest command.

Corrected RED after isolating test imports and temporarily reverting only helper implementation in the working tree:

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py -q
```

Result: `1 error in 1.62s`, `AttributeError: module 'mpc_policy_eval_under_test' has no attribute 'SmallCollisionRoundAccumulator'`. The temporary negative state was restored before committing.

Final GREEN:

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py -q
```

Result: `3 passed in 1.48s`.

Pycompile:

```bash
python -m py_compile Go2Pvcnn/scripts/mpc_policy_eval.py
```

Result: exit `0`.

Diff check:

```bash
git diff --check -- Go2Pvcnn/scripts/mpc_policy_eval.py Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py
```

Result: exit `0`.

## Result

Pass. Task 2 metric helper behavior is implemented at `33cb1f8`, and the test import isolation review fix is implemented at `e84a78c`.

## Conclusion

The pure helper contract is accepted: tracking metrics report mean, p95, and per-leg error; command generation supports fixed, sweep, and deterministic random modes; small collision accounting counts each env once per round and records first collision step, body names, and max force.

## Follow-up

Continue Task 3 headless rollout skeleton and output files.
