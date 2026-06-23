# Train Keep Std Resume Option

## Purpose

Add an explicit `--keep_std` training CLI option for continued flat-small training from an existing checkpoint, so resume can optionally load the checkpoint policy action std instead of resetting it to the current initialized value.

## Stage

Training entrypoint / RSL-RL checkpoint loading.

## Related Todo

- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Command / Procedure

RED:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/test_train_script_static.py Go2Pvcnn/tests/test_rsl_runner_checkpoint_static.py
```

Result: expected failure before implementation, `3 failed, 3 passed`; missing `--keep_std`, missing runner `keep_std` load contract.

GREEN:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/test_train_script_static.py Go2Pvcnn/tests/test_rsl_runner_checkpoint_static.py
```

Result: `6 passed in 0.03s`.

Additional checks:

```bash
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/scripts/train.py Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py Go2Pvcnn/tests/test_train_script_static.py Go2Pvcnn/tests/test_rsl_runner_checkpoint_static.py
git diff --check -- Go2Pvcnn/scripts/train.py Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py Go2Pvcnn/tests/test_train_script_static.py Go2Pvcnn/tests/test_rsl_runner_checkpoint_static.py
```

Result: both exit `0`.

## Input Conditions

- Current branch already had local flat-small eval and reset-diagnostic changes.
- Existing runner behavior unconditionally dropped `std` from loaded checkpoints, so normal resume reset action std to init value.

## Key Metrics

- Focused static tests: `6 passed`.
- Pycompile: exit `0`.
- Diff check: exit `0`.

## Result

- [../../Go2Pvcnn/scripts/train.py](../../Go2Pvcnn/scripts/train.py) now exposes `--keep_std`.
- Resume calls `runner.load(checkpoint_path, keep_std=args_cli.keep_std)`.
- [../../Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py](../../Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py) keeps the previous default behavior with `keep_std=False`: checkpoint `std` is removed before `load_state_dict`.
- With `--keep_std`, checkpoint `std` remains in the loaded model state dict and is restored with the rest of actor-critic parameters.

## Conclusion

For continuing from `2026-06-17_12-01-10/model_14700.pt` while preserving that checkpoint's learned action noise, run training with both `--resume` and `--keep_std`. Omitting `--keep_std` keeps the previous safer behavior of resetting policy std to the current initialization.

## Follow-Up

No IsaacLab smoke was run because the change is isolated to CLI parsing and checkpoint state loading. A short resumed training smoke is still useful before a long run if GPU time is available.

## Git Refs

- Baseline Ref: `704db79`
- Candidate Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/scripts/train.py](../../Go2Pvcnn/scripts/train.py)
  - [../../Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py](../../Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py)
  - [../../Go2Pvcnn/tests/test_train_script_static.py](../../Go2Pvcnn/tests/test_train_script_static.py)
  - [../../Go2Pvcnn/tests/test_rsl_runner_checkpoint_static.py](../../Go2Pvcnn/tests/test_rsl_runner_checkpoint_static.py)
