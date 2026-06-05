# T302o Task 3 Rollout Skeleton

## Purpose

Verify Task 3 rollout skeleton/output files and real IsaacLab smoke.

## Stage

MPC semantic policy evaluation / rollout skeleton.

## Related Todo

- [T302o](../todo/T302o-mpc-policy-eval-plan.md)

## Git Refs

- Baseline Ref: `7f5e481`
- Candidate Ref: `2fe1870`
- Current Work Ref: `costmap-teacher-ablation`
- Key Files:
  - [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
  - [../../Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py](../../Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py)

## Commands And Results

RED static:

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py -q
```

Result: `3 failed, 2 passed in 0.04s`.

GREEN static:

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py -q
```

Result: `5 passed in 0.02s`.

Pycompile:

```bash
python -m py_compile Go2Pvcnn/scripts/mpc_policy_eval.py
```

Result: exit `0`.

Diff check:

```bash
git diff --check -- Go2Pvcnn/scripts/mpc_policy_eval.py Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py
```

Result: exit `0`.

Real IsaacLab smoke:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 300s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode tracking \
  --headless \
  --device cuda:0 \
  --num-envs 1 \
  --num-rounds 1 \
  --max-steps 2 \
  --run-dir 2026-05-31_20-03-27 \
  --checkpoint model_14000.pt \
  --terrain-rows 0 \
  --terrain-cols 0 \
  --command-mode fixed \
  --command "0.1 0.0 0.0" \
  --output-dir logs/mpc_policy_eval/task3_smoke
```

Result: exit `0`.

## Output

- Output dir: `logs/mpc_policy_eval/task3_smoke/2026-06-05_17-00-41`
- Output files:
  - `config.json`
  - `metrics.jsonl`: 2 lines
  - `rounds.jsonl`: 1 line
  - `summary.json`: `total_steps` 2, `round_count` 1

## Checkpoint Note

`Go2Pvcnn/logs/...` was absent; the script used the repo-root `logs/rsl_rl/...` fallback.

## Result

Pass.

## Conclusion

Task 3 rollout skeleton and output-file contract are verified against static tests, pycompile, diff check, and a real IsaacLab smoke.

## Follow-up

Task 4 tracking metrics next.
