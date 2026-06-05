# T302o MPC Policy Eval Smoke

## Purpose

Verify the Python-only MPC policy evaluation entry, tracking metrics, small-collision env-rate metric, and livestream marker startup path.

## Stage

MPC semantic policy evaluation.

## Related Todo

- [T302o](../todo/T302o-mpc-policy-eval-plan.md)

## Git Refs

- Baseline Ref: `f46eab8`
- Candidate Ref: `996ce1f`
- Current Work Ref: `costmap-teacher-ablation`
- Pre-existing unrelated dirty files:
  - `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
  - deleted legacy `.sh` files under `Go2Pvcnn/scripts/`

## Commands

Static regression:

```bash
pytest Go2Pvcnn/tests/test_mpc_policy_eval_metrics.py \
  Go2Pvcnn/tests/test_mpc_policy_eval_script_static.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py::test_mpc_policy_eval_cfgs_enable_reference_without_changing_play -q
python -m py_compile Go2Pvcnn/scripts/mpc_policy_eval.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
```

Tracking smoke:

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

Small-collision smoke:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 300s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode small_collision \
  --headless \
  --device cuda:0 \
  --num-envs 4 \
  --num-rounds 1 \
  --max-steps 20 \
  --run-dir 2026-05-31_20-03-27 \
  --checkpoint model_14000.pt \
  --command-mode random \
  --random-command-interval 5 \
  --small-count-per-tile 80 \
  --output-dir logs/mpc_policy_eval/small_collision_smoke
```

Livestream startup smoke:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 300s \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/mpc_policy_eval.py \
  --mode tracking \
  --livestream 2 \
  --device cuda:0 \
  --num-envs 1 \
  --num-rounds 1 \
  --max-steps 2 \
  --run-dir 2026-05-31_20-03-27 \
  --checkpoint model_14000.pt \
  --command-mode fixed \
  --command "0.4 0.0 0.0" \
  --output-dir logs/mpc_policy_eval/visual_tracking_smoke
```

## Key Metrics

- Static regression: `16 passed in 1.78s`.
- Pycompile: exit `0`.
- Tracking smoke: exit `0`; output `logs/mpc_policy_eval/tracking_smoke/2026-06-05_17-43-26-548577`; `metrics.jsonl` 20 lines; tracking mean `0.11181626562029123`, p95 `0.2721988260746002`, `reference_valid_ratio=1.0`, valid steps `20`; summary finite.
- Small-collision smoke: exit `0`; output `logs/mpc_policy_eval/small_collision_smoke/2026-06-05_17-44-18-878827`; `metrics.jsonl` 20 lines; round `collided_env_count=0`, `num_envs=4`, `small_collision_env_rate_per_round=0.0`; top-level `total_env_rounds=4`, `aggregate_small_collision_env_rate=0.0`; summary finite.
- Livestream startup smoke: exit `0`; output `logs/mpc_policy_eval/visual_tracking_smoke/2026-06-05_17-45-11-376192`; log included `Streaming server started`; `metrics.jsonl` 2 lines; tracking `reference_valid_ratio=1.0`.

## Result

Pass.

## Conclusion

`Go2Pvcnn/scripts/mpc_policy_eval.py` is accepted for the two requested headless evaluation modes and the livestream startup path. Tracking metrics use the MPC reference cache, small-collision metrics use `semantic_contact_small.data.force_matrix_w`, and the small-collision denominator is environment count per round rather than step count.

## Follow-Up

- `--terrain-rows/--terrain-cols` currently resize the generated terrain grid rather than selecting original terrain row/col IDs; fix before claiming true multi-terrain comparison semantics.
- Livestream was verified to start and create marker objects without crashing, but browser-side visual confirmation of marker appearance was not inspected in this run.
