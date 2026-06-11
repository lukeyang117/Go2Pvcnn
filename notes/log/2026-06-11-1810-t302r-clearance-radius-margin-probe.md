# T302r Clearance Radius/Margin Probe

## Purpose

Test whether increasing `semantic_body_part_clearance` query radius can make the flat-small clearance reward nonzero, and separate semantic-hit failure from positive-deficit failure.

## Stage

RL reward diagnostics / flat-small semantic clearance.

## Related Todo

- [../todo/T302r-go2-geometry-clearance-reward-plan.md](../todo/T302r-go2-geometry-clearance-reward-plan.md)
- [../todo/T302q-flat-small-avoidance-reward-plan.md](../todo/T302q-flat-small-avoidance-reward-plan.md)

## Command / Procedure

Added a focused radius sweep probe:

```bash
CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/semantic_body_part_clearance_radius_probe.py \
  --num-envs 64 --steps 1 --radii 0.12,0.50 --margin-scale 1.0 \
  --output-jsonl Go2Pvcnn/tests/artifacts/semantic_body_part_clearance_radius_probe_64_diagnostics.jsonl

CUDA_VISIBLE_DEVICES=0 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/semantic_body_part_clearance_radius_probe.py \
  --num-envs 64 --steps 1 --radii 0.50 --margin-scale 5.0 \
  --output-jsonl Go2Pvcnn/tests/artifacts/semantic_body_part_clearance_radius_probe_64_margin5.jsonl
```

Also ran:

```bash
python -m py_compile Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py \
  Go2Pvcnn/tests/semantic_body_part_clearance_radius_probe.py
pytest -q Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py
```

## Input Conditions

- Probe cfg disables `reference_foot_pos` and planner-owned trajectory cache to isolate the clearance reward during `env.step`.
- Scanner grid shape: `[64, 151, 151]`.
- Fixed-shape query offset cap raised from `13` to default `81`.
- Flat-small cfg is changed to signal-first params:
  - query radius `0.50m` for foot/calf/thigh/base
  - foot/base margin `0.20m`
  - calf/thigh margin `0.40m`

## Key Metrics

- RED unit test confirmed old large-radius behavior stayed capped at `13` offsets.
- Focused reward tests after fix: `15 passed`.
- 64-env diagnostics, margin scale `1.0`:
  - scanner small cells: `145`
  - radius `0.12`: `body_small_hit_count=2`, `positive_deficit=0`, reward nonzero `0/64`
  - radius `0.50`: `body_small_hit_count=20`, `positive_deficit=0`, reward nonzero `0/64`
- 64-env diagnostics, radius `0.50`, margin scale `5.0`:
  - scanner small cells: `199`
  - `body_small_hit_count=13`
  - `body_small_positive_deficit_count=6`
  - reward nonzero `1/64`
  - min reward `-5.163e-05`
  - mean reward call time `12.9ms`
- 64-env diagnostics, radius `0.12`, margin scale `5.0`:
  - scanner small cells: `163`
  - `body_small_hit_count=0`
  - reward nonzero `0/64`

## Result

Diagnostic and tuning pass. Increasing radius alone is not enough; the reward chain only became nonzero once both radius and margin were enlarged.

## Conclusion

The semantic query path can hit small cells, but the previous surface-height deficit condition was too conservative. Radius `0.50m` plus enlarged margins is a signal-first setting that should make the next short training run expose nonzero TensorBoard clearance values, though it is intentionally aggressive and should be tightened after confirming the reward is alive.

## Follow-Up

- Run a short flat-small train/TensorBoard sanity with signal-first params and check `Episode_Reward/semantic_body_part_clearance`.
- If nonzero, reduce radius/margins toward a less broad near-body risk field while preserving nonzero rate.
- Add part-level TensorBoard diagnostics only if the next training run still shows zero.

## Git Refs

- Baseline Ref: `da46138`
- Candidate Ref: working tree on 2026-06-11
- Key Files:
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/tests/semantic_body_part_clearance_radius_probe.py](../../Go2Pvcnn/tests/semantic_body_part_clearance_radius_probe.py)
  - [../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py](../../Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py)
