# Semantic Map Contact Collision Implementation

## Purpose

Implement train/play/viewer defaults that avoid loading global semantic contact sensors and fold semantic collision inference into the existing body-part clearance reward path.

## Stage

Train reward/curriculum wiring and cfg defaults.

## Related Todo

- [../todo/T302u-semantic-map-contact-collision-plan.md](../todo/T302u-semantic-map-contact-collision-plan.md)

## Baseline Ref

- Current working tree before the T302u implementation.

## Candidate Ref

- Working tree after the semantic-map contact inference and cfg cleanup.

## Key Files

- [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
- [../../Go2Pvcnn/extension/semantic_curriculum.py](../../Go2Pvcnn/extension/semantic_curriculum.py)
- [../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py](../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py)
- [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)

## Result

Pass. The new path now:

- keeps `semantic_contact_small` / `semantic_contact_large` disabled in train/play/viewer defaults;
- infers small semantic collision from ordinary body contact force plus 0.01m semantic/elevation map queries;
- combines the map-contact penalty into `semantic_body_part_clearance_reward`;
- updates curriculum sticky collision bookkeeping to use the inferred map-contact helper.

## Verification

Focused tests:

```text
2 passed
3 passed
15 passed
55 passed
17 passed
```

Compile:

```text
python -m py_compile ...  # exit 0
git diff --check          # exit 0
```

Real IsaacLab smoke:

```bash
CUDA_VISIBLE_DEVICES=2 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --headless \
  --device cuda:0 \
  --num_envs 8 \
  --max_iterations 1 \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --planner-backend mpc
```

Observed exit `0`, run directory:

```text
logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance/2026-06-14_00-08-53
```

Runtime evidence:

```text
Available scene entities: terrain, robot, contact_forces, semantic_height_scanner, sky_light
Reward Manager has semantic_body_part_clearance and semantic_foot_over_clearance
Reward Manager does not have semantic_contact_collision
Curriculum Manager has only terrain_levels
Simulation start completed in 10.309492s
```

## Follow-Up

Run a 1024-env startup check after the user stops any older stuck process, then resume training if memory and scene creation time are acceptable.
