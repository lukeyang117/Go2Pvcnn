# Flat-Small 1024 Simulation Start Stall

## Purpose

Diagnose why the user's 1024-env resumed flat-small training command appears stuck after scene creation at IsaacLab simulation start.

## Stage

Flat-small train startup / semantic static course / global semantic contact sensor initialization.

## Related Todo

- [../todo/T302t-goal-anchored-flat-small-command-plan.md](../todo/T302t-goal-anchored-flat-small-command-plan.md)
- [../todo/T302s-env-level-collision-curriculum-plan.md](../todo/T302s-env-level-collision-curriculum-plan.md)

## Baseline Ref

- `01e1059` plus current working tree.

## Candidate Ref

- Diagnostic only; no code change.

## Command

```bash
CUDA_VISIBLE_DEVICES=2 python Go2Pvcnn/scripts/train.py \
  --headless --livestream 2 --device cuda:0 --num_envs 1024 \
  --max_iterations 10000 \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --planner-backend mpc \
  --resume \
  --load_run /mnt/mydisk/lhy/testPvcnnWithIsaacsim/logs/rsl_rl/teacher_elevation_trajectory_mpc_semantic/2026-06-04_18-16-07 \
  --load_checkpoint model_19999.pt
```

The visible log reached:

```text
Time taken for scene creation : 191.653672 seconds
Number of environments: 1024
Starting the simulation. This may take a few seconds. Please wait...
```

## Runtime Evidence

Process probe while stalled:

```text
PID 4142175
STAT Rl+
ELAPSED 09:29
CPU 1556%
MEM 8.6%
```

GPU probe:

```text
GPU2 memory used about 11993 MiB / 24564 MiB
GPU2 utilization 0%
```

This means the process is alive and CPU-heavy, but has not entered GPU-heavy training. The likely stall point is CPU/PhysX/USD/contact-view initialization during simulation start.

## Code Evidence

The recent flat-small static course column fix intentionally changed the flat-small course from one populated column to all 20 populated terrain columns. The real cfg probe in [2026-06-13-2207-flat-small-semantic-course-column-fix.md](2026-06-13-2207-flat-small-semantic-course-column-fix.md) reported:

```text
row00_col_count=20
row09_col_count=20
total_small=8000
```

The flat-small curriculum currently uses:

```text
plane small counts = 8, 12, 16, 24, 32, 40, 52, 64, 72, 80
num_cols = 20
total small objects = 400 * 20 = 8000
```

Training still enables global semantic contact sensors:

```text
semantic_contact_small = _semantic_global_contact_sensor(SEMANTIC_COURSE_SMALL_ROOT)
semantic_contact_large = _semantic_global_contact_sensor(SEMANTIC_COURSE_LARGE_ROOT)
```

`SemanticGlobalContactSensor` resolves 13 robot bodies and calls:

```python
self._physics_sim_view.create_rigid_contact_view(
    sensor_paths,
    filter_patterns=[filter_paths] * len(sensor_paths),
)
```

For flat-small after the column fix, `filter_paths` can contain 8000 small obstacle prims. The contact tensor allocation shape is:

```text
[num_envs, 13 bodies, 8000 filters, 3]
```

At 1024 envs this is:

```text
1024 * 13 * 8000 * 3 = 319,488,000 float values
```

That is about 1.28 GB for one dense float32 matrix, before PhysX contact-view metadata and initialization overhead. The filtered contact view construction itself is likely much more expensive than this raw tensor estimate.

## Conclusion

The stall is most likely not PPO, not checkpoint loading, and not the new goal-anchored command. It is most likely the global semantic contact filter scale caused by the combination of:

1. flat-small now correctly spawning small objects in all 20 columns;
2. total small obstacles rising to 8000;
3. global semantic contact sensors filtering 13 robot bodies against all 8000 objects;
4. 1024 envs amplifying the force matrix and PhysX view initialization cost.

The semantic objects being numerous matters, but the sharper bottleneck is the global contact-filter matrix, not rendering.

## Follow-Up

Recommended minimal confirmation:

1. Run a 1024-env startup probe with `semantic_contact_small/large` disabled but visual/static semantic objects kept. If it passes simulation start quickly, the root cause is contact-filter scale.
2. Run a second 1024-env probe with semantic contact enabled but much lower flat-small `plane_counts`. If it passes, the sensitivity is object/filter count.
3. Long-term fix should avoid global all-object contact filters for training. Prefer a scalable collision source such as row/env-local contact subsets, map/raycast clearance for dense rewards, or a capped near-field contact filter.
