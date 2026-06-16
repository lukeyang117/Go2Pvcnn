# MPC Proximity Field Semantic Avoidance

## Purpose

Reduce MPC semantic avoidance GPU memory so `TeacherElevationTrajectoryMpcSemanticEnvCfg` can run with `num_envs == mpc_num_envs`, specifically `1024` RL envs and `1024` MPC envs, without adding a new loss name or regressing the existing low-small acceptance contract.

## Stage

Batch MPC planner / parametric semantic avoidance / IsaacLab runtime probe.

## Related Todo

- [../todo/T302u-semantic-map-contact-collision-plan.md](../todo/T302u-semantic-map-contact-collision-plan.md)
- [../../docs/superpowers/plans/2026-06-16-mpc-proximity-field-semantic-avoidance-plan.md](../../docs/superpowers/plans/2026-06-16-mpc-proximity-field-semantic-avoidance-plan.md)
- [../../docs/superpowers/specs/2026-06-16-mpc-proximity-field-semantic-avoidance-design.md](../../docs/superpowers/specs/2026-06-16-mpc-proximity-field-semantic-avoidance-design.md)

## Baseline Ref

- Working tree before this task already had unrelated local edits under flat-small command/curriculum files.
- Problem shape: old `parametric_semantic_avoidance` materialized dense root/foot/touchdown pairwise tensors against the `150 x 150 = 22500` semantic scanner cells.

## Candidate Ref

- Current Work Ref: local dirty tree on branch `costmap-teacher-ablation`.
- Key files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_parametric.py](../../Go2Pvcnn/tests/test_batch_mpc_parametric.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
  - [../../Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py](../../Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py)

## Change Summary

- Kept the existing `parametric_semantic_avoidance` loss key and existing MPC loss config surface.
- Added internal planner helpers to build a command-gated high-small/large soft proximity field from the semantic/elevation map.
- Replaced dense `[B,H,4,22500,2]` semantic avoidance distance tensors with `grid_sample` queries for root, foot, and touchdown xy positions.
- Extended `mpc_rl_epoch_perf_probe.py` with `--num-envs`, `--mpc-num-envs`, `--steps`, `--require-replan`, `--print-cuda-memory`, and `--summary-path`.
- The probe writes incremental JSON summaries because Isaac/Kit can exit cleanly while swallowing tail output in some step paths.

## Commands And Results

Environment for all Python/pytest checks:

```bash
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
```

Focused MPC regression:

```bash
pytest Go2Pvcnn/tests/test_batch_mpc_parametric.py Go2Pvcnn/tests/test_batch_mpc_backend.py Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py -q
```

Result:

```text
225 passed in 6.57s
```

Pycompile:

```bash
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py Go2Pvcnn/tests/test_batch_mpc_parametric.py Go2Pvcnn/tests/test_batch_mpc_backend.py
```

Result: exit `0`.

Diff whitespace:

```bash
git diff --check
```

Result: exit `0`.

Small real probe:

```bash
CUDA_VISIBLE_DEVICES=1 python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 8 --mpc-num-envs 8 --steps 30 --require-replan --print-cuda-memory --summary-path /tmp/mpc_probe_8x8_summary.json
```

Result: exit `0`; summary `phase=complete`, `completed_steps=30`, `max_sampled_plan_count_seen=8`, `replan_event_count=2`, CUDA max allocated `67298304`, reserved `94371840`.

Required 1024 real probe:

```bash
CUDA_VISIBLE_DEVICES=1 python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py --num-envs 1024 --mpc-num-envs 1024 --steps 30 --require-replan --print-cuda-memory --summary-path /tmp/mpc_probe_1024x1024_summary.json
```

Result: exit `0`.

Key summary:

```json
{
  "phase": "complete",
  "num_envs": 1024,
  "mpc_num_envs": 1024,
  "parallel_plan_batch_size": 1024,
  "horizon_steps": 25,
  "replan_interval_steps": 25,
  "steps": 30,
  "completed_steps": 30,
  "epoch_seconds": 64.36930854804814,
  "max_sampled_plan_count_seen": 1024,
  "replan_event_count": 2,
  "cuda_max_memory_allocated": 7431256576,
  "cuda_max_memory_reserved": 9265217536
}
```

Planner-level 1024 semantic GPU probe:

```bash
CUDA_VISIBLE_DEVICES=1 python - <<'PY'
# Constructs batch=1024, horizon=25, 150x150 semantic terrain with one large obstacle per env,
# then calls plan_segment with diagnostics enabled and optimize_steps=1.
PY
```

Result: exit `0`; `semantic_avoidance_mean=0.7900127172470093`, `finite_cost=true`, CUDA max allocated `1945641472`, reserved `2111832064`.

Low-small headless acceptance command:

```bash
PYTHONPATH=/mnt/mydisk/lhy/testPvcnnWithIsaacsim:/mnt/mydisk/lhy/testPvcnnWithIsaacsim/Go2Pvcnn MPC_T302_HEADLESS=1 CUDA_VISIBLE_DEVICES=1 pytest Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py -q
```

Result: exit `0`, but pytest emitted no summary in this tool session. A collect-only rerun collected `6` tests successfully. Treat this as exit-code evidence only, not a detailed metric printout.

## Constraints Checked

- No new MPC loss name was added.
- `parametric_semantic_avoidance` remains present.
- `MpcPlannerCfg().losses` has no proximity/distance-field term.
- Static regression confirms the old dense semantic avoidance pairwise patterns are absent from `planner.py`.
- Low-small thresholds were not edited or relaxed in this task.
- Real 1024/1024 `TeacherElevationTrajectoryMpcSemanticEnvCfg` probe did not CUDA OOM.

## Conclusion

The memory-risky semantic avoidance hot path has been replaced with proximity-field sampling. The 1024 RL env / 1024 MPC env acceptance gate passes on GPU1 with about `7.43GB` max allocated and `9.27GB` max reserved in the real IsaacLab probe.

## Follow-Up

- Runtime latency still deserves a separate pass: one 1024 probe profile showed a long `mixed_zero_split_ms` interval even though memory was safe.
- The headless low-small pytest file should be made less silent in this tool environment so future acceptance logs include explicit per-test summaries.

## Git Refs

- Last Feature Commit: not recorded in this dirty working tree.
- Last Verified Commit: not recorded in this dirty working tree.
- Current Work Ref: local branch `costmap-teacher-ablation`.
