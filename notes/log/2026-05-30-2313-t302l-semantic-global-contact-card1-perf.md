# T302l Semantic Global Contact Card1 Performance

## Purpose

Validate the repaired semantic global contact sensor path on GPU card 1 after the earlier 1024-env probe stalled in `gym.make(...)`.

## Stage

MPC RL semantic contact integration:

- `Go2Pvcnn/go2_pvcnn/sensor/semantic_contacter/semantic_global_contact_sensor.py`
- `Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py`
- `Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py`

## Related Todo

- [T302l](../todo/T302l-mpc-rl-participation-and-reward-plan.md)

## Baseline Ref

- `0f78099`

## Candidate Ref

- Worktree change on top of `0f78099`

## Root Cause

The first card1 repro timed out at `gym.make(...)`. A `faulthandler.dump_traceback_later(...)` run showed the main thread inside:

```text
isaaclab/sim/utils.py:665 find_first_matching_prim
go2_pvcnn/sensor/semantic_contacter/semantic_global_contact_sensor.py:88 _initialize_impl
```

The sensor was resolving 1024 environments times 13 robot bodies with `find_first_matching_prim(body_path)`. That function traverses the whole USD stage for each path, so initialization scaled badly.

## Change

Replaced per-body regex stage traversal with exact USD path lookup:

- Added `resolve_contact_body_paths(...)` helper.
- `_initialize_impl()` now uses `stage.GetPrimAtPath(body_path)` and checks `PhysxContactReportAPI`.
- Semantic object coverage remains unchanged: all leaf `row_*/col_*/slot_*` objects are still included.

## Commands And Results

### Red Test

```bash
pytest Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py::test_semantic_global_contact_body_resolution_uses_exact_paths -q
```

Result: FAIL before implementation because `resolve_contact_body_paths` was missing.

### Focused Unit Tests

```bash
pytest Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py Go2Pvcnn/tests/test_semantic_contact_rewards.py -q
```

Result: PASS, `8 passed`.

### IsaacLab Quantity Alignment

```bash
CUDA_VISIBLE_DEVICES=1 timeout 120s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/test_mpc_semantic_contact_isaaclab.py::test_mpc_semantic_global_contact_sensors_quantity_alignment_1024 -q
```

Result: PASS.

### Minimal 1024 Env Shape Repro

Command: custom one-off Python script using `env_isaacsim`, `CUDA_VISIBLE_DEVICES=1`, `TeacherElevationTrajectoryMpcSemanticEnvCfg`, and `num_envs=1024`.

Result: PASS.

Key metrics:

- `small_shape = (1024, 13, 640, 3)`
- `large_shape = (1024, 13, 100, 3)`
- `expected_small = 640`
- `expected_large = 100`
- `body_count = 13`

### 1024 Env / 64 MPC Env / 25 Step Performance Probe

```bash
CUDA_VISIBLE_DEVICES=1 timeout 300s /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/mpc_rl_epoch_perf_probe.py
```

Result: PASS.

Key metrics:

- `num_envs = 1024`
- `selected_mpc_envs = 64`
- `epoch_seconds = 5.64891067519784`
- Acceptance target: `epoch_seconds <= 10`

## Notes

- IsaacSim still prints `CUDA_VISIBLE_DEVICES` and GPU bad-state warnings on card1, but the run completed and the measured epoch time is within target.
- Environment startup remains heavy (`simulation start` around 65 seconds in this run), but the acceptance metric is the 25-step epoch time.
- No PhysX semantic filter-count mismatch like `/World/semantic_course/small/* expected 1024, found 7` appeared in the successful card1 runs.

## Follow-Up

No code follow-up is required for this specific bottleneck. If startup time itself becomes an acceptance target later, it should be tracked as a separate issue from the 25-step training epoch metric.
