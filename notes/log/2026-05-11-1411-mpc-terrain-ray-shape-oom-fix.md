# T300d MPC terrain ray-shape OOM fix

- timestamp: 2026-05-11 14:11 CST
- todo: [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)
- result: pass (targeted fix + regression tests)

## Purpose

Fix training-time CUDA OOM reported from `batch_mpc_planner/terrain.py` during `subset_mpc_terrain()` under `--planner-backend mpc`.

## Stage

MPC runtime terrain ingestion contract hardening (`ray_hits_w` shape handling) + focused test coverage.

## Trigger

User runtime traceback:

- `manager.py: sub_terrain = subset_mpc_terrain(terrain, selected_ids)`
- `terrain.py: height = terrain.height_map.index_select(0, ids)`
- `torch.OutOfMemoryError: Tried to allocate 89.07 GiB`

## Root Cause

`scanner.data.ray_hits_w` in IsaacLab runtime is `[B, H*W, 3]` for grid scanners.  
Old `build_mpc_terrain_from_scanner()` treated 3D input as `[H, W, 3]` by unconditional `unsqueeze(0)`, producing a misinterpreted `height_map` batch axis and causing oversized `index_select` allocations in `subset_mpc_terrain`.

## Changes

- Updated [../../Go2Pvcnn/extension/batch_mpc_planner/terrain.py](../../Go2Pvcnn/extension/batch_mpc_planner/terrain.py):
  - added robust ray-hit reshape support for:
    - `[B,H,W,3]`
    - `[B,H*W,3]`
    - `[H,W,3]`
    - `[H*W,3]`
  - added semantic-map reshape/broadcast support for:
    - `[B,H,W]`, `[H,W]`, `[B,H*W]`, `[H*W]`
  - added `nan/inf` sanitization on ray-hit and semantic inputs
  - added bounds/shape guards in `subset_mpc_terrain()`
- Updated [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py):
  - added terrain contract regression test for flattened ray-hit input
  - added manager refresh regression test for flattened scanner ray-hits

## Verification

- `python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q` -> `11 passed`
- `python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/terrain.py Go2Pvcnn/tests/test_batch_mpc_backend.py` -> pass
- `PYTHONPATH=Go2Pvcnn python - <<'PY' ...` terrain-shape sanity:
  - input `ray_hits_w`: `(4096, 151*151, 3)`
  - built terrain `height_map`: `(4096, 151, 151)`
  - subset (`256` ids) shape: `(256, 151, 151)`

## Key Metrics

- focused backend suite increased from `9` to `11` passing tests
- no failing tests in targeted MPC backend suite

## Conclusion

The OOM mechanism from flattened scanner-grid misinterpretation is addressed in terrain ingestion and guarded by regression tests.

## Follow-Up

- Re-run user training command on `env_isaacsim` (`--planner-backend mpc`) to confirm no early OOM at first reward refresh step.
- Keep 4096-env large-scale runtime acceptance as separate residual risk (PhysX/CUDA runtime stability path).

## Git Refs

- Baseline Ref: `979b2b5`
- Candidate Ref: working tree with terrain shape fix + tests
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/terrain.py](../../Go2Pvcnn/extension/batch_mpc_planner/terrain.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
