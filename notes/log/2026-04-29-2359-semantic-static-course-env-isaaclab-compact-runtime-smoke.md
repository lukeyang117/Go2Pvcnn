# Semantic Static Course `env_isaaclab` Compact Runtime Smoke

## Meta

- Time: `2026-04-29 23:59 +0800`
- Stage: `semantic static course env_isaaclab compact runtime smoke`
- Result: `pass with scoped caveat`
- Todo: [T200](../todo/T200-semantic-static-course-viewer.md)

## Purpose

- Verify that the semantic static-course viewer path can boot and pass real headless runtime checks in `/home/lhy/anaconda3/envs/env_isaaclab`.
- Keep the smoke bounded by using the compact `4 x 1` terrain grid in the runtime fixture while preserving `S1..S4` stage coverage.

## Commands

```bash
source /home/lhy/anaconda3/bin/activate env_isaaclab && \
timeout 120s pytest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q -k semantic_height_scanner_contract
```

```bash
source /home/lhy/anaconda3/bin/activate env_isaaclab && \
timeout 120s pytest Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py -q -k together_semantic_smoke
```

## Result

- `semantic_height_scanner_contract`: pass
- `together_semantic_smoke`: pass

## Key Evidence

- Isaac headless app launched successfully in `env_isaaclab`.
- `/World/semantic_course`, `/World/semantic_course/small`, and `/World/semantic_course/large` were created in the live stage.
- semantic per-tile cuboid descendants were spawned during real runtime startup.
- the compact runtime fixture was enough to bring the semantic contract and default `together` smoke to pytest verdict within the timeout window.

## Scoped Caveat

- This smoke uses the compact runtime fixture terrain grid (`4 x 1`) to keep startup bounded while preserving the four semantic stages.
- It does **not** prove that the full viewer default terrain grid has acceptable startup latency for an interactive session.

## Git Refs

- Baseline Ref: `d267461`
- Candidate Ref: `working tree on top of d267461 (2026-04-29 23:59 +0800); semantic static-course implementation uncommitted`
- Key Files:
  - [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
  - [../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/test_viewer_runtime_diagnostics.py)
  - [index.md](index.md)
