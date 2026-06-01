# 2026-05-31 22:49 T302m Teacher Elevation MPC Semantic Cleanup

## Purpose

Clean the current working tree around the single active `teacher_elevation_trajectory_mpc_semantic + mpc` route, preserving current train/play/viewer, low-small MPC, MPC-RL participation, semantic contactor, and semantic raycaster behavior.

## Stage

Repository cleanup / entrypoint narrowing / legacy backend retirement.

## Related Todo

- [../todo/T302m-teacher-elevation-mpc-semantic-cleanup-plan.md](../todo/T302m-teacher-elevation-mpc-semantic-cleanup-plan.md)

## Commands And Results

- `pytest Go2Pvcnn/tests/test_batch_mpc_backend.py::test_cleanup_entrypoints_only_expose_mpc_semantic_experiment Go2Pvcnn/tests/test_batch_mpc_backend.py::test_cleanup_mpc_factory_has_no_legacy_or_together_backend Go2Pvcnn/tests/test_batch_mpc_backend.py::test_cleanup_batch_mpc_planner_has_no_debug_variants_module -q`
  - Result: `3 passed in 1.69s`
- `pytest Go2Pvcnn/tests/test_viewer_reset.py Go2Pvcnn/tests/test_viewer_entrypoint_import_order.py -q`
  - Result: `16 passed in 1.69s`
- `pytest Go2Pvcnn/tests/test_batch_mpc_parametric.py Go2Pvcnn/tests/test_mpc_rl_participation.py Go2Pvcnn/tests/test_semantic_contact_rewards.py Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py -q`
  - Initial failure: `test_batch_mpc_parametric.py` lacked `Go2Pvcnn` on `sys.path`; fixed test prelude.
  - Result after fix: `27 passed in 1.69s`
- `pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q`
  - Initial failure: PLAY cfg inherited base `fk_body_leg_collision.weight=120.0` instead of duplicating assignment; updated static test to accept inheritance.
  - Result after fix: `128 passed, 1 warning in 4.85s`
- `pytest Go2Pvcnn/tests/test_batch_mpc_parametric.py Go2Pvcnn/tests/test_mpc_rl_participation.py Go2Pvcnn/tests/test_semantic_contact_rewards.py Go2Pvcnn/tests/test_mpc_semantic_rl_env_cfg.py Go2Pvcnn/tests/test_viewer_reset.py Go2Pvcnn/tests/test_viewer_entrypoint_import_order.py -q`
  - Result: `43 passed in 1.72s`
- `find Go2Pvcnn/extension Go2Pvcnn/scripts Go2Pvcnn/go2_pvcnn/tasks -path '*/__pycache__' -prune -o -type f -name '*.py' -print0 | xargs -0 python -m py_compile`
  - Result: pass.
- Production old-route scan:
  - `rg -n "batched_planner|batched_together_planner|teacher_without_semantic|teacher_semantic|teacher_elevation_semantic_map|teacher_elevation_trajectory_env_cfg|teacher_elevation_env_cfg|debug_variants|debug_loss_variant|apply_mpc_debug_variant_cfg" Go2Pvcnn/extension Go2Pvcnn/scripts Go2Pvcnn/go2_pvcnn/tasks Go2Pvcnn/agent --glob '*.py'`
  - Result: no production matches.

## IsaacLab Attempt

- Command attempted on card3:
  - `CUDA_VISIBLE_DEVICES=3 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py --headless --device cuda:0 --num_envs 1024 --max_iterations 1 --experiment teacher_elevation_trajectory_mpc_semantic --planner-backend mpc`
- Result: blocked by external GPU memory pressure, not treated as code regression.
  - Existing process `2959859` was already running `Go2Pvcnn/scripts/train.py --num_envs 1024 --max_iterations 20000 ... --planner-backend mpc` and occupied about `20.6GB` on the visible GPU.
  - The smoke reached env registration, scene creation, and simulation start, then failed in PhysX/semantic contact allocation with CUDA OOM.
- Semantic contact drop probe was also started on the same saturated card and was stopped after confirming the resource conflict.

## Key Changes Verified

- Train/play/register/factory are narrowed to `teacher_elevation_trajectory_mpc_semantic` and `mpc` only.
- `teacher_elevation_trajectory_mpc_semantic_env_cfg.py` is self-contained and no longer imports old teacher cfg modules.
- Production `extension/batched_planner`, `extension/batched_together_planner`, old task cfgs, old script entrypoints, and production debug variants are removed from the working tree.
- Viewer is MPC-only and compiles.
- Current MPC/RL/semantic static tests pass after test cleanup.

## Follow-Up

- Re-run real IsaacLab acceptance after card3 is free:
  - semantic contact drop probe.
  - 1024 env / 1 iteration train smoke.
  - 1024 env / 64 MPC / 25-step performance probe if runtime wiring changes again.
- Task 7 duplicate-computation refactor was intentionally not performed in this cleanup pass because it is not required for route cleanup and could change planner internals.

## Git Refs

- Current Work Ref: local working tree on branch `costmap-teacher-ablation`.
- Key Files: `Go2Pvcnn/scripts/train.py`, `Go2Pvcnn/scripts/play.py`, `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`, `Go2Pvcnn/extension/trajectory_manager_factory.py`, `Go2Pvcnn/extension/viz/go2_foostep_planner.py`, `Go2Pvcnn/extension/batch_mpc_planner/`, `Go2Pvcnn/tests/`.
