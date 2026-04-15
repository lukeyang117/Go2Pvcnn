# Batched Planner Single-Shot Decoupled Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the batched planner runtime so it uses raw-aligned single-shot semantics, per-env masked replanning, planner-owned observability, and viewer direct playback without changing downstream reward cache contracts.

**Architecture:** Keep `Go2Pvcnn/extension/batched_planner` as the single runtime planner for train and viewer, but move lifecycle ownership fully into `BatchedTrajectoryManager`. The manager will maintain a full-shaped canonical reference cache, build subset planner inputs for replanning envs only, and masked-write results back into that cache. The planner core will produce only `motion` or `standstill` trajectories, while train/viewer hooks expose instrumentation and playback controls around the shared runtime.

**Tech Stack:** Python, PyTorch, Isaac Lab, Gymnasium, RSL-RL, pytest, git

---

## File Structure

### Planner Core

- Modify: `Go2Pvcnn/extension/batched_planner/trajectory.py`
  Responsibility: Remove multi-candidate recovery, keep only single-shot evaluation, and return binary `motion` or `standstill` planner outputs.

- Modify: `Go2Pvcnn/extension/batched_planner/config.py`
  Responsibility: Delete obsolete multi-candidate recovery knobs and add any planner instrumentation config knobs that must travel with planner config.

- Create: `Go2Pvcnn/extension/batched_planner/instrumentation.py`
  Responsibility: Provide lightweight planner-stage timing containers and manager-facing summaries that can stay quiet by default.

### Runtime Manager and Cache Boundary

- Modify: `Go2Pvcnn/extension/batched_planner/manager.py`
  Responsibility: Compute per-env replan masks, subset planner inputs, masked-write subset outputs back into the canonical cache, track per-env phase/replan state, and expose optional diagnostics.

- Modify: `Go2Pvcnn/extension/convention.py`
  Responsibility: Preserve canonical full-cache conversion and add any helper needed for subset result normalization or masked writeback.

- Modify: `Go2Pvcnn/extension/reference/cache.py`
  Responsibility: Preserve the canonical full-shaped cache ABI and add any explicit helper needed for safe env-row replacement without changing reward-facing contracts.

### Runtime Consumers

- Modify: `Go2Pvcnn/extension/mdp/rewards_reference.py`
  Responsibility: Continue consuming the shared full cache as-is, while adding focused assertions/tests so partial replans do not leak a new consumer contract.

- Modify: `Go2Pvcnn/scripts/train.py`
  Responsibility: Add planner verbosity wiring and benchmark-friendly diagnostics while keeping planner-owned runtime attachment for `teacher_elevation_trajectory`.

- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
  Responsibility: Add explicit planner direct playback mode on top of the shared manager/cache path without introducing viewer-only planner semantics.

- Modify: `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py`
  Responsibility: Remove obsolete replan recovery config fields and expose any viewer/train playback or debug flags that must live in env config.

### Tests and Benchmarks

- Modify: `Go2Pvcnn/tests/test_batched_trajectory.py`
  Responsibility: Lock down single-shot success/failure behavior and explicit standstill fallback.

- Modify: `Go2Pvcnn/tests/test_batched_trajectory_batch.py`
  Responsibility: Verify mixed motion/standstill batch behavior without candidate recovery loops.

- Modify: `Go2Pvcnn/tests/test_batched_manager.py`
  Responsibility: Cover per-env reset, command-change, interval-based replanning, masked cache writes, and standstill cache persistence.

- Modify: `Go2Pvcnn/tests/test_batched_reference_integration.py`
  Responsibility: Verify reward-side cache consumers still work against a full canonical cache after partial replans.

- Modify: `Go2Pvcnn/tests/test_batched_planner_runtime_path.py`
  Responsibility: Cover train/viewer runtime hooks, verbose planner wiring, and direct playback mode selection.

- Create: `Go2Pvcnn/tests/test_batched_planner_instrumentation.py`
  Responsibility: Validate planner timing summary accumulation and quiet-by-default diagnostics behavior.

- Create: `Go2Pvcnn/scripts/bench_batched_planner.py`
  Responsibility: Run planner micro-benchmarks for required env counts and report per-stage timings.

### Documentation

- Modify: `notes/human/human-10-extension-planner-runtime.md`
  Responsibility: Update runtime notes to reflect per-env lifecycle and single-shot semantics.

- Modify: `notes/ai/ai-10-extension-planner-runtime.md`
  Responsibility: Keep the AI-facing planner runtime note aligned with the new manager/cache contract.

## Task 1: Lock Planner Single-Shot Semantics

**Files:**
- Modify: `Go2Pvcnn/extension/batched_planner/trajectory.py`
- Modify: `Go2Pvcnn/extension/batched_planner/config.py`
- Test: `Go2Pvcnn/tests/test_batched_trajectory.py`
- Test: `Go2Pvcnn/tests/test_batched_trajectory_batch.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_single_shot_infeasible_command_returns_standstill():
    result = batched_generate_trajectory(terrain, states, commands, requested_n_frames=5, cfg=cfg)
    assert torch.allclose(result.root_lin_vel_w, torch.zeros_like(result.root_lin_vel_w))


def test_single_shot_planner_does_not_iterate_recovery_commands():
    with patch("extension.batched_planner.trajectory._iter_replan_commands") as helper:
        batched_generate_trajectory(terrain, states, commands, requested_n_frames=5, cfg=cfg)
    helper.assert_not_called()


def test_mixed_batch_keeps_motion_envs_and_standstill_envs_together():
    result = batched_generate_trajectory(terrain, states, commands, requested_n_frames=5, cfg=cfg)
    assert result.root_pos_w.shape[0] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_trajectory.py Go2Pvcnn/tests/test_batched_trajectory_batch.py -k "single_shot or standstill or mixed_batch" -v`
Expected: FAIL because the planner still uses candidate recovery loops and lacks explicit single-shot-only assertions.

- [ ] **Step 3: Write minimal implementation**

Implement in `Go2Pvcnn/extension/batched_planner/trajectory.py` and `Go2Pvcnn/extension/batched_planner/config.py`:
- delete `_iter_replan_commands(...)`
- remove velocity-scale / yaw-bias / vy-bias candidate search
- treat touchdown feasibility failure as immediate standstill fallback
- preserve the existing standstill trajectory builder
- remove obsolete config fields and call sites tied to multi-candidate recovery

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_trajectory.py Go2Pvcnn/tests/test_batched_trajectory_batch.py -k "single_shot or standstill or mixed_batch" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/trajectory.py Go2Pvcnn/extension/batched_planner/config.py Go2Pvcnn/tests/test_batched_trajectory.py Go2Pvcnn/tests/test_batched_trajectory_batch.py
git commit -m "refactor: make batched planner single shot"
```

## Task 2: Rebuild Manager Around Per-Env Masked Replanning

**Files:**
- Modify: `Go2Pvcnn/extension/batched_planner/manager.py`
- Modify: `Go2Pvcnn/extension/convention.py`
- Modify: `Go2Pvcnn/extension/reference/cache.py`
- Test: `Go2Pvcnn/tests/test_batched_manager.py`
- Test: `Go2Pvcnn/tests/test_batched_reference_integration.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_reset_replans_only_selected_env_rows():
    manager.refresh_from_env(env)
    manager.reset_envs(torch.tensor([False, True, False]))
    cache = manager.refresh_from_env(env)
    assert cache.root_pos_w.shape[0] == 3


def test_command_change_replans_only_changed_env_rows():
    manager.refresh_from_env(env)
    env.command_manager.command[1] = torch.tensor([0.3, 0.0, 0.0], dtype=torch.float64)
    cache = manager.refresh_from_env(env)
    assert torch.allclose(cache.root_pos_w[0], first_cache.root_pos_w[0])
    assert not torch.allclose(cache.root_pos_w[1], first_cache.root_pos_w[1])


def test_failed_env_keeps_standstill_cache_until_its_next_replan():
    first_cache = manager.refresh_from_env(env)
    mark_env_one_as_infeasible_for_next_replan()
    cache = manager.refresh_from_env(env)
    assert is_standstill_row(cache, env_id=1)
    cache_again = manager.refresh_from_env(env)
    assert torch.allclose(cache_again.root_pos_w[1], cache.root_pos_w[1])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_manager.py Go2Pvcnn/tests/test_batched_reference_integration.py -k "selected_env_rows or changed_env_rows or standstill_cache" -v`
Expected: FAIL because the current manager still triggers whole-batch replans and replaces the whole cache.

- [ ] **Step 3: Write minimal implementation**

Implement in `Go2Pvcnn/extension/batched_planner/manager.py`, `Go2Pvcnn/extension/convention.py`, and `Go2Pvcnn/extension/reference/cache.py`:
- compute per-env `replan_mask`
- subset states / commands / terrain inputs for replanning envs only
- build or preserve a full canonical cache for all envs
- masked-write subset planner outputs into the selected env rows
- reset `phase_index` only for replanned envs and advance/clamp others
- preserve standstill rows until that same env hits its next replan trigger
- keep reward-facing cache ABI unchanged

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_manager.py Go2Pvcnn/tests/test_batched_reference_integration.py -k "selected_env_rows or changed_env_rows or standstill_cache" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/manager.py Go2Pvcnn/extension/convention.py Go2Pvcnn/extension/reference/cache.py Go2Pvcnn/tests/test_batched_manager.py Go2Pvcnn/tests/test_batched_reference_integration.py
git commit -m "refactor: decouple batched planner manager per env"
```

## Task 3: Add Planner-Owned Instrumentation and Train Verbosity

**Files:**
- Create: `Go2Pvcnn/extension/batched_planner/instrumentation.py`
- Modify: `Go2Pvcnn/extension/batched_planner/trajectory.py`
- Modify: `Go2Pvcnn/extension/batched_planner/manager.py`
- Modify: `Go2Pvcnn/scripts/train.py`
- Modify: `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py`
- Test: `Go2Pvcnn/tests/test_batched_planner_instrumentation.py`
- Test: `Go2Pvcnn/tests/test_batched_planner_runtime_path.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_planner_timing_summary_accumulates_named_stages():
    summary = PlannerTimingSummary()
    summary.record("ik", 0.001)
    assert "ik" in summary.stage_names()


def test_train_parser_accepts_verbose_planner_flag():
    parser = build_arg_parser()
    parsed = parser.parse_args(["--verbose-planner"])
    assert parsed.verbose_planner is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_planner_instrumentation.py Go2Pvcnn/tests/test_batched_planner_runtime_path.py -k "timing_summary or verbose_planner" -v`
Expected: FAIL because no instrumentation module or verbose train flag exists yet.

- [ ] **Step 3: Write minimal implementation**

Implement:
- a lightweight planner-stage timing helper in `Go2Pvcnn/extension/batched_planner/instrumentation.py`
- timing capture around planner stages and manager refresh
- compact manager-facing timing summary objects
- `--verbose-planner` in `Go2Pvcnn/scripts/train.py`
- quiet-by-default train behavior with optional periodic planner diagnostics when verbose mode is enabled
- any env-config plumbing required for debug cadence

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_planner_instrumentation.py Go2Pvcnn/tests/test_batched_planner_runtime_path.py -k "timing_summary or verbose_planner" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/batched_planner/instrumentation.py Go2Pvcnn/extension/batched_planner/trajectory.py Go2Pvcnn/extension/batched_planner/manager.py Go2Pvcnn/scripts/train.py Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py Go2Pvcnn/tests/test_batched_planner_instrumentation.py Go2Pvcnn/tests/test_batched_planner_runtime_path.py
git commit -m "feat: add planner instrumentation and train verbosity"
```

## Task 4: Add Viewer Direct Planner Playback

**Files:**
- Modify: `Go2Pvcnn/extension/viz/go2_foostep_planner.py`
- Modify: `Go2Pvcnn/tests/test_batched_planner_runtime_path.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_viewer_parser_accepts_direct_playback_mode():
    parser = build_arg_parser()
    parsed = parser.parse_args(["--planner-playback-mode", "direct"])
    assert parsed.planner_playback_mode == "direct"


def test_direct_playback_uses_planner_cache_pose_without_physics_enforcement():
    mode = resolve_playback_mode(args)
    assert mode == "direct"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_planner_runtime_path.py -k "direct_playback or planner_playback_mode" -v`
Expected: FAIL because the viewer does not yet expose an explicit direct playback mode.

- [ ] **Step 3: Write minimal implementation**

Implement in `Go2Pvcnn/extension/viz/go2_foostep_planner.py`:
- an explicit playback-mode CLI flag such as `--planner-playback-mode`
- a `direct` mode that drives displayed root/joint state from planner cache output
- clear separation between direct trajectory inspection and physics-constrained stepping
- shared manager/cache path for all viewer modes

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_planner_runtime_path.py -k "direct_playback or planner_playback_mode" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/test_batched_planner_runtime_path.py
git commit -m "feat: add direct planner playback mode"
```

## Task 5: Add Required Planner Benchmarks

**Files:**
- Create: `Go2Pvcnn/scripts/bench_batched_planner.py`
- Modify: `Go2Pvcnn/scripts/train.py`
- Test: `Go2Pvcnn/tests/test_batched_planner_instrumentation.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_benchmark_cli_defaults_cover_required_env_counts():
    args = parse_benchmark_args([])
    assert args.env_counts == [1, 16, 64, 100, 256, 512, 1024, 2048]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_planner_instrumentation.py -k "benchmark_cli_defaults" -v`
Expected: FAIL because the benchmark script and parser do not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement:
- planner micro-benchmark CLI in `Go2Pvcnn/scripts/bench_batched_planner.py`
- required env-count sweep defaults
- output fields for absolute time, per-env time, standstill env count, and replanned env count
- any train-side helper needed to run the one-iteration macro-benchmark sweep reproducibly

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_planner_instrumentation.py -k "benchmark_cli_defaults" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/scripts/bench_batched_planner.py Go2Pvcnn/scripts/train.py Go2Pvcnn/tests/test_batched_planner_instrumentation.py
git commit -m "feat: add batched planner benchmark entrypoints"
```

## Task 6: Verify End-to-End Runtime Path and Update Notes

**Files:**
- Modify: `Go2Pvcnn/tests/test_batched_manager.py`
- Modify: `Go2Pvcnn/tests/test_batched_planner_runtime_path.py`
- Modify: `notes/human/human-10-extension-planner-runtime.md`
- Modify: `notes/ai/ai-10-extension-planner-runtime.md`

- [ ] **Step 1: Write the failing tests**

```python
def test_teacher_trajectory_runtime_uses_planner_owned_cache_only():
    env = make_env_fixture()
    cache = ensure_reference_cache(env)
    assert cache.is_ready()


def test_partial_replan_keeps_reward_cache_contract_full_shaped():
    cache = ensure_reference_cache(env)
    assert cache.root_pos_w.ndim == 3
    assert cache.root_pos_w.shape[0] == env.num_envs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_manager.py Go2Pvcnn/tests/test_batched_planner_runtime_path.py -k "planner_owned_cache_only or partial_replan_keeps_reward_cache_contract" -v`
Expected: FAIL until the manager/cache/runtime integration is fully aligned.

- [ ] **Step 3: Write minimal implementation**

Implement:
- any final runtime-path fixes discovered by the end-to-end tests
- notes updates describing per-env replan semantics, standstill cache persistence, verbose planner diagnostics, and direct playback mode

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_manager.py Go2Pvcnn/tests/test_batched_planner_runtime_path.py -k "planner_owned_cache_only or partial_replan_keeps_reward_cache_contract" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Go2Pvcnn/tests/test_batched_manager.py Go2Pvcnn/tests/test_batched_planner_runtime_path.py notes/human/human-10-extension-planner-runtime.md notes/ai/ai-10-extension-planner-runtime.md
git commit -m "docs: update planner runtime notes"
```

## Final Verification

- [ ] Run: `python3 -m pytest Go2Pvcnn/tests/test_batched_trajectory.py Go2Pvcnn/tests/test_batched_trajectory_batch.py Go2Pvcnn/tests/test_batched_manager.py Go2Pvcnn/tests/test_batched_reference_integration.py Go2Pvcnn/tests/test_batched_planner_runtime_path.py Go2Pvcnn/tests/test_batched_planner_instrumentation.py -v`
Expected: PASS

- [ ] Run: `python3 Go2Pvcnn/scripts/bench_batched_planner.py --env-counts 1 16 64 100 256 512 1024 2048`
Expected: benchmark output includes planner stage timings and per-env normalized metrics for each env count

- [ ] Run: `python3 Go2Pvcnn/scripts/train.py --experiment teacher_elevation_trajectory --num_envs 1 --max_iterations 1 --headless --verbose-planner`
Expected: planner-owned runtime attaches successfully, one-iteration train run completes, and verbose planner diagnostics print without crashing

- [ ] Run: `for n in 1 16 64 100 256 512 1024 2048; do python3 Go2Pvcnn/scripts/train.py --experiment teacher_elevation_trajectory --num_envs "$n" --max_iterations 1 --headless --verbose-planner; done`
Expected: each run reports the existing train-side `Steps per second` / `Collection time` metrics and, when verbose mode is enabled, planner diagnostics sufficient to compare the env-count sweep required by the spec

- [ ] Run: `python3 Go2Pvcnn/extension/viz/go2_foostep_planner.py --num_envs 1 --planner-playback-mode direct --headless`
Expected: viewer starts with explicit direct planner playback mode available on the shared runtime path

- [ ] Commit final integration verification:

```bash
git add docs/superpowers/plans/2026-04-14-batched-planner-single-shot-decoupled-runtime.md
git commit -m "docs: add single-shot decoupled runtime implementation plan"
```
