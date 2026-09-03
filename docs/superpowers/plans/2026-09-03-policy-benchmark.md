# Unified Policy Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, single-checkpoint-at-a-time Isaac Lab benchmark for Parallelism AMP, Distillation, Pure PPO, and the planner-conditioned Teacher across mixed terrain, large-obstacle runway, and small-obstacle runway suites.

**Architecture:** Keep training environments and reward definitions unchanged. Add a small `evaluation/` package containing deterministic command/layout manifests, GPU-batch metric accumulators, valid-only 24-frame trajectory alignment, atomic result shards, and model metadata. Add one Isaac entry script that reuses `play.py`'s observation wrapper and `OnPolicyRunner` loading contract, and one shell launcher for explicit checkpoint paths. Run each model in a separate process against the same manifest.

**Tech Stack:** Python 3.10, PyTorch, Gymnasium, Isaac Lab, RSL-RL `OnPolicyRunner`, JSONL/JSON result files, pytest.

## Global Constraints

- AMP, Distillation student, and Pure PPO use the same deployment observation contract; Teacher is marked as a planner-conditioned upper bound.
- The Parallelism reference horizon is exactly 24 state frames (`B0...B23`) and each segment contains 23 transitions.
- Behavior metrics include all episodes; planner tracking MSE includes only complete, continuous, planner-valid windows with one `plan_id` and no reset.
- Mixed commands are `vx={-1,-0.5,-0.25,0.25,0.5,1}`, `vy={-0.5,-0.25,0.25,0.5}`, `yaw={-1,-0.5,0,0.5,1}`: exactly 120 combinations.
- Runway commands use `vy=0`, `vx={0.25,0.5,1.0}`, and recompute yaw toward the goal at each 23-transition replan boundary.
- The environment is initialized once per model process; metrics and transition masks use Torch batch operations without per-environment Python loops in the step path.
- Checkpoints must be explicit files; smoke outputs carry `is_smoke_test=true` and cannot enter formal summaries.
- Preserve unrelated working-tree changes and never modify `raw/` reference code.

---

### Task 1: Deterministic benchmark protocols and manifest

**Files:**
- Create: `Go2Pvcnn/evaluation/__init__.py`
- Create: `Go2Pvcnn/evaluation/protocols.py`
- Create: `Go2Pvcnn/evaluation/manifest.py`
- Test: `Go2Pvcnn/tests/evaluation/test_protocols.py`
- Test: `Go2Pvcnn/tests/evaluation/test_manifest.py`

**Interfaces:**
- `protocols.mixed_velocity_commands() -> tuple[tuple[float, float, float], ...]` returns the ordered 120-command protocol.
- `protocols.runway_velocity_commands(layout_count: int, seed: int) -> tuple[tuple[float, float, float], ...]` returns a deterministic balanced vx assignment with `vy=0` and zero initial yaw placeholder.
- `protocols.replan_boundary(step: int, interval: int = 23) -> bool` is true at reset and at 23, 46, 69, ... transitions.
- `manifest.Condition` is a frozen dataclass with `condition_id`, `suite`, `layout_id`, `seed`, `command`, `goal_xy`, `terrain_row`, `terrain_col`, `large_obstacle_count`, `small_obstacle_count`, and `timeout_steps`.
- `manifest.build_conditions(suite: str, layout_count: int, seed: int) -> list[Condition]` produces the exact suite condition set.
- `manifest.write_manifest(path, conditions)` and `manifest.read_manifest(path)` use one JSON object per line and reject duplicate condition IDs.

- [ ] **Step 1: Write failing protocol and manifest tests**

```python
def test_mixed_protocol_has_exact_nonzero_velocity_grid():
    commands = mixed_velocity_commands()
    assert len(commands) == 120
    assert sorted({row[0] for row in commands}) == [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]
    assert 0.0 not in {row[1] for row in commands}

def test_replan_boundary_uses_23_transitions():
    assert replan_boundary(0)
    assert replan_boundary(23)
    assert replan_boundary(46)
    assert not replan_boundary(24)

def test_manifest_round_trip_rejects_duplicate_ids(tmp_path):
    conditions = build_conditions("large_runway", layout_count=2, seed=7)
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, conditions)
    assert read_manifest(path) == conditions
    path.write_text(path.read_text() + path.read_text())
    with pytest.raises(ValueError, match="duplicate condition_id"):
        read_manifest(path)
```

- [ ] **Step 2: Run the focused tests and verify they fail because the package is missing**

Run: `pytest -q tests/evaluation/test_protocols.py tests/evaluation/test_manifest.py`

Expected: collection failure mentioning missing `evaluation.protocols` or `evaluation.manifest`.

- [ ] **Step 3: Implement protocols and manifest generation**

Implement the Cartesian mixed grid in the listed order, deterministic seeded runway layout IDs, density mapping `2.5/5/7.5/10 -> 40/80/120/160` small obstacles, large counts `4/6/8/10`, and timeout `ceil(1.5 * 10 / vx / 0.02)` for each positive runway vx. Serialize tuples as JSON arrays and normalize all numeric values to JSON-safe floats/ints.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `pytest -q tests/evaluation/test_protocols.py tests/evaluation/test_manifest.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add evaluation tests/evaluation/test_protocols.py tests/evaluation/test_manifest.py
git commit -m "feat: add deterministic benchmark protocols and manifests"
```

### Task 2: Batch metrics and 24-frame valid-only alignment

**Files:**
- Create: `Go2Pvcnn/evaluation/metrics.py`
- Create: `Go2Pvcnn/evaluation/trajectory_alignment.py`
- Test: `Go2Pvcnn/tests/evaluation/test_metrics.py`
- Test: `Go2Pvcnn/tests/evaluation/test_trajectory_alignment.py`

**Interfaces:**
- `metrics.EpisodeAccumulator(num_envs, device)` owns sticky large/small collision, fall, path length, command progress, contact steps, and termination state tensors.
- `EpisodeAccumulator.update(root_pos_w, command, large_contact, small_contact, fallen, done, valid_step)` updates every row in one Torch call.
- `EpisodeAccumulator.finish(env_ids) -> list[dict]` returns per-episode records and clears only finished rows.
- `trajectory_alignment.ValidWindowBuffer(num_envs, horizon=24, device)` accepts actual and reference state frames plus `plan_id`, `plan_valid`, `reset`, and `transition_valid` tensors.
- `ValidWindowBuffer.push(...) -> torch.Tensor` returns a per-env valid-window mask and never copies a discontinuous replan segment into the current window.
- `trajectory_alignment.tracking_mse(actual, reference, valid_mask, anchor_root) -> dict[str, torch.Tensor]` computes joint position/velocity, root position/rotation/velocity, and active swing-foot errors after anchoring root positions to `B0`.

- [ ] **Step 1: Write failing metric and alignment tests**

```python
def test_episode_accumulator_counts_each_collision_once():
    acc = EpisodeAccumulator(2, torch.device("cpu"))
    acc.update(torch.zeros(2, 3), torch.zeros(2, 3), torch.tensor([True, False]), torch.tensor([False, True]), torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), torch.ones(2, dtype=torch.bool))
    records = acc.finish(torch.tensor([0, 1]))
    assert records[0]["large_collision_episode"] is True
    assert records[1]["small_collision_episode"] is True

def test_alignment_rejects_plan_id_change_and_reset():
    buffer = ValidWindowBuffer(1, horizon=24, device=torch.device("cpu"))
    frame = torch.zeros(1, 39)
    for step in range(24):
        mask = buffer.push(frame, frame, torch.tensor([0 if step < 23 else 1]), torch.tensor([True]), torch.tensor([step == 12]), torch.tensor([True]))
    assert not bool(mask.item())

def test_tracking_mse_anchors_root_to_window_start():
    actual = torch.zeros(1, 24, 39)
    reference = actual.clone()
    actual[:, :, 21] = torch.arange(24, dtype=torch.float32)
    reference[:, :, 21] = torch.arange(24, dtype=torch.float32) + 0.1
    metrics = tracking_mse(actual, reference, torch.ones(1, 24, dtype=torch.bool), actual[:, :1, 21:24])
    assert metrics["root_position_mse"].item() == pytest.approx(0.01)
```

- [ ] **Step 2: Run focused tests and verify the expected missing-module failures**

Run: `pytest -q tests/evaluation/test_metrics.py tests/evaluation/test_trajectory_alignment.py`

Expected: collection failure for the not-yet-created evaluation modules.

- [ ] **Step 3: Implement Torch-batch accumulators and alignment**

Use sticky boolean tensors for episode collision/fall state, cumulative tensors for distance/progress/contact counts, and indexed assignment for finished rows. Require 24 frames, exactly 23 valid transitions, stable `plan_id`, no reset, and `plan_valid=True` for a valid window. Treat invalid planner windows as excluded from MSE, never as zero error. Keep rotation error finite with quaternion geodesic distance and clamp dot products to `[-1, 1]`.

- [ ] **Step 4: Run focused tests and the existing tracking metric tests**

Run: `pytest -q tests/evaluation/test_metrics.py tests/evaluation/test_trajectory_alignment.py tests/test_mpc_policy_eval_metrics.py`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add evaluation/metrics.py evaluation/trajectory_alignment.py tests/evaluation/test_metrics.py tests/evaluation/test_trajectory_alignment.py
git commit -m "feat: add batched episode metrics and valid trajectory alignment"
```

### Task 3: Model adapters and resumable result writer

**Files:**
- Create: `Go2Pvcnn/evaluation/model_adapters.py`
- Create: `Go2Pvcnn/evaluation/result_writer.py`
- Test: `Go2Pvcnn/tests/evaluation/test_model_adapters.py`
- Test: `Go2Pvcnn/tests/evaluation/test_result_writer.py`

**Interfaces:**
- `model_adapters.ModelSpec` is a frozen dataclass containing `name`, `experiment`, `task_id`, `checkpoint`, `observation_kind`, `requires_reference_manager`.
- `model_adapters.resolve_model_spec(kind: str, checkpoint: Path) -> ModelSpec` supports `amp`, `distillation`, `ppo`, `teacher`, validates an existing file, and never accepts a directory.
- `model_adapters.build_play_mapping() -> dict[str, tuple[type, str]]` mirrors the four `play.py` task IDs and their PLAY config classes.
- `result_writer.ResultWriter(root, run_manifest)`: `write_episode`, `write_progress`, `load_completed_condition_ids`, and atomic `write_summary` methods.

- [ ] **Step 1: Write failing adapter and resume tests**

```python
def test_resolve_model_spec_rejects_directory(tmp_path):
    with pytest.raises(ValueError, match="explicit checkpoint file"):
        resolve_model_spec("ppo", tmp_path)

def test_result_writer_resume_skips_completed_conditions(tmp_path):
    writer = ResultWriter(tmp_path, {"manifest_hash": "x"})
    writer.write_episode({"condition_id": "c0"})
    writer.write_progress(["c0"])
    assert writer.load_completed_condition_ids() == {"c0"}
```

- [ ] **Step 2: Run focused tests and verify they fail**

Run: `pytest -q tests/evaluation/test_model_adapters.py tests/evaluation/test_result_writer.py`

Expected: collection failure for the not-yet-created modules.

- [ ] **Step 3: Implement explicit model mapping and atomic JSON output**

Use `play.py`'s task IDs and PLAY config classes for AMP, Distillation, and PPO; use the Parallelism teacher task for `teacher`. Store checkpoint SHA256, manifest SHA256, config hash, smoke flag, and completed condition IDs. Write each JSONL shard through a temporary sibling file followed by `os.replace`, and reject resume when hashes differ.

- [ ] **Step 4: Run focused tests**

Run: `pytest -q tests/evaluation/test_model_adapters.py tests/evaluation/test_result_writer.py`

Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add evaluation/model_adapters.py evaluation/result_writer.py tests/evaluation/test_model_adapters.py tests/evaluation/test_result_writer.py
git commit -m "feat: add benchmark model adapters and resumable results"
```

### Task 4: Isaac benchmark environment and single-model CLI

**Files:**
- Create: `Go2Pvcnn/evaluation/benchmark_suites.py`
- Create: `Go2Pvcnn/scripts/policy_benchmark.py`
- Create: `Go2Pvcnn/scripts/run_policy_benchmark.sh`
- Modify: `Go2Pvcnn/extension/semantic_course.py:62-70, 551-614`
- Test: `Go2Pvcnn/tests/evaluation/test_benchmark_suites.py`
- Test: `Go2Pvcnn/tests/evaluation/test_policy_benchmark_static.py`

**Interfaces:**
- `benchmark_suites.SUITE_NAMES` is `("complex_mixed", "large_runway", "small_runway")`.
- `benchmark_suites.build_suite_config(suite, base_cfg, condition) -> base_cfg` applies frozen terrain/curriculum, obstacle counts, command hold, and runway goal metadata without changing training configs in place.
- `benchmark_suites.goal_command(root_pos_w, root_yaw, goal_xy, vx) -> Tensor[N,3]` returns `[vx, 0, clipped_heading_error]`.
- `scripts/policy_benchmark.py` accepts `--experiment-type`, `--checkpoint`, `--suite`, `--num-envs`, `--layout-manifest`, `--max-steps`, `--output-dir`, `--smoke-test`, and Isaac `AppLauncher` flags.
- `scripts/run_policy_benchmark.sh` requires explicit `--experiment-type` and `--checkpoint`, forwards all remaining arguments, and defaults `PYTHONPATH` to the project and RSL-RL roots.

- [ ] **Step 1: Write failing suite and CLI contract tests**

```python
def test_goal_command_has_zero_lateral_velocity():
    command = goal_command(torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([0.0]), torch.tensor([[1.0, 0.0]]), 0.5)
    assert command.shape == (1, 3)
    assert command[0, 0].item() == pytest.approx(0.5)
    assert command[0, 1].item() == pytest.approx(0.0)

def test_benchmark_cli_requires_explicit_checkpoint_and_exposes_smoke_flags():
    source = Path("scripts/policy_benchmark.py").read_text()
    assert "--experiment-type" in source
    assert "--checkpoint" in source
    assert "--layout-manifest" in source
    assert "AppLauncher.add_app_launcher_args(parser)" in source
```

- [ ] **Step 2: Run focused tests and verify they fail**

Run: `pytest -q tests/evaluation/test_benchmark_suites.py tests/evaluation/test_policy_benchmark_static.py`

Expected: collection or assertion failure because the new suite/CLI files do not exist.

- [ ] **Step 3: Implement fixed-layout suite configuration and semantic large-point support**

Extend `SemanticCourseLayoutCfg` with optional `fixed_large_obstacle_local_xy` and apply it in `_stage_slots` with the same bounds/spacing checks as fixed small points. `build_suite_config` must copy the incoming config, set `terrain_generator.curriculum=False`, disable command resampling, set the suite counts, and attach per-condition goal metadata. Use a 10x2 m runway generator for the two runway suites and preserve the existing mixed terrain generator for `complex_mixed`. The evaluator must attach a read-only `ParallelismReferenceManager` for PPO/Distillation so all four models can produce comparable valid-only MSE; it must not alter actor observations.

- [ ] **Step 4: Implement the single-model Isaac rollout**

Reuse `play.py`'s `SimpleRslRlEnvWrapper` behavior and `OnPolicyRunner.load(checkpoint, load_optimizer=False)`. Use deterministic inference (`runner.get_inference_policy`), update runway command only at `replan_boundary`, call `env.step` once per transition, feed manager state/reference frames to `ValidWindowBuffer`, update GPU-batch metrics, and write completed episode records through `ResultWriter`. Isolate non-finite rows as `numerical_failure` and continue remaining env rows. Keep a `--smoke-test` marker in the run manifest.

- [ ] **Step 5: Run static and compile tests**

Run: `pytest -q tests/evaluation/test_benchmark_suites.py tests/evaluation/test_policy_benchmark_static.py && python -m py_compile evaluation/*.py scripts/policy_benchmark.py`

Expected: all selected tests pass and py_compile exits 0.

- [ ] **Step 6: Commit**

```bash
git add evaluation/benchmark_suites.py scripts/policy_benchmark.py scripts/run_policy_benchmark.sh extension/semantic_course.py tests/evaluation/test_benchmark_suites.py tests/evaluation/test_policy_benchmark_static.py
git commit -m "feat: add single-model Isaac policy benchmark runner"
```

### Task 5: Summary statistics, tests, notes, and real Isaac acceptance

**Files:**
- Create: `Go2Pvcnn/scripts/summarize_policy_benchmark.py`
- Create: `Go2Pvcnn/tests/evaluation/test_summarize_policy_benchmark.py`
- Create: `Go2Pvcnn/tests/evaluation/run_policy_benchmark_1024_smoke.py`
- Modify: `Go2Pvcnn/.gitignore`
- Modify: `notes/todo/T305-policy-benchmark.md`
- Modify: `notes/todo.md`
- Modify: `notes/log/index.md`
- Create: `notes/log/2026-09-03-policy-benchmark-implementation-and-smoke.md`

**Interfaces:**
- `summarize_policy_benchmark.py` reads four completed model result roots, rejects smoke runs, checks manifest/config hashes, and writes paired macro means, bootstrap 95% CIs, Wilcoxon/McNemar-compatible paired tables, and valid-window statistics.
- `run_policy_benchmark_1024_smoke.py` launches one explicit checkpoint with 1024 environments for 48 transitions, asserts finite actions/observations/metrics, confirms a complete 24-frame window, and exits nonzero on any traceback/NaN/Inf/OOM.

- [ ] **Step 1: Write failing summary and smoke contract tests**

```python
def test_summary_rejects_smoke_results(tmp_path):
    root = tmp_path / "amp"
    root.mkdir()
    (root / "run_manifest.json").write_text(json.dumps({"is_smoke_test": True}))
    with pytest.raises(ValueError, match="smoke"):
        load_formal_run(root)
```

- [ ] **Step 2: Run focused tests and verify they fail**

Run: `pytest -q tests/evaluation/test_summarize_policy_benchmark.py`

Expected: collection failure for missing summary module.

- [ ] **Step 3: Implement formal summary and ignore generated results**

Aggregate only matching `condition_id` rows, compute macro averages over the 120 mixed commands and density/vx groups, use layout-level bootstrap with a fixed seed, and write CSV/JSON output. Add `results/policy_benchmark/` to `.gitignore`; leave code, schemas, and notes tracked.

- [ ] **Step 4: Implement and run the real 1024-env smoke**

Run:

```bash
python tests/evaluation/run_policy_benchmark_1024_smoke.py \
  --experiment-type amp \
  --checkpoint /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn/logs/rsl_rl/parallelism_tracking_cross_large_complex_amp/2026-08-31_22-29-10/4234fbe/model_*.pt \
  --num-envs 1024 \
  --transitions 48
```

Expected: Isaac exits 0, all 1024 rows remain finite, at least one valid 24-frame window is recorded, and no OOM/NaN/Inf/Traceback appears. Resolve the glob to one concrete checkpoint before invoking the script; the launcher must reject an unresolved directory or wildcard.

- [ ] **Step 5: Run the complete evaluation unit suite and record evidence**

Run: `pytest -q tests/evaluation tests/tracking/test_parallelism_reference_manager.py tests/test_mpc_policy_eval_metrics.py`

Expected: all selected tests pass. Record exact duration, transition count, env-steps/s, and output paths in `notes/log/2026-09-03-policy-benchmark-implementation-and-smoke.md`.

- [ ] **Step 6: Update dashboard and commit**

Update T305 state, open leaves, latest verified ref, and test log links in the required notes files, then run `git diff --check` and commit only benchmark files/notes:

```bash
git add scripts/summarize_policy_benchmark.py tests/evaluation .gitignore notes/todo/T305-policy-benchmark.md notes/todo.md notes/log/index.md notes/log/2026-09-03-policy-benchmark-implementation-and-smoke.md
git commit -m "test: verify unified policy benchmark harness"
```

## Review Checklist

- [ ] Plan covers all sections of the approved Chinese HTML design: four model loading, three suites, command grid, 24/23 timing, fixed layouts, valid-only MSE, all-episode behavior metrics, resumable JSONL output, and 1024-env real acceptance.
- [ ] No production code was written before its focused failing test.
- [ ] No task accepts a checkpoint directory or silently falls back to a different model.
- [ ] PPO/Distillation shadow planning cannot change their actor observations.
- [ ] Generated results are ignored and unrelated user changes remain unstaged.
