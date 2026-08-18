# Parallelism Distillation Command/Replan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (&#96;- [ ]&#96;) syntax for tracking.

**Goal:** Align the distillation task's velocity rewards and linear-velocity curriculum with Go2Pvcnn while making Parallelism replan independently after the 23 future control frames in its 24-frame reference.

**Architecture:** Keep command resampling at 100 seconds and reuse Go2Pvcnn's velocity reward/curriculum functions. Add an explicit per-environment Parallelism control-step counter in the reference manager; reset, command changes, and the 23-step timer each invalidate and refresh only the affected environments. Reference frame 0 is the measured state and frames 1 through 23 are the 23 control targets.

**Tech Stack:** Python, PyTorch, Isaac Lab configuration classes, pytest, headless Isaac Sim smoke test.

## Global Constraints

- Angular velocity starts at [-1.0, 1.0] and never participates in curriculum expansion.
- Command resampling is (100.0, 100.0) and is independent from Parallelism replanning.
- Parallelism replans after reset, command change, or 23 completed control steps.
- Every new trajectory writes the measured state to frame 0; policy-facing references use frame 1.
- Existing user changes in Go2Pvcnn/scripts/train_parallelism_large_obstacles_rl_headless_distilation_resume.sh must be preserved.
- Do not modify shared Go2Pvcnn reward or curriculum functions.

---

### Task 1: Update failing configuration and timing tests

**Files:**
- Modify: Go2Pvcnn/tests/tracking/test_parallelism_official_velocity_curriculum.py
- Modify: Go2Pvcnn/tests/tracking/test_parallelism_distillation_env_cfg.py
- Modify: Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py

**Interfaces:**
- Tests consume the existing distillation config source and ParallelismReferenceManager.
- Later tasks make these tests pass without weakening their assertions.

- [ ] Step 1: Replace stale velocity-reward assertions

Assert source/config values for weights 1.5, 0.75, and numeric std 0.5; assert command resampling (100.0, 100.0), initial lin_vel_y=(-0.1, 0.1), and initial yaw (-1.0, 1.0).

- [ ] Step 2: Add explicit 24-step timer tests

Add tests showing that unchanged command does not replan for 22 completed control steps and does replan on the 23rd. Add tests for command-change immediate replan, reset replan, combined trigger deduplication, and independent per-environment counters.

- [ ] Step 3: Run the focused tests before implementation

~~~bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn
pytest -q tests/tracking/test_parallelism_official_velocity_curriculum.py \
  tests/tracking/test_parallelism_distillation_env_cfg.py \
  tests/tracking/test_parallelism_reference_manager.py
~~~

Expected: FAIL because the current config still has old reward/command values and the manager has no independent counter.

- [ ] Step 4: Commit the red tests

~~~bash
git add tests/tracking/test_parallelism_official_velocity_curriculum.py \
  tests/tracking/test_parallelism_distillation_env_cfg.py \
  tests/tracking/test_parallelism_reference_manager.py
git commit -m "test: specify distillation command and replan timing"
~~~

### Task 2: Align the distillation configuration

**Files:**
- Modify: Go2Pvcnn/tracking/parallelism_cross_large_complex_distillation_env_cfg.py
- Test: Go2Pvcnn/tests/tracking/test_parallelism_official_velocity_curriculum.py

**Interfaces:**
- Produces the effective ParallelismTrackingCrossLargeComplexDistillationEnvCfg command, reward, and curriculum objects.
- Reuses go2_mdp.track_lin_vel_xy_exp, go2_mdp.track_ang_vel_z_exp, and go2_mdp.lin_vel_cmd_levels.

- [ ] Step 1: Set the Go2Pvcnn velocity reward values

Set:

~~~python
track_lin_vel_xy.weight = 1.5
track_ang_vel_z.weight = 0.75
params["std"] = math.sqrt(0.25)
~~~

Import math if required. Leave all non-velocity rewards unchanged.

- [ ] Step 2: Define distillation-local command values

In the distillation environment post-init, set:

~~~python
self.commands.base_velocity.resampling_time_range = (100.0, 100.0)
self.commands.base_velocity.rel_standing_envs = 0.1
self.commands.base_velocity.ranges.lin_vel_x = (-0.1, 0.1)
self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
self.commands.base_velocity.limit_ranges.ang_vel_z = (-1.0, 1.0)
~~~

Keep the existing linear-velocity limits and parallelism_velocity = None.

- [ ] Step 3: Run the config tests

Run the focused config tests and expect PASS.

### Task 3: Add an explicit Parallelism replan interval

**Files:**
- Modify: Go2Pvcnn/extension/parallelism/config.py
- Modify: Go2Pvcnn/tracking/managers/parallelism_reference_manager.py
- Test: Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py

**Interfaces:**
- ParallelismCfg.replan_interval_steps: int = 23.
- ParallelismReferenceManager.parallelism_step_count: Tensor[int64] is per environment.
- Existing reset, mark_command_changed, refresh, and prepare_step_reference APIs remain compatible.

- [ ] Step 1: Add the per-environment counter test cases

Use the existing fake environment and monkeypatched _plan to assert plan counts and counter values. Keep existing frame-0/frame-1 tests intact, updating only stale assumptions about the old 23-step boundary.

- [ ] Step 2: Add the interval field

Add replan_interval_steps next to horizon and dt, with default value 23. This is horizon - 1 because frame 0 is the measured state and is not a control target.

- [ ] Step 3: Replace episode-derived cycle timing

Initialize a tensor counter with one value per environment. refresh() must:

~~~python
phase = torch.clamp(parallelism_step_count, max=horizon - 1)
timer_due = parallelism_step_count >= replan_interval_steps
needs_plan = (~initialized) | timer_due
~~~

When a plan is installed, reset the affected environments' counter and phase to zero. Keep _cached_cycle as a monotonic plan generation token for diagnostics/tests, but stop using episode_length_buf or horizon - 1 to determine the timer.

- [ ] Step 4: Keep command-change and reset triggers

mark_command_changed() must invalidate only the selected environments and reset their Parallelism counter. on_environment_reset() must clear the selected counter, phase, reference validity, standstill state, and initialization state. The existing reset hook remains the immediate reset trigger.

- [ ] Step 5: Advance the counter once per control action

At the end of prepare_step_reference(), increment the selected per-environment counter exactly once for the prepared control step. Do not advance it from reward/observation property reads. A command-change or timer-triggered refresh before target-frame selection must plan first, then the next policy target remains frame 1. The 24th cached frame is not an additional action: frame 0 is state-only, and frames 1 through 23 are the control sequence.

- [ ] Step 6: Prevent duplicate plans for combined triggers

The refresh path must build one boolean needs_plan mask and call _plan once per refresh, so reset/command/timer overlap cannot issue duplicate plans.

- [ ] Step 7: Run manager tests

~~~bash
pytest -q tests/tracking/test_parallelism_reference_manager.py
~~~

Expected: PASS, including reset, command-change, timer, multi-environment isolation, and frame alignment tests.

### Task 4: Run the complete validation suite

**Files:**
- Test: existing tracking and distillation tests
- Runtime: Go2Pvcnn/scripts/train.py

- [ ] Step 1: Run all affected pytest tests

~~~bash
pytest -q \
  tests/tracking/test_parallelism_official_velocity_curriculum.py \
  tests/tracking/test_parallelism_distillation_env_cfg.py \
  tests/tracking/test_parallelism_distillation_static.py \
  tests/tracking/test_parallelism_reference_manager.py
~~~

- [ ] Step 2: Run syntax and compile checks

~~~bash
python -m py_compile \
  tracking/parallelism_cross_large_complex_distillation_env_cfg.py \
  tracking/managers/parallelism_reference_manager.py \
  extension/parallelism/config.py
python -m compileall -q tracking extension/parallelism
bash -n scripts/train_parallelism_large_obstacles_rl_headless_distilation.sh
bash -n scripts/train_parallelism_large_obstacles_rl_headless_distilation_resume.sh
~~~

- [ ] Step 3: Run the 1024-environment four-iteration smoke test

Use the existing headless distillation training entrypoint with --num_envs 1024 and --max_iterations 4, preserving the current script's teacher/student and std arguments. Verify startup, reset, command initialization, reference frame shapes, and at least one timer-driven replanning event after 23 control steps without a tensor-shape error.

- [ ] Step 4: Record timer evidence

The smoke test must report or expose plan counts showing that an unchanged command produces a replan after 23 control steps while the command itself remains unchanged.

### Task 5: Commit the implementation

**Files:**
- Commit only the implementation and tests from this plan.
- Preserve the pre-existing modified resume script unless it is directly required by this feature.

- [ ] Step 1: Inspect the final diff

~~~bash
git diff --check
git status --short
git diff --stat
~~~

- [ ] Step 2: Commit

~~~bash
git add Go2Pvcnn/tracking/parallelism_cross_large_complex_distillation_env_cfg.py \
  Go2Pvcnn/extension/parallelism/config.py \
  Go2Pvcnn/tracking/managers/parallelism_reference_manager.py \
  Go2Pvcnn/tests/tracking/test_parallelism_official_velocity_curriculum.py \
  Go2Pvcnn/tests/tracking/test_parallelism_distillation_env_cfg.py \
  Go2Pvcnn/tests/tracking/test_parallelism_reference_manager.py
git commit -m "feat: decouple parallelism replanning from command sampling"
~~~

- [ ] Step 3: Verify the final worktree

~~~bash
git status --short --branch
git log -2 --oneline
~~~
