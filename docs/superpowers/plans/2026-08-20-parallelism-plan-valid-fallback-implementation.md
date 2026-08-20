# Parallelism Plan Valid Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 在 large teacher RL 中，让有效 Parallelism 轨迹用于 reference tracking，规划失败时改为 command locomotion fallback，并完整区分有效/无效轨迹的统计。

**Architecture:** Reference manager 维护与 t+1 reference 同步的 step-level \`plan_valid\`。Actor/Critic 保留原 reference 维度并在 invalid 时置零，同时增加一维状态标志。Reference reward 和 reference termination 仅在 valid 时启用；速度、碰撞和机器人安全约束始终生效。Episode tracking、obstacle、termination 和 curriculum 统计按 valid/invalid 分桶。

**Tech Stack:** Python 3.10, PyTorch, IsaacLab manager/config API, pytest.

## Global Constraints

- 不新增 planner root velocity observation。
- 不修改 Parallelism planner 的候选、IK、重规划和 standstill 判定逻辑。
- large teacher 的 \`parallelism_geometry_collision.weight\` 为 \`-10.0\`。
- invalid reference 不得触发 reference-dependent termination。
- \`base_contact\`、\`bad_orientation\`、\`time_out\` 在 valid/invalid 两种状态都保留。
- small-obstacles 基础配置仍保持碰撞权重 \`-2.0\`。
- distillation 可以继承 large teacher 的 planner-valid 环境接口；保留其 student 专用 reward 和 observation 定义。
- large teacher 的 command 刷新周期为 `(10.0, 10.0)`；Parallelism 仍按 reset、command 改变和第 23 个控制步独立重规划。
- large teacher 的 `active_swing_foot_on_small_obstacle.weight` 为 `10.0`；small-obstacles 基础配置仍为 `0.5`。
- 保留工作区已有训练脚本改动，不回滚或覆盖。

---

### Task 1: Add step-level plan validity and masked observations

**Status: completed.**

**Files:**
- Modify: \`Go2Pvcnn/tracking/managers/parallelism_reference_manager.py\`
- Modify: \`Go2Pvcnn/tracking/mdp/observations.py\`
- Modify: \`Go2Pvcnn/tracking/parallelism_cross_large_complex_env_cfg.py\`
- Modify: \`Go2Pvcnn/tracking/mdp/__init__.py\`
- Test: \`Go2Pvcnn/tests/tracking/test_parallelism_plan_valid.py\`

**Interfaces:**
- Manager produces \`step_plan_valid: Tensor[num_envs]\`.
- Observation function \`parallelism_plan_valid(env) -> Tensor[num_envs, 1]\`.
- Existing reference observation functions preserve their shapes and return zeros where \`step_plan_valid=False\`.

- [ ] **Step 1: Write failing tests**
  - Build lightweight manager/env fixtures with one valid and one invalid environment.
  - Assert \`step_plan_valid\` follows \`trajectory.valid\`.
  - Assert invalid reference joint/root observations are zero and valid observations are unchanged.
  - Assert \`parallelism_plan_valid\` has shape \`[num_envs, 1]\` and values \`0/1\`.

- [ ] **Step 2: Run the focused tests**
  - Run: \`pytest Go2Pvcnn/tests/tracking/test_parallelism_plan_valid.py -q\`
  - Expected: FAIL because the new manager field and observation function do not exist.

- [ ] **Step 3: Implement the manager state**
  - Initialize \`step_plan_valid\` beside the step reference cache.
  - Set it from \`plan_valid\` in \`prepare_step_reference()\`.
  - Clear it on environment reset.
  - Keep the planner and trajectory generation unchanged.

- [ ] **Step 4: Implement observation masking**
  - Add a small helper that broadcasts \`step_plan_valid[:, None]\`.
  - Apply it to joint position, joint velocity, root position and root rotation reference observations.
  - Add \`parallelism_plan_valid()\`.
  - Export the new function through \`tracking/mdp/__init__.py\`.

- [ ] **Step 5: Enable the flag only for large teacher**
  - Add \`parallelism_plan_valid = ObsTerm(func=tracking_mdp.parallelism_plan_valid)\` to both large teacher policy and critic state configs.
  - Keep \`velocity_commands\` unchanged.
  - Do not add planner velocity terms.

- [ ] **Step 6: Run the focused tests**
  - Run: \`pytest Go2Pvcnn/tests/tracking/test_parallelism_plan_valid.py -q\`
  - Expected: PASS.

---

### Task 2: Gate reference rewards and split tracking/obstacle metrics

**Status: completed.**

**Files:**
- Modify: \`Go2Pvcnn/tracking/mdp/rewards.py\`
- Modify: \`Go2Pvcnn/tracking/env.py\`
- Modify: \`Go2Pvcnn/tracking/parallelism_cross_large_complex_env_cfg.py\`
- Test: \`Go2Pvcnn/tests/tracking/test_parallelism_plan_valid.py\`
- Test: \`Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py\`

**Interfaces:**
- Reference rewards return zero for invalid environments.
- Existing raw tracking statistics are accumulated separately for valid and invalid frames.
- Obstacle statistics expose valid/invalid collision and small-foot ratios.
- Existing metric names remain available for compatibility; new names add explicit \`valid_\` or \`invalid_\` prefixes.

- [ ] **Step 1: Write failing reward and metric tests**
  - Assert every reference reward returns zero for an invalid environment.
  - Assert command/locomotion rewards are not gated.
  - Assert geometry collision raw event remains nonzero in invalid mode.
  - Assert valid and invalid tracking accumulators use separate denominators.
  - Assert obstacle metrics expose \`valid_geometry_collision_ratio\` and \`invalid_geometry_collision_ratio\`.

- [ ] **Step 2: Run the focused tests**
  - Run: \`pytest Go2Pvcnn/tests/tracking/test_parallelism_plan_valid.py Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py -q\`
  - Expected: FAIL on reward gating and missing split metrics.

- [ ] **Step 3: Add the reference reward gate**
  - Add an internal helper returning \`step_plan_valid\` as the reward dtype.
  - Multiply root, joint and foot reference rewards by that gate after computing the raw reward.
  - Do not gate \`parallelism_geometry_collision_penalty\`, command velocity rewards, or generic locomotion rewards.

- [ ] **Step 4: Split tracking accumulators**
  - Add valid and invalid sums/counts for joint mean/max, foot mean/max, root position/rotation and active swing metrics.
  - In \`update_parallelism_tracking_error_stats()\`, route each frame using \`step_plan_valid\`.
  - Return both legacy aggregate values and explicit valid/invalid values; legacy values must be based on valid frames for reference curriculum compatibility.

- [ ] **Step 5: Split obstacle accumulators**
  - Route collision, safe small-foot and standstill events into valid/invalid buckets.
  - Keep \`parallelism_geometry_collision_penalty()\` unchanged in event semantics.
  - Return valid/invalid ratios with clamped per-bucket denominators and expose sample counts.

- [ ] **Step 6: Set the large collision weight**
  - In \`ParallelismCrossLargeTeacherRewardsCfg\` or its large config initialization, set:
    \`\`\`python
    self.rewards.parallelism_geometry_collision.weight = -10.0
    \`\`\`
  - Leave \`ParallelismSmallObstaclesRewardsCfg.weight=-2.0\` unchanged.

- [ ] **Step 7: Log split metrics**
  - Update \`tracking/env.py\` to log valid and invalid tracking metrics, obstacle ratios and sample counts.
  - Log termination buckets in Task 3.
  - Keep old aggregate metric names during the transition.

- [ ] **Step 8: Run focused tests**
  - Run: \`pytest Go2Pvcnn/tests/tracking/test_parallelism_plan_valid.py Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py -q\`
  - Expected: PASS.

---

### Task 3: Mask reference terminations and separate termination diagnostics

**Status: completed.**

**Files:**
- Modify: \`Go2Pvcnn/tracking/mdp/terminations.py\`
- Modify: \`Go2Pvcnn/tracking/env.py\`
- Modify: \`Go2Pvcnn/tracking/parallelism_cross_large_complex_env_cfg.py\`
- Test: \`Go2Pvcnn/tests/tracking/test_parallelism_plan_valid.py\`

**Interfaces:**
- Reference-dependent termination functions return false when \`step_plan_valid=False\`.
- Safety terminations are untouched.
- Termination diagnostic events are accumulated into valid/invalid buckets.

- [ ] **Step 1: Write failing termination tests**
  - Create valid and invalid manager states with the same large reference error.
  - Assert root-z, projected-gravity and joint-position reference termination are false for invalid envs.
  - Assert the same functions can still return true for valid envs.
  - Assert \`base_contact\` and \`bad_orientation\` are not gated by plan validity.

- [ ] **Step 2: Run focused tests**
  - Run: \`pytest Go2Pvcnn/tests/tracking/test_parallelism_plan_valid.py -q\`
  - Expected: FAIL because reference termination functions currently ignore \`plan_valid\`.

- [ ] **Step 3: Add the reference termination gate**
  - Add a helper that returns the manager step-valid mask.
  - Compute each existing termination exactly as before, then AND it with the valid mask.
  - Keep threshold and consecutive-step behavior unchanged.
  - Leave \`parallelism_consecutive_standstill\` disabled in large config.

- [ ] **Step 4: Add termination event buckets**
  - In the environment reset/logging path, classify the termination cause using the plan-valid state at the terminating step.
  - Emit names such as:
    \`\`\`
    Episode_Termination/valid_base_contact
    Episode_Termination/invalid_base_contact
    Episode_Termination/valid_bad_orientation
    Episode_Termination/invalid_bad_orientation
    Episode_Termination/valid_parallelism_ref_joint_pos_too_far
    Episode_Termination/invalid_parallelism_ref_joint_pos_too_far
    \`\`\`
  - Invalid reference termination buckets should remain zero by construction.

- [ ] **Step 5: Run focused tests**
  - Run: \`pytest Go2Pvcnn/tests/tracking/test_parallelism_plan_valid.py -q\`
  - Expected: PASS.

---

### Task 4: Make curriculum use valid reference or invalid fallback metrics

**Status: completed.**

**Files:**
- Modify: \`Go2Pvcnn/tracking/mdp/curriculums.py\`
- Modify: \`Go2Pvcnn/tracking/mdp/rewards.py\`
- Test: \`Go2Pvcnn/tests/tracking/test_parallelism_plan_valid.py\`

**Interfaces:**
- \`parallelism_velocity_curriculum()\` consumes valid reference errors and fallback command/safety metrics.
- Planner invalid ratio never directly decrements the curriculum.
- Reference errors from invalid frames never enter the reference success test.

- [ ] **Step 1: Write failing curriculum tests**
  - A timeout with invalid reference and good command fallback must not fail only because reference errors are large.
  - A valid reference episode must still use root/joint thresholds.
  - A safety termination must fail the curriculum in either mode.
  - Resetting curriculum stats must clear both valid and invalid buckets for selected environments only.

- [ ] **Step 2: Run focused tests**
  - Run: \`pytest Go2Pvcnn/tests/tracking/test_parallelism_plan_valid.py -q\`
  - Expected: FAIL because the curriculum currently consumes a single mixed error set.

- [ ] **Step 3: Implement split curriculum success**
  - Read valid reference errors from the valid bucket.
  - Read command tracking error/progress from fallback buckets for invalid frames.
  - Treat an episode with no valid frames as fallback-only.
  - Require safety success independently of plan validity.
  - Preserve existing level range interpolation and command range update.

- [ ] **Step 4: Run focused tests**
  - Run: \`pytest Go2Pvcnn/tests/tracking/test_parallelism_plan_valid.py -q\`
  - Expected: PASS.

---

### Task 5: Static registration checks, config validation and regression tests

**Status: completed.**

**Files:**
- Modify: \`Go2Pvcnn/tests/tracking/test_parallelism_tracking_registration_static.py\` only if existing expected metric names need extension.
- Create: \`Go2Pvcnn/tests/tracking/test_parallelism_plan_valid.py\`
- Modify: \`Go2Pvcnn/tracking/mdp/__init__.py\` if exports are missing.

- [ ] **Step 1: Run all focused tracking tests**
  - Run:
    \`\`\`
    pytest Go2Pvcnn/tests/tracking/test_parallelism_plan_valid.py \\
      Go2Pvcnn/tests/tracking/test_parallelism_tracking_mdp.py \\
      Go2Pvcnn/tests/tracking/test_parallelism_tracking_registration_static.py -q
    \`\`\`
  - Expected: PASS.

- [ ] **Step 2: Validate large config statically**
  - Run a Python import/config inspection that verifies:
    \`\`\`
    cfg.rewards.parallelism_geometry_collision.weight == -10.0
    cfg.terminations.parallelism_consecutive_standstill is None
    \`\`\`
  - Verify policy and critic observation groups each contain \`parallelism_plan_valid\`.
  - Verify no planner velocity observation was registered.

- [ ] **Step 3: Run the broader tracking regression**
  - Run: \`pytest Go2Pvcnn/tests/tracking -q\`
  - Expected: all existing tracking tests pass; unrelated pre-existing failures must be reported separately.

- [ ] **Step 4: Run a lightweight 1024-environment smoke test when IsaacLab is available**
  - Start the existing large headless training entrypoint with \`--num_envs 1024\` and a four-iteration limit.
  - Verify reset, first collection, replan at step 23, reward computation and logging complete without an exception.
  - Do not claim learning quality from four iterations.

- [ ] **Step 5: Commit implementation**
  - Run:
    \`\`\`
    git add Go2Pvcnn/tracking Go2Pvcnn/tests/tracking
    git commit -m "feat: add plan-valid fallback for large teacher"
    \`\`\`
  - Do not stage the pre-existing training-script changes.

---

## Verification Notes

- The old checkpoint cannot be resumed after adding one Actor/Critic observation dimension.
- The planner remains unchanged; this feature only changes the RL interface and diagnostics.
- A valid/invalid metric with zero samples must be reported as zero together with its sample count, never as an averaged value over unrelated frames.
