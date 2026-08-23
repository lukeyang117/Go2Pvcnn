# Parallelism Terrain-Weighted Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add GPU-parallel, terrain-aware imitation weighting to the Parallelism distillation task while keeping the student as the only environment controller and guaranteeing zero imitation gradient for invalid plans.

**Architecture:** IsaacLab computes a two-column distillation context per environment: [imitation_multiplier, plan_valid]. The wrapper passes this context alongside student, teacher, and critic observations. RSL-RL stores the context per transition and applies teacher_coef * mean(multiplier * per-sample MSE) during PPO updates. Terrain family and curriculum level are resolved on the environment side; the generic optimizer never accesses IsaacLab objects.

**Tech Stack:** Python 3.10, PyTorch, IsaacLab ManagerBasedRLEnv, RSL-RL rollout storage, pytest, IsaacSim headless smoke test.

## Global Constraints

- teacher_coef=0.1, ppo_coef=1.0.
- teacher_ratio_start=0.0 and teacher_ratio_end=0.0; every environment executes student action.
- flat_dense_small_obstacles and flat keep multiplier 1.0 at every terrain level.
- All other terrain multipliers start at 1.0 and decay by terrain-specific endpoint and power.
- plan_valid=0 must produce exactly zero imitation gradient while the same sample remains active for PPO.
- Student keeps elevation/semantic maps and does not receive Parallelism reference trajectory.
- New teacher checkpoint requires teacher observation dimension 1103; student critic remains independent.
- Preserve existing unrelated dirty files and do not modify non-distillation training behavior.

---

### Task 1: Add pure terrain-weight and invalid-plan tests

**Files:**
- Create: Go2Pvcnn/tests/tracking/test_parallelism_distillation_weights.py
- Modify: none

**Interfaces:**
- Define terrain_imitation_context_from_metadata(terrain_types, terrain_levels, terrain_column_names, end_multipliers, powers, num_rows, plan_valid).
- Return a float tensor [N, 2]; column 0 is the terrain multiplier after the valid mask and column 1 is the valid mask.

- [x] Step 1: Write endpoint, monotonicity, invalid-plan, and unknown-terrain tests.

~~~python
def test_flat_weights_stay_one_and_complex_weights_reach_endpoints():
    context = terrain_imitation_context_from_metadata(
        terrain_types=torch.tensor([0, 1, 2, 3]),
        terrain_levels=torch.tensor([0, 9, 0, 9]),
        terrain_column_names=("flat_dense_small_obstacles", "flat", "boxes", "pyramid_stairs"),
        end_multipliers={"flat_dense_small_obstacles": 1.0, "flat": 1.0, "boxes": 0.0, "pyramid_stairs": 0.0},
        powers={"flat_dense_small_obstacles": 1.0, "flat": 1.0, "boxes": 1.5, "pyramid_stairs": 2.0},
        num_rows=10,
        plan_valid=torch.ones(4),
    )
    assert torch.equal(context[:, 0], torch.tensor([1.0, 1.0, 1.0, 0.0]))
    assert torch.equal(context[:, 1], torch.ones(4))

def test_invalid_plan_zeroes_only_imitation_context():
    context = terrain_imitation_context_from_metadata(
        terrain_types=torch.tensor([0, 0]),
        terrain_levels=torch.tensor([0, 0]),
        terrain_column_names=("flat",),
        end_multipliers={"flat": 1.0},
        powers={"flat": 1.0},
        num_rows=10,
        plan_valid=torch.tensor([1.0, 0.0]),
    )
    assert torch.equal(context[:, 0], torch.tensor([1.0, 0.0]))
    assert torch.equal(context[:, 1], torch.tensor([1.0, 0.0]))
~~~

Also assert unknown terrain columns return [0, 0] and complex terrain multipliers are non-increasing from Level 0 to Level 9.

- [x] Step 2: Run the focused test before implementation.

Run: pytest -q Go2Pvcnn/tests/tracking/test_parallelism_distillation_weights.py

Expected: FAIL because tracking.mdp.distillation and the helper do not exist.

---

### Task 2: Implement GPU-parallel context and observation configuration

**Files:**
- Create: Go2Pvcnn/tracking/mdp/distillation.py
- Modify: Go2Pvcnn/tracking/mdp/__init__.py
- Modify: Go2Pvcnn/tracking/parallelism_cross_large_complex_distillation_env_cfg.py
- Test: Go2Pvcnn/tests/tracking/test_parallelism_distillation_weights.py

**Interfaces:**
- terrain_imitation_context_from_metadata(...) -> Tensor[N, 2] is pure and testable.
- parallelism_distillation_context(env, end_multipliers, powers, num_rows=10) -> Tensor[N, 2] reads terrain buffers and the live Parallelism manager.

- [x] Step 1: Implement the pure helper with device-preserving Torch operations.

~~~python
difficulty = (terrain_levels.float() / max(num_rows - 1, 1)).clamp(0.0, 1.0)
value = end + (1.0 - end) * torch.pow(1.0 - difficulty, power)
multiplier = torch.where(terrain_types == column, value, multiplier)
multiplier = multiplier * plan_valid.float()
return torch.stack((multiplier, plan_valid.float()), dim=-1)
~~~

Unknown terrain names produce zero multiplier. If plan validity or required terrain metadata is unavailable, return an all-zero context so unknown state cannot trust the teacher.

- [x] Step 2: Resolve generated terrain column names from sub-terrain proportions and num_cols. Loop only over terrain families, never over environments. Read terrain_types, terrain_levels, and step_plan_valid on the active device.

- [x] Step 3: Export parallelism_distillation_context from tracking/mdp/__init__.py.

- [x] Step 4: Add a non-corrupted context observation group.

~~~python
@configclass
class DistillationContextCfg(ObsGroup):
    imitation_context = ObsTerm(
        func=tracking_mdp.parallelism_distillation_context,
        params={
            "end_multipliers": {
                "flat_dense_small_obstacles": 1.0,
                "flat": 1.0,
                "random_rough": 0.30,
                "hf_pyramid_slope": 0.20,
                "hf_pyramid_slope_inv": 0.20,
                "boxes": 0.0,
                "pyramid_stairs": 0.0,
                "pyramid_stairs_inv": 0.0,
            },
            "powers": {
                "flat_dense_small_obstacles": 1.0,
                "flat": 1.0,
                "random_rough": 1.0,
                "hf_pyramid_slope": 1.25,
                "hf_pyramid_slope_inv": 1.25,
                "boxes": 1.5,
                "pyramid_stairs": 2.0,
                "pyramid_stairs_inv": 2.0,
            },
            "num_rows": 10,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
~~~

Add distillation_context: DistillationContextCfg to the observation config. Add parallelism_plan_valid to TeacherStateCfg, and explicitly set it to None in StudentStateCfg so the student input dimension remains unchanged.

- [x] Step 5: Run the focused weight tests.

Run: pytest -q Go2Pvcnn/tests/tracking/test_parallelism_distillation_weights.py

Expected: PASS.

---

### Task 3: Store context in rollout storage

**Files:**
- Modify: Go2Pvcnn/rsl_rl/rsl_rl/storage/rollout_storage.py:12-105,188-258
- Test: Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py

**Interfaces:**
- Transition.imitation_weight and Transition.plan_valid.
- RolloutStorage.imitation_weights and RolloutStorage.plan_valid_masks, both shaped [T, N, 1].
- mini_batch_generator(..., include_imitation_context=True) appends both fields after the PPO mask.

- [x] Step 1: Add a failing round-trip test that stores [1, 0] for both fields and asserts the minibatch returns the same tensors.

- [x] Step 2: Run the focused storage test and verify it fails before the fields exist.

Run: pytest -q Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py -k imitation_context

- [x] Step 3: Add transition fields, default-one storage buffers, copy logic, flattening, and the optional generator output. Legacy callers that omit context must receive default ones.

- [x] Step 4: Run the full existing distillation static test file.

Run: pytest -q Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py

Expected: PASS.

---

### Task 4: Pass context through the wrapper and runner

**Files:**
- Modify: Go2Pvcnn/scripts/train.py:745-816
- Modify: Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py:38-105,160-242
- Test: Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py

**Interfaces:**
- Wrapper returns extras[observations][distillation_context] with shape [N, 2].
- Runner calls alg.act(obs, teacher_obs, critic_obs, distillation_context) in hybrid mode.

- [x] Step 1: Add source contract assertions for the context name in the wrapper and runner.

- [x] Step 2: Run the contract test and verify it fails before the changes.

- [x] Step 3: In _format_observations, read obs_dict[distillation_context] and return it under extras[observations] together with teacher and critic. The same shared formatter covers reset and step.

- [x] Step 4: In runner initialization and every step, read context and move it to the training device. Do not apply observation normalization to the context.

- [x] Step 5: Run the tracking distillation static tests and expect PASS.

---

### Task 5: Apply masked imitation loss and metrics

**Files:**
- Modify: Go2Pvcnn/rsl_rl/rsl_rl/algorithms/hybrid_distillation_ppo.py:189-239,261-376
- Test: Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py

**Interfaces:**
- HybridDistillationPPO.act(obs, teacher_obs, critic_obs=None, distillation_context=None).
- update() returns weighted and unweighted imitation metrics, actual contribution, effective coefficient mean, valid ratio, and the epsilon-protected imitation/surrogate ratio.

- [x] Step 1: Add autograd regression tests.

~~~python
student_mean = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
teacher_action = torch.zeros_like(student_mean)
weights = torch.tensor([1.0, 0.0])
sample_mse = (student_mean - teacher_action).pow(2).mean(dim=-1)
loss = (sample_mse * weights).mean()
loss.backward()
assert torch.equal(student_mean.grad[1], torch.zeros(2))
assert torch.any(student_mean.grad[0] != 0)
~~~

Also test an all-zero weight batch: loss is finite and exactly zero, and all imitation gradients are zero.

- [x] Step 2: Add context parsing in act(). If no context is supplied by a legacy caller, use ones. Store detached multiplier and plan-valid tensors in the transition. Keep teacher actions deterministic, detached targets.

- [x] Step 3: In update(), compute the following without denominator normalization.

~~~python
sample_mse = torch.mean((mu_batch - privileged_actions_batch).pow(2), dim=-1)
imitation_loss_unweighted = sample_mse.mean()
imitation_loss_weighted = torch.mean(sample_mse * imitation_weight_batch.reshape(-1))
imitation_contribution = teacher_coef * imitation_loss_weighted
loss = ppo_coef * ppo_loss + imitation_contribution
~~~

When every weight is zero, the expression remains connected to mu_batch and gives exact zero imitation gradient. PPO terms remain active.

- [x] Step 4: Return and log imitation_loss_unweighted, imitation_loss_weighted, imitation_contribution, effective_teacher_coef_mean, plan_valid_ratio, and imitation_to_surrogate_ratio = abs(imitation_contribution) / (abs(surrogate_loss) + 1e-6).

- [x] Step 5: Run all distillation static tests.

Run: pytest -q Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py

Expected: PASS.

---

### Task 6: Set defaults and fresh teacher launcher

**Files:**
- Modify: Go2Pvcnn/agent/train_cfg.py:23-82
- Modify: Go2Pvcnn/scripts/train_parallelism_large_obstacles_rl_headless_distilation.sh:13-23
- Test: Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py

- [x] Step 1: Add assertions for teacher_coef=0.1, new teacher path, and zero teacher ratio in the fresh launcher.

- [x] Step 2: Set distillation config defaults to teacher_coef=0.1, ppo_coef=1.0, num_learning_epochs=5, entropy_coef=0.01, init_noise_std=1.0, and student-only teacher ratio. Keep direct algorithm schedule compatibility tests working.

- [x] Step 3: Update only the fresh launcher:

~~~bash
--teacher_checkpoint /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn/logs/rsl_rl/parallelism_tracking_cross_large_complex/2026-08-20_21-20-52/91b27a4/model_9899.pt
--ppo-coef 1.0
--teacher-coef 0.1
--teacher-ratio-start 0.0
--teacher-ratio-end 0.0
~~~

Do not overwrite the user's pre-existing dirty resume scripts.

- [x] Step 4: Run the distillation static tests and configuration tests.

---

### Task 7: Validate dimensions, context, and regression behavior

**Files:**
- Modify: Go2Pvcnn/tests/tracking/test_parallelism_distillation_env_cfg.py
- Modify: Go2Pvcnn/tests/tracking/test_parallelism_distillation_static.py

- [x] Step 1: Assert the config declares distillation_context, teacher declares parallelism_plan_valid, and student removes it.

- [x] Step 2: Assert teacher actor input is 1103, student actor remains the existing dimension after CNN flattening, and critic remains the existing dimension after CNN flattening.

- [x] Step 3: Run:

~~~bash
pytest -q Go2Pvcnn/tests/tracking
pytest -q Go2Pvcnn/tests/tracking Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py
~~~

Expected: PASS.

---

### Task 8: Run 1024-env smoke test and commit

**Files:**
- Test: Go2Pvcnn/scripts/train_parallelism_large_obstacles_rl_headless_distilation.sh

- [x] Step 1: Run four fresh IsaacSim iterations.

~~~bash
cd /share/home/tm884089579940000/a915071960/lhy/kinematic/Go2Pvcnn
MAX_ITERATIONS=4 ./Go2Pvcnn/scripts/train_parallelism_large_obstacles_rl_headless_distilation.sh
~~~

Expected: 1024 environments initialize, dimensions print correctly, four iterations complete, and no checkpoint shape error, NaN, or missing context field occurs.

- [x] Step 2: Verify TensorBoard/log output contains Distillation/imitation_loss_weighted, Distillation/imitation_contribution, Distillation/effective_teacher_coef_mean, Distillation/plan_valid_ratio, and Distillation/ppo_active_ratio.

- [x] Step 3: Review and commit only implementation files.

~~~bash
git diff --check
git status --short
git add Go2Pvcnn/tracking/mdp/distillation.py Go2Pvcnn/tracking/mdp/__init__.py Go2Pvcnn/tracking/parallelism_cross_large_complex_distillation_env_cfg.py Go2Pvcnn/rsl_rl/rsl_rl/storage/rollout_storage.py Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py Go2Pvcnn/rsl_rl/rsl_rl/algorithms/hybrid_distillation_ppo.py Go2Pvcnn/scripts/train.py Go2Pvcnn/agent/train_cfg.py Go2Pvcnn/scripts/train_parallelism_large_obstacles_rl_headless_distilation.sh Go2Pvcnn/tests/tracking
git commit -m "feat: add terrain-weighted distillation with invalid-plan masking"
~~~

The commit must not include the pre-existing dirty resume scripts or deleted teacher launcher.

## Verification Summary

The implementation is complete only when pure weighting tests, exact zero-gradient tests, tracking regression tests, and the 1024-environment four-iteration IsaacSim smoke test all pass. A passing unweighted imitation loss is insufficient; the invalid-plan autograd assertion is mandatory.
