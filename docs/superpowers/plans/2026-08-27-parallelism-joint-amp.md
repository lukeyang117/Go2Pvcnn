# Parallelism Joint AMP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated AMP training experiment that uses batched 24-frame Parallelism state trajectories as discriminator expert windows, preserves pure-PPO/distillation behavior, supports legacy-policy warm start and full AMP resume, and passes a real 1024-environment four-iteration Isaac Lab smoke.

**Architecture:** A new `ParallelismAmpManager` owns fixed-shape per-environment transition rings. It extracts relative joint/SE(3) deltas and reconstructs/localizes 24-frame windows with Torch batch operations, including across successful replans; invalid/standstill/reset rows clear only their own history. A new `ParallelismAMPPPO` combines the existing base GAE with a masked AMP GAE and trains a separate AMP value head plus discriminator, while the actor and base critic remain compatible with the existing `ActorCriticCNN` checkpoint format.

**Tech Stack:** Python 3.10, PyTorch, the local `rsl_rl` package, Isaac Lab/Gymnasium, pytest, TensorBoard, and the existing `extension.parallelism` planner.

## Global Constraints

- Discriminator input is exactly `[batch, 24, 39]` flattened to `[batch, 936]`: 12 joint positions, 12 joint velocities, 3 local root positions, 6D local root rotation, 3 local root linear velocity, and 3 local root angular velocity.
- Discriminator never receives terrain, semantic labels, commands, actions, plan id, `plan_valid`, `amp_active`, or history ratio.
- All transition extraction, replan-boundary handling, ring writes, reconstruction, and local encoding operate on the environment batch with Torch/GPU tensors; no per-environment Python loops or CPU round-trips.
- Successful replans preserve history and use `inverse(B0) * B1` for the first transition of the new plan. Invalid/standstill/reset clears only affected rows; AMP resumes only after 24 valid frames.
- `V_base` trains on every transition. `V_amp` and AMP GAE/loss are masked to active rows. For every inactive row, `A_actor == A_base_norm` elementwise.
- The new experiment is the only path that creates AMP modules. Existing pure PPO and distillation configurations, classes, observation groups, launchers, and resume behavior remain unchanged.
- The target smoke is a real `1024`-environment Isaac Lab run for exactly `4` PPO iterations through the new shell launcher, exit code `0`, no OOM/NaN/Inf/Traceback, and iteration completion logs 1 through 4.

---

### Task 1: Define the batched AMP history and state encoder

**Files:**
- Create: `Go2Pvcnn/tracking/managers/parallelism_amp_manager.py`
- Modify: `Go2Pvcnn/tracking/managers/__init__.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_amp_history.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_amp_time_alignment.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_amp_state_encoding.py`

**Interfaces:**
- `ParallelismAmpManager(num_envs: int, device: torch.device | str, window_frames: int = 24, dt: float = 0.02)` allocates `[B, 23, ...]` transition rings, `write_index[B]`, `valid_count[B]`, and `standstill_latched[B]`.
- `push_transition(start_state: Tensor, target_state: Tensor, expert_start: Tensor, expert_target: Tensor, valid: Tensor) -> AmpStepPayload` accepts batched state tensors and returns `agent_window[B,24,39]`, `expert_window[B,24,39]`, `amp_active[B]`, and `history_ratio[B]`.
- `reset(env_mask: Tensor) -> None` clears only masked rows.
- `state_to_frame(state: Tensor, previous: Tensor | None, dt: float) -> Tensor` returns the 39-dimensional frame payload.

- [ ] **Step 1: Write the failing tests**

```python
def test_push_is_batched_and_wraps_after_23_transitions():
    manager = ParallelismAmpManager(3, "cpu")
    for step in range(50):
        state = torch.full((3, 39), float(step))
        payload = manager.push_transition(state, state + 1, state, state + 1, torch.ones(3, dtype=torch.bool))
    assert payload.agent_window.shape == (3, 24, 39)
    assert torch.equal(manager.valid_count, torch.full((3,), 24))
    assert torch.allclose(payload.agent_window[:, -1], torch.full((3, 39), 50.0))

def test_replan_boundary_uses_new_plan_start():
    manager = ParallelismAmpManager(1, "cpu")
    for _ in range(22):
        manager.push_transition(torch.zeros(1, 39), torch.ones(1, 39), torch.zeros(1, 39), torch.ones(1, 39), torch.ones(1, dtype=torch.bool))
    a23 = torch.full((1, 39), 9.0)
    b0 = torch.full((1, 39), 100.0)
    b1 = torch.full((1, 39), 101.0)
    payload = manager.push_transition(a23, b1, b0, b1, torch.ones(1, dtype=torch.bool))
    assert torch.allclose(payload.expert_window[:, -1], b1)
    assert torch.allclose(manager.last_expert_delta[:, 0], torch.ones(1, 39))

def test_encoder_is_anchor_invariant_and_has_936_features():
    manager = ParallelismAmpManager(2, "cpu")
    deltas = torch.randn(2, 23, 39)
    first = manager.reconstruct_and_encode(torch.zeros(2, 39), deltas)
    second = manager.reconstruct_and_encode(torch.full((2, 39), 10.0), deltas)
    assert first.shape == (2, 936)
    assert torch.isfinite(first).all()
    assert torch.allclose(first[..., :-3], second[..., :-3], atol=1e-5)
```

- [ ] **Step 2: Run the focused tests and verify the expected missing-interface failure**

Run: `/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/tracking/test_parallelism_amp_history.py Go2Pvcnn/tests/tracking/test_parallelism_amp_time_alignment.py Go2Pvcnn/tests/tracking/test_parallelism_amp_state_encoding.py -q`

Expected: FAIL because `parallelism_amp_manager.py` and `ParallelismAmpManager` do not yet exist.

- [ ] **Step 3: Implement fixed-shape vectorized history**

Implement batched `torch.gather`/`scatter` indexing for all rows. Store joint and root relative deltas separately, calculate root velocities with batched quaternion/SE(3) logarithms, reconstruct in reverse transition order from the terminal anchor, and localize each window using only its terminal heading. Use `torch.where(valid[:, None], new_delta, old_delta)` so invalid rows retain no new samples and return `amp_active=False` until `valid_count == 24`.

- [ ] **Step 4: Run the focused tests and a CUDA batch probe**

Run: `/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/tracking/test_parallelism_amp_history.py Go2Pvcnn/tests/tracking/test_parallelism_amp_time_alignment.py Go2Pvcnn/tests/tracking/test_parallelism_amp_state_encoding.py -q`

Expected: all focused tests pass; when CUDA is available, the same tests run once with `device="cuda:0"` and report finite `[B,24,39]` windows without host transfers.

- [ ] **Step 5: Commit the history component**

```bash
git add Go2Pvcnn/tracking/managers/parallelism_amp_manager.py Go2Pvcnn/tracking/managers/__init__.py Go2Pvcnn/tests/tracking/test_parallelism_amp_history.py Go2Pvcnn/tests/tracking/test_parallelism_amp_time_alignment.py Go2Pvcnn/tests/tracking/test_parallelism_amp_state_encoding.py
git commit -m "feat: add batched parallelism AMP history"
```

### Task 2: Add standstill gating and AMP discriminator/normalizer

**Files:**
- Create: `Go2Pvcnn/rsl_rl/rsl_rl/modules/amp_discriminator.py`
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/modules/__init__.py`
- Modify: `Go2Pvcnn/tracking/managers/parallelism_amp_manager.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_amp_standstill.py`
- Test: `Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_discriminator.py`

**Interfaces:**
- `AMPDiscriminator(input_dim=936, hidden_dims=(1024,512,256))` returns one logit per window.
- `AMPDiscriminator.reward(agent_window, active, normalizer) -> Tensor[B]` computes `active * (-log1p(-sigmoid(logit)).clamp_max(...)) * 2.0`.
- `AMPDiscriminator.update(expert, agent, active) -> dict[str, float]` applies BCE, input gradient penalty, logit L2, and an Adam step only to active rows.
- `AMPObservationNormalizer.update(windows, mask)` updates running statistics only where `mask` is true and `normalize(windows)` returns float32 windows.

- [ ] **Step 1: Write failing tests for invalid/recovery gates and discriminator update**

```python
def test_invalid_clears_only_selected_history_and_recovery_needs_24_frames():
    manager = ParallelismAmpManager(2, "cpu")
    valid = torch.ones(2, dtype=torch.bool)
    for _ in range(23):
        payload = manager.push_transition(torch.zeros(2,39), torch.ones(2,39), torch.zeros(2,39), torch.ones(2,39), valid)
    assert not payload.amp_active.any()
    payload = manager.push_transition(torch.zeros(2,39), torch.ones(2,39), torch.zeros(2,39), torch.ones(2,39), torch.tensor([False, True]))
    assert manager.valid_count[0] == 0 and manager.valid_count[1] == 24
    assert payload.amp_active.tolist() == [False, True]

def test_discriminator_step_changes_parameters_and_zeroes_inactive_reward():
    disc = AMPDiscriminator(input_dim=936)
    expert = torch.randn(8, 936)
    agent = torch.randn(8, 936)
    active = torch.tensor([1,1,1,1,0,0,0,0], dtype=torch.bool)
    before = [p.detach().clone() for p in disc.parameters()]
    metrics = disc.update(expert, agent, active)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, disc.parameters()))
    assert torch.equal(disc.reward(agent, active, disc.normalizer)[4:], torch.zeros(4))
```

- [ ] **Step 2: Run tests to verify they fail for the missing discriminator**

Run: `/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/tracking/test_parallelism_amp_standstill.py Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_discriminator.py -q`

Expected: FAIL because the discriminator and gate methods are not implemented.

- [ ] **Step 3: Implement the discriminator and masked statistics**

Use `nn.Sequential(Linear, LeakyReLU, ...)` and `BCEWithLogitsLoss`. Compute gradient penalty with `torch.autograd.grad` on active expert/agent inputs only; return zero metrics for an empty active batch. The manager must clear ring rows and count on invalid/reset, keep history on successful replan, and expose `history_ratio = valid_count / 24`.

- [ ] **Step 4: Run the tests and finite-gradient check**

Run: `/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/tracking/test_parallelism_amp_standstill.py Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_discriminator.py -q`

Expected: PASS with finite loss/gradient metrics and exact zero inactive reward.

- [ ] **Step 5: Commit discriminator/gating**

```bash
git add Go2Pvcnn/rsl_rl/rsl_rl/modules/amp_discriminator.py Go2Pvcnn/rsl_rl/rsl_rl/modules/__init__.py Go2Pvcnn/tracking/managers/parallelism_amp_manager.py Go2Pvcnn/tests/tracking/test_parallelism_amp_standstill.py Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_discriminator.py
git commit -m "feat: add masked AMP discriminator"
```

### Task 3: Add AMP actor-critic with isolated value head

**Files:**
- Create: `Go2Pvcnn/rsl_rl/rsl_rl/modules/amp_actor_critic_cnn.py`
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/modules/__init__.py`
- Test: `Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_dual_value.py`

**Interfaces:**
- `AmpActorCriticCNN` subclasses or composes `ActorCriticCNN` while preserving `actor`, `critic`, CNN, and `std` key names.
- `evaluate_amp(critic_observations, amp_active, history_ratio) -> Tensor[B,1]` feeds a detached base critic feature plus two context scalars into `amp_value_head`.
- `evaluate(critic_observations)` remains the exact base value API used by legacy PPO.

- [ ] **Step 1: Write the failing gradient-isolation and API compatibility tests**

```python
def test_amp_value_gradient_does_not_reach_base_network():
    model = AmpActorCriticCNN(8, 8, 4, use_cost_map=False, actor_hidden_dims=[16], critic_hidden_dims=[16])
    obs = torch.randn(6, 8)
    active = torch.ones(6)
    ratio = torch.ones(6)
    loss = model.evaluate_amp(obs, active, ratio).square().mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.amp_value_head.parameters())
    assert all(p.grad is None for p in model.critic.parameters())

def test_base_state_dict_keys_and_output_match_actor_critic_cnn():
    base = ActorCriticCNN(8, 8, 4, use_cost_map=False, actor_hidden_dims=[16], critic_hidden_dims=[16])
    amp = AmpActorCriticCNN(8, 8, 4, use_cost_map=False, actor_hidden_dims=[16], critic_hidden_dims=[16])
    amp.load_common_state_dict(base.state_dict())
    assert torch.allclose(base.act_inference(torch.ones(2,8)), amp.act_inference(torch.ones(2,8)))
```

- [ ] **Step 2: Run the tests and confirm the missing class failure**

Run: `/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_dual_value.py -q`

Expected: FAIL because `AmpActorCriticCNN` is not defined.

- [ ] **Step 3: Implement detached AMP context and zero-output initialization**

Reuse the existing CNN feature extraction and base `evaluate`. Build `amp_value_head` from `critic_feature_dim + 2`; initialize hidden layers orthogonally and final weight/bias to zero. Keep all actor inference methods independent of AMP history and planner state.

- [ ] **Step 4: Run the dual-value tests**

Run: `/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_dual_value.py -q`

Expected: PASS; base output is compatible and AMP-only backward produces no base-network gradients.

- [ ] **Step 5: Commit the model**

```bash
git add Go2Pvcnn/rsl_rl/rsl_rl/modules/amp_actor_critic_cnn.py Go2Pvcnn/rsl_rl/rsl_rl/modules/__init__.py Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_dual_value.py
git commit -m "feat: add isolated AMP value head"
```

### Task 4: Add masked storage and dual-channel PPO

**Files:**
- Create: `Go2Pvcnn/rsl_rl/rsl_rl/storage/parallelism_amp_storage.py`
- Create: `Go2Pvcnn/rsl_rl/rsl_rl/algorithms/parallelism_amp_ppo.py`
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/storage/__init__.py`
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/algorithms/__init__.py`
- Test: `Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_gae.py`
- Test: `Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_update.py`

**Interfaces:**
- `ParallelismAMPStorage` extends the rollout contract with `amp_rewards`, `amp_values`, `amp_active`, `history_ratio`, `amp_returns`, and `amp_advantages` tensors shaped `[T,B,1]`.
- `compute_returns(last_base_value, last_amp_value, gamma, lam)` computes base GAE normally and AMP GAE with `m_t*m_next` recursion cuts.
- `ParallelismAMPPPO.act(...)` records base and AMP values; `process_env_step(...)` consumes `infos["amp"]` and stores the vectorized payload.
- `ParallelismAMPPPO.update() -> dict[str, float]` masked-normalizes AMP advantages, combines `base_norm + lambda_amp * amp_norm`, updates actor/base value/AMP value, and updates D/replay.

- [ ] **Step 1: Write failing GAE, inactive invariant, and full-update tests**

```python
def test_amp_gae_cuts_at_inactive_boundary():
    storage = ParallelismAMPStorage(1, 8, [2], [2], [1], "cpu")
    storage.amp_rewards[:, 0, 0] = torch.tensor([1,1,0,0,0,1,1,1], dtype=torch.float)
    storage.amp_active[:, 0, 0] = torch.tensor([1,1,0,0,0,1,1,1], dtype=torch.float)
    storage.compute_returns(torch.zeros(1,1), torch.zeros(1,1), .99, .95)
    assert torch.equal(storage.amp_advantages[2:5], torch.zeros(3,1,1))
    assert storage.amp_advantages[1].abs() < 1e-6

def test_inactive_rows_keep_exact_base_actor_advantage():
    base = torch.tensor([[1.0], [2.0], [3.0]])
    amp = torch.tensor([[10.0], [20.0], [30.0]])
    mask = torch.tensor([[1.0], [0.0], [1.0]])
    combined = combine_advantages(base, amp, mask, 0.1)
    assert combined[1] == base[1]

def test_one_update_is_finite_and_clears_rollout_only():
    alg = make_small_parallelism_amp_ppo(num_envs=4, steps=4)
    collect_four_steps(alg)
    metrics = alg.update()
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert alg.storage.step == 0
    assert alg.amp_manager.valid_count.max() >= 0
```

- [ ] **Step 2: Run the tests and verify the missing storage/algorithm failure**

Run: `/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_gae.py Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_update.py -q`

Expected: FAIL because the new storage and algorithm classes do not exist.

- [ ] **Step 3: Implement storage and exact masked GAE**

Use `[T,B,1]` tensors. For each reverse time step calculate `delta_amp = m_t * (r_t + gamma * (1-done_t) * m_next * V_{t+1} - V_t)` and recursively multiply by `m_t*m_next`. Normalize AMP advantages only over active rows and set inactive rows to zero; do not normalize the combined actor advantage a second time.

- [ ] **Step 4: Implement PPO and discriminator update sequencing**

Reuse the existing PPO ratio/value clipping logic, add `amp_value_loss_coef`, and return named metrics for base value, AMP value, surrogate, discriminator, active ratio, and history ratio. Keep replay and `ParallelismAmpManager` outside `storage.clear()`. Use vectorized flattening for discriminator windows and skip the D update when no active rows exist.

- [ ] **Step 5: Run algorithm tests**

Run: `/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_gae.py Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_update.py -q`

Expected: PASS with exact inactive-row equality and finite optimizer updates.

- [ ] **Step 6: Commit storage and algorithm**

```bash
git add Go2Pvcnn/rsl_rl/rsl_rl/storage/parallelism_amp_storage.py Go2Pvcnn/rsl_rl/rsl_rl/algorithms/parallelism_amp_ppo.py Go2Pvcnn/rsl_rl/rsl_rl/storage/__init__.py Go2Pvcnn/rsl_rl/rsl_rl/algorithms/__init__.py Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_gae.py Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_update.py
git commit -m "feat: add dual-channel parallelism AMP PPO"
```

### Task 5: Wire the isolated AMP environment and training configuration

**Files:**
- Create: `Go2Pvcnn/tracking/parallelism_amp_cross_large_complex_env_cfg.py`
- Create: `Go2Pvcnn/tracking/parallelism_amp_env.py`
- Modify: `Go2Pvcnn/tracking/register_envs.py`
- Modify: `Go2Pvcnn/agent/train_cfg.py`
- Modify: `Go2Pvcnn/tracking/__init__.py`
- Modify: `Go2Pvcnn/scripts/train.py`
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_amp_isolation.py`

**Interfaces:**
- New experiment name: `parallelism_tracking_cross_large_complex_amp`.
- New Gym id: `Isaac-Go2-Parallelism-Tracking-Cross-Large-Complex-AMP-v0`.
- Environment info key: `infos["amp"]` containing batched `expert_window`, `agent_window`, `amp_active`, and `history_ratio`.
- Runner creates `ParallelismAMPPPO` only when `algorithm.class_name == "ParallelismAMPPPO"`; all other class-name branches remain unchanged.

- [ ] **Step 1: Write failing isolation/config tests**

```python
def test_amp_config_is_additive_and_old_configs_keep_classes():
    amp = get_train_cfg("parallelism_tracking_cross_large_complex_amp")
    pure = get_train_cfg("cross_large_complex_ppo")
    distill = get_train_cfg("parallelism_tracking_cross_large_complex_distillation")
    assert amp["algorithm"]["class_name"] == "ParallelismAMPPPO"
    assert amp["policy"]["class_name"] == "AmpActorCriticCNN"
    assert pure["algorithm"]["class_name"] == "PPO"
    assert distill["algorithm"]["class_name"] == "HybridDistillationPPO"

def test_amp_env_is_the_only_env_with_amp_payload():
    assert amp_env_cfg().experiment_name.endswith("_amp")
    assert not hasattr(cross_large_env_cfg(), "amp_manager")
```

- [ ] **Step 2: Run the isolation tests and verify the new experiment is rejected**

Run: `/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/tracking/test_parallelism_amp_isolation.py -q`

Expected: FAIL because the AMP experiment/config/runner branch is not registered.

- [ ] **Step 3: Add the additive config, registration, and env hook**

Subclass `ParallelismTrackingCrossLargeComplexEnvCfg`, set only the AMP experiment name and manager settings, and register the new Gym id. In the environment wrapper, call `prepare_step_reference()` before `env.step`, collect the post-physics state, call the batched manager once, and put only the AMP payload in `infos`. Do not add AMP observations to policy/critic groups.

- [ ] **Step 4: Add the runner branch and 4096 guard compatibility**

Add the new name to CLI choices and `EXPERIMENT_ENV_MAP`, import the new algorithm/module, and preserve the existing 4096 step guard. `OnPolicyRunner` must use the AMP-specific `act`, `process_env_step`, `compute_returns`, `update`, `save`, and `load` methods only for the AMP algorithm class. Keep `num_steps_per_env=40` for 1024 and allow the existing `>=4096 -> 24` guard.

- [ ] **Step 5: Run isolation and existing focused regressions**

Run: `/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/tracking/test_parallelism_amp_isolation.py Go2Pvcnn/tests/tracking/test_parallelism_cross_large_complex_env_cfg.py Go2Pvcnn/tests/tracking/test_parallelism_distillation_resume_static.py -q`

Expected: PASS; pure PPO/distillation assertions remain unchanged.

- [ ] **Step 6: Commit environment wiring**

```bash
git add Go2Pvcnn/tracking/parallelism_amp_cross_large_complex_env_cfg.py Go2Pvcnn/tracking/parallelism_amp_env.py Go2Pvcnn/tracking/register_envs.py Go2Pvcnn/agent/train_cfg.py Go2Pvcnn/tracking/__init__.py Go2Pvcnn/scripts/train.py Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py Go2Pvcnn/tests/tracking/test_parallelism_amp_isolation.py
git commit -m "feat: wire isolated parallelism AMP experiment"
```

### Task 6: Implement checkpoint compatibility and AMP launcher

**Files:**
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py`
- Modify: `Go2Pvcnn/scripts/train.py`
- Create: `Go2Pvcnn/scripts/train_parallelism_amp_cross_large_complex_headless.sh`
- Test: `Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_checkpoint.py`
- Test: `Go2Pvcnn/tests/tracking/test_parallelism_amp_launcher.py`

**Interfaces:**
- `OnPolicyRunner.load_amp(path, load_optimizer=True, keep_std=True)` returns `legacy_policy_warm_start` or `full_amp_resume` and raises `IncompleteAMPCheckpointError` for a half-AMP checkpoint.
- Legacy mode loads common Actor/base-Critic/CNN/std/normalizer keys, initializes zero-output `amp_value_head`, creates fresh D/AMP optimizer/replay, and sets AMP iteration to zero.
- Full mode restores both value channels, D, both optimizers, normalizers, and iteration; it intentionally does not restore simulator history or replay.
- Launcher requires one checkpoint path, resolves it with `realpath`, passes `--resume --load_checkpoint ... --keep_std`, and exits nonzero for missing input.

- [ ] **Step 1: Write failing checkpoint/launcher tests**

```python
def test_legacy_checkpoint_preserves_base_outputs_and_zeroes_amp_head(tmp_path):
    checkpoint = save_actor_critic_checkpoint(tmp_path / "pure.pt")
    runner = make_amp_runner()
    mode = runner.load_amp(checkpoint, keep_std=True)
    assert mode == "legacy_policy_warm_start"
    assert torch.allclose(runner.alg.actor_critic.evaluate(test_obs), baseline_value)
    assert torch.equal(runner.alg.actor_critic.evaluate_amp(test_obs, torch.ones(2), torch.ones(2)), torch.zeros(2,1))

def test_half_amp_checkpoint_is_rejected(tmp_path):
    checkpoint = save_partial_amp_checkpoint(tmp_path / "partial.pt")
    with pytest.raises(IncompleteAMPCheckpointError):
        make_amp_runner().load_amp(checkpoint)

def test_launcher_requires_checkpoint_and_forwards_resume_flags():
    text = Path("Go2Pvcnn/scripts/train_parallelism_amp_cross_large_complex_headless.sh").read_text()
    assert "--resume" in text and "--keep_std" in text and "--load_checkpoint" in text
```

- [ ] **Step 2: Run tests and verify the missing loader/launcher failure**

Run: `/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_checkpoint.py Go2Pvcnn/tests/tracking/test_parallelism_amp_launcher.py -q`

Expected: FAIL because `load_amp` and the launcher do not exist.

- [ ] **Step 3: Implement strict key detection and restore modes**

Detect `amp_value_head.` keys and `amp_discriminator_state_dict`. Load only common keys in legacy mode with shape checks; initialize the AMP head and new optimizer/replay. In full mode load all required states and retain `iter`. Reject exactly one of the two AMP state groups. Emit the required `checkpoint_mode=...` log lines.

- [ ] **Step 4: Add the executable launcher**

Use `set -euo pipefail`, validate the first positional checkpoint file, set Isaac Sim library paths, default `NUM_ENVS=1024` and `MAX_ITERATIONS=10000`, and execute `scripts/train.py` with the AMP experiment and resume flags. Ensure `NUM_ENVS=1024 MAX_ITERATIONS=4` reaches the real target smoke unchanged.

- [ ] **Step 5: Run checkpoint tests and shell syntax checks**

Run: `/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_checkpoint.py Go2Pvcnn/tests/tracking/test_parallelism_amp_launcher.py -q && bash -n Go2Pvcnn/scripts/train_parallelism_amp_cross_large_complex_headless.sh`

Expected: PASS and shell syntax exit code `0`.

- [ ] **Step 6: Commit checkpoint/launcher support**

```bash
git add Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py Go2Pvcnn/scripts/train.py Go2Pvcnn/scripts/train_parallelism_amp_cross_large_complex_headless.sh Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_checkpoint.py Go2Pvcnn/tests/tracking/test_parallelism_amp_launcher.py
git commit -m "feat: add AMP checkpoint resume launcher"
```

### Task 7: Add real probes, logging, and notes

**Files:**
- Create: `Go2Pvcnn/tests/tracking/parallelism_amp_runtime_probe.py`
- Create: `Go2Pvcnn/tests/tracking/parallelism_amp_training_smoke_probe.py`
- Modify: `Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py`
- Modify: `notes/todo.md`
- Create: `notes/todo/T304-parallelism-joint-amp.md`
- Modify: `notes/log/index.md`
- Create: `notes/log/2026-08-27-parallelism-joint-amp-implementation.md`

**Interfaces:**
- Runtime probe accepts `--num-envs`, `--steps`, and `--device`, runs the actual registered AMP env and asserts finite payloads, active recovery, and batch timing.
- Training smoke probe invokes the actual launcher, captures exit code/stdout/stderr, and verifies four completed iteration lines plus forbidden error tokens.

- [ ] **Step 1: Write failing real-probe assertions**

```python
def test_training_smoke_requires_four_completed_iterations():
    result = run_amp_launcher(num_envs=1024, max_iterations=4)
    assert result.returncode == 0
    for iteration in range(1, 5):
        assert f"Learning iteration {iteration}/4" in result.stdout
    lowered = result.stdout.lower() + result.stderr.lower()
    assert "traceback" not in lowered
    assert "cuda out of memory" not in lowered
    assert "nan" not in lowered and "inf" not in lowered
```

- [ ] **Step 2: Run the probe test before implementation**

Run: `/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/tracking/parallelism_amp_training_smoke_probe.py -q`

Expected: FAIL because the AMP launcher and runtime instrumentation are not present.

- [ ] **Step 3: Implement real runtime instrumentation**

Record `amp_active_ratio`, `amp_history_ratio_mean`, `amp_invalid_reset_count`, `amp_recovery_count`, base/AMP rewards, D metrics, both value losses, planner valid ratio, and batch transition/reconstruction milliseconds plus peak CUDA memory. Print stable `Learning iteration N/4` lines through the existing runner logger.

- [ ] **Step 4: Run layered verification**

Run the focused tensor/optimizer suite, then the single-environment probe:

```bash
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/rsl_rl/test_parallelism_amp_*.py Go2Pvcnn/tests/tracking/test_parallelism_amp_*.py -q
/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/tracking/parallelism_amp_runtime_probe.py --num-envs 1 --steps 50 --device cuda:0
```

Expected: all focused tests pass; the real environment probe exits `0`, observes 24-frame activation after recovery, and reports finite batch timings.

- [ ] **Step 5: Run the mandatory 1024-env four-iteration acceptance**

Run from the repository root with a real pure-PPO checkpoint:

```bash
PURE_PPO_CHECKPOINT="$(realpath "${PURE_PPO_CHECKPOINT:?set PURE_PPO_CHECKPOINT to a trained pure-PPO .pt file}")"
NUM_ENVS=1024 MAX_ITERATIONS=4 bash Go2Pvcnn/scripts/train_parallelism_amp_cross_large_complex_headless.sh "$PURE_PPO_CHECKPOINT"
```

Expected: shell exit code `0`; four completed iteration records; no `Traceback`, CUDA OOM, NaN, or Inf; TensorBoard/run logs contain active ratio, D accuracy, base/AMP rewards, both value losses, and batch transition/reconstruction timing. This is a real Isaac Lab run and cannot be replaced by a static test or a shorter run.

- [ ] **Step 6: Run regressions and update notes/log**

Run: `/share/home/tm884089579940000/a915071960/lhy/miniconda3/envs/env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/tracking/test_cross_large_complex_ppo_static.py Go2Pvcnn/tests/tracking/test_parallelism_distillation_*.py -q`

Record exact commands, exit codes, iteration lines, active ratios, losses, timing, peak memory, and git refs in `notes/log/2026-08-27-parallelism-joint-amp-implementation.md`; update the T304 branch page, dashboard, and log index with only verified results.

- [ ] **Step 7: Commit probes and evidence**

```bash
git add Go2Pvcnn/tests/tracking/parallelism_amp_runtime_probe.py Go2Pvcnn/tests/tracking/parallelism_amp_training_smoke_probe.py Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py notes/todo.md notes/todo/T304-parallelism-joint-amp.md notes/log/index.md notes/log/2026-08-27-parallelism-joint-amp-implementation.md
git commit -m "test: verify parallelism AMP real training smoke"
```

## Self-Review Checklist

- [ ] Every design requirement has a task: 24-frame state windows, replan continuity, vectorized transition increments, standstill recovery, dual values, strict checkpoint modes, isolated old experiments, and real 1024/4-iteration exit-code acceptance.
- [ ] No task relies on a static-only test for functional behavior.
- [ ] All public names used by later tasks are defined in earlier task interfaces.
- [ ] No per-environment Python loop is introduced in the AMP manager or 1024 smoke path.
- [ ] Before claiming completion, run the exact mandatory 1024-env command and inspect the full exit/log output.
