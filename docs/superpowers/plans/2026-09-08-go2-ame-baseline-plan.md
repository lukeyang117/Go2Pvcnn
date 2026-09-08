# Go2 AME Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with review checkpoints.

**Goal:** Add an isolated AME PPO baseline for Go2 that uses the existing `cross_large_complex_ppo` task and AME's CNN/BatchNorm/MultiheadAttention architecture with a six-channel semantic terrain map, without changing existing PPO, AMP, distillation, teacher, or Parallelism behavior.

**Architecture:** A new observation helper converts the semantic ray scanner into `[x_local, y_local, z_local, terrain_one_hot, small_one_hot, large_one_hot]` and area/max pools it to `6x16x16`. A new environment config subclasses the planner-free cross-large-complex PPO config and replaces only observation groups; a standalone VecEnv adapter and training entrypoint feed those groups to a new `ActorCriticAME`. The existing local PPO implementation and rollout storage are reused, while class registration is local to the AME entrypoint so existing imports remain unchanged.

**Tech Stack:** Python 3.10, Isaac Lab/Isaac Sim, Gymnasium, PyTorch, local `rsl_rl` PPO, pytest, TensorBoard.

## Global Constraints

- Use `num_steps_per_env=40`, `num_learning_epochs=5`, `num_mini_batches=4`, `learning_rate=1e-3`, `clip_param=0.2`, `gamma=0.99`, `lam=0.95`, `value_loss_coef=1.0`, `entropy_coef=0.01`, `max_grad_norm=1.0`, `schedule=adaptive`, and `desired_kl=0.01`.
- Use exactly six map channels: local `x,y,z` plus one-hot terrain/small-obstacle/large-obstacle semantic channels, output shape `[N,6,16,16]`.
- Do not add observation noise; retain finite-value sanitization and deterministic clipping for local coordinates.
- Keep `raw/` ignored and do not modify existing PPO/AMP/distillation/teacher source files.
- AME checkpoints and TensorBoard logs live under `logs/rsl_rl/cross_large_complex_ame/`.
- The real smoke test must run Isaac Lab with 1024 environments for 4 PPO iterations and exit normally with a checkpoint.

---

### Task 1: Six-channel AME observation helper

**Files:**
- Create: `Go2Pvcnn/ame_baseline/ame_observations.py`
- Create: `Go2Pvcnn/ame_baseline/__init__.py`
- Test: `Go2Pvcnn/tests/test_ame_observations.py`

**Interfaces:**
- Consumes: a duck-typed Isaac Lab environment with `scene.sensors[sensor_cfg.name].data.ray_hits_w`, `pos_w`, `quat_w`, and `semantic_map`.
- Produces: `downsampled_ame_scan(env, sensor_cfg, target_size=16) -> torch.Tensor` with shape `[N,6,target_size,target_size]`.

- [x] **Step 1: Write the failing tests**

```python
def test_downsampled_ame_scan_has_xyz_and_one_hot_channels():
    out = downsampled_ame_scan(_FakeEnv(), _FakeCfg(), target_size=2)
    assert out.shape == (1, 6, 2, 2)
    assert torch.equal(out[:, 3:].sum(dim=1), torch.ones(1, 2, 2))
    assert torch.allclose(out[:, :3].mean(), torch.tensor(0.0))

def test_downsampled_ame_scan_rejects_non_square_scan():
    with pytest.raises(ValueError, match="perfect square"):
        downsampled_ame_scan(_FakeEnv(num_rays=3), _FakeCfg(), target_size=2)
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd Go2Pvcnn && pytest -q tests/test_ame_observations.py`

Expected: FAIL because `extension.mdp.ame_observations` does not exist.

- [x] **Step 3: Implement the helper**

Convert world hits to sensor-local coordinates using `quat_apply_inverse`, sanitize with `torch.nan_to_num`, clamp xyz to `[-5,5]`, pool xyz with `adaptive_avg_pool2d`, pool semantic ids with `adaptive_max_pool2d`, and create one-hot channels for ids `0,1,2`. Return `torch.cat((xyz, one_hot), dim=1)`.

- [x] **Step 4: Export and run tests**

Run: `cd Go2Pvcnn && pytest -q tests/test_ame_observations.py`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add Go2Pvcnn/ame_baseline/ame_observations.py Go2Pvcnn/ame_baseline/__init__.py Go2Pvcnn/tests/test_ame_observations.py
git commit -m "feat: add AME six-channel semantic observation"
```

### Task 2: Isolated AME environment configuration

**Files:**
- Create: `Go2Pvcnn/ame_baseline/ame_env_cfg.py`
- Test: `Go2Pvcnn/tests/tracking/test_ame_cross_large_complex_env_cfg.py`

**Interfaces:**
- Consumes: `CrossLargeComplexPpoEnvCfg`, existing Go2 scene/rewards/terrain/curriculum.
- Produces: `AmeCrossLargeComplexEnvCfg` and `AmeCrossLargeComplexEnvCfg_PLAY`, with observation groups named `policy_elevation_semantic_map`, `policy_state`, `critic_elevation_semantic_map`, and `critic_state`.

- [x] **Step 1: Write the failing static/config tests**

```python
def test_ame_config_uses_six_channel_helper_without_noise():
    source = Path("ame_baseline/ame_env_cfg.py").read_text()
    assert "downsampled_ame_scan" in source
    assert "noise=None" in source
    assert "num_envs=1024" in source

def test_ame_observation_groups_have_matching_policy_and_critic_maps():
    cfg = AmeCrossLargeComplexEnvCfg()
    assert cfg.observations.policy_elevation_semantic_map.elevation_semantic_map.params["target_size"] == 16
    assert cfg.observations.critic_elevation_semantic_map.elevation_semantic_map.params["target_size"] == 16
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd Go2Pvcnn && pytest -q tests/tracking/test_ame_cross_large_complex_env_cfg.py`

Expected: FAIL because the AME config file does not exist.

- [x] **Step 3: Implement the config**

Subclass `CrossLargeComplexPpoEnvCfg`; define no-noise policy/critic map terms using `downsampled_ame_scan`; define no-noise state terms matching the existing PPO proprioception; set `experiment_name="cross_large_complex_ame"`, preserve `num_envs=1024`, and clear corruption in the play subclass.

- [x] **Step 4: Run tests to verify they pass**

Run: `cd Go2Pvcnn && pytest -q tests/tracking/test_ame_cross_large_complex_env_cfg.py`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add Go2Pvcnn/ame_baseline/ame_env_cfg.py Go2Pvcnn/tests/tracking/test_ame_cross_large_complex_env_cfg.py
git commit -m "feat: add isolated AME cross-large environment"
```

### Task 3: AME actor-critic network

**Files:**
- Create: `Go2Pvcnn/ame_baseline/actor_critic_ame.py`
- Test: `Go2Pvcnn/tests/test_actor_critic_ame.py`

**Interfaces:**
- Consumes: flattened observations with map first (`6*16*16`) followed by proprioception; PPO calls `act`, `evaluate`, `get_actions_log_prob`, `action_mean`, `action_std`, `entropy`, and `reset`.
- Produces: 12-D stochastic actions and scalar critic values; architecture `Conv2d(6,16,5,stride=2,pad=2) -> ReLU -> BatchNorm2d -> Conv2d(16,64,3,pad=1) -> ReLU -> BatchNorm2d -> MultiheadAttention(64,16) -> [512,256,128] ELU MLP` for actor and critic.

- [x] **Step 1: Write the failing tests**

```python
def test_ame_forward_shapes_and_finite_distribution():
    net = ActorCriticAME(1581, 1584, 12, map_channels=6, map_size=16)
    obs = torch.randn(8, 1581)
    actions = net.act(obs)
    assert actions.shape == (8, 12)
    assert net.evaluate(torch.randn(8, 1584)).shape == (8, 1)
    assert torch.isfinite(net.action_std).all()

def test_ame_has_expected_encoder_channels_and_attention():
    net = ActorCriticAME(1581, 1584, 12, map_channels=6, map_size=16)
    assert net.map_cnn[0].in_channels == 6
    assert net.map_cnn[0].out_channels == 16
    assert net.mha.embed_dim == 64 and net.mha.num_heads == 16
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd Go2Pvcnn && pytest -q tests/test_actor_critic_ame.py`

Expected: FAIL because `ActorCriticAME` does not exist.

- [x] **Step 3: Implement the minimal network**

Split the first `map_channels*map_size*map_size` values, reshape to NCHW, encode with the AME CNN, flatten spatial tokens, embed proprioception to 64 dimensions, use proprioception as the attention query, concatenate attention output with raw proprioception, and apply independent actor/critic MLPs. Store a 12-D learnable `std` parameter initialized to exactly `1.0`, matching AME; PPO's post-update lower clamp keeps it positive.

- [x] **Step 4: Run tests to verify they pass**

Run: `cd Go2Pvcnn && pytest -q tests/test_actor_critic_ame.py`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add Go2Pvcnn/ame_baseline/actor_critic_ame.py Go2Pvcnn/tests/test_actor_critic_ame.py
git commit -m "feat: add AME CNN attention actor critic"
```

### Task 4: AME VecEnv adapter and PPO configuration

**Files:**
- Create: `Go2Pvcnn/ame_baseline/ame_env_wrapper.py`
- Create: `Go2Pvcnn/ame_baseline/ame_train_cfg.py`
- Create: `Go2Pvcnn/ame_baseline/ame_runner.py`
- Test: `Go2Pvcnn/tests/test_ame_vec_env_and_cfg.py`

**Interfaces:**
- Consumes: Isaac `ManagerBasedRLEnv` observation dict from Task 2.
- Produces: `AmeRslRlEnvWrapper` implementing local `VecEnv` plus `get_ame_train_cfg()` with the exact PPO values in Global Constraints.

- [x] **Step 1: Write failing tests**

```python
def test_ame_wrapper_flattens_map_before_state():
    wrapper = AmeRslRlEnvWrapper.__new__(AmeRslRlEnvWrapper)
    obs, extras = wrapper._format_observations({"policy_elevation_semantic_map": torch.zeros(2,6,16,16), "policy_state": torch.ones(2,45), "critic_elevation_semantic_map": torch.zeros(2,6,16,16), "critic_state": torch.ones(2,48)})
    assert obs.shape == (2, 6*16*16+45)
    assert extras["observations"]["critic"].shape == (2, 6*16*16+48)

def test_ame_cfg_matches_ppo_baseline():
    cfg = get_ame_train_cfg()
    assert cfg["num_steps_per_env"] == 40
    assert cfg["algorithm"]["num_mini_batches"] == 4
    assert cfg["algorithm"]["desired_kl"] == 0.01
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd Go2Pvcnn && pytest -q tests/test_ame_vec_env_and_cfg.py`

Expected: FAIL because the wrapper/config module does not exist.

- [x] **Step 3: Implement wrapper/config**

Flatten map then state, combine terminated/truncated into long `dones`, expose `time_outs`, and leave all observation tensors unchanged. Return a fresh AME PPO config with `ActorCriticAME`, six channels, 16 map size, and the exact baseline hyperparameters.

- [x] **Step 4: Run tests to verify they pass**

Run: `cd Go2Pvcnn && pytest -q tests/test_ame_vec_env_and_cfg.py`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add Go2Pvcnn/ame_baseline/ame_env_wrapper.py Go2Pvcnn/ame_baseline/ame_train_cfg.py Go2Pvcnn/ame_baseline/ame_runner.py Go2Pvcnn/tests/test_ame_vec_env_and_cfg.py
git commit -m "feat: add AME PPO wrapper and config"
```

### Task 5: Standalone AME train/play entrypoints

**Files:**
- Create: `Go2Pvcnn/scripts/train_cross_large_complex_ame.py`
- Create: `Go2Pvcnn/scripts/play_cross_large_complex_ame.py`
- Create: `Go2Pvcnn/scripts/train_cross_large_complex_ame_headless.sh`
- Test: `Go2Pvcnn/tests/test_ame_entrypoints_static.py`

**Interfaces:**
- Consumes: `--num_envs`, `--max_iterations`, `--headless`, optional `--resume`/`--checkpoint`.
- Produces: isolated TensorBoard/checkpoints under `logs/rsl_rl/cross_large_complex_ame/<timestamp>/<git-hash>/`, and a deterministic play entrypoint loading AME checkpoints.

- [x] **Step 1: Write the failing static tests**

```python
def test_ame_train_entrypoint_isolated_from_existing_experiments():
    source = Path("scripts/train_cross_large_complex_ame.py").read_text()
    assert "cross_large_complex_ame" in source
    assert "ActorCriticAME" in source
    assert "OnPolicyRunner" in source

def test_ame_headless_script_defaults_to_1024_envs_and_four_iterations_for_smoke_override():
    source = Path("scripts/train_cross_large_complex_ame_headless.sh").read_text()
    assert "train_cross_large_complex_ame.py" in source
    assert "--headless" in source
```

- [x] **Step 2: Run tests to verify they fail**

Run: `cd Go2Pvcnn && pytest -q tests/test_ame_entrypoints_static.py`

Expected: FAIL because the entrypoint files do not exist.

- [x] **Step 3: Implement train/play/shell**

Launch `AppLauncher`, register a private Gym id, construct Task 2 config and Task 4 wrapper, inject `ActorCriticAME` into the local runner module namespace, support optional checkpoint loading, and close environment/simulation in `finally`. The shell sets `PYTHONPATH`, accepts `NUM_ENVS`, `MAX_ITERATIONS`, and `CHECKPOINT`, and never touches other experiment directories.

- [x] **Step 4: Run static tests**

Run: `cd Go2Pvcnn && pytest -q tests/test_ame_entrypoints_static.py`

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add Go2Pvcnn/scripts/train_cross_large_complex_ame.py Go2Pvcnn/scripts/play_cross_large_complex_ame.py Go2Pvcnn/scripts/train_cross_large_complex_ame_headless.sh Go2Pvcnn/tests/test_ame_entrypoints_static.py
git commit -m "feat: add standalone AME train and play entrypoints"
```

### Task 6: Real Isaac Lab smoke test and regression verification

**Files:**
- Create: `Go2Pvcnn/tests/test_ame_smoke_command.py`
- Modify: `Go2Pvcnn/docs/superpowers/specs/2026-09-08-go2-ame-baseline-design.html` only if implementation details need correction.

- [x] **Step 1: Run all AME unit/static tests**

Run: `cd Go2Pvcnn && pytest -q tests/test_ame_observations.py tests/tracking/test_ame_cross_large_complex_env_cfg.py tests/test_actor_critic_ame.py tests/test_ame_vec_env_and_cfg.py tests/test_ame_entrypoints_static.py`

Expected: all tests PASS.

- [x] **Step 2: Run the real 1024-env, 4-iteration Isaac smoke**

Run:

```bash
cd Go2Pvcnn && NUM_ENVS=1024 MAX_ITERATIONS=4 bash scripts/train_cross_large_complex_ame_headless.sh
```

Expected: Isaac Lab starts headlessly, prints `Starting Training - cross_large_complex_ame`, reaches four learning iterations (`0..3`), exits with code 0, and creates `logs/rsl_rl/cross_large_complex_ame/<timestamp>/<git-hash>/model_3.pt` plus TensorBoard event data.

- [x] **Step 3: Verify isolation and checkpoint contents**

Run: `cd Go2Pvcnn && find logs/rsl_rl/cross_large_complex_ame -name 'model_3.pt' -o -name 'events.out.tfevents.*'`

Expected: both files exist and no existing experiment directory is modified.

- [x] **Step 4: Run the focused regression suite**

Run: `cd Go2Pvcnn && pytest -q tests/test_train_script_static.py tests/tracking/test_cross_large_complex_ppo_static.py`

Expected: existing PPO static tests remain PASS.

- [x] **Step 5: Commit verification metadata**

```bash
git add Go2Pvcnn/tests/test_ame_smoke_command.py
git commit -m "test: verify AME Isaac Lab smoke command"
```
