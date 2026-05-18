# MPC Semantic RL Training Config Design

## Purpose

Create an independent RL task that trains with the MPC backend and a high-resolution semantic scanner, without changing the existing `teacher_elevation_trajectory` / `together` path or reopening T302 tuning.

The new task teaches the policy from MPC foot trajectories only, gives the CNN a downsampled elevation+semantic map, and adds a swing/leg collision reward that measures the simulated robot against the high-resolution scanner.

## Scope

In scope:

- Add a new task config file under `Go2Pvcnn/go2_pvcnn/tasks/`.
- Add a new train/play experiment name and Gym ids.
- Add observation helpers for `2 x 16 x 16` high-resolution semantic scanner downsampling.
- Add a reward helper that uses IsaacLab robot body buffers, not planner FK, to penalize swing/leg collision risk.
- Add tests for config wiring, semantic priority pooling, foot-only imitation reward wiring, dirty-subset MPC replanning, and 4096 real IsaacLab collect-data timing.
- Add notes/log entries under the T302 tree.
- Use `notes/todo/T302g-mpc-semantic-rl-training-config.md` as the execution todo source of truth. Do not create a separate new implementation plan document for the next implementation pass.

Out of scope:

- Changing default `teacher_elevation_trajectory` behavior.
- Changing T302 MPC loss defaults.
- Replacing the MPC optimizer or changing planner semantics to gain speed.
- Adding MPC reference trajectories to observations.

## New Branch Memory

New child branch page:

- `notes/todo/T302g-mpc-semantic-rl-training-config.md`

Parent relationship:

- child-of `T302`
- related-to `T300e`

T302 parent remains the authority for collision-safety implementation and strict metric baselines. The new child records RL integration and performance evidence.

Implementation tracking:

- the T302g branch page owns the detailed child todo tree
- each child node records files, acceptance, and verification evidence
- the existing initial plan document is historical context only after this spec hardening
- no new plan file should be created for the next implementation pass unless the user explicitly asks for one

## Task And File Layout

Create:

- `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
  - Owns the new train/play env configs.
  - Imports existing trajectory config pieces only where reuse is explicit.

Modify:

- `Go2Pvcnn/go2_pvcnn/tasks/register_envs.py`
  - Registers new train/play Gym ids.
- `Go2Pvcnn/scripts/train.py`
  - Adds `teacher_elevation_trajectory_mpc_semantic` experiment mapping.
- `Go2Pvcnn/scripts/play.py`
  - Adds `teacher_elevation_trajectory_mpc_semantic` experiment mapping and allows `mpc` in `--planner-backend`.
- `Go2Pvcnn/extension/mdp/observations.py`
  - Adds reusable high-resolution semantic scanner downsampling helpers.
- `Go2Pvcnn/extension/mdp/rewards_reference.py`
  - Adds `swing_leg_collision_reward`.
- `Go2Pvcnn/extension/mdp/__init__.py`
  - Exports new trajectory-guided helpers.
- `Go2Pvcnn/tests/test_batch_mpc_backend.py`
  - Adds focused local/unit tests for config and helper contracts.
- `Go2Pvcnn/tests/test_mpc_runtime_headless.py`
  - Adds real 4096 IsaacLab timing/counter acceptance that instantiates the new task, not the old viewer/play task.

## New Env Config

The new config lives in its own file and does not enlarge `teacher_elevation_trajectory_env_cfg.py`.

Classes:

- `TeacherElevationTrajectoryMpcSemanticSceneCfg`
- `TeacherElevationTrajectoryMpcSemanticObservationsCfg`
- `TeacherElevationTrajectoryMpcSemanticRewardsCfg`
- `TeacherElevationTrajectoryMpcSemanticEnvCfg`
- `TeacherElevationTrajectoryMpcSemanticEnvCfg_PLAY`

Gym ids:

- `Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0`
- `Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-Play-v0`

CLI experiment:

- `teacher_elevation_trajectory_mpc_semantic`

The existing `teacher_elevation_trajectory` experiment continues to use its current defaults.

## Scanner Contract

The new scene uses a single high-resolution semantic scanner named `semantic_height_scanner`.

Configuration:

- `SemanticGridRayCasterCfg`
- `prim_path="{ENV_REGEX_NS}/Robot/base"`
- yaw-attached grid
- `resolution=0.01`
- `size=[1.5, 1.5]`
- `mesh_prim_paths=["/World/ground", SEMANTIC_COURSE_SMALL_ROOT, SEMANTIC_COURSE_LARGE_ROOT]`
- semantic ids:
  - terrain: `0`
  - small obstacle: `1`
  - large obstacle: `2`

The new task sets:

```python
reference_height_scanner_name = "semantic_height_scanner"
```

MPC and `swing_leg_collision_reward` consume the high-resolution scanner directly. The policy/critic only consume the downsampled map.

Coordinate query contract:

- reward-side high-resolution map queries must not map world `xy` into a fixed unrotated range
- helper code must transform world body/link points into the scanner grid frame with the scanner pose and yaw
- for the current yaw-attached scanner, the transform must use `scanner.data.pos_w`, `scanner.data.quat_w`, and the scanner pattern `size/resolution`
- the index order must match `SemanticGridRayCaster` map layout and its `grid_pattern` flatten order
- tests must include a translated and yawed scanner case so a fixed-world-range implementation fails
- if a future scanner helper exposes an official world-point query API, the reward should call that helper instead of duplicating coordinate math

## CNN Observation Contract

Observation shape:

```text
(num_envs, 2, 16, 16)
```

This matches the current `teacher_elevation_trajectory_env_cfg.py` CNN map size.

Channels:

- channel 0: elevation/height, downsampled from high-resolution `elevation_map` with area pooling
- channel 1: semantic id, downsampled with priority pooling

Priority pooling:

- if a target cell contains any large obstacle id, output `2`
- else if it contains any small obstacle id, output `1`
- else output `0`

Semantic ids must not be averaged.

## MPC Runtime Contract

The new task uses:

```python
planner_owned_reference_cache = True
use_batched_reference_trajectory = True
planner_backend = "mpc"
reference_height_scanner_name = "semantic_height_scanner"
```

Both train and play config classes must set all four fields explicitly. They must not rely on inherited defaults for `planner_owned_reference_cache` or `use_batched_reference_trajectory`.

Every MPC replan must read current IsaacLab robot buffers:

- `robot.data.root_pos_w`
- `robot.data.root_quat_w`
- `robot.data.joint_pos`
- `robot.data.body_pos_w`

It must not roll forward from the previous MPC reference/cache as the source of current robot state.

For 4096 training, the new semantic task uses a global synchronized MPC replan mode, not the earlier dirty-subset refresh mode:

- all environments share the same global MPC replan tick
- the default target is full 4096-env MPC planning at that tick
- `mpc_parallel_plan_batch_size` controls the maximum number of environments passed to one `plan_segment(...)` call
- the default `mpc_parallel_plan_batch_size` is `4096`
- if a 4096 call OOMs or proves unstable on the current GPU, only this resource knob should be reduced first, for example to `256`, `128`, or `64`
- reducing `mpc_parallel_plan_batch_size` is an execution-detail fallback; it must preserve one logical global replan generation
- each global generation snapshots all current IsaacLab robot states, commands, and scanner terrain before planning
- chunked execution, when used, consumes that frozen snapshot rather than rereading live env state between chunks
- result rows scatter back into a full-shaped cache
- only environments with a successful new MPC plan receive MPC imitation reward for that generation

Reset and command-change handling inside a generation:

- if an environment resets after the global MPC plan, its `reference_foot_pos` reward is disabled until the next global replan tick
- if an environment's velocity command changes after the global MPC plan, its `reference_foot_pos` reward is disabled until the next global replan tick
- those environments are not replanned immediately in the middle of the interval
- non-MPC rewards, including `swing_leg_collision_reward`, remain active
- the disabled imitation reward must be implemented as a mask, not by tracking stale MPC foot targets

Optional MPC reward sampling:

- the default is to enable `reference_foot_pos` reward for every successfully planned environment
- a later resource/learning experiment may add reward sampling across the successful set
- reward sampling must be explicit and must not be confused with planning batch size; sampling rewards does not by itself reduce MPC planning time

Default training parameters:

```python
reference_trajectory_horizon = 50
reference_replan_interval_steps = 50
mpc_replan_mode = "global_sync"
mpc_parallel_plan_batch_size = 4096
mpc_max_stale_steps = 100
mpc_optimize_steps = 24
mpc_diagnostics_emit_runtime_counters = False
```

Diagnostics may be enabled in tests.

Performance tuning constraint:

- default `mpc_optimize_steps` remains `24`
- timing work may optimize tensor paths, batching, caching, scanner queries, global snapshot handling, and `mpc_parallel_plan_batch_size`
- timing work must not lower MPC collision/semantic loss quality, disable T302 safety losses, shorten the policy-facing foot reference horizon, or reduce optimizer quality just to pass the `10s` gate
- any proposed MPC-quality change must first pass the T302 strict metric gate and preserve policy-facing foot trajectory shape/horizon

## Reward Contract

The new task keeps the base locomotion reward terms for velocity/stability/energy.

MPC imitation rewards:

- enabled: `reference_foot_pos`
- disabled or not registered:
  - `reference_root_pose`
  - `reference_joint_pos`
  - `reference_contact`
  - `reference_touchdown`

This means the policy learns only MPC foot trajectories, not planner root pose, planner joint pose, or planner contact labels.

## Swing/Leg Collision Reward

Add `swing_leg_collision_reward` as an RL reward term.

Source of current robot geometry:

- read IsaacLab simulated body/link positions from `robot.data.body_pos_w`
- use link/body samples for `.*_thigh`, `.*_calf`, and `.*_foot`; do not reconstruct those samples from joint angles
- prefer body-name filtering for `.*_thigh`, `.*_calf`, and `.*_foot`
- do not solve FK from joint angles inside the reward
- do not use MPC cache/reference as current state

Terrain source:

- `semantic_height_scanner.data.elevation_map`
- `semantic_height_scanner.data.semantic_map`
- scanner pose/yaw for local-to-world query alignment as needed

Swing/stance signal:

- classify swing/stance from current IsaacLab contact state, not from planner contact imitation or MPC cache
- use `contact_forces` with `body_names=".*_foot"` as the default contact source
- a leg is stance when its current foot contact force magnitude is above `swing_collision_contact_force_threshold`
- a leg is swing otherwise
- map each `thigh/calf/foot` body sample to its leg by body name and broadcast that leg's swing/stance weight to all samples from the leg
- if the current task lacks the configured contact sensor or foot body mapping, raise a clear `RuntimeError`

Behavior:

- penalize leg sample points below or too close to the height field
- penalize semantic obstacle overlap
- apply stronger weight to swing legs than stance legs
- large obstacle penalty is stronger than small obstacle penalty

Suggested exposed params:

```python
swing_collision_height_margin_m = 0.04
swing_collision_semantic_margin_m = 0.02
swing_collision_contact_force_threshold = 1.0
swing_collision_swing_weight = 1.0
swing_collision_stance_weight = 0.25
swing_collision_small_weight = 1.0
swing_collision_large_weight = 5.0
swing_collision_reward_scale = -1.0
```

The reward must be vectorized and must not introduce per-env Python loops.

Reward tests must prove:

- same body/sample collision is penalized more strongly in swing than stance
- large obstacle overlap is penalized more strongly than small obstacle overlap
- translated/yawed scanner map queries hit the expected cell
- the reward source does not call planner FK/IK helpers and does not read MPC reference cache as current state

## 4096 Collect-Data Performance Gate

Add a real IsaacLab headless test for the new task at `num_envs=4096`.

Acceptance:

- the test instantiates `Isaac-Teacher-Elevation-Trajectory-Mpc-Semantic-Go2-v0` or directly instantiates `TeacherElevationTrajectoryMpcSemanticEnvCfg`
- the test asserts the active config class, `scene.num_envs == 4096`, `reference_height_scanner_name == "semantic_height_scanner"`, `planner_backend == "mpc"`, and the new observation/reward terms are active
- the test must not rely on the current `viewer_runtime_diagnostics` fixture unless that fixture is extended with explicit `task_id` / `env_cfg_cls` selection and assertions proving the new task is active
- collect-data / rollout collection timing is measured through the RSL-RL rollout collection path when available; raw `env.step(...)` timing alone is not sufficient for final acceptance
- each measured collect-data pass must be under `10s`.
- runtime counters show global synchronized replanning:
  - `replan_mode == "global_sync"`
  - `planned_env_count == 4096` on full successful generations
  - `parallel_plan_batch_size == mpc_parallel_plan_batch_size`
  - `plan_chunk_count` records how many `plan_segment(...)` chunks were used
  - `plan_success_count` and `reference_reward_enabled_count` are recorded
  - reset/command-changed environments inside the interval reduce `reference_reward_enabled_count` until the next global generation
  - `planner_ms` and `cache_ms` are recorded when diagnostics are enabled.

If timing exceeds `10s`, implementation is not complete. First try full `mpc_parallel_plan_batch_size=4096`; if it OOMs or is unstable, reduce that parameter while preserving the global-generation semantics. Optimize tensor paths, snapshot reuse, reward masking, terrain building, and avoid CPU/GPU sync. Do not reduce MPC effect quality to pass timing.

## T302 Non-Regression Gate

The T302 baseline is:

- `notes/log/2026-05-17-0804-t302-strict-collision-metric-tuning.md`

Required non-regression:

- T302 backend tests continue passing.
- T302 strict collision metric matrix remains passing.
- root-bottom, swing-foot, knee, and shank collision ratios stay `0.0` in the strict matrix.
- low-small crossing remains accepted.
- high-small/large obstacle deweighting/avoidance remains accepted.
- stance semantic count stays `0`.

The new RL reward must not change MPC planner loss defaults or weaken T302 metrics.

## Error Handling

- If `semantic_height_scanner` is missing or lacks `elevation_map` / `semantic_map`, observation and reward helpers raise `RuntimeError` with the scanner name.
- If semantic downsampling receives non-square maps or invalid target size, it raises `ValueError`.
- If reward body-name selection finds no leg bodies, it raises `RuntimeError`.
- If `planner_backend` is not `mpc` in the new config, tests fail.

## Verification Summary

Local/unit:

- semantic priority pooling
- dual map observation shape
- config defaults
- train/play parser experiment choices
- Gym registration with `gym.spec(...)` for both new Gym ids
- regression assertion that existing `teacher_elevation_trajectory` mapping/default planner path remains unchanged
- foot-only reference reward wiring
- swing reward reads current IsaacLab body/contact state, not planner FK/cache
- swing reward applies stronger swing than stance weighting
- scanner map query handles translation and yaw
- manager global-sync replan counters
- reset/command change disables MPC imitation reward until the next global replan
- `mpc_parallel_plan_batch_size` chunks planning without changing global generation semantics

Real IsaacLab:

- new task smoke with small env count
- 4096 RSL-RL collect-data timing under `10s` on the new MPC semantic task
- T302 strict/headless non-regression evidence
