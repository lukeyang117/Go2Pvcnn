# T302s Env-Level Collision Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 flat-small 课程改成每个 env 在 episode 结束时独立判断升降级，并把 curriculum TensorBoard 输出精简到 terrain difficulty。

**Architecture:** 保留现有 `terrain_levels_vel_semantic_plane_gate` 入口，避免改 cfg 的 curriculum term 名称；内部去掉 flat 全局 semantic gate，把 sticky small collision / base contact / bad orientation 直接映射到当前 reset env 的 `move_up` / `move_down`。`SemanticObstacleCurriculumState` 只保留 episode sticky 状态和最近一次调试值，不再驱动全局 gate；TensorBoard curriculum return 只暴露 `mean_terrain_level`。

**Tech Stack:** IsaacLab ManagerBased RL env、PyTorch tensor masks、`semantic_contact_small.data.force_matrix_w`、pytest、`env_isaacsim` real smoke。

---

## Source Design

- [../../docs/superpowers/specs/2026-06-11-flat-small-env-level-collision-curriculum-design.html](../../docs/superpowers/specs/2026-06-11-flat-small-env-level-collision-curriculum-design.html)
- Triggering TensorBoard readout: [../log/2026-06-11-1955-t302q-flat-small-1831-tensorboard-readout.md](../log/2026-06-11-1955-t302q-flat-small-1831-tensorboard-readout.md)

## Conflicting Old Todo Cleanup

The following old assumptions are closed for T302q/T302r and must not be reimplemented:

- Global `semantic_gate_pass` controls flat move-up after consecutive success windows.
- `min_completed_episodes` blocks all flat env upgrades when too few flat episodes reset in one curriculum call.
- `completed_flat_episodes`, `successful_full_no_collision_episodes`, `semantic_success_rate`, `consecutive_success_count`, `semantic_gate_pass`, `flat_move_up_count`, `non_flat_move_up_count`, and `plane_collision_rate` are public TensorBoard curriculum metrics.
- `small_collision` forces immediate downgrade in the first implementation.

## File Structure

- Modify `Go2Pvcnn/extension/semantic_curriculum.py`
  - Add env-level episode result helper.
  - Keep old count/layout helpers.
  - Keep compatibility fields only as internal debug state if needed.
- Modify `Go2Pvcnn/go2_pvcnn/mdp/curriculums.py`
  - Replace global flat gate with env-level `flat_move_up_i = terrain_move_up_i AND episode_success_i`.
  - Add downgrade for base contact / bad orientation.
  - Return only `mean_terrain_level`.
- Modify `Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py`
  - Add `clearance_scale=1000.0` to flat-small reward params.
  - Remove no-longer-used gate threshold overrides from flat-small config if they are only for global gate.
- Modify tests:
  - `Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py`
  - `Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py`
  - `Go2Pvcnn/tests/test_batch_mpc_backend.py`

## Task 1: RED Tests For Env-Level Curriculum

- [x] Replace old global-gate tests with tests proving:
  - a single successful flat env upgrades immediately even if only one flat env resets
  - a flat env with sticky small collision does not upgrade
  - small collision does not force downgrade
  - base contact / bad orientation force downgrade
  - non-flat terrain still follows IsaacLab distance curriculum
  - curriculum return keys are exactly `{"mean_terrain_level"}`

Run:

```bash
pytest Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py -q
```

Observed before implementation: `8 failed, 167 passed, 1 warning`; failures were old extra return keys and missing `clearance_scale`.

## Task 2: GREEN Env-Level Episode Logic

- [x] Implement helper logic that computes per-env episode success from reset ids:

```text
episode_success_i =
  time_out_i
  AND NOT episode_had_small_collision_i
  AND NOT base_contact_i
  AND NOT bad_orientation_i
```

- [x] In `terrain_levels_vel_semantic_plane_gate`, use:

```text
flat_move_up_i = terrain_move_up_i AND episode_success_i
flat_move_down_i = terrain_move_down_i OR base_contact_i OR bad_orientation_i
```

- [x] Clear sticky flags only for env ids passed to the curriculum call.
- [x] Return only `mean_terrain_level`.

Run:

```bash
pytest Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py -q
```

Observed after implementation: focused selected tests pass as part of the `184 passed, 1 warning` verification.

## Task 3: Reward Scale Wiring

- [x] Add `clearance_scale` to `semantic_body_part_clearance_reward`.
- [x] Apply scale to the raw negative reward before return while preserving clipping behavior.
- [x] Wire flat-small config with `clearance_scale=1000.0`.
- [x] Add static tests for the config param and pure reward scale behavior.

Run:

```bash
pytest Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Expected: reward scale test and cfg static tests pass.

## Task 4: Focused Verification

- [x] Run focused curriculum/reward/backend tests:

```bash
pytest \
  Go2Pvcnn/tests/test_semantic_obstacle_curriculum.py \
  Go2Pvcnn/tests/test_semantic_obstacle_curriculum_term.py \
  Go2Pvcnn/tests/test_semantic_body_part_clearance_reward.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py \
  -q
```

- [x] Run pycompile for touched production files:

```bash
python -m py_compile \
  Go2Pvcnn/extension/semantic_curriculum.py \
  Go2Pvcnn/go2_pvcnn/mdp/curriculums.py \
  Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py \
  Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py
```

## Task 5: Real IsaacLab Smoke

- [x] Run a small `env_isaacsim` smoke after local tests:

```bash
CUDA_VISIBLE_DEVICES=<card> /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py \
  --experiment teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance \
  --headless \
  --num_envs 8 \
  --max_iterations 1 \
  --device cuda:0
```

Observed: process exits `0`; Curriculum Manager has only `terrain_levels`; Reward Manager includes `semantic_body_part_clearance`.

## Task 6: Notes And Logs

- [x] Create a verification log under `notes/log/`.
- [x] Update `notes/log/index.md`.
- [x] Update `notes/todo.md`.
- [x] Update this branch page with actual command results and next step.

## Related Logs

- [../log/2026-06-11-2156-flat-small-env-level-collision-curriculum-html-design.md](../log/2026-06-11-2156-flat-small-env-level-collision-curriculum-html-design.md)
- [../log/2026-06-11-2211-t302s-env-level-collision-curriculum-implementation.md](../log/2026-06-11-2211-t302s-env-level-collision-curriculum-implementation.md)
- [../log/2026-06-12-1039-t302s-flat-small-2215-tensorboard-readout.md](../log/2026-06-12-1039-t302s-flat-small-2215-tensorboard-readout.md)
- [../log/2026-06-12-1054-t302s-flat-small-fixed-command-ranges.md](../log/2026-06-12-1054-t302s-flat-small-fixed-command-ranges.md)
- [../log/2026-06-12-1355-t302s-flat-small-1053-tensorboard-readout.md](../log/2026-06-12-1355-t302s-flat-small-1053-tensorboard-readout.md)
- [../log/2026-06-12-1548-t302s-flat-small-1053-tensorboard-readout.md](../log/2026-06-12-1548-t302s-flat-small-1053-tensorboard-readout.md)
- [../log/2026-06-12-1642-t302s-flat-small-1053-tensorboard-readout.md](../log/2026-06-12-1642-t302s-flat-small-1053-tensorboard-readout.md)
- [../log/2026-06-12-1722-t302s-model23600-first-layer-eval.md](../log/2026-06-12-1722-t302s-model23600-first-layer-eval.md)
- [../log/2026-06-12-1740-t302s-model23600-crossing-1000step-probe.md](../log/2026-06-12-1740-t302s-model23600-crossing-1000step-probe.md)
- [../log/2026-06-12-1815-t302s-model23600-controlled-crossing-eval.md](../log/2026-06-12-1815-t302s-model23600-controlled-crossing-eval.md)
- [../log/2026-06-12-1833-flat-small-foot-over-training-signal.md](../log/2026-06-12-1833-flat-small-foot-over-training-signal.md)
- [../log/2026-06-13-1505-t302s-model28900-controlled-crossing-eval.md](../log/2026-06-13-1505-t302s-model28900-controlled-crossing-eval.md)
- [../log/2026-06-13-1510-t302s-flat-small-1120-tensorboard-readout.md](../log/2026-06-13-1510-t302s-flat-small-1120-tensorboard-readout.md)
- [../log/2026-06-13-1651-train-single-env-livestream-follow-camera.md](../log/2026-06-13-1651-train-single-env-livestream-follow-camera.md)
- [../log/2026-06-13-1735-play-keyboard-control-terrain-selection.md](../log/2026-06-13-1735-play-keyboard-control-terrain-selection.md)
- [../log/2026-06-13-1756-play-pynput-install-headless-smoke.md](../log/2026-06-13-1756-play-pynput-install-headless-smoke.md)
- [../log/2026-06-13-1810-human-12-play-keyboard-command-update.md](../log/2026-06-13-1810-human-12-play-keyboard-command-update.md)
- [../log/2026-06-13-1839-play-terminal-keyboard-backend.md](../log/2026-06-13-1839-play-terminal-keyboard-backend.md)
- [../log/2026-06-13-2022-play-disable-timeout-refresh.md](../log/2026-06-13-2022-play-disable-timeout-refresh.md)
- [../log/2026-06-13-2207-flat-small-semantic-course-column-fix.md](../log/2026-06-13-2207-flat-small-semantic-course-column-fix.md)
- [../log/2026-06-23-model14700-flat-small-eval.md](../log/2026-06-23-model14700-flat-small-eval.md)
- [../log/2026-06-23-crossing-reset-diagnostics-and-stability-tuning.md](../log/2026-06-23-crossing-reset-diagnostics-and-stability-tuning.md)
- [../log/2026-06-23-flat-small-1210-tensorboard-reset-readout.md](../log/2026-06-23-flat-small-1210-tensorboard-reset-readout.md)
- [../log/2026-06-23-train-keep-std-resume-option.md](../log/2026-06-23-train-keep-std-resume-option.md)
- [../log/2026-06-23-human12-keep-std-command-update.md](../log/2026-06-23-human12-keep-std-command-update.md)
- [../log/2026-06-24-flat-small-214649-tensorboard-collapse-readout.md](../log/2026-06-24-flat-small-214649-tensorboard-collapse-readout.md)
- [../log/2026-06-24-flat-small-214649-checkpoint-eval-and-reward-correlation.md](../log/2026-06-24-flat-small-214649-checkpoint-eval-and-reward-correlation.md)
- [../log/2026-06-11-1955-t302q-flat-small-1831-tensorboard-readout.md](../log/2026-06-11-1955-t302q-flat-small-1831-tensorboard-readout.md)

## Git Refs

- Last Feature Commit: `da46138`
- Last Verified Commit: `da46138`
- Current Work Ref: working tree
- Key Files:
  - [../../Go2Pvcnn/extension/semantic_curriculum.py](../../Go2Pvcnn/extension/semantic_curriculum.py)
  - [../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py](../../Go2Pvcnn/go2_pvcnn/mdp/curriculums.py)
  - [../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py](../../Go2Pvcnn/extension/mdp/semantic_body_part_clearance.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_mpc_semantic_env_cfg.py)
  - [../../Go2Pvcnn/scripts/mpc_policy_eval.py](../../Go2Pvcnn/scripts/mpc_policy_eval.py)
  - [../../Go2Pvcnn/scripts/train.py](../../Go2Pvcnn/scripts/train.py)
  - [../../Go2Pvcnn/scripts/play.py](../../Go2Pvcnn/scripts/play.py)
  - [../../Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py](../../Go2Pvcnn/rsl_rl/rsl_rl/runners/on_policy_runner.py)

## Next Step

- Start a short warm-start run from the latest useful checkpoint and watch whether `Episode_Reward/semantic_foot_over_clearance` becomes nonzero; then re-run controlled crossing eval and use `foot_over_count` as the first acceptance signal.

## Post-Implementation Finding

- Run `2026-06-11_22-15-56` confirms old curriculum TensorBoard noise is gone and clearance reward is visible, but terrain progression still fails.
- Saved config shows `lin_vel_x=(-0.1,0.1)` and `lin_vel_y=(-0.1,0.1)` while `lin_vel_cmd_levels` is disabled.
- Terrain move-up still requires `distance > 4m` because terrain tile size is `8m`.
- This creates a new immediate fix: flat-small should use fixed useful command ranges once velocity curriculum is disabled.
- Fixed locally: flat-small training now uses `lin_vel_x=(0.6,1.0)`, `lin_vel_y=(-0.2,0.2)`, `ang_vel_z=(-0.3,0.3)`. Real 8-env smoke generated `2026-06-12_10-53-23` and saved cfg confirms the new ranges.
- Current run `2026-06-12_10-53-23` confirms the curriculum now opens: `mean_terrain_level` reaches `7.475` and last-100 mean is `5.97`. The remaining question is not curriculum opening but whether semantic contact decreases after more training at high levels.
- Later readout at event step `22868` still supports continuing briefly: `mean_terrain_level` last-100 is `5.805`, contact last-100 improves to `-0.000684`, clearance last-100 improves to `-0.002979`, and episode length last-100 remains `980.84`; watch mean reward drift and stop for retuning if episode length falls below about `950`.
- Readout at event step `23389` shows stability recovered but semantic trend is noisy: terrain last-100 `5.871`, episode length last-100 `991.84`, reward last-100 `28.15`, but contact last-100 worsened to `-0.000858` and clearance last-100 worsened to `-0.003496`. Continue only to about `model_24000`, then evaluate behavior or retune.
- Model `23600` first-layer eval is mixed: strict flat-small training scene short sample has `0/8` collided envs over `200` steps, but existing dense small-collision eval has `8/16` collided envs over `300` steps. Since neither computes true path-obstacle foot-over success, reliable low-small overpass is not proven.
- Model `23600` 1000-step crossing probe on 16 training-scene envs has zero true small contact but also zero overpass successes: only one env had a path-obstacle opportunity, the root crossed it, but foot-over stayed false and scanner-derived clearance was negative. This rejects a success claim but also shows the random training scene has too few path-obstacle opportunities for a stable success rate.
- Formal controlled crossing eval for `model_23600.pt` removes that ambiguity: `15/16` envs had a path-small opportunity and `14` crossed with root, but `foot_over_count=0`, true small contact hit `11/16`, and overpass success is `0/15`. The current checkpoint has not learned clean low-small overpass.
- Flat-small training signal is retuned locally: low-row obstacle layout now uses smaller center safety holes, contact penalty is stronger for small objects, and `semantic_foot_over_clearance` gives positive reward for feet clearing low-small path cells. Real 8-env smoke exits `0`, but one iteration is too short to see nonzero foot-over reward.
- Latest `2026-06-12_19-05-27/model_28900.pt` controlled crossing eval still rejects the overpass claim: `15/16` opportunities, `14` root crossed, `foot_over_count=0`, true small contact `7/16`, overpass success `0/15`. Contact improved versus `model_23600.pt`, but the intended foot-over behavior did not appear.
- Model `2026-06-17_12-01-10/model_14700.pt` controlled crossing eval uses the current map-contact fallback because flat-small no longer mounts the old `semantic_contact_small` sensor. It has lower small-contact count (`3/16`) and the first recorded foot-over envs (`2/16`), but root completion drops to `7/16` and clean overpass remains `0/16`. Treat this as partial improvement, not solved behavior. See [../log/2026-06-23-model14700-flat-small-eval.md](../log/2026-06-23-model14700-flat-small-eval.md).
- Crossing reset diagnostics and stability retuning are implemented locally after user visualization showed "tries to step over, then falls/resets": flat-small now lowers foot-over reward weight to `0.12` and strengthens stability penalties (`flat_orientation_l2=-3.5`, `base_angular_velocity=-0.12`, `feet_slide=-0.18`) without changing MPC planner loss. A short 4-env reset-diagnostic smoke emitted `bad_orientation=2` and verified the new weights are active. See [../log/2026-06-23-crossing-reset-diagnostics-and-stability-tuning.md](../log/2026-06-23-crossing-reset-diagnostics-and-stability-tuning.md).
- TensorBoard readout for `2026-06-17_12-01-10` argues against globally relaxing reset as the first fix: saved `bad_orientation.limit_angle=1.1`, last100 `bad_orientation=2.183`, `base_contact=0.001`, `semantic_foot_over_clearance=0`, and terrain remains low (`1.485`). The better first move is stability shaping plus reset-stage diagnostics, only considering a small temporary angle relaxation if future diagnostics show recoverable post-foot-over resets. See [../log/2026-06-23-flat-small-1210-tensorboard-reset-readout.md](../log/2026-06-23-flat-small-1210-tensorboard-reset-readout.md).
- Resume training from `2026-06-17_12-01-10/model_14700.pt` can now explicitly preserve learned policy action noise with `--keep_std`. Default resume still drops checkpoint `std` and resets to the current initialized value, matching prior behavior; `--keep_std` keeps the checkpoint `std` in `OnPolicyRunner.load()`. Focused RED/GREEN tests pass (`6 passed`), pycompile exits `0`, and diff check exits `0`. See [../log/2026-06-23-train-keep-std-resume-option.md](../log/2026-06-23-train-keep-std-resume-option.md).
- Human command guide [../human/human-12-batched-planner-train-viewer-commands.md](../human/human-12-batched-planner-train-viewer-commands.md) now carries the current `model_14700.pt` continuation command with `--keep_std` and `--mpc_num_envs 1024`, plus parameter explanations for resume std and MPC env count. See [../log/2026-06-23-human12-keep-std-command-update.md](../log/2026-06-23-human12-keep-std-command-update.md).
- Run `2026-06-23_21-46-49` should not continue from `model_17300.pt`: TensorBoard shows a collapse into short bad-orientation resets. Last100 episode length is `12.55`, terrain level is `0`, bad_orientation is `161.48`, reference reward is effectively gone, and bucketed metrics show the collapse starts around `16800-16900`. Evaluate pre-collapse checkpoints (`14800`, `14900`, `16700`) before choosing any restart. See [../log/2026-06-24-flat-small-214649-tensorboard-collapse-readout.md](../log/2026-06-24-flat-small-214649-tensorboard-collapse-readout.md).
- Checkpoint sweep for `2026-06-23_21-46-49` evaluated `14700`, `14800`, `14900`, `16700`, and `17300` in controlled crossing. `17300` is not a successful foot-over policy: it has `foot_over=5/16`, but `small_contact=15/16`, `bad_orientation=16/16`, and avg bad reset step `10.5`. TensorBoard reward correlation supports this: `semantic_foot_over_clearance` has near-zero correlation with bad reset (`-0.004`), while std/action/stability metrics correlate strongly. See [../log/2026-06-24-flat-small-214649-checkpoint-eval-and-reward-correlation.md](../log/2026-06-24-flat-small-214649-checkpoint-eval-and-reward-correlation.md).
- User-requested `bad_orientation=None` ablation was applied and tested, then reverted locally to finite `bad_orientation.limit_angle=1.1` after the new run showed base-contact collapse. See [../log/2026-06-24-flat-small-disable-bad-orientation-reset-experiment.md](../log/2026-06-24-flat-small-disable-bad-orientation-reset-experiment.md) and [../log/2026-06-24-flat-small-094941-tensorboard-and-checkpoint-eval.md](../log/2026-06-24-flat-small-094941-tensorboard-and-checkpoint-eval.md).
- Flat-small PPO exploration is now reduced without affecting the base semantic experiment: `get_train_cfg("teacher_elevation_trajectory_mpc_semantic_flat_small_avoidance")` overrides `algorithm.entropy_coef` to `0.002`, while base semantic stays `0.01`. See [../log/2026-06-24-flat-small-lower-entropy-coef.md](../log/2026-06-24-flat-small-lower-entropy-coef.md).
- New ablation run `2026-06-24_09-49-41` should not continue from `model_17100.pt`: TensorBoard last100 episode length is `17.43`, base_contact `117.8`, terrain level `0`, and reference rewards are zero. Controlled crossing shows more foot-over events (`9/16`) but no clean success (`0/16`) and base-contact reset in all envs (`16/16`). `model_15600.pt` is the only stable candidate from this run, with tracking mean `0.0775m` and base-contact reset `3/16`, but it still has overpass success `0/16`. See [../log/2026-06-24-flat-small-094941-tensorboard-and-checkpoint-eval.md](../log/2026-06-24-flat-small-094941-tensorboard-and-checkpoint-eval.md).
- Run `2026-06-13_11-20-40` with larger foot-over scale shows the scale is active but the trigger is still sparse: `semantic_foot_over_clearance` nonzero `9/2167`, max `0.1018`, last-100 mean `0`; contact remains dense with last-100 `-0.00210`. Larger reward magnitude alone did not create a reliable learning signal.
- Train single-env livestream now follows env0: `train.py` preserves the requested `--livestream` value before `AppLauncher` can mutate args, then installs a camera update wrapper only for `rank==0`, `num_envs==1`, and livestream `1/2`. Real `env_isaacsim` smoke with the user’s resume checkpoint exits `0` and prints `Single-env livestream follow camera enabled`.
- Play keyboard visualization now supports terminal-thread hold-to-move body commands and deterministic env0 terrain selection: `--keyboard-control`, `W/S/A/D/Q/E`, `+/-`, `--terrain-row`, and `--terrain-col`. The flat-small PLAY cfg disables training curriculum because it does not mount semantic contact sensors. The old `pynput` route is removed; livestream control now reads from the SSH terminal when stdin is a TTY. Static `32 passed`, pycompile exits `0`, and real `--keyboard-control` smoke exits `0` with expected non-TTY warning in the automated tool runner. [../human/human-12-batched-planner-train-viewer-commands.md](../human/human-12-batched-planner-train-viewer-commands.md) is updated with the terminal-input contract.
- PLAY timeout refresh is disabled for visualization: both semantic PLAY and flat-small PLAY set `terminations.time_out=None`, while training cfg keeps timeout for curriculum success. Real flat-small PLAY smoke exits `0` and Termination Manager contains only `base_contact` and `bad_orientation`.
- Flat-small semantic objects appearing in only one visual column was a static semantic-course column-name bug, separate from the earlier curriculum flat-mask bug. Flat-small has `sub_terrains={"flat": ...}` but `num_cols=20`; before the fix, `terrain_name_for_col()` returned `"flat"` only for column 0 and `None` for columns 1-19, so columns 1-19 used zero `non_plane_counts`. The fix repeats the single terrain name for all columns. Real train probe now reports row 0 and row 9 both cover all 20 columns with first/last counts `[8,8]` and `[80,80]`.
