# AI Extension Planner Runtime

## Navigation

- doc role: AI runtime reference for batched planner
- paired human doc: [../human/human-10-extension-planner-runtime.md](../human/human-10-extension-planner-runtime.md)
- previous: [ai-09-extension-planner-mapping.md](ai-09-extension-planner-mapping.md)
- next: [ai-11-extension-trajectory-reward.md](ai-11-extension-trajectory-reward.md)
- master index: [../index.md](../index.md)
- raw index: [../../raw/kinematic_footsteps/notes/index.md](../../raw/kinematic_footsteps/notes/index.md)

## Runtime Pattern

Updated 2026-04-15: planner-owned reference cache with single-shot, per-env masked replanning.

## Main Flow

- Isaac Lab robot state
- high-resolution `height_scanner`
- `extension.batched_planner.manager.BatchedTrajectoryManager.refresh_from_env(env)`
- optional single planner call per refresh: `extension.batched_planner.trajectory.batched_generate_trajectory`
- `extension.convention.planner_result_to_reference_cache` (canonical `ReferenceTrajectoryCache`)
- full-shaped cache `(num_envs, horizon, ...)` is consumed by reward/viewer

## Runtime Graph

```mermaid
graph LR
    env["trajectory env cfg\n../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py"]
    convention["convention.py\n../../Go2Pvcnn/extension/convention.py"]
    traj["trajectory.py\n../../Go2Pvcnn/extension/batched_planner/trajectory.py"]
    manager["manager.py\n../../Go2Pvcnn/extension/batched_planner/manager.py"]
    cache["planner-owned ReferenceTrajectoryCache\n(env.unwrapped._trajectory_manager.refresh_from_env)\n+ runtime mirror: env.unwrapped._trajectory_reference_cache"]
    reward["rewards_reference.py\n../../Go2Pvcnn/extension/mdp/rewards_reference.py"]

    env --> convention
    env --> traj
    convention --> traj
    traj --> manager
    manager --> cache
    cache --> reward
```

## Interface Layers

Runtime is not a direct `Isaac -> trajectory.py` call.

- Isaac Lab provides batched robot state, commands, and scanner terrain
- `extension.convention.py` normalizes conventions at the boundary
- `manager.py` owns the cache lifecycle, computes the per-env replan mask, and advances phases
- `trajectory.py` is the planner entrypoint used only when a replan is needed
- `rewards_reference.py` calls `ensure_reference_cache(env)` which requires `env.unwrapped._trajectory_manager`

## Counters

- `_step_counter`: global, not reset per env
- `_phase_counter`: per-env current reference frame
- `_last_episode_length_buf`: last refresh episode steps
- `_last_replan_episode_length_buf`: per-env last (attempted) replan episode steps (for interval bookkeeping)
- `_pending_reset_mask`: per-env pending reset flags

## Replan Trigger

Per-env masked replanning (decoupled semantics):

- reset / pending reset: only those env rows are replanned
- command delta: only env rows with changed commands are replanned
- interval elapsed: per env if `episode_length_buf - last_replan_episode_length_buf >= reference_replan_interval_steps`
- cache/horizon mismatch: falls back to a full replan for safety

## Cache Contract

Reward/viewer consume `ReferenceTrajectoryCache`, not the raw planner result, and they expect a full-shaped cache:

- even partial replans keep the cache layout `(num_envs, horizon, ...)`
- manager writes replanned rows back into the full cache via masked writes

## MPC Semantic Internal Planning Command

Updated 2026-05-24 for `extension/batch_mpc_planner` T302h:

- The external runtime command passed into `plan_segment(...)` remains unchanged.
- Inside `plan_segment(...)`, semantic corridor policy can derive an internal `planning_command`.
- Low-small obstacles keep the requested command and are handled by crossing / foot-contact / stepcap losses.
- High-small and large obstacles reduce forward velocity and add lateral velocity toward the freer side.
- The internal `planning_command` is used for both nominal construction and optimizer/loss terms. A nominal-only shaping pass failed large-forward continuity because tracking still pulled the plan toward the original command.

Evidence: [../log/2026-05-24-1948-t302h-production-v10-implementation.md](../log/2026-05-24-1948-t302h-production-v10-implementation.md)

## MPC Replan Phase Default

Updated 2026-05-24 for `extension/batch_mpc_planner` T302h:

- `MpcRuntimeCfg.randomize_replan_phase` now defaults to `False`.
- Real IsaacLab multi-cycle probes showed that randomized replan phase can switch diagonal swing/stance timing between consecutive plan segments and create a frame-0 foot discontinuity.
- The production config keeps an explicit task override: `mpc_randomize_replan_phase=True`.
- In the large-forward semantic obstacle probe, deterministic phase improved multi-cycle task failures from `1/6` to `0/6` and continuity failures from `1/6` to `0/6`.

Evidence: [../log/2026-05-24-2109-t302h-deterministic-replan-phase.md](../log/2026-05-24-2109-t302h-deterministic-replan-phase.md)

### Standstill Persistence

If replanning fails for a subset and a cache already exists, the manager fills those rows with a standstill (time-constant) trajectory and keeps them until that env hits its next trigger. Interval bookkeeping is updated so failures do not retrigger every step.

### Verbose Planner Diagnostics

With `verbose_planner` (or `planner_instrumentation`) enabled, the manager prints a compact timing summary every `verbose_planner_interval_steps`.

### Viewer Direct Playback

Viewer supports `--planner-playback-mode direct` to drive robot pose/joints directly from the planner result/reference cache (as opposed to the default `physics` mode).

### Together Viewer Handoff Root Z

Updated 2026-04-28 for `Go2Pvcnn/extension/viz/go2_foostep_planner.py`:

- The together viewer preserves single-segment raw/together planner parity.
- Segment chaining no longer uses the previous segment's terminal root z directly as the next initial base height.
- `_together_state_from_reference_result()` calls `_together_handoff_root_pos()` and reconstructs handoff root z from current contact-foot support height plus the segment's initial support clearance.
- This prevents repeated viewer replans from accumulating segment root-z bias into visible lift-off.
- For hold-like full-contact segments with no planar/yaw motion, the viewer now bypasses that clearance reconstruction and keeps the terminal root z directly, preventing zero-command recovery from replaying every segment.

Evidence: [../log/2026-04-28-1007-viewer-together-root-z-ratchet.md](../log/2026-04-28-1007-viewer-together-root-z-ratchet.md).

### Together Zero-Command Rehome

Updated 2026-04-28 for `Go2Pvcnn/extension/batched_together_planner/parameterization.py`:

- Zero-command hold keeps root `xy` and yaw.
- Feet move toward current-yaw root-frame nominal slots, with z from terrain support.
- Root `z` recovers toward support height plus `hip_height`.
- Root roll/pitch recover toward the support plane; flat terrain recovers toward `0/0`.
- The support-plane normal is computed with a vectorized four-foot midpoint cross product, not `torch.linalg.svd`.
- The training-path guardrail now forbids `torch.linalg.svd` and `torch.svd` under `extension/batched_together_planner`.

Evidence: [../log/2026-04-28-1132-together-zero-command-rehome.md](../log/2026-04-28-1132-together-zero-command-rehome.md).

## CPU vs Pure-GPU Runtime Role

- raw CPU path remains the semantic parity baseline
- batched pure-GPU path is the intended runtime path for Isaac Lab training
- legacy EventTerm / raw bridge runtime should be treated as historical, not default

## Related deep-dive

- swing/stance semantics and IK time complexity (single env, code anchors): [ai-13-batched-planner-swing-stance-ik-complexity.md](ai-13-batched-planner-swing-stance-ik-complexity.md)

## Joint MPC RTI Runtime (2026-07-15)

- Every refresh injects measured `x0`; no segment phase playback.
- First call initializes solver state; steady-state calls use `JointMpcCudaGraphRunner` with static state/command storage and stable field-cache addresses.
- RayCaster callbacks queue updated env ids only; `latest_field()` builds/publishes rows outside the Warp/Isaac sensor callback. A newly captured graph replays once before its first result is returned.
- Production profile: compiled kernels, diagonal state Riccati, alphas `(1.0,0.25)`, named breakdown off.
- Acceptance: 1024 env, H16, 1000 calls `2896.49ms`, nonfinite `0`, peak `282.58MiB`.
- Real Isaac 1-env and 16-env three-step probes pass with field version `2`, finite references, and `x0_error_max=0`; real 1024 first-time initialization/timing remains open.
