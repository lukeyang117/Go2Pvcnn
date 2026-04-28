# T100 Batched Together Planner GPU Migration

## Current State

- The native IsaacLab GPU migration is implemented under `Go2Pvcnn/extension/batched_together_planner/`.
- Subagents implemented code/tests, while the main agent reviewed flow and verification.
- The new backend aligns to the updated raw planner under `raw/kinematic_footsteps/scripts/go2fp/batch_planner/` for flat P0 parity while exposing a legacy-compatible training and reward interface.
- Default backend should be `planner_backend = "together"`, with `planner_backend = "legacy"` available as a rollback path.
- Raw timing contract is strict for the new backend: `35` frames, `dt = 0.02`, `0.7s` horizon, `event_cap = 2`.
- Replanning semantics are fixed-shape full-batch: every planner call receives all envs `[N, ...]`; per-env replacement is decided by tensor masks and `torch.where`.
- Hard revision from review: the together training path must not branch on GPU masks before a planner call. No `if torch.any(replan_mask)`, no `nonzero`, no dynamic sub-batch planner call.
- User cadence decision: together should update trajectory only when velocity command changes, reset occurs, or the `0.7s` replan interval is reached.
- Together training cache must be GPU resident. The existing CPU-canonical cache converter remains legacy/reference-only.
- Reward frame selection must come from the manager phase/current-reference snapshot, not from `episode_length_buf % horizon`.
- Viewer may keep CPU logging/camera/visualization exceptions. Training path may not.
- Viewer backend fix verified: `--planner-backend together` now calls the native together `plan_segment` path with the scripted/teleop command, while `legacy` keeps `batched_generate_trajectory`.
- Final conda verification in `/home/lhy/anaconda3/envs/env_isaaclab` passed: full together suite `50 passed`, CUDA smoke at `N=1024/4096`, py_compile, diff check, together viewer headless smoke, and subagent train/play smoke for both together and legacy.
- Continued testing passed: together 1-iteration training at `32` and `128` envs, legacy rollback 1-iteration training at `16` envs, real Isaac cadence/full-N checks, and updated regression tests for the new factory/default-together contract.
- Human command guide [human-12-batched-planner-train-viewer-commands.md](../human/human-12-batched-planner-train-viewer-commands.md) now matches the current train/play/viewer CLI: `env_isaaclab` Python, explicit `--planner-backend together`, legacy rollback, and viewer `task/35/0.02` contract.
- New viewer issue fixed to verification: launching the together viewer could make the displayed Go2 walk for a while and then lift into the air. Root cause was repeated viewer-style replan handoff/root-z accumulation, not a CLI startup problem.
- New viewer stop-command issue under T110 is fixed to verification: together zero-command hold now rehomes feet, root height, and root roll/pitch while keeping root `xy` and yaw.
- Follow-up viewer handoff fix under T110: hold-like zero-command segments now hand off terminal root z directly, so recovery plays once and then stays settled instead of repeating every `0.7s`.

## Open Children

- T110: together zero-command rehome / upright recovery is implemented and smoke-verified; manual interactive viewer confirmation remains useful.
- T109: viewer together root-z ratchet / visual lift-off is fixed in code and awaiting manual visual confirmation.
- T103: complex raw planner core semantic parity beyond flat P0.
- T107: long training, env counts beyond 128, and multi-device Isaac runtime throughput.

## Closed Children Archive

- T101: architecture and fixed-shape runtime design implemented.
- T102: module API/data structure design implemented with together result/cache contracts.
- T104: fixed-shape full-batch manager/cache blending implemented.
- T105: env cfg, train/play/viewer backend factory switch implemented.
- T106: A+B parity and behavior test matrix added for P0 flat/core/runtime cases.
- T108: static training-path guardrail added and passing.
- Regression test contract drift after default-together/factory migration: fixed in `Go2Pvcnn/tests/test_batched_planner_runtime_path.py`.

## Related Logs

- [2026-04-28-1254-viewer-zero-command-handoff-idempotence.md](../log/2026-04-28-1254-viewer-zero-command-handoff-idempotence.md)
- [2026-04-28-1132-together-zero-command-rehome.md](../log/2026-04-28-1132-together-zero-command-rehome.md)
- [2026-04-28-1007-viewer-together-root-z-ratchet.md](../log/2026-04-28-1007-viewer-together-root-z-ratchet.md)
- [2026-04-28-0952-human-12-command-guide-update.md](../log/2026-04-28-0952-human-12-command-guide-update.md)
- [2026-04-27-1914-batched-together-continued-testing.md](../log/2026-04-27-1914-batched-together-continued-testing.md)
- [2026-04-27-1836-batched-together-env-isaaclab-final-verification.md](../log/2026-04-27-1836-batched-together-env-isaaclab-final-verification.md)
- [2026-04-27-1828-viewer-together-backend-smoke.md](../log/2026-04-27-1828-viewer-together-backend-smoke.md)
- [2026-04-27-1711-batched-together-cadence-decision.md](../log/2026-04-27-1711-batched-together-cadence-decision.md)
- [2026-04-27-1630-batched-together-design-review-revisions.md](../log/2026-04-27-1630-batched-together-design-review-revisions.md)
- [2026-04-27-1622-batched-together-planner-gpu-migration-design.md](../log/2026-04-27-1622-batched-together-planner-gpu-migration-design.md)

## Git Refs

- Last Feature Commit: `pending`
- Last Verified Commit: `working tree verification on top of 7cf6c11`
- Current Work Ref: `working tree on top of 7cf6c11 (2026-04-27 19:14 +0800); implementation and test fixes uncommitted`
- Key Files:
  - [../../Go2Pvcnn/extension/batched_together_planner](../../Go2Pvcnn/extension/batched_together_planner)
  - [../../Go2Pvcnn/extension/batched_planner/manager.py](../../Go2Pvcnn/extension/batched_planner/manager.py)
  - [../../Go2Pvcnn/extension/batched_planner/trajectory.py](../../Go2Pvcnn/extension/batched_planner/trajectory.py)
  - [../../Go2Pvcnn/extension/convention.py](../../Go2Pvcnn/extension/convention.py)
  - [../../Go2Pvcnn/extension/mdp/rewards_reference.py](../../Go2Pvcnn/extension/mdp/rewards_reference.py)
  - [../../Go2Pvcnn/extension/reference/cache.py](../../Go2Pvcnn/extension/reference/cache.py)
  - [../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py](../../Go2Pvcnn/go2_pvcnn/tasks/teacher_elevation_trajectory_env_cfg.py)
  - [../../Go2Pvcnn/scripts/train.py](../../Go2Pvcnn/scripts/train.py)
  - [../../Go2Pvcnn/scripts/play.py](../../Go2Pvcnn/scripts/play.py)
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../raw/kinematic_footsteps/scripts/go2fp/batch_planner](../../raw/kinematic_footsteps/scripts/go2fp/batch_planner)

## Next Step

- Rerun the interactive together viewer manually and confirm both T109 lift-off and T110 zero-command crouch recovery are visually resolved.
- Extend T103 with complex terrain/support/CEM parity cases beyond flat P0.
- Extend T107 with multi-iteration train profiling, env counts beyond `128`, and multi-device scaling once the user wants performance numbers beyond smoke.

## Node Details

### T110 zero-command rehome upright recovery

- why-created: The user reported that giving speed `0` in the together viewer lacks the raw-style "回正" effect; screenshot shows the robot ending crouched/twisted after stop.
- observed-current-behavior-before-fix:
  - `Go2Pvcnn/extension/batched_together_planner/parameterization.py` has `_hold_rehome_targets()` that moves feet toward root-frame nominal slots.
  - The hold branch still uses `hold_root = root_pos[:, None, :]` and `hold_rpy = root_rpy[:, None, :]`, so root height and roll/pitch are frozen for the full horizon.
  - Existing tests only allowed terminal roll/pitch to be unchanged, so they did not catch the missing recovery.
- implemented-behavior:
  - For zero command, keep current yaw and root `xy`.
  - Move feet toward current-yaw nominal root-frame slots with support heights from the local terrain.
  - Move root `z` toward support height plus nominal `hip_height`.
  - Move root roll/pitch toward the support-plane or flat upright target.
  - Keep full-contact / no-touchdown semantics for hold.
- parallelism-constraints:
  - Training hot path must remain batched over all envs.
  - No per-env Python loops, comprehensions, NumPy/CPU packages, `.cpu()`, `.item()`, `nonzero`, dynamic sub-batch planner calls, or GPU-mask host branching.
  - Use tensor masks and `torch.where`/broadcasting so mixed batches of zero and moving commands remain independent.
- implementation-notes:
  - Support-plane target uses vectorized four-foot midpoint cross product rather than `torch.linalg.svd`, avoiding solver/sync risk in the training hot path.
  - Static guardrail now forbids `torch.linalg.svd` and `torch.svd` in together training files.
- verification:
  - Red test caught frozen terminal root z before implementation.
  - `test_batched_together_core.py`, `test_batched_together_parity.py`, and `test_batched_together_guardrails.py`: `26 passed`.
  - Runtime/viz subset: `29 passed`.
  - Combined core/parity/guardrail/runtime/viz subset: `55 passed`.
  - CUDA diagnostic recovered root rpy `[0.22, -0.16, 0.70] -> [0.0, 0.0, 0.70]` and root z `0.18 -> 0.30`.
  - Headless viewer scripted forward then zero command reached playback with plan/actual near `z=+0.30` and `rpy=0`; process was timeout-terminated by design with no residual GPU compute process.
  - Follow-up viewer handoff regression: repeated zero-command replans now show `0.4 -> 0.3` once, then `0.3 -> 0.3` on later segments instead of replaying the same recovery each cycle.

### T101 architecture and fixed-shape runtime design

- why-created: The new planner must be native IsaacLab GPU code, not a direct copy of raw CPU/viewer compatibility layers.
- decision:
  - Build `Go2Pvcnn/extension/batched_together_planner/` as a new backend.
  - Keep `extension.batched_planner` as legacy rollback.
  - Use a backend factory/attach helper rather than mixing legacy and together internals in one manager.
- data-flow:
  - Isaac tensors -> `TogetherTrajectoryManager.refresh_from_env()`
  - `TogetherPlannerTerrain.from_ray_hits()`
  - `plan_segment()`
  - device-preserving adapter to `TogetherPlannerResult` / legacy-compatible output surface
  - GPU resident reference cache
  - manager-owned reward gather snapshot.
- hard runtime rules:
  - Planner calls always receive full `[N, ...]` tensors.
  - `replan_mask [N]` controls whether a new result is accepted per env, not which envs are planned.
  - The together training path must never branch on GPU tensor masks before planner call.
  - Planner call cadence may be controlled only by host-known scalar state such as manager step token or a fixed host cadence.
  - No dynamic shapes created from `replan_mask`.

### T102 module API, manager protocol, and GPU cache ABI

- requirement:
  - New modules: `types.py`, `config.py`, `terrain.py`, `schedule.py`, `optimizer.py`, `parameterization.py`, `kinematics.py`, `costs.py`, `planner.py`, `adapter.py`, `manager.py`.
  - Expose a legacy-compatible external surface without copying legacy internals.
- manager protocol:
  - `refresh_from_env(env) -> ReferenceTrajectoryCache`
  - `current_reference() -> dict[str, Tensor]`
  - `current_frame_ids() -> Tensor`
  - `reset_envs(env_mask: Tensor) -> None`
  - `mark_command_changed(...)` or equivalent host-side dirty token from command manager / command wrapper
  - maintain `env.unwrapped._trajectory_reference_cache`
  - same env step is idempotent: repeated reward terms read the same snapshot and do not advance phase twice.
- GPU cache ABI:
  - Together training path must not call the existing CPU-canonical `planner_result_to_reference_cache()` converter.
  - Add a device-preserving cache adapter for together training.
  - `ReferenceTrajectoryCache.is_ready()` shape contract may be reused, but canonical CPU status is not required for together training.
  - Legacy/raw parity code may still use canonical CPU conversion.
- result schema:
  - Internal `TogetherPlannerResult` should include raw-like fields:
    - `root_pos`
    - `root_rpy`
    - `foot_pos`
    - `joint_angles`
    - `contact_state`
    - `touchdown_seq`
    - `touchdown_mask`
    - `cost_total`
    - `cost_breakdown`
    - `status`
    - `feasible`
    - `safe_fallback`
    - IK diagnostics: `joint_limit_violation`, `workspace_margin`
    - support diagnostics: selected `support_xy`, `support_height`, `support_slope` when available.
  - Adapter maps result fields to reward-facing `root_pos_w`, `root_quat_w`, `joint_angles`, `foot_pos_root`, `contact_state`, `planned_touchdown_w`, `phase_index`, `valid_mask`.

### T103 raw planner core semantic migration

- requirement:
  - Align to raw `batch_planner` semantics while excluding raw CPU/viewer boundaries.
  - Do not migrate raw `adapter.py` NumPy conversion, raw terrain `height_at/local_window` CPU sampling, raw cache-key `.item()` logic, or viewer playback/logging into training hot path.
- raw semantic contract:
  - `config`: mirror raw `BatchPlannerConfig` fixed contract and tuning fields or provide explicit field-by-field parity tests.
  - `schedule`: exact `contact_state` and `touchdown_mask` semantics; Isaac version must avoid raw `nonzero`/leg loops.
  - `seeds/CEM`: preserve seed templates, command/state gates, CEM selection semantics, and cost ranking.
  - `parameterization`: preserve root-frame template priority, swing phase, support stance reference, base-frame command integration, touchdown/support target rules.
  - `terrain support`: preserve local patch height, geometry smoothing, slope, support relocation, and no low-ground fallback semantics using GPU batched queries.
  - `costs`: preserve `J_td`, `J_swing`, `J_ik`, `J_base`, `J_vel` and total weighting.
  - `IK`: preserve joint angle, joint-limit, and workspace-margin semantics.
  - `fallback/rehome`: preserve zero-command recovery, safe fallback, and all-infeasible display/training-safe behavior.
  - `result schema`: preserve raw field shape/meaning where exposed through together result.
- guardrails from raw active work:
  - T412 root-frame template-first is core planner behavior, not viewer-only.
  - T409/T410 no support penetration and no low-ground fallback must remain semantic-free: no terrain labels, only command/state/heightmap/support/IK signals.
  - T413 centralized config must be mirrored or parity-tested.
  - T414 host-sync removal must be treated as a minimum bar, then tightened for Isaac training.

### T104 fixed-shape full-batch manager/cache blending

- requirement:
  - Replace legacy dynamic sub-batch behavior for the `together` backend.
  - No `nonzero -> index_select -> planner(sub_batch)` training path.
  - No `if torch.any(replan_mask)` or `torch.allclose`/`torch.equal` GPU-tensor Python branch to decide planner execution.
- design:
  - Maintain `old_cache`, `new_cache`, `fallback_cache`, `phase_counter`, `replan_mask`, `new_ok_mask`, `replace_mask`, and `fallback_mask`.
  - Planner attempts produce full `[N, 35, ...]` results only when a host-visible trigger fires:
    - velocity command changed;
    - reset occurred;
    - the `0.7s` / `35` frame replan interval expired.
  - Manager updates cache rows with `torch.where`.
  - Phase resets to `0` where `replace_mask OR fallback_mask` is true; otherwise advances/clamps by tensor operation.
  - Reward frame ids come from `TogetherTrajectoryManager.current_frame_ids()` / `current_reference()`, not `episode_length_buf % horizon`.
  - Step idempotence guard uses a host step token, not GPU tensor equality.
  - Command-change detection must be driven by a host-side dirty token, version counter, or explicit hook from command/reset handling. It must not compare GPU command tensors and branch on the resulting tensor.
- trigger semantics:
  - If a trigger fires, the planner still receives all env rows `[N, ...]`.
  - Per-env `replan_mask` can mark which rows should accept the new result, but it cannot determine planner batch shape or planner execution.
  - If only one env command changes, the planner call is still full `N`; unaffected envs normally keep old cache rows through `keep_mask`.
- masks:
  - `must_replace_mask = first_cache OR reset_mask OR cache_invalid`
  - `soft_replan_mask = replan_mask AND NOT must_replace_mask`
  - `new_ok_mask = result.feasible OR result.safe_fallback`
  - `replace_mask = must_replace_mask AND new_ok_mask OR soft_replan_mask AND new_ok_mask`
  - `fallback_mask = must_replace_mask AND NOT new_ok_mask`
  - `keep_mask = NOT replace_mask AND NOT fallback_mask`
- fallback rule:
  - `fallback_mask` rows receive a GPU-generated current-state standstill/rehome cache, not an invalid planner row.
  - `safe_fallback` is true only for finite, shape-valid, training-safe fallback rows whose IK/workspace diagnostics pass configured fallback thresholds.
- cache blend shape rule:
  - `[N, H, 3]` fields use mask shape `[N, 1, 1]`.
  - `[N, H, 4]` fields use `[N, 1, 1]`.
  - `[N, H, 4, 3]` fields use `[N, 1, 1, 1]`.
  - `index_copy_`, masked row writes, and dynamic row gathering are forbidden in together training path.

### T105 env cfg, train/play/viewer backend switch

- requirement:
  - Add `planner_backend = "together" | "legacy"` to `teacher_elevation_trajectory_env_cfg.py`.
  - Default `together` uses `reference_trajectory_horizon = 35`, `plan_dt`/`dt = 0.02`.
  - Viewer uses the same backend name but remains excluded from training-path CPU/static guardrails.
- backend factory:
  - Add a neutral factory such as `extension/trajectory_manager_factory.py`.
  - `planner_backend="legacy"` attaches the existing `BatchedTrajectoryManager`.
  - `planner_backend="together"` attaches `TogetherTrajectoryManager`.
  - Invalid backend raises a clear error.
  - `train.py`, `play.py`, and viewer attach helper should route through the factory or a shared attach helper.
- isolation:
  - Static together guardrail must not scan `extension/batched_planner/**` as if it were together code.
  - Viewer CPU exceptions are limited to `extension/viz/**`.

### T106 parity and behavior tests

- requirement:
  - A-level behavior alignment and B-level tensor parity are both required.
  - Tests must be executable, not just narrative.
- parity matrix:

| Area | B-level fields | Exact/tolerance | P0 scenarios |
| --- | --- | --- | --- |
| schedule | `contact_state`, `touchdown_mask` | exact | flat zero/forward/lateral/yaw/combo, mixed batch |
| kinematics | `joint_angles`, `joint_limit_violation`, `workspace_margin` | flat `1e-5` to `1e-4`; terrain measured | flat commands, all-infeasible fallback |
| trajectory | `root_pos`, `root_rpy`, `foot_pos`, `touchdown_seq` | flat tight; terrain z/support 1-3 cm | forward/lateral/yaw, yaw=`pi/2`, stair turn |
| costs | `cost_total`, `J_td`, `J_swing`, `J_ik`, `J_base`, `J_vel` | close on identical fixtures; trend on complex terrain | flat, step, rocky |
| support query | `support_xy`, `support_height`, `support_slope` | XY <= 2-3 cm, support gap <= 5 cm | stair top/riser, stairs->rocky |
| status | `status`, `feasible`, `safe_fallback` | exact semantics | feasible, all-infeasible, safe fallback |

- A-level behavior:
  - forward/lateral/yaw command response;
  - base-frame command with yaw `pi/2`;
  - zero-command root-frame template recovery;
  - stair-turn event-root-frame touchdown template;
  - stairs->rocky no low-ground fallback;
  - no riser/support penetration;
  - mixed full batch with different per-row commands.
- raw guardrail thresholds to carry forward:
  - `forward_mid` late segment amplitude `>= 0.60`;
  - `forward_max` late segment amplitude `>= 0.52`;
  - direction error should be recorded and must not regress without an explicit note.
- together manager tests:
  - reset/command-change/interval triggers still call planner with full `N`.
  - no planner call occurs when no host trigger is pending and the interval has not expired.
  - host command dirty token changes cause exactly one full-`N` planner attempt for the affected env step.
  - old/new cache truth table covers `first_cache`, `reset`, `cache_invalid`, `soft_replan`, `feasible`, `safe_fallback`, and infeasible/no fallback.
  - multiple reward terms in one step do not advance phase twice.
  - accepted rows phase reset to `0`; kept rows phase advances/clamps.
  - cache tensors stay on CUDA when CUDA is available.

### T107 performance and scaling benchmarks

- P0 smoke:
  - `num_envs=1`
  - `num_envs=32`
  - `num_envs=128`
  - check full-batch call size, cache shape, phase behavior, and no CPU transfer.
- latest evidence:
  - `num_envs=32`, `max_iterations=1`, together training passed with `Total steps: 1280`.
  - `num_envs=128`, `max_iterations=1`, together training passed with `Total steps: 5120`.
  - `num_envs=16`, `max_iterations=1`, legacy rollback training passed.
  - Real Isaac env cadence test at `num_envs=4` recorded full-N planner call batch sizes `[4, 4, 4, 4]` for reset, one-env command dirty, interval, and command hook triggers.
- Optional CUDA benchmark:
  - `num_envs=1024`
  - `num_envs=4096`
  - use CUDA events/median timing where possible.
  - record planner call count, batch size, GPU memory, and planner stage times.
- benchmark meaning:
  - Legacy partial-replan benchmarks are not together acceptance tests.
  - Together benchmarks must measure fixed-shape full-batch behavior.

### T108 static training-path guardrail

- why-created: User explicitly requested code tests that detect forbidden CPU package use and forbidden loops in the training path.
- test file:
  - `Go2Pvcnn/tests/test_batched_together_guardrails.py`
- scan manifest:
  - `Go2Pvcnn/extension/batched_together_planner/**/*.py`
  - backend factory / attach helper
  - together-related `teacher_elevation_trajectory_env_cfg.py` wiring
  - reward/cache gather changes introduced for together.
- exclude:
  - `Go2Pvcnn/extension/viz/**`
  - `Go2Pvcnn/extension/batched_planner/**`
  - raw bridge/reference-only code
  - tests and benchmarks unless a specific guardrail test targets them.
- forbidden in scanned training path:
  - `import numpy`, `from numpy`, `np.`
  - `.cpu()`
  - `.item()`
  - `.numpy()`
  - `.tolist()`
  - `.detach().cpu()`
  - `bool(tensor...)`, `int(tensor...)`, `float(tensor...)` where the argument is tensor-derived
  - `torch.equal`, `torch.allclose` for manager step idempotence or planner execution decisions
  - `torch.cuda.synchronize`
  - `nonzero`
  - `index_select` for env/candidate dynamic sub-batches
  - `index_copy_`
  - `masked_select`
  - `torch.split` / `chunk` over candidate/env hot-path dimensions
  - dynamic planner call on a sub-batch.
- `for` rule:
  - No Python `for` loops in together hot-path files/functions.
  - Hot path includes `plan_segment`, `refresh_from_env`, adapter/cache conversion, terrain query, schedule, optimizer, parameterization, costs, kinematics, and reward gather changes.
  - Fixed CEM iteration loops are not automatically exempt; if CEM iteration remains, it must be vectorized/unrolled or explicitly approved by the user before implementation.
- path coverage:
  - If a future file is added under `batched_together_planner/` and is not covered by the scan manifest or an allowlist with reason, guardrail fails.

### T109 viewer together root-z ratchet

- why-created: User reported, with a viewer screenshot, that the robot "walks then flies" when launching the viewer.
- status: `verify`; fixed by stabilizing together viewer handoff root z, but interactive visual confirmation remains.
- scope:
  - Primary affected workflow: `Go2Pvcnn/extension/viz/go2_foostep_planner.py` together viewer playback and segment handoff.
  - Related planner workflow: `Go2Pvcnn/extension/batched_together_planner/parameterization.py` root z trajectory and raw single-segment parity.
  - Training path must not gain CPU packages, NumPy, or dynamic sub-batch logic from this fix.
- initial hypothesis:
  - The together planner's single segment can end with a positive root-z bias.
  - The viewer uses the last played frame as the next segment's initial state.
  - Repeated walking replans can therefore accumulate base height even on flat terrain, producing visual lift-off.
- implemented fix:
  - Added `_together_handoff_root_pos()` in [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py).
  - `_together_state_from_reference_result()` now reconstructs the next segment root z from current contact-foot support height plus the segment's initial support clearance.
  - The fix is viewer-only and preserves raw/together single-segment parity.
- evidence:
  - New regression reproduced `0.3964m` terminal z after 4 viewer-style replans before the fix.
  - `Go2Pvcnn/tests/test_viz_playback.py`: `20 passed`.
  - Together parity/core/runtime/guardrail set: `35 passed`.
  - Headless viewer reached real together playback and was terminated by timeout; later standstill evidence stayed near `plan.z=+0.30`, `actual.z=+0.30`.
- acceptance:
  - Regression, relevant unit tests, parity/core/runtime/guardrail tests, py_compile, and headless viewer smoke evidence are complete.
  - Remaining: manual interactive visual confirmation.
