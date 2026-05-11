# Unified Dense MPC Backend Design

## 1. Problem Statement

The current `together` backend is a fixed-table candidate search planner with strong semantic constraints and diagnostics. The next target is to replace the search-style inner loop with a fully differentiable dense MPC optimizer while preserving:

- planner-owned cache runtime contract
- reward consumption contract
- viewer replay/debug contract
- pure GPU, fixed-shape, no CPU hot-path behavior

The user requires:

- a new backend (`planner_backend="mpc"`) instead of replacing `together`
- no legacy mode concepts (`CRUISE/APPROACH/CROSS/BYPASS`) in the new optimizer
- unified optimization variables:
  - `root_pos_residual [B,T,3]`
  - `root_rpy_residual [B,T,3]`
  - `foot_pos_residual [B,T,4,3]`
  - `contact_logits [B,T,4]`
- touchdown behavior derived from optimized foot trajectories (not separate decision tables)
- per-loss term configuration similar to RL reward configuration (weight + tunable parameters)
- optional hard diagnostics via an enable switch
- per-env asynchronous replanning for speed at large parallel scale (`num_envs=4096`)

## 2. Goals

- Add a new MPC backend that can handle:
  - normal locomotion on terrain
  - small obstacle crossing
  - large obstacle bypass/avoidance
  - infeasible-state diagnostics
- Keep training path pure GPU and fixed-shape.
- Keep existing manager/cache/reward/viewer integration stable by preserving output ABI.
- Support high parallel throughput with async per-env replan masking.
- Make optimizer behavior tunable through config classes only.

## 3. Non-Goals

- No reuse of old mode classifiers or candidate tables inside MPC core.
- No CPU/Numpy hot-path operations (`.cpu()`, `.item()`, `.tolist()`, `argwhere`, `masked_select`, dynamic subbatch loops).
- No direct joint-angle optimization in first iteration (joint quality is enforced through IK/workspace/joint-limit losses).
- No implementation-phase code changes in this design document.

## 4. Architecture Overview

### 4.1 Backend Integration

`trajectory_manager_factory.py` adds:

- `planner_backend="mpc"`
- `create_trajectory_manager(...) -> MpcTrajectoryManager`

Backends remain:

- `legacy`
- `together`
- `mpc` (new)

### 4.2 New Module Layout

```text
Go2Pvcnn/extension/
  trajectory_contracts.py

Go2Pvcnn/extension/batch_mpc_planner/
  __init__.py
  config.py
  profiles.py
  types.py
  terrain.py
  nominal.py
  variables.py
  optimizer.py
  planner.py
  adapter.py
  manager.py
  diagnostics.py
  losses/
    __init__.py
    base.py
    registry.py
    tracking.py
    smoothness.py
    contact.py
    terrain_clearance.py
    kinematics.py
```

### 4.3 Runtime Dataflow

```text
env state + command + scanner
-> MpcTrajectoryManager.refresh_from_env(...)
-> batch_mpc_planner.planner.plan_segment(...)
-> ReferenceTrajectoryCache-compatible core ABI output
-> ReferenceTrajectoryCache (adapter)
-> rewards_reference.py current-frame gather
```

Viewer uses the same backend switch and consumes the same trajectory output shape.

## 5. Optimization Variables and Trajectory Decode

For each replan subset (`B_dirty`), optimize dense tensors:

- `root_pos_residual [B,T,3]`
- `root_rpy_residual [B,T,3]`
- `foot_pos_residual [B,T,4,3]`
- `contact_logits [B,T,4]`

Decoded trajectory:

- `root_pos = root_nominal + root_pos_residual`
- `root_rpy = root_rpy_nominal + root_rpy_residual`
- `foot_pos = foot_nominal + foot_pos_residual`
- `contact_prob = sigmoid(contact_logits / contact_temperature)`

Export for existing contracts:

- `contact_state_bool = contact_prob > contact_threshold`
- `touchdown_seq/planned_touchdown_w` extracted from foot trajectory and contact transitions

No explicit mode is produced or consumed in this backend.

### 5.1 Differentiable Contact Timing Contract

- `contact_logits` is an optimizer-owned variable, not only an export field.
- Stance/swing timing losses must consume differentiable `contact_prob` (or differentiable masks derived from it).
- Hard thresholding (`contact_prob > contact_threshold`) is allowed only for export ABI, diagnostics, and viewer summaries; it must not drive optimization-path losses.
- The MPC core must not use precomputed legacy gait/mode schedules as timing sources.
- Contact transitions, support duration, swing duration, and touchdown timing are optimization outcomes shaped by loss terms.

### 5.2 Touchdown Extraction Contract

- `touchdown_seq` and `planned_touchdown_w` are derived outputs, not independent optimizer variables and not table-generated outputs.
- Export touchdown events from per-leg low->high contact transitions on decoded contact sequence.
- The touchdown position is sampled from optimized `foot_pos` at transition frame.
- Export must preserve full cache shape `[B, T, 4, 3]` with deterministic hold-forward/backfill policy.

## 6. Loss System (Reward-Style Configurable)

Each loss term is a first-class config entry with:

- `enabled`
- `weight`
- term-specific tunable parameters (margins, temperatures, sample counts, penalties)

### 6.1 Required Core Loss Families

- command tracking
- root/foot smoothness
- stance slip minimization (when contact is high)
- swing clearance (when contact is low)
- terrain/body clearance
- small obstacle clearance/crossing consistency
- large obstacle avoidance/bypass consistency
- touchdown support validity
- IK/workspace/joint-limit constraints (through existing kinematics evaluation)
- contact regularization (binary tendency, transition cost, support stability)
- progress-direction consistency

### 6.2 Config Contract Example

```python
@dataclass
class MpcRuntimeCfg:
    horizon_steps: int = 80
    dt: float = 0.02
    optimize_steps: int = 12
    lr: float = 3e-2
    optimizer: str = "adam"
    grad_clip_norm: float = 10.0
    contact_temperature: float = 0.20
    contact_threshold: float = 0.55
    replan_interval_steps: int = 50
    max_stale_steps: int = 100
    warm_start_from_previous_plan: bool = True
    detach_warm_start: bool = True
    detach_cache_on_write: bool = True
    heavy_loss_stride: int = 2
    heavy_loss_enable_from_iter: int = 8
    selection_mode: str = "fixed_topk_priority"
    max_dirty_envs_per_step: int = 256
    target_dirty_ratio: float = 0.05
    randomize_replan_phase: bool = True
    randomize_command_phase: bool = True
    command_hard_lin_delta: float = 0.25
    command_hard_yaw_delta: float = 0.35
    command_soft_lin_delta: float = 0.05
    command_soft_yaw_delta: float = 0.10
    command_blend_steps: int = 8
    terrain_subset_before_build: bool = True
    step_local_reference_cache: bool = True
    train_dtype: str = "float32"
    amp_enabled: bool = False
    optimizer_unroll_graph: bool = False
    profile_4096_required: bool = True


@dataclass
class MpcPlannerCfg:
    runtime: MpcRuntimeCfg = field(default_factory=MpcRuntimeCfg)
    diagnostics: MpcDiagnosticsCfg = field(default_factory=MpcDiagnosticsCfg)
    losses: MpcLossesCfg = field(default_factory=MpcLossesCfg)
    profile_name: str = "train_4096"
```

Term-level examples:

- `SwingClearanceLossCfg(min_clearance_m, obstacle_clearance_m, temperature, weight, enabled)`
- `LargeObstacleAvoidanceLossCfg(body_margin_m, foot_margin_m, repulsion_radius_m, sample_count, weight, enabled)`
- `ContactRegularizationLossCfg(binary_weight, transition_weight, min_support_legs, max_airborne_steps, weight, enabled)`

## 7. Hard Diagnostics Layer (Config-Gated)

Diagnostics are not mode logic. They are post-optimization safety/explainability checks.

### 7.1 Purpose

- training guardrails for invalid plans
- readable failure reasons in viewer/tests
- manager fallback decisions (`ALL_INFEASIBLE`, hold old cache row, or standstill row)

### 7.2 Config Switch

```python
@dataclass
class MpcDiagnosticsCfg:
    enabled: bool = False
    strict_failure_mask: bool = True
    emit_viewer_fields: bool = True
```

Policy:

- training default: `enabled=False` for speed
- validation/viewer/tests: `enabled=True`

Typical reasons:

- large collision
- small collision/insufficient clearance
- touchdown unsupported
- IK/workspace impossible
- airborne-support instability
- command progress violation

## 8. Asynchronous Replan Semantics (4096-Env Target)

### 8.1 Trigger Conditions

Per env, mark dirty on:

- env reset
- command hard change
- interval timeout (`replan_interval_steps`)
- explicit manager events
- first-cache / cache-shape-invalid / NaN-fallback recovery paths

Soft command changes can be deferred to interval boundaries.

### 8.2 Dirty-Subset Replan

Each step:

- choose dirty rows using fixed-budget GPU scheduling (priority + capacity)
- only selected dirty rows enter MPC optimization
- non-dirty rows advance cached reference frame index
- cache row replacement uses fixed-shape masked blending (`torch.where`)

This avoids full-batch replanning on every step.

Dirty backlog priority:

1. `reset`
2. hard command change
3. stale timeout (`max_stale_steps`)
4. interval timeout
5. soft command change

### 8.3 GPU-Only Contract

- Dirty-mask computation, selection, warm-start selection, and cache blending remain GPU-resident.
- Hot path forbids host-sync operations and Python env-row loops (`.cpu()`, `.item()`, `.tolist()`, NumPy conversion, per-env dynamic loop over selected rows).
- Command and reset hooks must preserve per-env masks; if unknown mask arrives, manager should record explicit reason and avoid silent all-env dirty expansion in training mode.
- Cache outputs stay full-shaped even when only a subset is replanned.

### 8.4 Command Asynchrony and Stability

Command update phase should be randomized per env (not globally synchronized) so dirty ratio remains bounded.

Command adaptation policy:

- hard delta -> immediate dirty
- soft delta -> deferred to interval boundary
- planned command blending over `command_blend_steps` to avoid warm-start shock
- if command cycle is too short relative to interval (e.g. `< 4 * replan_interval_steps`), reduce optimize steps or increase budget to prevent backlog growth

## 9. Performance Strategy for 4096 Envs

To balance speed and quality:

- train profile:
  - `horizon_steps=80`
  - `optimize_steps=10~12`
  - `replan_interval_steps=50`
  - `max_dirty_envs_per_step=256` (profile-gated scale-up to `384/512`)
  - `target_dirty_ratio=0.05`
  - `heavy_loss_stride=2~4`
  - `diagnostics.enabled=False`
  - warm-start enabled
  - `train_dtype=float32`, `amp_enabled=False` by default
- eval/viewer profile:
  - `optimize_steps=48~64`
  - `heavy_loss_stride=1`
  - `diagnostics.enabled=True`

Expected behavior target (single 4090 class GPU, non-authoritative estimate):

- non-replan step: `<0.5~1.0 ms` planner/cache overhead target
- dirty-subset replan: roughly proportional to selected dirty budget
- stable average throughput depends on bounded dirty ratio and bounded `max_dirty_envs_per_step`

Per-step timing model:

- non-replan: `T_non = T_guard + T_phase + T_current_frame_gather + T_reward_compare`
- replan: `T_replan = T_dirty_select + T_subset_gather(B) + T_warm_start(B,T) + I*(T_decode + T_light_loss + 1/stride*T_heavy_loss + T_backward + T_adam) + T_export + T_cache_write(B,T)`

Required runtime profiling counters for 4096-env approval:

- `dirty_count`
- `selected_dirty_count`
- `dirty_backlog`
- `max_stale_observed`
- `planner_ms`
- `cache_ms`
- `reward_gather_ms`

## 10. Output ABI Compatibility

The new backend should produce a ReferenceTrajectoryCache-compatible core ABI required by:

- cache adapter
- reward gather
- viewer playback

Always-required fields:

- `root_pos`, `root_rpy`, `foot_pos`, `joint_angles`
- `contact_state`, `touchdown_seq`
- `cost_total`, `status`, `feasible/safe_fallback`

Diagnostics-enabled additional fields:

- `hard_reason_mask`
- `loss_breakdown`
- selected clearance/support summaries

## 11. Error Handling

- shape/device/type mismatches fail fast with explicit messages
- NaN/Inf in optimization objective handled by finite guards and fallback status
- impossible rows produce deterministic failure status and fallback-compatible outputs
- manager never leaves cache uninitialized after first valid refresh path

## 12. Config/Implementation Boundaries and Contracts

### 12.1 Config Layering

- `MpcPlannerCfg` is top-level and composes:
  - `runtime: MpcRuntimeCfg`
  - `losses: MpcLossesCfg`
  - `diagnostics: MpcDiagnosticsCfg`
  - `profile_name: str`
- `config.py` holds only config definitions (no torch ops, no optimizer/loss object creation).
- `profiles.py` exposes preset builders (`train/eval/viewer`) for direct RL config import and override.
- Planner runtime should use `mpc_planner.runtime.*` as source of truth to avoid dual-authority drift with legacy planner fields.

### 12.2 Protocol Contracts

- `TrajectoryManagerProtocol`: `planner_backend`, `refresh_from_env`, `current_reference`, `current_frame_ids`, `reset_envs`, `mark_command_changed`, `horizon_steps`.
- `PlannerCoreProtocol`: `plan_segment(terrain, state, command, cfg, warm_start=None) -> MpcPlannerResult`.
- `PlannerResultProtocol`: includes `root_pos/root_rpy/foot_pos/joint_angles/contact_state/touchdown_seq/cost_total/status/feasible/safe_fallback`.
- `CacheAdapterProtocol`: includes result-to-cache conversion, standstill fallback cache, and masked blending.
- `ViewerDiagnosticsProtocol`: `diagnostic_names`, `hard_reason_mask`, `loss_breakdown`, `status_name`.

### 12.3 Coupling Rules

- Do not force MPC to carry old together-mode/candidate semantics.
- Avoid reward/viewer hardcoding to `"together"` capability checks; use backend-agnostic protocol checks.
- Keep cache/output ABI fixed-shape even if internal dirty scheduling is budgeted.

## 13. Verification Plan

### 13.1 Unit/Deterministic

- variable decode shape and dtype contracts
- contact logits differentiable timing contract (loss path uses `contact_prob`, not bool threshold)
- contact logits -> bool contact extraction contract (export-only path)
- touchdown extraction contract
- loss registry toggles and parameter plumbing
- diagnostics enable/disable behavior
- fixed-shape dirty-row blend behavior
- static GPU guardrail scan for forbidden hot-path APIs

### 13.2 Integration

- backend factory switch: `mpc/together/legacy/invalid`
- manager dirty scheduling with bounded `max_dirty_envs_per_step`
- adapter/cache/reward contract without old mode/candidate dependencies
- viewer payload contract with diagnostics fields and no together-only assumptions

### 13.3 Runtime Acceptance (`env_isaaclab`, GPU-only)

- asynchronous command updates do not force full-batch replan each step
- dirty-subset refresh correctness
- small obstacle crossing success signals
- large obstacle avoidance signals
- infeasible diagnostics correctness when enabled
- 4096-env throughput profiling counters are emitted and stable

### 13.4 Regression Compatibility and Isolation

- reward consumers still read current frame via manager cache contract
- viewer backend switch path remains functional
- legacy/together backends unaffected by mpc addition
- new MPC tests do not import together-specific mode/candidate semantics
- runtime acceptance uses IsaacLab environment and CUDA tensors (no silent CPU fallback)

### 13.5 Detailed Test Matrix

| ID | Level | Input | Key Assertions | Environment |
| --- | --- | --- | --- | --- |
| U1 | unit | `B=4,T=80` nominal/residual tensors on CUDA | decode shape/dtype/device are correct; backward is valid; outputs finite | `env_isaaclab` Python + CUDA |
| U2 | unit | synthetic stance->swing->stance `contact_logits` | stance/swing losses change with `contact_logits`; export bool contact is thresholded output only | `env_isaaclab` Python + CUDA |
| U3 | unit | loss cfg toggles/weights/params | disabled term contributes zero; weighted scaling is monotonic; `loss_breakdown` schema fixed | `env_isaaclab` Python + CUDA |
| U4 | unit | valid + invalid plan tensors with diagnostics on/off | `enabled=False` omits heavy fields; `enabled=True` returns fixed-shape reason mask | `env_isaaclab` Python + CUDA |
| U5 | unit | cache with mixed dirty/clean env rows | clean rows advance phase only; dirty rows replaced; full-shape cache preserved | `env_isaaclab` Python + CUDA |
| U6 | static | hot-path source scan | forbid `.cpu()/.item()/.tolist()/.numpy()` and host-loop dirty dispatch in MPC hot path | repo static gate |
| I1 | integration | backend switch `mpc/together/legacy/invalid` | factory attaches correct manager; invalid backend fails fast | `env_isaaclab` |
| I2 | integration | planner result -> adapter -> reward gather | reward reads cache ABI only; no mode/candidate dependency | `env_isaaclab` |
| I3 | integration | mixed reset/command/interval dirty events | priority + `max_dirty_envs_per_step` enforced; `max_stale_steps` respected | `env_isaaclab` + CUDA |
| I4 | integration | viewer backend path with MPC result | viewer prints MPC diagnostics fields and no together-only fields | `env_isaaclab` |
| R1 | runtime | headless train smoke (`num_envs=32`) | startup + one-iteration path stable; cache/reward finite; exit `0` | `env_isaaclab` + GPU |
| R2 | runtime | throughput profile (`num_envs=512/4096`) | bounded dirty selection, no full-batch spikes, profiling counters logged | `env_isaaclab` + GPU |
| R3 | runtime | small-obstacle crossing scenarios | progress + clearance constraints satisfied when feasible | `env_isaaclab` + GPU |
| R4 | runtime | large-obstacle bypass/avoid scenarios | accepted trajectories clear margins or deterministic infeasible fallback | `env_isaaclab` + GPU |
| R5 | runtime | infeasible IK/support/collision scenarios | status/fallback deterministic; hard reasons match expected set | `env_isaaclab` + GPU |
| R6 | runtime | viewer headless with diagnostics enabled | `loss_breakdown/hard_reason` visible; no legacy mode/beta/route output | `env_isaaclab` + GPU |

### 13.6 Diagnostics-Oriented Oracle Rules

- Use `MpcDiagnosticsCfg(enabled=True, strict_failure_mask=True, emit_viewer_fields=True)` for validation and runtime acceptance.
- Valid scenario oracle:
  - `hard_reason_mask.any() == False`
  - clearance/support/progress summaries stay within configured thresholds
- Invalid scenario oracle:
  - reason set contains expected labels (e.g. collision / touchdown unsupported / IK-workspace impossible / support instability / command progress violation)
  - fallback status and cache remain finite and deterministic

## 14. Risks and Mitigations

- Risk: dense optimization converges slowly at 4096 scale.
  - Mitigation: warm-start + dynamic optimize-steps + heavy-loss staging.
- Risk: contact logits become noisy/chattering.
  - Mitigation: transition/binary/support regularizers + temperature schedule.
- Risk: no explicit mode leads to ambiguous local minima.
  - Mitigation: strong obstacle/clearance/progress terms + diagnostics-driven tuning.
- Risk: synchronized command updates collapse throughput.
  - Mitigation: per-env random phases + hysteresis + command blending + bounded dirty budget.
- Risk: budgeted scheduling starves some env rows.
  - Mitigation: stale-priority promotion + `max_stale_steps` hard cap.
- Risk: hidden host-sync appears in hot path.
  - Mitigation: static guardrail scan + runtime profiling counters + explicit forbidden-API list.

## 15. Scope Boundary for Next Phase

This design covers architecture and runtime contracts only.

Implementation planning should next define:

- file-by-file ownership
- phased rollout (factory -> manager -> planner core -> adapter -> tests)
- minimal first-pass losses and acceptance targets
- profiling checkpoints for 4096 env throughput
