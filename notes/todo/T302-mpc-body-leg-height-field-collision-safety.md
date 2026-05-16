# T302 MPC Body/Leg Height-Field Collision Safety

## Current State

- T302 is a new design branch related to [T300e](T300e-mpc-continuous-swing-window-plan.md).
- Purpose: add body/leg/foot height-field collision safety, semantic touchdown/stance obstacle rejection, and high-obstacle speed/yaw risk scaling to the active `batch_mpc_planner` MPC backend.
- Written design: [../../docs/superpowers/specs/2026-05-16-mpc-body-leg-height-field-collision-safety-design.md](../../docs/superpowers/specs/2026-05-16-mpc-body-leg-height-field-collision-safety-design.md)
- Status: design recorded; subagent requirements-coverage review found no P0 gaps before implementation planning.
- Subagent coverage review found no P0 gaps; P1 clarifications were folded into the spec for semantic-small classification, all-direction scanner-mask risk detection, and T300e regression baseline reuse.
- Implementation has not started. The current user gate is design/spec review.

## Open Children

| Child | Status | Priority | Purpose | Primary Files |
| --- | --- | --- | --- | --- |
| T302a | verify | P0 | User spec review gate before writing an implementation plan; subagent requirement coverage review is clean | `docs/superpowers/specs/2026-05-16-mpc-body-leg-height-field-collision-safety-design.md`, this page |
| T302b | pending | P0 | Convert approved design into a TDD implementation plan with test slices and file ownership | `Go2Pvcnn/tests/`, `Go2Pvcnn/extension/batch_mpc_planner/` |
| T302c | pending | P0 | Implement GPU kinematics outputs for knee/shank world samples without adding production files | `Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py` |
| T302d | pending | P0 | Implement height-field collision and semantic touchdown/stance losses | `terrain_clearance.py`, `registry.py`, `terrain.py`, `config.py` |
| T302e | pending | P0 | Implement high-small/large command corridor and yaw-swept risk scaling for tracking losses | `tracking.py`, `registry.py`, `planner.py`, `config.py` |
| T302f | pending | P0 | Add headless `env_isaacsim` acceptance for COBBLESTONE and flat semantic obstacles while preserving T300e metrics | `Go2Pvcnn/tests/` |

## Closed Children Archive

- None yet.

## Related Logs

- [../log/2026-05-16-2200-t302-mpc-body-leg-collision-design.md](../log/2026-05-16-2200-t302-mpc-body-leg-collision-design.md)
- T300e baseline acceptance: [../log/2026-05-15-2001-mpc-contact-support-touchdown-anchor-acceptance.md](../log/2026-05-15-2001-mpc-contact-support-touchdown-anchor-acceptance.md)

## Git Refs

- Last Feature Commit: `pending`
- Last Verified Commit: `pending`
- Current Work Ref: `working tree on top of 65f0d99`
- Key Files:
  - [../../docs/superpowers/specs/2026-05-16-mpc-body-leg-height-field-collision-safety-design.md](../../docs/superpowers/specs/2026-05-16-mpc-body-leg-height-field-collision-safety-design.md)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/terrain.py](../../Go2Pvcnn/extension/batch_mpc_planner/terrain.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/tracking.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/tracking.py)
  - [../../Go2Pvcnn/tests](../../Go2Pvcnn/tests)

## Next Step

- Ask the user to review the written spec before implementation planning.

## Node Details

### T302a Design/Spec Review Gate

- why-created: user wants the MPC redesign to include body/leg collision safety, low-small crossing, high-small/large avoidance, real IsaacLab headless tests, GPU-only implementation, TDD flow, and no loss of T300e behavior.
- design basis:
  - collisions for root/body/knee/shank/swing foot use height map clearance;
  - touchdown and stance use semantic ids, with ground `0` allowed and obstacle ids such as `1/2` penalized;
  - low small obstacles are crossable when obstacle top is within `0.3m` of root-projected ground height;
  - high small and large obstacles can reduce linear/yaw tracking weight when they affect command direction or yaw swept region;
  - tests must include `COBBLESTONE_ROAD_CFG` and flat semantic obstacle scenes;
  - all runtime planner logic must stay GPU-batched.
- evidence:
  - design spec written at [../../docs/superpowers/specs/2026-05-16-mpc-body-leg-height-field-collision-safety-design.md](../../docs/superpowers/specs/2026-05-16-mpc-body-leg-height-field-collision-safety-design.md)
  - subagent review: no P0 gaps; P1 clarifications integrated.
- next:
  - request user review.

### T302b TDD Implementation Plan

- why-created: implementation must follow TDD and preserve future RL throughput.
- pending design-to-plan points:
  - failing backend tests for knee/shank kinematics outputs;
  - failing loss tests for height-field collision and semantic stance/touchdown penalties;
  - failing tests for high-obstacle linear/yaw risk scaling;
  - headless `env_isaacsim` probes for COBBLESTONE and semantic obstacle scenes;
  - runtime metrics and vectorization guardrails.

### T302c GPU Kinematics For Knee/Shank

- why-created: knee/shank collisions need future-horizon positions, so they must come from planned `root + foot` IK/FK, not IsaacLab current link poses.
- constraints:
  - no duplicate IK pass;
  - no CPU geometry loop;
  - outputs remain batched `[B,T,4,3]` and `[B,T,4,K,3]`.

### T302d Height-Field And Semantic Collision Losses

- why-created: current T300e covers foot/terrain grounding but not body/knee/shank swept collisions.
- constraints:
  - height map for root/body/knee/shank/swing-foot clearance;
  - semantic map for touchdown and stance obstacle rejection;
  - no privileged obstacle prim positions in planner runtime.

### T302e High-Obstacle Tracking Weight Scaling

- why-created: high small obstacles and large obstacles should let MPC reduce speed/yaw tracking pressure rather than hard-follow commands into collisions.
- constraints:
  - translation corridor handles nonzero `[Vx,Vy]`;
  - yaw swept region handles yaw-only and mixed-yaw commands;
  - scale tracking losses inside optimization, not after trajectory generation.

### T302f Headless Acceptance Matrix

- why-created: user requires real IsaacLab headless evidence under `env_isaacsim`, not only unit tests.
- cases:
  - `COBBLESTONE_ROAD_CFG` complex terrain with multiple command combinations;
  - flat semantic course with low-small crossing;
  - flat semantic course with high-small avoidance;
  - flat semantic course with large avoidance;
  - yaw-only near obstacle.
- required metrics:
  - T300e gait/grounding regression metrics;
  - T302 collision/semantic/cross/avoid/risk-scale/runtime metrics.
