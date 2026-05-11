# T300 Unified Dense MPC Backend Design

## Purpose

Capture the approved brainstorming design for a new `planner_backend="mpc"` that replaces together-style trajectory search with GPU dense gradient optimization while preserving runtime cache/reward/viewer contracts.

## Stage

Planner architecture design and integration planning for:

- `Go2Pvcnn/extension/batched_mpc_planner/*` (new backend family)
- `Go2Pvcnn/extension/trajectory_manager_factory.py` backend switch extension
- manager/cache/reward/viewer contract compatibility

## Related Todo

[T300/T300a](../todo/T300-unified-dense-mpc-backend.md#t300a-written-spec-review-gate-and-implementation-plan-handoff)

## Command/Procedure

- Read mandatory repo notes and rules:
  - `notes/index.md`
  - `.codex/RULES.md`
  - `notes/todo.md`
  - `notes/log/index.md`
  - planner preread notes (`human-08/09/10/11/12/13`)
- Review current `batched_together_planner` + `go2_foostep_planner.py` call chain and contracts.
- Confirm user requirements:
  - new backend (`mpc`) instead of replacing `together`
  - no old mode concept
  - dense optimization variables:
    - `root_pos_residual [B,T,3]`
    - `root_rpy_residual [B,T,3]`
    - `foot_pos_residual [B,T,4,3]`
    - `contact_logits [B,T,4]`
  - touchdown derived from optimized `foot_pos` + contact transitions
  - per-loss tunable config (not weight-only)
  - diagnostics layer with `enabled` switch
  - async per-env replanning with dirty masks for 4096 env scale
- Write approved design spec:
  - `docs/superpowers/specs/2026-05-11-unified-dense-mpc-backend-design.md`

## Input Conditions

- Current authoritative implementation baseline remains `together` T116/T116i.
- Working tree is dirty from previous branches; this log/spec update is documentation-only and does not modify runtime code.
- Local raw reference directory `raw/kinematic_footsteps` is absent in this workspace; design is based on current project contracts and notes.

## Key Metrics

- Design scope finalized: `planner_backend="mpc"` with unified dense optimization and no mode classifier.
- Performance intent finalized for 4096 env training:
  - async dirty-subset replanning
  - warm-start
  - configurable optimize steps and heavy-loss staging
- Config contract finalized:
  - top-level MPC runtime config
  - per-loss tunable parameter blocks
  - diagnostics `enabled` gate

## Result

Pass. A full approved design spec is recorded for T300 and is ready for user review before implementation planning.

## Conclusion

The next step is implementation planning based on this spec, with explicit profiling checkpoints for 4096 env throughput and staged backend integration (`factory -> manager -> planner core -> adapter -> tests`).

## Follow-Up

- User review gate for the written spec.
- After user approval, produce a dedicated implementation plan (no code changes yet in this stage).

## Git Refs

- Baseline Ref: working tree on top of `130c635`
- Candidate Ref: documentation-only working tree update
- Key Files:
  - [../../docs/superpowers/specs/2026-05-11-unified-dense-mpc-backend-design.md](../../docs/superpowers/specs/2026-05-11-unified-dense-mpc-backend-design.md)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)
  - [../todo.md](../todo.md)
  - [index.md](index.md)
