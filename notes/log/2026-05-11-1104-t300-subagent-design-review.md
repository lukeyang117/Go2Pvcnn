# T300 Unified Dense MPC Subagent Design Review

## Purpose

Converge a multi-subagent design review for the unified dense MPC backend spec and check readiness for implementation planning under user constraints:

- GPU-only, no CPU hot path
- async per-env replan at 4096 env scale
- no legacy mode/candidate-table concepts
- contact timing in optimization variables
- new IsaacLab-oriented tests with diagnostics-oracle support
- clear config-class vs implementation-class boundaries

## Stage

Spec review and implementation-readiness assessment for:

- `docs/superpowers/specs/2026-05-11-unified-dense-mpc-backend-design.md`
- `Go2Pvcnn/extension/trajectory_manager_factory.py`
- `Go2Pvcnn/extension/mdp/rewards_reference.py`
- `Go2Pvcnn/extension/viz/go2_foostep_planner.py`

## Related Todo

[T300/T300b](../todo/T300-unified-dense-mpc-backend.md#t300b-subagent-review-convergence-and-spec-hardening-before-implementation-plan)

## Command/Procedure

- Read mandatory notes/rules and planner preread docs.
- Spawn four subagents in parallel:
  - A: requirement-completeness audit
  - B: RL parallelism + 4096 throughput risk audit
  - C (worker): IsaacLab GPU-only testing strategy audit
  - D: config/implementation boundary + file-structure audit
- Main agent cross-checked current code contracts in factory/reward/viewer/manager files.

## Input Conditions

- No runtime code changes in this step.
- No tests/training executed.
- Review target is design-level readiness.

## Key Metrics

- Subagent count: `4`
- Major blockers found: `3` P0 categories
- High-priority spec updates proposed: `>10` concrete clauses/params

## Result

Partial pass with blockers. Direction is correct, but spec is not yet implementation-ready for 4096-env training without additional hard contracts.

## Findings Summary

- P0-1: Contact timing optimization contract is underspecified.
  - Must explicitly require stance/swing timing losses to use differentiable `contact_prob` from `contact_logits`; hard-threshold bool may only be used for export/diagnostics.
- P0-2: GPU-only async replan mechanism is underspecified.
  - Must define fixed-budget dirty scheduling, priority selection, and forbidden host-sync patterns in hot path.
- P0-3: 4096 training profile defaults are too aggressive.
  - Current `horizon=80`, `opt_steps=16`, `replan_interval=10` likely causes excessive dirty load; needs budgeted scheduler + command hysteresis/ramp parameters.
- P1: Test plan currently too coarse.
  - Needs explicit new MPC test matrix (unit/integration/runtime), old-together isolation rules, diagnostics-enabled oracle assertions, and IsaacLab GPU-only acceptance gates.
- P1: Config/implementation boundaries need stronger separation.
  - Add layered config (`runtime/losses/diagnostics/profiles`) and backend-agnostic protocol contracts.
- P1: Runtime acceptance section still labels `env_isaacsim`; should be `env_isaaclab`.

## Conclusion

Move T300 from "spec review gate" to "spec hardening before implementation planning". After patching the spec with P0 clauses and parameter contracts, implementation planning can proceed.

## Follow-Up

- Patch spec with:
  - differentiable contact timing contract
  - GPU-only dirty-budget scheduler contract
  - 4096 profile parameters and command-hysteresis settings
  - expanded IsaacLab GPU-only test matrix and diagnostics-oracle rules
  - config/implementation/protocol boundary section
- Then produce file-by-file implementation plan.

## Git Refs

- Baseline Ref: working tree on top of `130c635`
- Candidate Ref: docs/notes/log update only
- Key Files:
  - [../../docs/superpowers/specs/2026-05-11-unified-dense-mpc-backend-design.md](../../docs/superpowers/specs/2026-05-11-unified-dense-mpc-backend-design.md)
  - [../todo/T300-unified-dense-mpc-backend.md](../todo/T300-unified-dense-mpc-backend.md)
  - [index.md](index.md)
