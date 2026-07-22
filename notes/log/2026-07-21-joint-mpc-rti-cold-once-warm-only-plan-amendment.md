# Joint MPC RTI Cold-Once Warm-Only Plan Amendment

## Purpose

Record the user-approved lifecycle boundary and the corresponding redesign of the 2026-07-20 pure-kinematic implementation plan.

## Stage

- T302v pure-kinematic RTI nominal/runtime contract
- Blocks further Task 13 tuning until the lifecycle migration and focused regressions are complete

## Contract

```text
UNINITIALIZED --one cold build--> WARM_ONLY --explicit env reset--> UNINITIALIZED
```

- Each environment may execute cold once after creation or explicit reset.
- Every later control cycle uses rolling warm start.
- Candidate rejection and `alpha=0` retain a finite initialized warm cache.
- Candidate/query validity cannot clear lifecycle initialization.
- Missing, wrong-shape, or nonfinite initialized cache is an explicit invariant fault, never a cold fallback.
- Warm nominal remains limited to shift, root rebase, joint mismatch decay, tail fill, and exact x0 overwrite.

## Plan Changes

- Replace solver-state `valid` lifecycle overloading with `initialized`.
- Redesign Task 4 around per-row cold-once/warm-only tests and invariant failures.
- Extend Tasks 10-11 with alpha-zero cache retention and reset isolation.
- Extend Task 12 and final acceptance with lifecycle counters and zero unexpected restarts.
- Preserve the existing first-edge, stance-anchor, command-loss, root-trust, seven-loss, KKT, scan, and five-alpha amendments.

## Verification

- Documentation diff and placeholder scans only; no production code or runtime behavior changed in this pass.
- `git diff --check` must pass for the amended design, plan, and notes.

## Result

Plan amended. Implementation and tests remain pending.

## Git Refs

- Baseline Ref: `41f1b18`
- Candidate Ref: dirty `work/joint-mpc-kinematic`
- Key Files:
  - `docs/superpowers/specs/2026-07-20-joint-mpc-rti-kinematic-flat-small-obstacle-design.html`
  - `docs/superpowers/plans/2026-07-20-joint-mpc-rti-kinematic-flat-small-obstacle-implementation-plan.md`
  - `Go2Pvcnn/extension/joint_mpc_rti/types.py`
  - `Go2Pvcnn/extension/joint_mpc_rti/model/nominal.py`
  - `Go2Pvcnn/extension/joint_mpc_rti/runtime/warm_start.py`
  - `Go2Pvcnn/extension/joint_mpc_rti/runtime/manager.py`

## Next Step

Implement the lifecycle contract with TDD before resuming the flat behavior gate.
