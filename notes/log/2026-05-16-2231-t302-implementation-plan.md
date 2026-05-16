# T302 MPC Collision Implementation Plan

## Purpose

Record that the T302 body/leg height-field collision safety design has been converted into a detailed TDD implementation plan in the branch page.

## Stage

- production planner path: `Go2Pvcnn/extension/batch_mpc_planner`
- test path: `Go2Pvcnn/tests`
- notes path: `notes/todo/T302-mpc-body-leg-height-field-collision-safety.md`

## Related Todo

- [../todo/T302-mpc-body-leg-height-field-collision-safety.md](../todo/T302-mpc-body-leg-height-field-collision-safety.md)

## Procedure

- Used the `writing-plans` skill as requested.
- Read repository entry constraints and active planner notes.
- Wrote `## T302 Implementation Plan` directly into the T302 branch page.
- Updated the dashboard and log index so the next working session enters through T302b instead of the completed design gate.

## Key Contents

- `T302c.1`: GPU FK helper returns foot, knee, and shank sample world points.
- `T302d.1`: height-field root/body, knee/shank, and swing-foot collision losses.
- `T302d.2`: semantic touchdown and stance obstacle rejection.
- `T302e.1`: high-small/large obstacle linear and yaw tracking-risk scaling.
- `T302d.3`: config overrides and static GPU hot-path guardrails.
- `T302f.1`: backend regression suite.
- `T302f.2`: real IsaacLab headless COBBLESTONE acceptance.
- `T302f.3`: flat semantic low-small crossing, high-small avoidance, large/yaw risk acceptance.
- `T302f.4`: final T300e regression and T302 verification notes.

## Result

- Plan recorded.
- Implementation not started.
- No production code changed.

## Checks

- Red-flag scan over the T302 branch page, dashboard, log index, and this log: exit code `1`, meaning no matches.

- Diff whitespace check:

```bash
git diff --check
```

Result: exit code `0`.

## Follow-Up

- Execute T302 plan task-by-task with TDD.
- Prefer subagent-driven execution with disjoint write scopes and main-agent review between slices.

## Git Refs

- Baseline Ref: `3843555`
- Candidate Ref: working tree on top of `3843555`
- Key Files:
  - [../todo/T302-mpc-body-leg-height-field-collision-safety.md](../todo/T302-mpc-body-leg-height-field-collision-safety.md)
  - [../todo.md](../todo.md)
  - [index.md](index.md)
