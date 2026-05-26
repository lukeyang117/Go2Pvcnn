# T302k Parametric MPC Todo And Dashboard Cleanup

## Purpose

Record the approved parametric MPC trajectory direction as the new execution front and clean the dashboard so stale T302h/T302i/T302j loss-tuning leaves do not steer implementation.

## Stage

`extension/batch_mpc_planner` planning / repository memory.

## Related Todo

- [../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md)
- [../todo.md](../todo.md)

## Command / Procedure

- Read repository rules, todo dashboard, active T302h/T302i/T302j pages, recent log index, and the approved spec.
- Created a new T302k branch page with implementation child nodes T302k.1-T302k.8.
- Moved `notes/todo.md` back to dashboard form and made T302k the active execution front.
- Kept T302h/T302i/T302j as context/evidence branches; no archive/delete action was taken.

## Input Conditions

- User approved the parametric trajectory design and requested the implementation plan as a new todo Markdown file with child nodes.
- User also stated that many current todos are stale and should be cleaned so they do not affect execution.
- Approved design commit: `d922eef`.

## Key Metrics

- New branch page: `notes/todo/T302k-parametric-mpc-trajectory-contract.md`.
- New implementation children: `8`.
- Destructive actions: `0`.
- Code changes: `0`.

## Result

Pass for planning and memory grooming.

## Conclusion

T302k is now the active implementation route. T302h/T302i/T302j remain available as evidence, but the next code work should start from T302k.1 parametric geometry helpers instead of continuing old V9/V10/V11/V12 scalar-loss tuning.

## Follow-up

Execute T302k.1 using the implementation plan embedded in [../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md).

## Git Refs

- Baseline Ref: `d922eef`
- Candidate Ref: `working tree on top of d922eef`
- Key Files:
  - [../todo.md](../todo.md)
  - [../todo/T302k-parametric-mpc-trajectory-contract.md](../todo/T302k-parametric-mpc-trajectory-contract.md)
  - [../../docs/superpowers/specs/2026-05-26-mpc-parametric-trajectory-contract-design.md](../../docs/superpowers/specs/2026-05-26-mpc-parametric-trajectory-contract-design.md)
