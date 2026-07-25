# Joint MPC Cross Common-Metric Gate

- Purpose: amend the small-obstacle crossing design and Task 17 plan so crossing is accepted only when the complete flat gait-quality contract also passes.
- Stage: Task 17 planning boundary; no production code changed in this pass.
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)
- Procedure: reviewed the final design, Task 17 plan, `strict_crossing_event`, `simulate_small_trace`, and `evaluate_trace` behavior.
- Decision: every small crossing trace must pass all `M_common` metrics. Continuing and touchdown-onset stance feet must be on current-refresh normal ground, not small/large/unknown semantic cells, with existing stance gap and penetration limits. Root lateral offset may use a bounded `0.10m` crossing-window threshold while retaining the flat `0.06m` diagnostic; no other common threshold is relaxed.
- Test boundary: the controlled crossing test must consume the formal `evaluate_trace`/`require_small_gate` path. A geometric crossing with stance on a semantic object is a failure.
- Result: design and Task 17 plan updated; implementation remains pending.
- Follow-up: implement the shared common-metric gate and normal-ground stance detector, then rerun the controlled cuboid test and the full small matrix.
- Current Work Ref: `work/joint-mpc-kinematic`
- Key Files: `docs/superpowers/specs/2026-07-22-joint-mpc-rti-perceptive-kinematic-final-design.html`, `docs/superpowers/plans/2026-07-23-joint-mpc-rti-perceptive-kinematic-implementation-plan.md`
