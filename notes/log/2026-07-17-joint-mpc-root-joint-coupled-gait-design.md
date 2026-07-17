# Joint MPC Root-Joint Coupled Gait Design

- Purpose: record the approved Chinese HTML design for true root-joint contact coupling, foot-leading-root startup, and scenario-metric joint acceptance.
- Stage: T302v.7 design; no planner behavior changed.
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md).
- Baseline Ref: `e5516c8`.
- Candidate Ref: design working tree.
- Key File: `docs/superpowers/specs/2026-07-17-joint-mpc-rti-root-joint-coupled-gait-design.html`.

## Design Decisions

- Explicitly inherits the 2026-07-15 GPU RTI design, 2026-07-16 joint-order/stance-grounding design, and 2026-07-16 small-obstacle crossing/stop-support continuation.
- Replaces joint-only foot linearization and diagonal/root-separated LQ behavior with a complete root-translation/root-rotation/leg-joint FK Jacobian and nonzero GGN cross blocks.
- Uses scheduled stance world-anchor equality constraints, horizon command progress, command-conditioned touchdown references, and a 20-80ms foot-leading-root startup contract.
- Chooses a root-centered arrowhead/Schur Riccati structure, with a dense 18x18 solver as the correctness baseline; coupling may not be discarded for speed.
- Defines one shared JointMetrics accumulator. Every scenario computes every universal metric; event metrics may be N/A only with explicit applicability and opportunity coverage.
- Preserves all previous crossing, collision, support, grounding, viewer, rolling, field, and dynamic-environment-count gates.
- Stage A now treats the global fixed gait period as an explicit exploration direction instead of preselecting H16: candidates remain in H16-H50, keep `Horizon = 2 * half_cycle_steps`, `dt=0.02`, measured x0, rolling RTI and x1-only publication, and are ranked only after the complete old/new JointMetrics gate.
- Longer periods do not replace the existing solver/loss directions. Trust-region scale, collision warm-start fallback, denser parallel line search and merit/LQ phase consistency remain required investigations.

## Performance Order

- Current idle-GPU realistic baseline remains `7402.55ms/1000`, field `4.5884ms`, MPC `2.8416ms`, full mean `7.4025ms`, P95 `7.4742ms`, peak `1127.79MiB`, nonfinite `0`.
- The old single-cell `4768ms` result remains rejected.
- Stage A first closes all old/new walking and safety metrics.
- Stage A selects one final `H_selected`; Stage B freezes that exact horizon and behavior contract, then optimizes exact field plus coupled MPC to `<=5.0s/1000`. The threshold is unchanged even if the selected horizon is H24/H40/H50.
- Stage B explicitly references `raw/mpx` for batched environment/time-node vectorization, temporal associative-scan TVLQR/Riccati, root-centered state-space factorization, parallel line search and multiple-shooting/KKT organization. MPX remains reference-only and is not a runtime dependency.
- Every performance change reruns Stage A.
- Stage C finally reruns both complete gates from the same final candidate; historical Stage A/Stage B passes cannot substitute for the joint final verification.

## Result

- Chinese HTML design amended and user-approved for Stage A period exploration plus MPX-referenced Stage B closure on the selected horizon.
- HTML parser check passes and placeholder/contract review found no unresolved items.
- No production code or test implementation changed.
