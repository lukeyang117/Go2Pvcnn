# Joint MPC RTI Mixed Swing-Floor Viewer

## Scope

- Branch: `work/joint-mpc-kinematic`
- Backend: `joint_mpc_rti`, pure kinematic H30
- Scenario: actual-state S4 sphere `small_forward`
- Device: `cuda:1`
- Report: `/tmp/joint_mpc_viewer_mixed_swing_floor.json`
- Execution boundary: representative viewer only; ranked/formal not run

## Architecture Under Test

The published-x1 fixed-rank KKT has six rows with mixed semantics:

```text
4 rows = two stance feet, world XY
2 rows = two swing feet, world Z floor
```

The swing target uses the same effective surface as Terrain:

```text
safe_z = h_effective(candidate_xy) + foot_contact_offset + buffer
J_foot_z delta_z1 = max(safe_z - nominal_foot_z, 0)
```

The fourth and final line-search filter is `published_kinematics`; it combines continuing-stance exact-FK XY anchors with swing exact-FK Z floors. The ABI remains one QP, one RTI, five alphas, four filters, and seven loss families.

## Verification

- Initial architecture RED: `5 failed`
- Core GREEN: `5 passed`
- Candidate-height and original-bound coverage: `2 passed`
- Safe-Z diagnostic RED: `2 failed`
- Safe-Z diagnostic GREEN: `2 passed`
- Combined focused union: `93 passed in 22.85s`
- `py_compile`: exit `0`
- `git diff --check`: exit `0`

## Viewer Evidence

The mixed KKT closes the swing-floor defect at the solver layer:

| Layer | Safe-floor deficit |
| --- | ---: |
| nominal x1 | `+2.4730805mm` |
| full-QP x1 | `-5.2713um` |
| selected x1 | `-5.2936um` |

Positive deficit means below the floor. The `published_kinematics` filter accepts only alpha `1.0`; alphas `0.5`, `0.25`, `0.125`, and `0` are rejected. Alpha `1.0` is selected.

Green representative metrics include:

- swing clearance `+6.219um`
- trajectory validity `1.0`
- joint step `0.320307rad`
- stance XY slip max `0.136290mm`
- stance anchor residual `0.117573mm`
- strict crossing `1.0`
- collision and maximum penetration `0`
- lifecycle `1 cold + 48 warm`, unexpected restart `0`

The sole failed metric is root velocity error `0.2239295m/s` against the `0.2m/s` threshold. Overall `passed=false` is therefore correct.

## Decision

Task 14C's explicit swing-floor contract is behaviorally verified, but the complete representative gate is not closed. Do not run ranked or formal acceptance and do not start another scalar sweep. The next investigation must identify whether root tracking is lost in nominal generation, the full QP under root trust, line-search selection, or transient averaging in the acceptance metric.
