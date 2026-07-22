# Joint MPC RTI Affine Support KKT

## Purpose

Implement and evaluate the approved published-x1 stance-support constraint without changing the full 18D root-joint variable, one-SQP/RTI lifecycle, five alphas, or seven loss families.

## Implementation

- `TrajectoryQp` now carries `support_jacobian[B,6,18]` and `support_target[B,6]`.
- The two x1 stance feet use complete root/RPY/joint FK Jacobians and the affine equality `A_s delta z_1 = b_s`.
- Dense KKT appends the six nonzero target rows.
- H30 scan applies the fixed-rank Schur correction `lambda=(A_sY)^-1(A_sd_0-b_s)`.
- Active-set refinement starts from the minimum-norm affine-feasible seed so feasible-step interpolation preserves the equality.
- Line search retains exactly five alphas and adds exact-FK `0.5mm` XY filtering for continuing stance only.
- Continuing support keeps persistent XY and refreshes z from terrain height plus `foot_contact_offset`; touchdown onset uses the current touchdown reference.

## TDD And Regression

- RED proved missing affine target, dense affine feasibility, exact-FK filtering, touchdown-onset target selection, continuing-stance filter masking, and grounded support z.
- Focused QP/scan/line-search/RTI/backend union: `63 passed`.
- Earlier scan/QP compile regression: `42 passed`, including CUDA B1 compile smoke; only the existing TF32 warning was emitted.
- `git diff --check`: pass.

## Real Viewer Evidence

All runs used the same actual-state S4 sphere `small_forward` case on `cuda:1`, direct published-x1 playback, real scanner field, shared detector, and 49 executed RTI cycles.

| Candidate | Joint | Stance slip / anchor | Root velocity | Clearance | Penetration | Valid | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| homogeneous support | `0.43369rad` | `1.442/1.090mm` | `0.20893m/s` | `-3.333mm` | `21.890mm` | `1.00` | reject |
| affine, measured persistent xyz | `0.20128rad` | `0.175/0.211mm` | `0.19428m/s` | `+3.680mm` | `4.621mm` | `0.92` | reject |
| affine, grounded z + continuing filter | `0.34094rad` | `0.136/0.181mm` | `0.21931m/s` | `-3.296mm` | approximately `0` | `0.98` | reject |

The final report is `/tmp/joint_mpc_viewer_support_affine_grounded.json`. It passes joint, stance XY, stationary, penetration, collision, semantic, crossing, map, and lifecycle gates, but fails `trajectory_valid_ratio=0.98`, `root_velocity_error=0.21931m/s`, and `swing_surface_clearance_min_m=-3.296mm`.

## Conclusion

The affine support layer fixes the original root-carried x1 stance defect, but a full 3D x1 support correction transfers error into root tracking and swing clearance under the current one-RTI, no-dynamics, direct-root architecture. Stop support-anchor variants here. Ranked small/flat, remaining signed viewers, formal `29,640`, and Stage B remain blocked pending a new explicit architecture decision.

## Git Refs

- Baseline Ref: `41f1b18`
- Candidate Ref: `41f1b18` plus current Task 14 working tree
- Reports: `/tmp/joint_mpc_viewer_support_kkt.json`, `/tmp/joint_mpc_viewer_support_affine.json`, `/tmp/joint_mpc_viewer_support_affine_grounded.json`
