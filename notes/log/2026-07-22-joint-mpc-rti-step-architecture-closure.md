# Joint MPC RTI Step Architecture Closure

## Purpose

Close the Task 14 final-swing Step experiments with real representative viewer evidence, restore the accepted production baseline, and record the remaining architecture contradiction.

## Stage

- Planner: `Go2Pvcnn/extension/joint_mpc_rti`
- Gate: Task 14 representative S4 sphere `small_forward`
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)

## Runtime Fixture Fix

The previous one-environment viewer inherited training-sized PhysX buffers and failed before planner behavior was measured. `viewer_runtime_diagnostics.py` now applies compact PhysX capacities only for `num_envs <= 16`; large-runtime expansion is unchanged. TDD was RED on the missing helper and GREEN with `test_viewer_reset.py` at `36 passed`. Real Isaac viewer startup then succeeded on `cuda:1`.

## Experiments

All candidates kept H30, one SQP/RTI, five alphas, seven losses, original Terrain, stance-only Contact ground, weak future onset, `step_z=4`, `joint_velocity_limit=30`, and cold-once/warm-only.

| Candidate | Joint step | Stance slip/anchor | Stationary | Root velocity error | Swing clearance | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| trajectory-relative midpoint | `0.32291rad` | `1.2969mm` | `0.9889` | `0.22145m/s` | `-2.080mm` | reject |
| fixed world reference | `0.32251rad` | `1.2939mm` | `0.9889` | `0.22194m/s` | `-2.076mm` | reject |
| body-relative phase 9/10 | `0.32122rad` | `1.2951mm` | `0.9889` | `0.22139m/s` | `-2.053mm` | reject |

The body-relative candidate first passed RED/GREEN root-invariance and world-touchdown tests, then the focused gait/nominal/loss/QP/RTI union at `70 passed`. The real report is `/tmp/joint_mpc_viewer_step_body_relative.json` and has `strict_cross_success=1`, zero semantic/collision failures, exact cold count `1`, and valid published x1, but the five metrics above remain red.

At the worst FR phase `0->1` event, the measured and nominal anchor errors before solving are `0.001mm` and `0.070mm`. The full QP direction increases x1 anchor error to `1.296mm`, and the selected root x correction is `+10mm`, exactly the root trust limit. Removing phase 9/10 Step's direct root Jacobian therefore did not remove the indirect coupling through Smooth, Contact, FK, and the full-horizon QP.

## Production Restoration

The rejected approach code and `step_approach_edges` viewer/config tuning surface were removed. Production Step is restored to the phase-11 world touchdown residual only. The restored focused union, including backend wiring, is `82 passed`; `git diff --check` passes.

## Conclusion

The Step-approach family is exhausted. Under the current direct root optimization, soft Contact, no dynamics, and one-RTI contract, stronger pre-touchdown shaping closes the joint gate only while reopening stance/root/clearance gates. Do not add another Step target/profile/frame, hard constraint, recovery, nominal IK repair, or second SQP. Task 14 remains blocked on an explicit architecture decision; ranked and formal matrices remain forbidden while the representative viewer is red.

## Git Refs

- Baseline Ref: `41f1b18`
- Candidate Ref: `41f1b18` plus current Task 14 working tree
- Key Files: `losses/step.py`, `losses/objective.py`, `solver/linearization.py`, `planner.py`, `viewer_runtime_diagnostics.py`, Task 14 design and plan
