# Joint MPC RTI Phase-12 Touchdown Amendment Rejected

## Purpose

Test whether defining phase 11 as the final swing sample and phase 12 as the unique touchdown/first-stance node removes the representative warm-nominal joint kink.

## Stage

- Planner: `extension/joint_mpc_rti`
- Plan: Task 14 representative viewer blocker
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)

## Procedure

- Changed `swing_tau` experimentally from `phase/11` to `phase/12`.
- Moved the existing Step event experimentally from phase 11 to phase 12.
- Kept H30, one SQP/RTI, seven losses, five alphas, Contact, Terrain, KKT, filters, warm start, and all scalar defaults unchanged.
- Added RED tests for gait tau, Step event placement, and cold nominal edge interpolation; all three failed before implementation and passed after it.
- Ran focused CPU regression, ranked small on `cuda:1`, and the real S4 sphere `small_forward` viewer on `cuda:1`.

## Results

- Experimental focused union: `67 passed`.
- Ranked small: `6/7`; only pure negative yaw failed with `joint_step_max_rad=0.353788` over the `0.35` gate.
- Real viewer: 49 cycles, strict crossing and collision/semantic/map/lifecycle gates remained green.
- Real viewer joint step worsened from the production baseline `0.419717rad` to `0.446783rad` at RR thigh phase `11->12`.
- The experimental nominal already contained `-0.458563rad`; same-cycle QP corrected only `+0.011781rad`.
- Phase-11 nominal foot remained `91.88mm` above its surface and moved `(113.91,54.65,-90.90)mm` into phase 12.
- Root velocity remained red at `0.204023m/s`; swing clearance regressed to `-3.831mm`.
- After restoring production semantics, the focused gait/nominal/loss/QP/RTI union is `66 passed`; `git diff --check` passes.

## Conclusion

Reject phase-12 Step and `tau=phase/12`. Removing phase-11 Step pressure lets Terrain keep the final swing node high and makes the final descent steeper. Production remains `tau=phase/11`, phase-11 Step, stance-only Contact ground from phase 12, weak future onset, original Terrain, `step_z=4`, `contact_future_onset_xy=1`, and `joint_velocity_limit=30rad/s`.

The next design checkpoint is a continuous final-swing touchdown-approach residual inside the existing Step family. Its exact reference formula must be reviewed before implementation. Do not launch the full `29,640` matrix while the representative viewer is red.

## Git Refs

- Baseline Ref: `41f1b18`
- Candidate Ref: `41f1b18` plus current Task 14 working tree
- Key Files: `model/gait_schedule.py`, `losses/step.py`, `losses/contact.py`, Task 14 design and plan
