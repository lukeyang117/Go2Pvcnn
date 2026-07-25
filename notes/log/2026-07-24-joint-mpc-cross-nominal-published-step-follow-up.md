# Joint MPC Cross Nominal and Published-Step Follow-up

## Result

- Swing warm retarget now blends foot references in Cartesian space before analytic IK.
- Swing descent adds a zero-endpoint landing buffer controlled by `swing_landing_buffer_m`.
- Line search has an independent `published_joint_step_limit_rad=0.35` gate for the measured `x0 -> x1` edge.
- Focused selector/nominal/line-search tests pass after the shape correction.
- Controlled cuboid crossing remains red: collision counts are zero, but selector/preview candidate intersections become empty around refresh 71 and the planner stops; strict crossing is not passed.

## Next owner

Nominal-level candidate ownership must evaluate a fixed small set of primary/preview candidate combinations with the exact nominal hard-safety contract when selector approximations disagree. Do not weaken common flat metrics or published collision gates.
