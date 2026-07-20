# Joint MPC RTI Kinematic Flat-Small Design

## Purpose

Record the approved replacement design for `Go2Pvcnn/extension/joint_mpc_rti/` before a new implementation plan is written.

## Scope

- Pure kinematic H30 state trajectory: root pose plus 12 joint positions.
- Fixed 24-frame diagonal trot with 12-frame swing and 12-frame stance.
- One vectorized `[B,31,18]` nominal construction with shifted rolling warm start.
- Seven fixed losses, joint-bound/trust-region KKT, H30/32 associative scan, and five-candidate parallel line search.
- First behavior milestone restricted to flat and small-obstacle tests with one shared JointMetrics contract.
- Inherited compile/process watchdog limits and unchanged Stage B `1024 x H30 x 1000 <=5.0s` contract.

## Result

The Chinese HTML design was written at:

- [../../docs/superpowers/specs/2026-07-20-joint-mpc-rti-kinematic-flat-small-obstacle-design.html](../../docs/superpowers/specs/2026-07-20-joint-mpc-rti-kinematic-flat-small-obstacle-design.html)

The document explicitly supersedes the old 15+15 adaptive-contact/recovery production architecture while retaining its useful metrics and resource-incident evidence. No production code or tests were changed in this design step.

## Verification

- Placeholder scan: no `TODO` or `TBD`.
- Contract scan: H30, gait 24/12+12, five alphas `(1, 0.5, 0.25, 0.125, 0)`, seven losses, flat/small gates, and Stage B five-second requirement are present.
- `xmllint --html --noout` returned exit code 0; its legacy HTML parser warns that the HTML5 `main` tag is unknown.

## Next Step

User reviews the written design. After explicit approval, write a replacement implementation plan; do not continue the old 2026-07-17 plan as-is.

## Approved Amendment

After initial approval, the user identified that semantic differentiability must explicitly come from convolution. The design now requires fixed grouped Gaussian `conv2d` over small/large masks and semantic-weighted height, smooth occupancy `1-exp(-gain*mass)`, propagated class height, fixed Scharr XY gradients, and bilinear trajectory queries. Raw semantic ids remain acceptance-detector inputs only and may not create hard optimizer branches.
