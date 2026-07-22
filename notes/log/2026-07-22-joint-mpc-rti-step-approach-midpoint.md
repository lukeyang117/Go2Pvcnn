# Joint MPC RTI Step Approach Midpoint Verification

> Superseded runtime status: a compact one-environment PhysX fixture configuration later removed the CUDA OOM blocker. The real midpoint viewer then ran and rejected this candidate; see [Step architecture closure](2026-07-22-joint-mpc-rti-step-architecture-closure.md).

## Purpose

Verify the single bounded midpoint profile for the Task 14 final-swing Step approach, and record whether the representative real viewer can execute it.

## Stage

- Planner: `extension/joint_mpc_rti`
- Plan: Task 14 small-obstacle behavior gate
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)

## Candidate

The existing Step family uses `step_approach_edges=3`. For phases `9,10,11`, the approach target is one remaining-edge interpolation toward the existing touchdown target. The residual energy weights are `[0.5, 0.75, 1.0]`; earlier swing nodes have zero approach weight. No loss family, variable, KKT row, candidate, repair, or SQP iteration was added.

## TDD Verification

- RED: the exact midpoint profile test failed against the prior `[1/3, 2/3, 1]` implementation.
- GREEN: `test_trajectory_losses.py -k step_approach`: `3 passed`.
- GREEN: full `test_trajectory_losses.py`: `25 passed`.

## Viewer Verification

Two attempts ran the same actual-state representative `small_forward` viewer case with H30, one RTI, direct published-x1 playback, and the shared detector:

```text
JOINT_MPC_VIEWER_REPRO_SCENARIO=small
JOINT_MPC_VIEWER_REPRO_CASES=small_forward
JOINT_MPC_VIEWER_REPRO_CYCLES=160
... joint_mpc_rti_viewer_reproduction_probe.py --device cuda:1
```

Both attempts were blocked before planner behavior was measured by Isaac/PhysX GPU allocation. The first failed allocating a `640 MiB` contact-pairs buffer with roughly `1.7 GiB` free on GPU 0. The second failed creating the PhysX scene while allocating `256 MiB` with other workloads leaving insufficient contiguous capacity on GPU 1, then had no active physics scene. No viewer pass/fail claim is made and no output report is accepted as evidence.

## Conclusion

This log records the original OOM attempts only. The later real run measured joint step `0.32291rad`, stance slip/anchor `1.2969mm`, stationary ratio `0.9889`, root velocity error `0.22145m/s`, and clearance `-2.080mm`; the candidate is rejected. Ranked flat/small and the formal `29,640` matrix remain blocked.

## Git Refs

- Baseline Ref: `41f1b18`
- Candidate Ref: `41f1b18` plus current Task 14 working tree
- Key Files: `losses/step.py`, `tests/joint_mpc_rti/test_trajectory_losses.py`, Task 14 design and plan
