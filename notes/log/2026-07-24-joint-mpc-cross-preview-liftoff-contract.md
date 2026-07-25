# Joint MPC Cross Preview Liftoff Contract

## Purpose

Localize the first controlled small-cross invalid refresh after the complete flat/common metric gate was added.

## Stage

Final perceptive-kinematic plan Task 17, controlled center-cuboid crossing.

## Related Todo

- [T302v](../todo/T302v-joint-mpc-rti-gpu.md)

## Procedure

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q \
  Go2Pvcnn/tests/joint_mpc_rti/test_perceptive_plan.py \
  Go2Pvcnn/tests/joint_mpc_rti/test_nominal.py
```

Then run the controlled `vx=0.2`, center cuboid trace for 90 warm refreshes and inspect refresh 59 with `evaluate_nodes` and `evaluate_swept_intervals`.

## Result

- Selector/nominal focused regression: `84 passed`.
- Controlled trace remains invalid from refresh 59; valid/publish ratio is `0.758242` over 91 recorded nodes.
- Primary touchdown is valid and has one safe FR candidate; preview reports five safe FR candidates.
- Nominal nodes are finite and inside joint limits; maximum H30 joint edge is `0.57621rad`, below the solver hard `0.6rad` edge limit but above the acceptance `0.35rad` continuity gate.
- Node collision clearance is nonnegative. The only negative clearance is swept edge 25: foot `-0.0139228m`, calf `-0.00992277m`.
- Edge 25 is preview liftoff. Selector samples a continuous IK foot curve, while publication and acceptance interpolate root/joints between discrete nodes and run FK on those interpolants. The two swept geometries are not equivalent.
- Complete cross/common stance checks remain active: no actual part collision, no stance/touchdown on small semantic, and no stance semantic bypass occurred before publication stopped.

## Conclusion

The remaining controlled failure is not missing flat metrics, stale-map reuse, candidate count, LQ weight, or line-search alpha. It is an owner-level trajectory contract mismatch between selector collision checking and the discrete joint trajectory that is actually published. The next implementation must make liftoff generation and selector safety use the same discrete joint-interpolation model, while preserving all hard gates.

## Git Refs

- Baseline Ref: `5250bf1`
- Candidate Ref: uncommitted `work/joint-mpc-kinematic`
- Key Files: `model/perceptive_plan.py`, `model/nominal.py`, `test_perceptive_plan.py`, `test_nominal.py`
