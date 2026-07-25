# Joint MPC Cross Stance/Common And SDF Follow-up

## Purpose

Verify that small crossing inherits the complete flat/common gate, including
current-world-map stance grounding, and repair the line-search small-footprint
query without weakening any safety threshold.

## Stage

- Final perceptive-kinematic plan Task 17
- Related todo: T302v joint MPC RTI GPU

## Changes

- `solver/line_search.py` now queries `context.terrain` with `query_world()` for
  `small_distance_m`; the perceptive query does not expose signed small SDF.
- The packed five-alpha query uses the current terrain field from `LossContext`.
- Common thresholds remain unchanged: stance gap `<=0.012m`, penetration
  `<=0.001m`, forbidden stance/touchdown semantic `0`, plus anchor/slip,
  continuity, command/root, lifecycle, numerical, KKT, and collision gates.

## Verification

```text
test_line_search_v2.py: 7 passed
focused line-search plus stance/ground/semantic metrics: 14 passed
controlled center-cuboid 144 refresh: failed strict_cross_success only
```

Controlled trace:

```text
valid_ratio = 1.0
root x = 0.0 -> 0.4634197m
touchdown_on_small = 0
stance_on_small = 0
stance_on_forbidden_semantic = 0
touchdown_on_forbidden_semantic = 0
all applicable flat/common metrics = passed
strict_cross_success = 0
over_xy = false
over_z = false
direction_ok = false
before/after/land_ok/body_ok = true
```

## Root Cause Evidence

- The selector commits leg 1 to crossing near refresh 50 with about `0.081m`
  inward swing offset, but the commitment is lost at the current-swing KFE
  boundary (`0.0494rad` margin versus unchanged `0.050rad`).
- Retarget expansion can preserve commitment, but exact line search then rejects
  a published joint-linear swing edge with foot clearance `-1.109mm` while all
  stance soles remain safe.
- Selector continuous-IK geometry and nominal discrete joint-linear publication
  therefore still disagree.
- Current-swing retarget expansion, an extra outward candidate, and a higher
  apex reserve were reverted because they introduced publish-stop regressions.

## Conclusion

Stance/common metric integration is active and green in the controlled small
trace. Task 17 remains red because no actual swing-foot event intersects the
small footprint. The next correction must make selector validation and the
nominal/published discrete root, joint, and swing path identical. Do not weaken
collision, stance-ground, semantic, or joint thresholds.

## Key Files

- `Go2Pvcnn/extension/joint_mpc_rti/solver/line_search.py`
- `Go2Pvcnn/extension/joint_mpc_rti/model/perceptive_plan.py`
- `Go2Pvcnn/extension/joint_mpc_rti/model/nominal.py`
- `Go2Pvcnn/tests/joint_mpc_rti/joint_metrics.py`
- `Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py`
