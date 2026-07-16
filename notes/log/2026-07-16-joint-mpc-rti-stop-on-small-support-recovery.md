# Joint MPC RTI Stop-On-Small Support Recovery

- Purpose: fix sustained floating when the robot walks onto a crossable small object and the command becomes zero, while preserving strict crossing and zero per-part collision.
- Stage: `extension/joint_mpc_rti` rolling x1, RTI warm start, semantic/stance loss, native-shape verification, and real viewer.
- Related todo: [T302v.5](../todo/T302v-joint-mpc-rti-gpu.md).
- Baseline Ref: `c2093a9`.
- Candidate Ref: `e2eb672`.
- Key Files: `config.py`, `planner.py`, `losses/contact.py`, `losses/semantic.py`, `losses/rollout_objective.py`, `small_obstacle_stop_probe.py`, `small_obstacle_crossing_probe.py`.

## Root Causes

1. `small_object_foot_over` kept substantial pressure too late in swing, so rolling one-step execution repeatedly published elevated feet near swing-to-stance handoff.
2. Shifted RTI warm starts retained the old root velocity for up to seven x1 cycles after the command changed; the measured `0.028m` drift was exactly `7 * 0.2m/s * 0.02s`.
3. The old stop acceptance covered only five offsets in the implementation draft; final verification restores 13 offsets from `-0.24m` through `+0.24m` for 65 native-shape cases.
4. Collision diagnostics used a constant `0.16m` top for all shapes, over-counting sphere/cone links against an outer box instead of the queried local height surface.
5. Flat stance and semantic support need different continuous emphasis: support viability must remain strong near small objects, while ordinary stance grounding needs additional far-field strength without changing semantic line-search choices.

## Implementation

- Concentrate foot-over at mid-swing with a phase exponent and add continuous late-swing safe landing in both merit and LQ/GGN.
- Add harmonic/smooth-min `stance_support_viability` in merit and LQ/GGN, using signed-distance safety weights.
- Separate continuous margins for touchdown avoidance, safe landing, and support safety; each concept uses the same margin in merit and LQ/GGN.
- Re-base shifted root controls to the current command reference every RTI frame while preserving shifted joint-control warm starts. This is command-agnostic and has no zero-command gate.
- Add sharp continuous far-field stance-grounding strength for empty/flat space; the gain smoothly vanishes in the small-object neighborhood.
- Increase existing calf/thigh clearance and stance XY-lock weights; no new cross mode, specified leg, shape branch, projection, snapping, or repair was added.
- Native-shape collision probes now use each sample's queried local `height_w` rather than a constant outer-box top.

## Final Functional Metrics

Final 65-case stop matrix, five shapes x 13 stop offsets, 32 hold cycles:

```text
support recovery: 65/65 = 100%
maximum total zero-support frames per case: 2
maximum consecutive zero-support frames per case: 1
maximum stop root XY drift: 0m
stance-on-small frames: 0
foot/calf/thigh/base collision frames: 0/0/0/0
invalid count: 0
```

Final crossing matrix, five shapes x three speeds with 160 rolling steps:

```text
overall cross success: 100%
minimum shape-speed success: 100%
foot/calf/thigh/base collision frames: 0/0/0/0
stance-on-small frames: 0
invalid count: 0
```

Real IsaacLab nine-command viewer, eight rolling cycles:

```text
passed: true
joint position/velocity order error max: 0
stance ground gap max: 0.0114493668m
joint step max: 0.185368538rad
actual-vs-planner foot error max: 5.23877e-7m
standstill root XY drift: 2.08953e-11m
standstill root yaw drift: 2.45247e-12rad
```

## Regression

```text
Go2Pvcnn/tests/joint_mpc_rti: 117 passed
legacy backend/parametric/eval/viewer: 213 passed
```

## Performance Boundary

Per user direction, the signed-field `1024 x H16 x 1000` synchronous field+MPC `<5s` gate remains open until an uncontended GPU is available. Functional metrics and collision thresholds were not relaxed for performance.

## Conclusion

The original sustained floating is removed without a zero-command/semantic hard gate. The planner now stops without root drift, keeps at least one real safe support through handoff, preserves strict crossing, and reports zero collision frames for foot/calf/thigh/base on the native-shape matrices.

## Git Refs

- Last Feature Commit: `e2eb672`.
- Last Verified Commit: `e2eb672`.
