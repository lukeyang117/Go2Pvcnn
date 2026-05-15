# T300e MPC Continuous Swing Window Implementation

## Purpose

Record the implementation and local verification of the T300e redesign for `Go2Pvcnn/extension/batch_mpc_planner`: continuous swing-window contact timing, scanner-driven terrain/semantic losses, IK-aware feasibility losses, and removal of planner-owned foothold memory plus output-side foot grounding.

## Stage

Production MPC backend implementation and focused local verification.

## Related Todo

- [T300e continuous swing-window plan](../todo/T300e-mpc-continuous-swing-window-plan.md)
- [T300 unified dense MPC backend](../todo/T300-unified-dense-mpc-backend.md)

## Input Conditions

- Candidate ref: working tree on top of `65f0d991c201ed50df8be1fa7ad62b3525dae20a`.
- Active target: `Go2Pvcnn/extension/batch_mpc_planner/`.
- Runtime IsaacLab environment was not available in the current shell: `python -c "import isaaclab"` failed with `ModuleNotFoundError`.
- Raw planner note path `raw/kinematic_footsteps/notes/` is absent in this checkout; repository planner notes under `notes/human/` were used as the active pre-read source.

## Implementation Summary

- Replaced `contact_logits` with continuous `swing_center_raw[B,4]` and `swing_width_raw[B,4]`.
- Decode now exposes `swing_center`, `swing_width`, `swing_start`, `swing_end`, `swing_prob`, and `contact_prob`.
- Rebuilt nominal generation around body-frame command integration, world-frame foot trajectories, random diagonal prior, swing-time root-frame touchdown targets, and terrain-height target z.
- Added scanner terrain helpers: `height_at`, `semantic_at`, `slope_at`, and `support_at`, including scanner pose/yaw support.
- Reworked loss registry to use scanner height/semantic information instead of post-grounding or true obstacle positions.
- Added/activated stance-ground, swing-clearance terrain, touchdown surface/semantic, semantic obstacle, swing direction, swing-center urgency, IK joint limit, IK/FK residual, root-foot center, and support-plane roll/pitch losses.
- Removed `MpcFootholdMemory`, manager/viewer foothold memory paths, and `_ground_contact_feet_to_terrain`.
- Planner output now keeps optimized `foot_pos` and derives `joint_angles` by IK from optimized root/foot trajectories.

## Subagent Review Integration

The requested subagent review found issues in scanner pose handling, circular averaging, terrain/semantic urgency proxy, swing obstacle clearance, hot-path `.item()` counters, and tracking weights. The final local code includes targeted coverage for those areas:

- scanner pose/yaw sampling through `MpcPlannerTerrain.sensor_pos_w/sensor_yaw`
- circular pair averaging in diagonal/urgency losses
- urgency proxy from touchdown terrain height/slope/semantic
- swing semantic obstacle penalty that depends on clearance
- manager counter host sync guarded behind diagnostics counters
- tracking velocity/yaw weights honored

## Commands And Results

```bash
python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q
```

Result: `32 passed in 3.88s`.

```bash
python -m py_compile \
  Go2Pvcnn/extension/batch_mpc_planner/config.py \
  Go2Pvcnn/extension/batch_mpc_planner/terrain.py \
  Go2Pvcnn/extension/batch_mpc_planner/nominal.py \
  Go2Pvcnn/extension/batch_mpc_planner/variables.py \
  Go2Pvcnn/extension/batch_mpc_planner/optimizer.py \
  Go2Pvcnn/extension/batch_mpc_planner/planner.py \
  Go2Pvcnn/extension/batch_mpc_planner/manager.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/gait_coupling.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/kinematics.py \
  Go2Pvcnn/extension/batch_mpc_planner/losses/tracking.py \
  Go2Pvcnn/extension/viz/go2_foostep_planner.py \
  Go2Pvcnn/tests/test_batch_mpc_backend.py \
  Go2Pvcnn/tests/test_mpc_runtime_headless.py \
  Go2Pvcnn/tests/mpc_touchdown_sequence_probe.py \
  Go2Pvcnn/tests/mpc_root_cause_probe.py \
  Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py
```

Result: exit code `0`.

```bash
python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py -q --collect-only
```

Result: `12 tests collected in 1.69s`.

```bash
python -c "import isaaclab"
```

Result: exit code `1`, `ModuleNotFoundError: No module named 'isaaclab'`.

## Key Metrics

- Focused backend suite: `32 passed`.
- Targeted compile check: exit `0`.
- Runtime test collection: `12` tests collected.
- IsaacLab runtime execution: blocked by missing package in current shell.

## Result

Local implementation verification passed. Runtime acceptance in IsaacLab remains open.

## Conclusion

The active MPC backend now follows the T300e design locally: contact timing is a continuous per-leg swing window, foot placement losses sample scanner height/semantic maps, touchdown losses are loss-side instead of output-side grounding, joint losses are derived from IK over optimized root/foot, and each replan reads current IsaacLab state rather than planner-owned foothold memory.

## Follow-Up

- Run T300e runtime probes after activating an IsaacLab-capable environment:
  - `MPC_ROOT_CAUSE_CYCLES=4 ... python Go2Pvcnn/tests/mpc_root_cause_probe.py`
  - `MPC_YAW_GAIT_CYCLES=8 ... python Go2Pvcnn/tests/mpc_yaw_gait_failure_probe.py`
  - selected `Go2Pvcnn/tests/test_mpc_runtime_headless.py` MPC cases
- Compare yaw actual-air ratio, IK/FK contact error, touchdown semantic collision count, and continuous swing-window behavior against T300d probe logs before tuning weights.

## Git Refs

- Baseline Ref: `65f0d991c201ed50df8be1fa7ad62b3525dae20a`
- Candidate Ref: working tree on top of `65f0d991c201ed50df8be1fa7ad62b3525dae20a`
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/variables.py](../../Go2Pvcnn/extension/batch_mpc_planner/variables.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py](../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/terrain.py](../../Go2Pvcnn/extension/batch_mpc_planner/terrain.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/registry.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/test_batch_mpc_backend.py](../../Go2Pvcnn/tests/test_batch_mpc_backend.py)
