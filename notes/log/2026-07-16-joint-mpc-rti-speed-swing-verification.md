# Joint MPC RTI Speed And Swing Verification

- Purpose: verify root velocity tracking and swing-leg lift after the joint-order and stance-grounding fixes.
- Stage: joint MPC RTI planner, rolling x1, direct Isaac viewer playback.
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md).
- Baseline Ref: `09957d4`.
- Candidate Ref: `09957d4`.
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/planner.py`, `Go2Pvcnn/extension/viz/go2_foostep_planner.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_command_dynamics.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_behavior.py`.

## Procedure

- Planner-side H16 and 32 rolling x1 evaluation in `env_isaacsim` on CUDA.
- Real viewer direct-playback evaluation with `dt=0.02`, 16 cycles per command.
- Commands: standstill, forward `0.1/0.4`, backward `-0.25`, lateral `+/-0.25`, yaw `0.5`, mixed `0.2/0.15/0.3`, mixed reverse `0.35/-0.2/-0.35`.

## Results

- Planner root body velocity error: `0.0 m/s` for all nine commands over H16 and rolling x1.
- Planner world-frame linear velocity error: `0.0 m/s` for all nine commands.
- Planner yaw-rate error: `0.0 rad/s` in the tensor path.
- Real viewer yaw-rate pose difference: maximum `1.76e-6 rad/s`.
- Real viewer swing surface clearance (foot-center height minus terrain height minus `0.022m`): minimum `-0.00124m` at the discrete touchdown boundary, maximum `0.06392m`.
- Real viewer swing peak lift by leg stayed positive for every leg; peak range across commands was approximately `0.0340-0.0641m` above the contact surface.
- Each leg contributed eight swing samples in the 16-cycle real viewer probe.
- No nonfinite planner state, terrain penetration in the planner trajectory, or stance grounding regression was observed.

## Important Measurement Boundary

`_apply_direct_playback_to_robot()` writes the planned root pose and explicitly writes a zero root velocity to Isaac. Therefore Isaac's instantaneous `root_lin_vel_w` is not a valid velocity-tracking metric for this viewer mode. The valid planner metric is the RTI control/root trajectory velocity; the viewer metric is pose displacement between written x1 frames. The planner velocity contract passes exactly.

## Tests

```text
Go2Pvcnn/tests/joint_mpc_rti/test_command_dynamics.py
Go2Pvcnn/tests/joint_mpc_rti/test_behavior.py
17 passed in 7.21s
```

## Conclusion

Root velocity tracking is correct in the joint MPC RTI trajectory/control path for all tested directions and magnitudes. Swing legs lift by roughly `34-64mm` above the physical contact surface; the small negative boundary residual is a touchdown discretization effect, not a mid-swing failure.
