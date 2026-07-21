# Joint MPC RTI Command Scale 0.50 After Liftoff Fix

## Purpose

Run the first rolling parameter point selected from post-fix cold geometry: `command_scale=0.5` with all trust regions and losses at default.

## Stage

Task 13 ranked flat parameter diagnosis. Read-only config variant; production config unchanged.

## Result

- Monitor completed in `12.54s`; task GPU memory about `2.09GiB`; no trigger.
- Zero remains `25/25` valid; drift `1.40e-4m` and clearance `-0.060mm` fail.
- Forward is `22/25` valid; backward is `24/25` valid, a substantial improvement over the pre-liftoff-fix default cold behavior.
- Root velocity error fails badly at `0.447m/s` forward and `0.509m/s` backward because the nominal supplies half speed and default root trust is only `0.005m/node`.
- Both nonzero cells still have stance, margin/fallback, lead, yaw, and clearance failures.

## Conclusion

Scale 0.5 is a feasible cold starting point but cannot track command under the default root trust. Test the informed combination `command_scale=0.5` plus `root_position_trust=0.01m/node`, which permits one RTI to recover the missing `0.5m/s` per interval.

## Git Refs

- Baseline ref: working tree on `724a1c3`, command scale `1.0`
- Candidate ref: read-only config variant `command_scale=0.5`
- Key files: `config.py`, `model/nominal.py`, `solver/trajectory_qp.py`
