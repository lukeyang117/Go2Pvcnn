# Joint MPC RTI Scale 0.50 Plus Root Trust 0.01

## Purpose

Test the informed combination that starts from the post-fix feasible half-speed cold nominal and allows one RTI up to `0.01m/node` root-position correction.

## Stage

Task 13 ranked flat parameter diagnosis. Read-only config variant; production config unchanged.

## Result

- Monitor completed in `11.65s`; task GPU memory about `2.09GiB`; no trigger.
- Backward becomes `25/25` valid with joint margin `0.154rad`, joint step `0.214rad`, and all stance metrics passing.
- Forward improves to `23/25` valid; yaw passes, but joint margin is `0.034rad` and stance slip/penetration remain slightly above threshold.
- Root velocity errors remain `0.358m/s` forward and `0.367m/s` backward; root-position trust alone does not make the optimizer consume the full available correction.
- Backward root-step jump reaches `0.0779m`, above the `0.05m` gate.

## Conclusion

This combination materially improves feasibility but is not adoptable as-is. Locate the two forward invalid frames before increasing command weight; if they are constraint-bound, stronger command pressure would be counterproductive.

## Git Refs

- Baseline ref: working tree on `724a1c3`
- Candidate ref: read-only config variant `command_scale=0.5`, `root_position_trust=0.01`
- Key files: `config.py`, `model/nominal.py`, `solver/trajectory_qp.py`, `losses/command.py`
