# Joint MPC RTI Command Scale 0.80 Diagnostic

## Purpose

Re-test approved nominal `command_scale=0.8` after infeasible fallback status and universal trajectory validity were corrected.

## Stage

Task 13 ranked flat parameter diagnosis. Read-only config variant; production config unchanged.

## Result

- Monitor completed in `12.12s`; task GPU memory about `2.09GiB`; no trigger.
- Forward remains `13/25` valid (`trajectory_valid_ratio=0.52`).
- Backward improves to `22/25` valid but still fails margin, tracking, stance, lead, clearance, and alpha-zero ratio.
- Zero is unchanged and still fails drift and clearance.

## Conclusion

Do not adopt `command_scale=0.8`. The cold foot placement uses raw command multiplied by the separate `step_reference_scale`; reducing root integration alone does not remove the overextended touchdown geometry. Test `step_reference_scale` next with command scale restored.

## Git Refs

- Baseline ref: working tree on `724a1c3`, command scale `1.0`
- Candidate ref: read-only config variant `command_scale=0.8`
- Key files: `model/nominal.py`, `config.py`, `solver/line_search.py`
