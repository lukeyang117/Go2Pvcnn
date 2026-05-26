# T302j Touchdown Endpoint Consistency Todo

## Purpose

Create a dedicated branch todo for the next T302i fix direction: make planner-exported touchdowns consistent with swing endpoints, FK reachability, and foot-height limits instead of changing viewer markers.

## Stage

Planning / todo split for `extension/batch_mpc_planner`.

## Related Todo

- [../todo/T302j-touchdown-endpoint-consistency.md](../todo/T302j-touchdown-endpoint-consistency.md)
- Parent: [../todo/T302i-viewer-realized-foot-mismatch.md](../todo/T302i-viewer-realized-foot-mismatch.md)

## Procedure

Created a new branch page:

- [../todo/T302j-touchdown-endpoint-consistency.md](../todo/T302j-touchdown-endpoint-consistency.md)

The page records:

- T302i evidence for forward-swing/rear-touchdown conflict.
- T302i evidence for V9 above-root foot arcs.
- The corrected touchdown-chain interpretation: nominal was not pulled backward in the mixed-yaw baseline row; the main remaining issue is weak coupling between swing path extrema/endpoints and sampled/exported touchdown, then IK clamp/FK adds planned-vs-realized mismatch.
- Probe-only next direction: `reachable_fk_cross_v11` with endpoint consistency, sampled-touchdown FK reachability, and foot-above-root guard.

## Result

Pass as todo creation. No planner behavior changed.

## Follow-Up

Next implementation should start with tests and a probe-only debug variant before any production default change.

## Git Refs

- Baseline Ref: `c54dc5c`
- Candidate Ref: `working tree`
- Key Files:
  - [../todo/T302j-touchdown-endpoint-consistency.md](../todo/T302j-touchdown-endpoint-consistency.md)
