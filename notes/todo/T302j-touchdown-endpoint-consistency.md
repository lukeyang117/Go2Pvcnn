# T302j Touchdown Endpoint Consistency

## Current State

- T302j is a child of [T302i](T302i-viewer-realized-foot-mismatch.md).
- User diagnosis to preserve:
  - During low-small obstacle crossing, the swing foot can follow the command direction and move far forward, while the exported touchdown is behind the swing/planned foot in command-frame coordinates.
  - V9 also shows foot arcs rising above root height.
  - Viewer markers should remain planner outputs; do not hide the mismatch by drawing FK-realized touchdown markers.
- Evidence from T302i:
  - Baseline mixed-yaw reproduces the endpoint conflict: `planned_swing_along_forward_step_max_m=0.347791`, `touchdown_behind_swing_foot_along_max_m=0.569856`.
  - V9 reduces but does not remove it: touchdown remains behind by about `0.314-0.316m`.
  - V9 reproduces above-root feet: `planned_swing_foot_above_root_z_max_m=0.120522`, `fk_swing_foot_above_root_z_max_m=0.069278`.
  - Touchdown chain trace partially rejects the exact "nominal far forward then pulled back" theory for the mixed-yaw row: nominal starts behind, grounding does not change along, optimization moves touchdown forward, and the remaining conflict is weak coupling between swing path extrema/endpoints and sampled/exported touchdown.
- 2026-05-26 V11 probe:
  - Implemented `reachable_fk_cross_v11` as a probe/debug variant with endpoint consistency, sampled-touchdown FK reachability, and foot-above-root guard.
  - Local verification passes: `py_compile` and `pytest -q Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py` -> `31 passed`.
  - Mixed-yaw real IsaacLab V11 improves the targeted endpoint/height metrics:
    - touchdown behind swing `0.569856 -> 0.124781`;
    - planned swing foot above root `0.120522` in V9 -> `0.013994`;
    - FK swing foot above root `0.069278` in V9 -> `0.007172`;
    - direction remains good `0.995655`;
    - small contact/penetration remains `0`.
  - V11 is rejected as a fix because touchdown IK/FK worsens to `0.786326` and swing continuity is worse than V9 (`fk_swing_foot_step_max_to_median=10.653`).
  - Current interpretation: endpoint/height terms are useful, but sampled-touchdown reachability is not strong or aligned enough yet. T302j.2 remains the next focus.
- 2026-05-26 V12 farthest-export probe:
  - User updated priority: first make `touchdowns` the farthest point along the velocity/translation command direction; IK/FK, continuity, and height can be secondary for this slice.
  - Loss-only V12 did not fully satisfy that contract (`touchdown_behind_swing_foot_along_max_m=0.180266`), and a stronger hinge only reduced it to `0.150514` while worsening direction/lateral drift.
  - Implemented V12 export contract: for `reachable_fk_cross_v12`, exported `touchdown_seq` / `planned_touchdown_w` uses each leg's command-direction farthest swing point, grounded to terrain height.
  - Real IsaacLab mixed-yaw result now satisfies the P0 marker/endpoint contract: `touchdown_behind_swing_foot_along_max_m=0.0`.
  - Remaining secondary issues are explicit, not hidden: FK-realized endpoint can still be ahead of exported touchdown (`touchdown_behind_fk_foot_along_max_m=0.211595`), touchdown IK/FK remains `0.563487`, direction/lateral drift regressed vs V11 (`command_direction_cosine=0.881158`, `lateral_drift_m=0.786369`), and planned/FK foot-above-root remains `0.089203/0.137184`.
- 2026-05-26 V12 viewer runtime port:
  - Viewer/runtime path now selects V12 with `--planner-backend mpc --mpc-debug-variant reachable_fk_cross_v12`.
  - Smoke reached `[Viewer] Attached mpc trajectory manager`, horizon `25`, and kinematic playback path.
  - The smoke was stopped after startup confirmation to avoid occupying the user's visual session.
- 2026-05-26 default MPC port correction:
  - User rejected the debug-only invocation path. The farthest-touchdown export is now part of the default `plan_segment()` MPC export path.
  - The user's command without `--mpc-debug-variant`, without `--n-frames 25`, and without scripted command reached `[Viewer] Attached mpc trajectory manager`, horizon `50`, terrain row `6`, and kinematic playback.
  - Current required viewer invocation is just the normal MPC backend, for example `--planner-backend mpc --terrain-row 6`.
- 2026-05-26 foot-above-root reproduction:
  - Default MPC forward low-small crossing reproduces the user's "swing foot above root" complaint after the farthest-touchdown export change.
  - Forward metrics: planned/FK swing foot above root `+0.025513/+0.073637m`, root height min `0.142901m`, base bottom clearance min `0.042901m`, small contact/penetration `0`.
  - Mixed-yaw in the same run did not exceed root height (`-0.005118/-0.008270m`) but still failed direction tracking (`command_direction_cosine=-0.595472`).
  - Interpretation: high-foot swing shape is reproduced in forward low-small crossing and is separate from the mixed-yaw direction failure.
- 2026-05-26 low-small acceptance test contract:
  - Added metric tests that combine the three user-required conditions:
    1. `touchdown_behind_swing_foot_along_max_m == 0`;
    2. `fk_foot_over_low_small_success/lift_then_land/touchdown_after == 1`;
    3. planned and FK swing foot above root `<= 0`.
  - Current default MPC forward low-small crossing passes endpoint and FK foot-over but fails no-above-root with planned/FK `+0.025513/+0.073637m`.
- 2026-05-26 structured low-small touchdown runtime slice:
  - User clarified desired behavior: derive low-small `touchdowns` from current IsaacLab foot positions, translation command direction, and each leg's obstacle-relative position; do not hard-code front legs; keep normal gait alternation but let the crossing leg and approach leg have different low-small semantics.
  - Default `plan_segment()` now attempts low-small structured touchdown generation before falling back to command-direction farthest export:
    - find nearest semantic small obstacle in the command corridor;
    - compute each current foot's command-frame `along/lateral` relative to that obstacle;
    - choose one eligible current swing leg to cross behind the obstacle, while other approaching legs target the pre-obstacle side;
    - align low-small swing Cartesian path toward the same exported touchdown target.
  - This is a runtime code change, not a test-only/debug variant.
  - Real `env_isaacsim` probe result is partial and not accepted:
    - structured-touchdown-only made endpoint worse (`forward/mixed touchdown_behind_swing=0.426/0.690m`), confirming marker-only changes are insufficient;
    - swing alignment reduced endpoint mismatch (`forward=0.053-0.063m`, `mixed=0.103m`) and kept small contact/penetration `0`;
    - forward FK foot-over fails (`fk_foot_over_low_small_success=0`), and FK can still put swing foot above root (`+0.048m`) even when planned swing is below root.
  - Interpretation: target generation and swing path must be coupled, but the coupling must also be FK-reachable; planned Cartesian postprocess alone can still be deformed by IK clamp.

## Open Children

| Child | Status | Priority | Purpose | Primary Files |
| --- | --- | --- | --- | --- |
| T302j.1 | verify | P0 | V12 farthest-export contract makes exported touchdown the command-direction farthest swing point; mixed-yaw now has `touchdown_behind_swing=0.0`; all-direction not yet run | `Go2Pvcnn/extension/batch_mpc_planner/planner.py`, `Go2Pvcnn/tests/test_batch_mpc_backend.py` |
| T302j.2 | active | P1 | FK-realized endpoint can still move ahead of exported touchdown (`touchdown_behind_fk=0.211595`); next fix should align exported marker with reachable/FK path without hiding mismatch in viewer | `Go2Pvcnn/extension/batch_mpc_planner/kinematics.py`, `Go2Pvcnn/extension/batch_mpc_planner/planner.py` |
| T302j.3 | active | P0 | Default MPC forward low-small crossing reproduces foot-above-root: planned/FK swing above root `+0.025513/+0.073637m`; needs a default guard/shape fix that preserves clearance | `Go2Pvcnn/extension/batch_mpc_planner/debug_variants.py`, `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py` |
| T302j.4 | done | P0 | Run focused `env_isaacsim` mixed-yaw V12 comparison and report endpoint, reachability, foot-height, contact, and continuity metrics | `tmp/t302i-viewer-realized-foot-mismatch/` |
| T302j.5 | todo | P0 | If mixed-yaw improves, run all-direction low-small coverage including forward, lateral, diagonal, mixed-yaw, and pure-yaw | `Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py` |
| T302j.6 | todo | P1 | Decide whether the accepted probe-only loss should become production behavior or whether the planner output contract needs FK-reachable touchdown generation/export | `Go2Pvcnn/extension/batch_mpc_planner/planner.py`, `Go2Pvcnn/extension/batch_mpc_planner/variables.py` |
| T302j.7 | active | P0 | Inspect structured low-small per-leg state in viewer/probe: forward endpoint mismatch improved but FK foot-over fails; decide if the remaining blocker is target timing, lateral lane, or IK/FK clamp | `Go2Pvcnn/extension/batch_mpc_planner/planner.py`, `tmp/t302i-viewer-realized-foot-mismatch/` |

## Closed Children Archive

- None yet.

## Related Logs

- [../log/2026-05-26-1002-t302i-command-frame-endpoint-height-reproduction.md](../log/2026-05-26-1002-t302i-command-frame-endpoint-height-reproduction.md)
- [../log/2026-05-26-1018-t302i-touchdown-chain-trace.md](../log/2026-05-26-1018-t302i-touchdown-chain-trace.md)
- [../log/2026-05-25-2326-t302i-v9-viewer-runtime-port.md](../log/2026-05-25-2326-t302i-v9-viewer-runtime-port.md)
- [../log/2026-05-25-2244-t302i-v10-soft-combo-probe.md](../log/2026-05-25-2244-t302i-v10-soft-combo-probe.md)
- [../log/2026-05-26-1113-t302j-v11-endpoint-consistency-probe.md](../log/2026-05-26-1113-t302j-v11-endpoint-consistency-probe.md)
- [../log/2026-05-26-1136-t302j-v12-touchdown-farthest-export.md](../log/2026-05-26-1136-t302j-v12-touchdown-farthest-export.md)
- [../log/2026-05-26-1154-t302j-v12-viewer-runtime-port.md](../log/2026-05-26-1154-t302j-v12-viewer-runtime-port.md)
- [../log/2026-05-26-1219-t302j-default-mpc-farthest-touchdown-port.md](../log/2026-05-26-1219-t302j-default-mpc-farthest-touchdown-port.md)
- [../log/2026-05-26-1249-t302j-default-mpc-foot-above-root-reproduction.md](../log/2026-05-26-1249-t302j-default-mpc-foot-above-root-reproduction.md)
- [../log/2026-05-26-1259-t302j-low-small-crossing-acceptance-test-contract.md](../log/2026-05-26-1259-t302j-low-small-crossing-acceptance-test-contract.md)
- [../log/2026-05-26-1336-t302j-structured-low-small-touchdown-runtime.md](../log/2026-05-26-1336-t302j-structured-low-small-touchdown-runtime.md)
- Parent: [T302i viewer realized-foot mismatch](T302i-viewer-realized-foot-mismatch.md)

## Git Refs

- Last Feature Commit: `pending`
- Last Verified Commit: `working tree @ c54dc5c`
- Current Work Ref: `working tree @ c54dc5c`
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py](../../Go2Pvcnn/tests/test_mpc_low_small_reachable_crossing_probe.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/debug_variants.py](../../Go2Pvcnn/extension/batch_mpc_planner/debug_variants.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/variables.py](../../Go2Pvcnn/extension/batch_mpc_planner/variables.py)

## Next Step

Next implementation slice should preserve the current low-small structured touchdown/swing coupling as a viewer-inspection candidate, but move the target generation into an FK-reachable space. Planned Cartesian touchdown/swing alone is not sufficient: forward still fails FK foot-over and FK can still exceed root height.

V12 status:

1. P0 latest user request is satisfied for mixed-yaw:
   - `touchdown_behind_swing_foot_along_max_m=0.0`.
2. V12 is not an all-metric fix:
   - `touchdown_behind_fk_foot_along_max_m=0.211595`;
   - `touchdown_ik_fk_error_max=0.563487`;
   - `command_direction_cosine=0.881158`;
   - `lateral_drift_m=0.786369`;
   - planned/FK above-root `0.089203/0.137184`.
3. All-direction coverage is still open, but the endpoint export behavior is no longer debug-gated; it is now default MPC behavior.

Previous V12 direction, now superseded for the immediate user priority:

V12 direction:

1. Reuse V11 endpoint/height terms with softer weights.
2. Strengthen and debug sampled-touchdown FK reachability:
   - compute on the same sampled/exported touchdown path as metrics;
   - inspect whether root/rpy sampling at touchdown phase is aligned with `planned_touchdown_w`;
   - penalize worst leg more directly.
3. Keep mixed-yaw direction and small contact gates.

Original V11 design kept here for reference:

1. Command-frame endpoint consistency:
   - compute command-frame `along` for FK/planned swing foot path and sampled touchdown;
   - penalize touchdown being behind the swing endpoint or forward swing extrema by more than a small tolerance;
   - gate on translation command and swing probability;
   - pure yaw should not require low-small crossing.
2. Sampled-touchdown FK reachability:
   - sample touchdown from final foot trajectory at `swing_center + 0.5 * swing_width`;
   - solve IK for that touchdown, clamp joints, FK back;
   - penalize `planned_touchdown - FK(clamped IK planned_touchdown)`.
3. Foot-above-root guard:
   - penalize `foot_z - root_z` above a small positive tolerance during swing;
   - preserve required obstacle clearance by making this an upper-bound guard, not a clearance removal.

Initial acceptance target for mixed-yaw:

- reduce `touchdown_behind_swing_foot_along_max_m` materially from `0.569856`;
- reduce or do not worsen `touchdown_ik_fk_error_max` from `0.661772`;
- keep `planned_swing_foot_above_root_z_max_m <= 0.02m` or at least below V9's `0.120522m`;
- keep small contact/penetration at `0`;
- do not regress swing continuity beyond V9/baseline tradeoff.

## Node Details

### T302j.1 Endpoint Consistency Loss

- why-created: T302i reproduced a command-frame geometry conflict where swing moves forward but exported touchdown remains behind the swing/planned foot.
- design intent: make touchdown and swing endpoint agree in command-frame coordinates, without forcing speed magnitude tracking.
- risk: if weighted too strongly, it may force unsafe crossing before the robot is in a good approach window.

### T302j.2 Sampled-Touchdown FK Reachability

- why-created: T302i clamp trace shows exported Cartesian targets can be unreachable after Go2 joint-limit clamp.
- design intent: apply reachability directly to the exported touchdown point, not only to whole-horizon foot positions.
- risk: FK reachability may conflict with semantic obstacle avoidance; if it does, target generation/output contract needs structural change.

### T302j.3 Foot-Above-Root Guard

- why-created: V9 mixed-yaw reproduced planned/FK swing feet above root height.
- design intent: preserve obstacle clearance while preventing unnatural high arcs or low-base/high-foot shortcuts.
- risk: too strict a guard can block legitimate high clearance over taller obstacles; start low-small gated only.
