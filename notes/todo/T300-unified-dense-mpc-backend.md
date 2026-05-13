# T300 Unified Dense MPC Backend

## Current State

- A new independent planning front is created outside `T100` for unified dense MPC backend design and follow-up implementation.
- The approved design direction is:
  - `planner_backend="mpc"` (new backend, not replacing `together`)
  - no old mode classifier/table in the new core
  - dense optimization variables:
    - `root_pos_residual [B,T,3]`
    - `root_rpy_residual [B,T,3]`
    - `foot_pos_residual [B,T,4,3]`
    - `contact_logits [B,T,4]`
  - touchdown sequence derived from foot trajectory + contact transitions
  - per-loss reward-style tunable config
  - diagnostics layer with `enabled` switch
  - async per-env dirty-mask replanning for 4096-env throughput
- Spec hardening is complete with P0/P1 review convergence integrated into the design doc:
  - differentiable contact-timing contract (`contact_logits` drives stance/swing optimization path)
  - GPU-only fixed-budget async replan contract (no host-sync hot path)
  - 4096 profile/scheduler defaults and command-hysteresis parameters
  - IsaacLab GPU-only test matrix and diagnostics-oracle rules
  - config/implementation boundary and protocol contract sections
- T300d implementation integration pass is complete with focused verification evidence:
  - factory `planner_backend="mpc"` routing verified
  - `batch_mpc_planner` manager/config/viewer integration updated
  - focused backend test suite passed (`11 passed` after terrain shape/OOM regression additions)
  - `env_isaacsim` headless MPC runtime test-layer selectors passed (`EXIT_CODE:0`)
  - runtime counters (`dirty/selected/backlog/max_stale/planner_ms/cache_ms`) are now implemented in manager
  - runtime terrain ingestion now correctly handles scanner `ray_hits_w` flattened grid shape `[B,H*W,3]` (OOM fix)
  - exact 4096-env training command (`max_iterations=1`, `planner-backend=mpc`) now passes end-to-end with checkpoint output
  - MPC viewer direct-script path now survives repeated replans after entrypoint import-order fix and autograd graph-detach handoff fix
  - leg-order diagnostics and command-matrix acceptance now align for backward/lateral/yaw after replacing quadrant-based foot reindexing with fixed planner leg order in viewer/runtime MPC paths
  - long-running IsaacLab headless reproduction now confirms the user-observed MPC foot drift across command directions; 120 cycles on `cuda:2` produced max absolute foot-base radius drift `0.1172m` on `yaw_left`, with `forward=0.0841m` and `backward=0.0722m`
  - long-running IsaacLab headless variant sweep now compares five proposed drift-improvement directions independently; no direct test-layer winner emerged:
    - input-level stance anchor proxy was neutral (`mean_abs=0.0627` vs baseline `0.0634`)
    - phase-continuity-only worsened `yaw_left` (`0.1709`)
    - hard anchor nominal replacement helped yaw but badly worsened linear/lateral commands (`mean_abs=0.0793`)
    - stronger global stance/root-frame losses were direction-dependent and worse overall (`mean_abs=0.0741`)
    - diagnostics-only stayed close to baseline (`mean_abs=0.0598`)
  - current drift-fix evidence points to a real contact-aware manager memory design rather than one-variable tuning: touchdown/stable-contact anchor update, anchor-aware nominal/loss masks, contact-event phase alignment, and direction-adaptive stance/root-frame weights
  - iterative follow-up testing selected `dir10_yaw_anchor_linear_seed_proxy` as the best current fix direction:
    - `dir6` yaw-only anchor nominal reduced full 120-cycle mean drift from `0.0618` to `0.0438` without linear/lateral side effects
    - `dir9` linear body-frame footprint seed fixed 40-cycle forward/backward drift while leaving yaw mostly unchanged
    - `dir10` combined both mechanisms and reduced full 120-cycle mean drift from `0.0646` to `0.0140`; forward/backward/lateral drift were near zero, yaw-left dropped from `0.1689` to `0.0624`, and yaw-right from `0.0228` to `0.0098`
  - next production fix should implement the `dir10` regime split in manager/nominal rather than as a test monkeypatch:
    - linear-dominant commands use persistent reset-time body-frame footprint seed transformed by current root pose
    - yaw-dominant commands use contact-gated stance anchor nominal replacement
    - retain command-regime gates and diagnostics in the manager-owned memory contract
  - second direction-expansion tests did not find a better replacement than `dir10`:
    - `dir11` (running linear footprint memory) was worse than `dir10` and lost the clean yaw benefit
    - `dir13`/`dir14` (strict vs soft regime gates) were effectively identical on the current discrete command matrix and remained worse than the best historical `dir10` run
    - `dir12` (stance-only yaw anchor) is still an unresolved experiment-harness branch and not yet evidence for production behavior
  - new mixed-command and command-switch long-horizon tests now confirm that recommendation:
    - `dir10` remains the best known discrete-command drift fix and strongly suppresses `yaw -> forward` switch drift
    - but `dir10` also introduces a new side effect that the old matrix hid: aggressive yaw-segment foot-step motion (`foot_step_mean` up to about `0.35`) and mixed-to-yaw transition overshoot
    - `dir14` soft regime weighting matters once mixed boundary commands are included: it improves the mixed boundary drift itself (`mix_diag_yaw_left 0.0991 -> 0.0127`) but does not fully remove yaw-transition aggressiveness
  - touchdown-grounding metrics are now wired into the real IsaacLab MPC runtime diagnostics:
    - metric bridge had to be adapted to current `batch_mpc_planner` terrain contract (`height_map + world_x/y_range`), not old `height_at()` terrain API
    - minimal forward probe shows planned touchdowns already airborne by about `0.0667m` on average with `touchdown_airborne_ratio=1.0`
    - short `forward_yaw_left_forward` sequence probe shows the airborne-touchdown issue persists across baseline, `dir10`, and `dir14`
    - `dir14` reduces touchdown gap versus baseline on the short sequence, but does not ground touchdowns; airborne ratio stays `1.0`
  - production `dir15 + dir19` synthesis is now implemented in `extension/batch_mpc_planner`:
    - manager-owned foothold memory stores body-frame footprint seed, stance anchors, previous contacts, and yaw-entry counters
    - soft linear/yaw gates select persistent footprint seed for linear commands and stance-anchor replacement for yaw-dominant commands
    - yaw-entry ramp reduces abrupt anchor influence when switching into yaw
    - contact foot z is grounded to sampled `MpcPlannerTerrain` height before touchdown extraction, IK, diagnostics, and result emission
    - viewer direct MPC path now carries the same lightweight memory so visual inspection exercises the production behavior
  - fresh scoped verification after production edit:
    - target `py_compile` checks passed
    - focused backend suite `python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q` passed (`12 passed`)
    - IsaacLab headless MPC smoke on `cuda:2` passed
  - remaining risk is narrowed to high-scale runtime diagnostics/counter stability, longer-run visual behavior, and possible future integration of terrain height into optimizer losses rather than output-only grounding

## Open Children

- [T300d](../todo.md#open-leaves): production MPC foothold memory/grounding is implemented and awaiting user visual inspection.

## Closed Children Archive

- T300a written spec review gate and implementation-plan handoff (closed after multi-subagent review pass identified concrete P0 hardening items).
- T300b subagent review convergence and spec hardening before implementation plan (closed after applying hardening edits to the design spec).

## Related Logs

- [2026-05-11-1050-t300-unified-dense-mpc-backend-design.md](../log/2026-05-11-1050-t300-unified-dense-mpc-backend-design.md)
- [2026-05-11-1104-t300-subagent-design-review.md](../log/2026-05-11-1104-t300-subagent-design-review.md)
- [2026-05-11-1110-t300-spec-hardening.md](../log/2026-05-11-1110-t300-spec-hardening.md)
- [2026-05-11-1157-t300d-subagent-implementation-review.md](../log/2026-05-11-1157-t300d-subagent-implementation-review.md)
- [2026-05-11-1243-t300d-env-isaacsim-mpc-headless-runtime.md](../log/2026-05-11-1243-t300d-env-isaacsim-mpc-headless-runtime.md)
- [2026-05-11-1318-t300d-4096-runtime-counters-attempt.md](../log/2026-05-11-1318-t300d-4096-runtime-counters-attempt.md)
- [2026-05-11-1343-human16-mpc-command-update.md](../log/2026-05-11-1343-human16-mpc-command-update.md)
- [2026-05-11-1411-mpc-terrain-ray-shape-oom-fix.md](../log/2026-05-11-1411-mpc-terrain-ray-shape-oom-fix.md)
- [2026-05-11-1428-mpc-4096-train-maxiter1-success.md](../log/2026-05-11-1428-mpc-4096-train-maxiter1-success.md)
- [2026-05-11-1528-mpc-viewer-flying-feet-order-regression.md](../log/2026-05-11-1528-mpc-viewer-flying-feet-order-regression.md)
- [2026-05-11-1543-mpc-viewer-forward-static-joint-ik-fix.md](../log/2026-05-11-1543-mpc-viewer-forward-static-joint-ik-fix.md)
- [2026-05-11-1635-mpc-gait-coupling-loss-minimal-fix.md](../log/2026-05-11-1635-mpc-gait-coupling-loss-minimal-fix.md)
- [2026-05-11-1655-mpc-long-replan-foot-motion-and-yaw-display.md](../log/2026-05-11-1655-mpc-long-replan-foot-motion-and-yaw-display.md)
- [2026-05-11-2005-mpc-leg-order-command-matrix-recovery.md](../log/2026-05-11-2005-mpc-leg-order-command-matrix-recovery.md)
- [2026-05-12-2147-mpc-long-replan-foot-drift-reproduction.md](../log/2026-05-12-2147-mpc-long-replan-foot-drift-reproduction.md)
- [2026-05-12-2243-mpc-long-replan-variant-sweep.md](../log/2026-05-12-2243-mpc-long-replan-variant-sweep.md)
- [2026-05-12-2346-mpc-long-replan-iterative-direction-search.md](../log/2026-05-12-2346-mpc-long-replan-iterative-direction-search.md)
- [2026-05-13-0030-mpc-long-replan-second-direction-expansion.md](../log/2026-05-13-0030-mpc-long-replan-second-direction-expansion.md)
- [2026-05-13-0932-mpc-mixed-sequence-long-horizon-sweep.md](../log/2026-05-13-0932-mpc-mixed-sequence-long-horizon-sweep.md)
- [2026-05-13-1025-mpc-touchdown-grounding-probe.md](../log/2026-05-13-1025-mpc-touchdown-grounding-probe.md)
- [2026-05-13-1253-mpc-dir15-dir19-production-grounding.md](../log/2026-05-13-1253-mpc-dir15-dir19-production-grounding.md)

## Git Refs

- Last Feature Commit: `pending`
- Last Verified Commit: `pending`
- Current Work Ref: `working tree on top of e90e3a4 (dir15/dir19 production foothold memory + terrain grounding)`
- Key Files:
  - [../../docs/superpowers/specs/2026-05-11-unified-dense-mpc-backend-design.md](../../docs/superpowers/specs/2026-05-11-unified-dense-mpc-backend-design.md)
  - [../log/2026-05-11-1050-t300-unified-dense-mpc-backend-design.md](../log/2026-05-11-1050-t300-unified-dense-mpc-backend-design.md)
  - [../log/2026-05-11-1104-t300-subagent-design-review.md](../log/2026-05-11-1104-t300-subagent-design-review.md)
  - [../log/2026-05-11-1110-t300-spec-hardening.md](../log/2026-05-11-1110-t300-spec-hardening.md)

## Next Step

- User visual inspection of `go2_foostep_planner.py --planner-backend mpc` with production foothold memory and terrain grounding.
- If visual artifacts remain, investigate terrain-height-in-loss and IK/contact smoothness before launching another broad drift sweep.

## Node Details

### T300a written spec review gate and implementation-plan handoff

- status: `done`
- why-created: Keep MPC work independent from T100 and gate implementation on explicit review of the written design spec.
- approved design source:
  - [../../docs/superpowers/specs/2026-05-11-unified-dense-mpc-backend-design.md](../../docs/superpowers/specs/2026-05-11-unified-dense-mpc-backend-design.md)
- constraints:
  - remain separate from T100/together cleanup track
  - keep `legacy/together` rollback paths
  - preserve cache/reward/viewer contracts during implementation
- related log:
  - [2026-05-11-1050-t300-unified-dense-mpc-backend-design.md](../log/2026-05-11-1050-t300-unified-dense-mpc-backend-design.md)
  - [2026-05-11-1104-t300-subagent-design-review.md](../log/2026-05-11-1104-t300-subagent-design-review.md)
- closure note:
  - Multi-subagent review completed and converted the gate into concrete spec-hardening work before implementation planning.

### T300b subagent review convergence and spec hardening before implementation plan

- status: `done`
- why-created: The written spec direction is approved, but implementation-readiness at 4096 env requires explicit hard contracts to avoid CPU hot path regressions and ambiguous timing semantics.
- must-fix before implementation planning:
  - define differentiable stance/swing timing contract from `contact_logits`
  - define GPU-only fixed-budget dirty scheduling contract
  - define 4096 training profile defaults and command hysteresis/ramp settings
  - expand IsaacLab GPU-only test matrix with diagnostics-oracle assertions and old-together isolation
  - strengthen config/implementation boundary and backend protocol contracts
- related log:
  - [2026-05-11-1104-t300-subagent-design-review.md](../log/2026-05-11-1104-t300-subagent-design-review.md)
  - [2026-05-11-1110-t300-spec-hardening.md](../log/2026-05-11-1110-t300-spec-hardening.md)

### T300c implementation-plan handoff after hardened spec approval

- status: `done`
- why-created: Spec hardening is complete; implementation should start only after confirming the hardened clauses align with user intent.
- next action:
  - produce a file-by-file implementation plan for `trajectory_contracts -> mpc backend -> factory/reward/viewer wiring -> tests`
  - keep rollout staged and verify each stage on `env_isaaclab` GPU path
- related log:
  - [2026-05-11-1110-t300-spec-hardening.md](../log/2026-05-11-1110-t300-spec-hardening.md)

### T300d subagent-driven implementation and test execution for `extension/batch_mpc_planner`

- status: `verify`
- why-created: User explicitly requested subagent execution for code changes/tests, with main agent acting as reviewer/integrator.
- execution split:
  - worker A: implement/repair `Go2Pvcnn/extension/batch_mpc_planner/*` scaffolding and factory wiring
  - worker B: add focused tests for new backend contracts and run quick verification
  - main agent: review worker outputs, merge/fix issues, update design/todo/log, report residual risks
- constraints:
  - keep path as `Go2Pvcnn/extension/batch_mpc_planner`
  - no CPU hot path in newly added MPC files where avoidable
  - preserve legacy/together behavior when backend is not `mpc`
- latest verification:
  - focused suite `python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q` passed (`12 passed`)
  - target `py_compile` checks passed
  - viewer backend path now accepts `--planner-backend mpc`
  - `env_isaacsim` headless selector runs for `test_mpc_runtime_headless.py` returned `EXIT_CODE:0` on all three MPC runtime smoke tests
  - runtime counters path is unit-verified and exposed through `MpcTrajectoryManager.runtime_counters()`
  - flattened scanner ray-hit shape contract (`[B,H*W,3]`) now has direct regression coverage and no longer mis-expands terrain batch axis
  - exact user command `python Go2Pvcnn/scripts/train.py --headless --device cuda:2 --num_envs 4096 --max_iterations 1 --experiment teacher_elevation_trajectory --planner-backend mpc` completed with `EXIT_CODE:0` and wrote `model_0.pt`
  - exact user viewer command `python Go2Pvcnn/extension/viz/go2_foostep_planner.py --headless --livestream 2 --webrtc-public-ip 172.31.179.75 --device cuda:2 --num_envs 1 --terrain task --planner-backend mpc` now passes bootstrap and repeated replan cycles (no `ModuleNotFoundError`, no camera `.numpy()` grad crash, no second-replan autograd graph reuse crash)
  - viewer flying-feet regression root-caused to MPC state joint-order mismatch; `_mpc_state_from_env` now converts robot-order joints to planner-order before planning, with headless `cuda:2/cuda:3` checks keeping `root/joint` exact and foot residuals at centimeter scale
  - viewer static-leg regression root-caused to planner returning horizon-constant seed joints; MPC now emits IK-solved `joint_angles[B,T,12]` from optimized root+foot trajectory, and forward-command headless probe reports nonzero joint span (`joint_tspan_max 0.9467`, prev `0.0`)
  - MPC optimizer now safely supports rollout-side `torch.inference_mode()` call context; 4096-run trainer mem-guard applied at runtime (`num_steps_per_env=24`, `num_mini_batches=8`)
  - Added minimal gait-coupling losses (`stance_slip`, `swing_stride`) with RL-configurable thresholds and verified wiring in focused/backend + Isaac headless runtime tests
  - Long viewer-style replan no-foot-motion was reproduced and fixed by adding a command-relative nominal foot seed plus `contact_schedule` loss; yaw display now uses `wxyz` as the primary RPY readout
  - Viewer kinematic diagnostics now use fixed planner leg order (`FL/FR/RL/RR`) rather than dynamic quadrant reordering, eliminating false high foot-error reports for `backward/yaw` while preserving command-direction motion checks
  - `test_mpc_runtime_headless.py -k command_matrix` now passes on `cuda:2`; full MPC runtime headless selector shows `8 passed, 1 skipped`
  - 2026-05-12 long replan foot drift reproduction added as an opt-in IsaacLab test under `Go2Pvcnn/tests/test_mpc_runtime_headless.py`; `MPC_RUNTIME_LONG_DRIFT=1`, 120 cycles on `cuda:2`, reproduced `mean_abs_drift=0.0569` and max `yaw_left=0.1172`
  - 2026-05-12 long replan variant sweep added as an opt-in IsaacLab test; `MPC_RUNTIME_LONG_DRIFT_SWEEP=1`, 120 cycles on `cuda:2`, tested baseline plus five proposed directions and found no single-variable fix, with strongest useful signal from contact-aware anchor nominal/loss design rather than hard replacement
  - 2026-05-12 iterative direction search extended the same opt-in test and selected `dir10_yaw_anchor_linear_seed_proxy`: full 120-cycle six-command run reduced `mean_abs_drift` from `0.0646` to `0.0140`
  - 2026-05-13 second direction expansion tested running linear footprint memory and gate-shape variants; no new direction beat `dir10`, so search focus should move toward mixed-command and command-switch horizons if further exploration is needed before production edits
  - 2026-05-13 mixed/sequence sweep validated that focus change: `dir10` still dominates discrete switch drift, but mixed-command and yaw-segment tests expose a new side effect, while `dir14` soft weighting helps the boundary cases without fully fixing yaw-anchor aggressiveness
  - 2026-05-13 touchdown-grounding probe added real-runtime `touchdown_ground_gap_mean / touchdown_airborne_ratio / touchdown_airborne_max_gap` and found a second major failure mode: planned touchdowns remain airborne and gap can grow over repeated replans; short evidence suggests `dir14` helps gap magnitude more than baseline but does not actually ground touchdowns
  - 2026-05-13 production `dir15 + dir19` implementation adds manager foothold memory, soft command gates, yaw-entry ramp, terrain-grounded contact foot z, and viewer direct-path memory; fresh `py_compile`, focused backend, and IsaacLab headless smoke passed
- residual risk:
  - true 4096-env throughput/timing counter extraction remains unstable in current semantic viewer runtime diagnostics path:
    - high-scale PhysX pair-capacity pressure and/or CUDA device-side asserts
    - long headless runtime loops can still prevent deterministic counter-capture assertions
  - long visual behavior is not re-swept after the production edit because the user explicitly moved the next acceptance step to visual inspection
  - terrain grounding is currently output-side; if snapping appears, move terrain height into nominal/loss terms
- related log:
  - [2026-05-11-1157-t300d-subagent-implementation-review.md](../log/2026-05-11-1157-t300d-subagent-implementation-review.md)
  - [2026-05-11-1243-t300d-env-isaacsim-mpc-headless-runtime.md](../log/2026-05-11-1243-t300d-env-isaacsim-mpc-headless-runtime.md)
  - [2026-05-11-1318-t300d-4096-runtime-counters-attempt.md](../log/2026-05-11-1318-t300d-4096-runtime-counters-attempt.md)
  - [2026-05-11-1411-mpc-terrain-ray-shape-oom-fix.md](../log/2026-05-11-1411-mpc-terrain-ray-shape-oom-fix.md)
  - [2026-05-11-1428-mpc-4096-train-maxiter1-success.md](../log/2026-05-11-1428-mpc-4096-train-maxiter1-success.md)
  - [2026-05-11-1505-t300d-mpc-viewer-entrypoint-and-autograd-replan-fix.md](../log/2026-05-11-1505-t300d-mpc-viewer-entrypoint-and-autograd-replan-fix.md)
  - [2026-05-11-1528-mpc-viewer-flying-feet-order-regression.md](../log/2026-05-11-1528-mpc-viewer-flying-feet-order-regression.md)
  - [2026-05-11-1543-mpc-viewer-forward-static-joint-ik-fix.md](../log/2026-05-11-1543-mpc-viewer-forward-static-joint-ik-fix.md)
  - [2026-05-11-1635-mpc-gait-coupling-loss-minimal-fix.md](../log/2026-05-11-1635-mpc-gait-coupling-loss-minimal-fix.md)
  - [2026-05-11-1655-mpc-long-replan-foot-motion-and-yaw-display.md](../log/2026-05-11-1655-mpc-long-replan-foot-motion-and-yaw-display.md)
  - [2026-05-11-2005-mpc-leg-order-command-matrix-recovery.md](../log/2026-05-11-2005-mpc-leg-order-command-matrix-recovery.md)
  - [2026-05-12-2147-mpc-long-replan-foot-drift-reproduction.md](../log/2026-05-12-2147-mpc-long-replan-foot-drift-reproduction.md)
  - [2026-05-12-2243-mpc-long-replan-variant-sweep.md](../log/2026-05-12-2243-mpc-long-replan-variant-sweep.md)
  - [2026-05-12-2346-mpc-long-replan-iterative-direction-search.md](../log/2026-05-12-2346-mpc-long-replan-iterative-direction-search.md)
  - [2026-05-13-0030-mpc-long-replan-second-direction-expansion.md](../log/2026-05-13-0030-mpc-long-replan-second-direction-expansion.md)
  - [2026-05-13-0932-mpc-mixed-sequence-long-horizon-sweep.md](../log/2026-05-13-0932-mpc-mixed-sequence-long-horizon-sweep.md)
  - [2026-05-13-1025-mpc-touchdown-grounding-probe.md](../log/2026-05-13-1025-mpc-touchdown-grounding-probe.md)
  - [2026-05-13-1253-mpc-dir15-dir19-production-grounding.md](../log/2026-05-13-1253-mpc-dir15-dir19-production-grounding.md)
