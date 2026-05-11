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
  - remaining risk is narrowed to high-scale runtime diagnostics/counter stability and longer-run throughput envelopes

## Open Children

- [T300d](../todo.md#open-leaves): subagent-driven implementation and test execution for `extension/batch_mpc_planner`.

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

## Git Refs

- Last Feature Commit: `pending`
- Last Verified Commit: `pending`
- Current Work Ref: `working tree on top of 130c635 (design hardened with P0/P1 review contracts)`
- Key Files:
  - [../../docs/superpowers/specs/2026-05-11-unified-dense-mpc-backend-design.md](../../docs/superpowers/specs/2026-05-11-unified-dense-mpc-backend-design.md)
  - [../log/2026-05-11-1050-t300-unified-dense-mpc-backend-design.md](../log/2026-05-11-1050-t300-unified-dense-mpc-backend-design.md)
  - [../log/2026-05-11-1104-t300-subagent-design-review.md](../log/2026-05-11-1104-t300-subagent-design-review.md)
  - [../log/2026-05-11-1110-t300-spec-hardening.md](../log/2026-05-11-1110-t300-spec-hardening.md)

## Next Step

- Run `env_isaaclab` GPU runtime acceptance for 4096-env async dirty scheduling and timing counters.
- Close T300d after runtime acceptance evidence is logged.

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
- residual risk:
  - true 4096-env throughput/timing counter extraction remains unstable in current semantic viewer runtime diagnostics path:
    - high-scale PhysX pair-capacity pressure and/or CUDA device-side asserts
    - long headless runtime loops can still prevent deterministic counter-capture assertions
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
