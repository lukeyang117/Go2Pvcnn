# T302h Semantic Obstacle Jitter Reproduction

## Current State

- T302h is a child of [T302](T302-mpc-body-leg-height-field-collision-safety.md), related to [T300f](T300-unified-dense-mpc-backend.md) swing trajectory quality and [T302g](T302g-mpc-semantic-rl-training-config.md) semantic MPC rollout.
- User report: when the robot reaches semantic small/large objects, root and feet shake, foot trajectories become discontinuous, collision avoidance is unreliable, and the planned feet are not smooth enough to look like normal walking.
- Scope for this branch: reproduce and quantify under real IsaacLab first. Do not change production MPC behavior yet.
- New probe added:
  - [../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py](../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py](../../Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py)
- The probe places env0 near S4 semantic-course anchors for `small` and `large`, runs 300-step MPC plans, reuses T300f swing metrics, and adds:
  - root/foot step and acceleration spike ratios
  - foot/stance/swing/touchdown semantic small/large rates
  - foot semantic penetration rates
  - root semantic occupancy/penetration rates
  - min root distance to the targeted semantic object and command-direction crossing flag
- First `env_isaacsim` reproduction ran on `CUDA_VISIBLE_DEVICES=2`, `--device cuda:0`, `small,large`, 3 commands, 300-step horizon.
- Reproduction signals:
  - `small + forward_v050`: `stance_on_small_rate=0.053571`, `foot_on_small_rate=0.037500`, `foot_accel_max_to_mean=30.885206`, `root_accel_max_to_mean=16.538178`, no command-direction crossing.
  - `small + forward_yaw_v050_vy025_yaw100`: crossed the small anchor path with `min_root_distance_to_obstacle=0.252402`, `root_accel_max_to_mean=40.758645`, `foot_accel_max_to_mean=24.883740`.
  - `large + forward_v050`: crossed the large anchor path with `foot_on_large_rate=0.020000`, `swing_over_large_rate=0.039474`, `large_penetration_rate=0.000833`, `root_accel_max_to_mean=35.116199`, `root_step_max_to_median=58.634496`.
  - `large + yaw100`: `foot_on_large_rate=0.032500`, `swing_over_large_rate=0.064784`, `large_penetration_rate=0.001667`, `min_z_quadratic_r2=0.351219`.
- 2026-05-24 variant sweeps cover low-small, large, and high-small (`semantic_small_height_m=0.46`) with test-only loss directions.
  - Broad variants did not produce a stable production winner: `contact_only_semantic` worsened semantic contact, `risk_crossing` helped root-on but retained stance/penetration side effects, and `high_body_margin` reduced stance while adding root/body side effects.
  - Focused `body_stance_crossing` is the best current test-only hypothesis: in one focused sweep low-small contact went to `0.0` with `2/3` crossing, high-small did not cross, and large min root distance reached `0.350m`.
  - `body_stance_crossing_smooth` is rejected because it reintroduced low-small stance contact and high-small crossing.
  - A temporary production-default attempt using `body_stance_crossing` failed real baseline verification (`low-small stance contact_sum=0.0690`, high-small `cross_count=1/3`, high-small `rootmax=0.0067`) and was reverted. Production MPC defaults remain at the prior accepted T300f/T302 settings.
- 2026-05-24 continued sweep tested additional test-only directions: lighter body margin, hard contact-only, progress-only crossing, long swing windows, opt40 convergence, foot soft fields, and touchdown support search.
  - Best scalar/config-only hypothesis so far: `opt40_body_hard_contact_progress`, which reached zero semantic contact on low-small and large in one single-cycle sweep and kept large jump lower than prior candidates.
  - It is still not robust: high-small later showed `min_dist=0.075` and `cross=1/3`, and risk/highbody variants that fix high-small tend to regress low-small stance.
  - Conclusion: more scalar-only tuning is likely insufficient; next test should be a structural test-only loss with class-conditioned high/large body margin and low-small stance/touchdown foot exclusion.
- 2026-05-24 structural/selector sweep added probe-only custom loss injection, semantic policy violation metrics, class-split low-small foot exclusion, large-only body avoidance, smooth variants, and a multi-candidate selector.
  - Best new evidence: `select_policy_pool` improved low-small/large to `policy_violation=1/6`, `max_stance=0`, `max_root_on=0`, `max_footacc=23.544`, `max_rootacc=17.625`; high-small reached `policy_violation=0/3`.
  - This is still not production-ready because low-small/large retained one policy violation and single-loss variants were run-variable.
  - Next direction should be a stronger test-only mode/candidate gate, not more scalar tuning: explicit low-small crossing candidate, explicit high/large avoidance candidate, and selector rejection of wrong crossing policy before smoothness ranking.
- 2026-05-24 clearance/jitter selector sweep tested stronger hard-cross candidates, jitter-aware ranking, risk/priority avoidance pools, and a clearance-aware policy metric.
  - `select_policy_class_hardcross_margin` is rejected: low/large regressed to `policy_violation=2/6`.
  - `select_policy_class_wide_margin` is policy-promising: low/large reached `0/6` and high-small `0/3`, but continuity spikes remained high in some runs.
  - `select_policy_class_jitter_margin` reduced low/large continuity spikes in one comparison (`footacc 26.383 -> 17.598`, `jump 61.397 -> 8.624`) but did not fully stabilize large avoidance.
  - `select_policy_class_clearance_jitter_margin` is the best next test-only direction: targeted large reached `clearance_policy=0/3`, high-small reached `clearance_policy=0/3`, stance/root contact stayed `0`, and high-small max jump was `5.632`.
  - Important metric correction: old `crossed_obstacle_along_command` over-penalizes valid large-obstacle bypass because passing the obstacle's along-command projection line can still be a safe lateral avoidance when margin/root/stance are clean.
- 2026-05-24 corrected task metric pass followed the user's clarified definition:
  - low-small crossing is no longer “root reached backside”; it requires commanded-path overpass, no stance/touchdown semantic contact, no foot penetration, and bounded continuity.
  - low-small crossing now allows `root_on_semantic_rate > 0` because the root may pass above the small obstacle.
  - yaw/body-frame test setup was corrected so linear commands start at yaw `0` instead of rotating body-frame `vx,vy` twice.
  - `select_policy_class_straight_task_jitter_margin` is the best current test-only selector for corrected task semantics: low-small forward succeeds, high-small reaches `semantic_task=0/3`, but low-small mixed forward+lateral+yaw and large forward remain open.
- 2026-05-24 local-overpass/post-blend selector pass resolved the remaining test-only gates under real IsaacLab 300-step sweeps:
  - Low-small crossing metric now uses local obstacle-lane overpass (`ever_crossed` plus local lateral lane) instead of global full-horizon path drift.
  - `path_tube_low_small_task` as an optimizer loss was rejected because it worsened mixed-command path drift and root acceleration.
  - Large-forward failure was traced to frame-0 foot handoff jump, not mid-obstacle semantic collision; probe-only `post_blend_body_hard_contact_only` validates first-frame blending as the likely production direction.
  - Final test-only selector `select_policy_class_large_smooth_metric_margin` reached default low-small/large `semantic_task=0/6`, low-small linear success `2/2`, large success `3/3`, and high-small `semantic_task=0/3`, high-small avoidance `3/3`.
  - Production planner/runtime code remains unchanged; this is evidence for the next production design, not an implementation.
- 2026-05-24 loss-only sweep followed the user's constraint to avoid selector, handoff blend, nominal-before logic, and postprocess repair.
  - Added probe-only loss variants: `loss_low_small_cross_v1`, `loss_high_large_avoid_v1`, `loss_continuity_anchor_v1`, `loss_semantic_all_v1`, and focused v2-v6 variants.
  - Best loss-only result: `loss_low_small_cont_v2` solved low-small mixed `vx=0.50, vy=0.25, yaw=1.00`: `semantic_task 1->0`, `small_overpass_success 0->1`, stance semantic `0.035313->0`, foot accel `18.57->14.54`, root accel `31.43->6.76`, score `945.5->221.2`.
  - Large-forward remains unresolved by loss-only tests: v2 can clear margin (`0.0305->0`) but fails continuity; v3/v5 make continuity clean (`jump 102.4->1.93/1.73`, foot accel `32.68->24.30/18.65`) but keep margin deficits (`0.066/0.077`); v6 S-curve introduces root semantic contact (`root_on=0.0667`).
  - High-small baseline was already clean in the tested rows; several loss-only variants regressed it, mainly through new margin or continuity failures.
  - Conclusion for this pass: loss-only is promising for low-small crossing and useful for isolating large-forward tradeoffs, but no tested loss-only large-forward candidate satisfies both avoidance clearance and trajectory continuity at the current metric gate.
- 2026-05-24 nominal-before command shaping sweep followed the user's accepted direction A in test/probe code only.
  - Added probe-only command shaping diagnostics and variants: `nominal_cmd_shape_a_v1`, conservative v4, low-small continuity controls v4/v5, and combined semantic routers v6-v8.
  - Large-forward targeted result: conservative command shaping passed (`semantic_task 1->0`, `large_avoid 0->1`, score `1196.588->236.173`, jump `102.418->2.778`, footacc `32.687->3.021`, rootacc `29.909->5.819`, margin deficit cleared).
  - High-small targeted result: v7 passed `3/3` and improved score `423.703->277.413`, jump `28.329->8.664`, rootacc `20.178->14.820`; v8 regressed one boundary gate.
  - Low-small mixed remains best with `loss_low_small_cont_v2`; combined command-shaping variants are not stable because stronger low-small continuity/anchor variants can create high `foot_accel_max_to_mean` spikes in repeat runs.
  - Conclusion: command shaping before nominal is useful for high/large avoidance, but a single combined test-only variant is not yet robust enough to productionize. The next test should trace optimizer initial residuals/per-loss breakdown for the inconsistent low-small and large-forward runs before adding more scalar variants.
- 2026-05-24 seeded diagnostics and v10 sweep corrected the prior comparison method:
  - The probe now records worst foot/root acceleration frame/leg/value and deterministic per-case/effective-candidate seeds.
  - This showed the earlier `combined_v8` inconsistency was caused by random nominal phase / run-order effects: with fixed seed, rows with the same `effective_candidate` are identical.
  - Low-small mixed failure in `loss_low_small_cont_v2` is a real mid-trajectory foot spike (`frame=103`, `leg=0`, `value=0.252m`, `foot_accel_max_to_mean=40.402`).
  - New test-only low-small `loss_low_small_stepcap_v4` solves mixed `vx=0.50, vy=0.25, yaw=1.00` with `semantic_task=0`, `small_overpass=1`, contact/continuity `0`, score `945.496 -> 234.565`.
  - Pure forward low-small needs a different loss: `struct_lowfoot_cross_hard` passes with score `63.825`, while stepcap v4 regresses pure forward.
  - New test-only `nominal_cmd_shape_a_combined_v10` routes low-small pure forward to `struct_lowfoot_cross_hard`, low-small mixed/yaw to `loss_low_small_stepcap_v4`, and high-small/large to conservative nominal-before command shaping.
  - v10 passes the seeded low-small/large four-row sweep: `semantic_task 3/4 -> 0/4`, `small_overpass 0/2 -> 2/2`, `large_avoid 1/2 -> 2/2`, contact/continuity violations `0`, score mean `816.388 -> 218.304`.
  - v10 also passes high-small `0.46m` three-command sweep with `semantic_task=0/3`, `large_avoid=3/3`, score `449.305 -> 323.873`.
  - Production planner/runtime code remains unchanged.
- 2026-05-24 production v10 implementation is now in the working tree.
  - Production added `semantic_policy.py`, high-small/large internal planning-command shaping, low-small foot-crossing loss, and low-small mixed/yaw stepcap continuity loss.
  - The implementation intentionally does not add selector/postprocess/handoff-blend machinery.
  - First production pass shaped only the nominal seed and failed large-forward continuity (`semantic_task=1/4`, large-forward jump `33.965`), proving optimizer/tracking still pulled toward the original command.
  - Final production pass uses the shaped command as the internal planning command for nominal and optimizer/loss terms when high-small/large avoidance is active.
  - Backend verification passes (`103 passed`).
  - Real IsaacLab 300-step low-small/large production baseline passes: `semantic_task=0/4`, `small_overpass=2/2`, `large_avoid=2/2`, contact/continuity violations `0`, score mean `217.941`.
  - Real IsaacLab high-small `0.46m` production baseline passes: `semantic_task=0/3`, `large_avoid=3/3`, contact/continuity violations `0`, score mean `276.147`.
- 2026-05-24 deterministic replan phase follow-up is now implemented in production runtime config.
  - Multi-cycle probes showed the remaining large-forward failure was not semantic contact; it was an intermittent replan-boundary gait phase discontinuity.
  - Probe-only `phase_fixed_probe` improved large-forward `cycles=6`, `playback_frame=20`: `semantic_task 1/6 -> 0/6`, `large_avoid 5/6 -> 6/6`, continuity `1/6 -> 0/6`, max jump `52.376 -> 23.012`, max root accel ratio `21.194 -> 9.172`, score mean `414.295 -> 336.443`.
  - Production now sets `MpcRuntimeCfg.randomize_replan_phase=False` by default and exposes `mpc_randomize_replan_phase` as a task override.
  - Production baseline after the default change matches the `phase_fixed_probe` large-forward result: `semantic_task=0/6`, `large_avoid=6/6`, contact/continuity `0`.
  - Low-small/large single-cycle acceptance remains clean: `semantic_task=0/4`, low-small overpass `2/2`, large avoid `2/2`, contact/continuity `0`.
  - High-small `0.46m` acceptance remains clean: `semantic_task=0/3`, large/high avoid `3/3`, contact/continuity `0`.
  - Remaining risk: long playback still shows high `playback_foot_error_max` in some frame-299 rows, so planned-foot vs realized IK/FK foot mismatch is not fully solved by deterministic phase.
- 2026-05-25 low-small foot-over reproduction follows the user's clarified failure.
  - Added probe-only `foot_over_low_small_*` metrics and tightened low-small task gate so root/local-lane overpass is not enough.
  - Real IsaacLab production baseline low-small now reproduces the issue: `foot_over_low_small_success=0/2`, `small_overpass=0/2`, `semantic_task=2/2`.
  - Contact and continuity remain clean (`stance=0`, `touchdown=0`, penetration `0`, max jump `4.222`, max foot accel ratio `6.692`), so this is specifically a "feet route around the small obstacle" failure.
  - Nearest foot-to-obstacle-center lateral distances are about `0.105-0.106m`, outside the current `0.08m` foot-over footprint threshold.
  - Production planner code remains unchanged in this reproduction pass.
- 2026-05-25 loss-only foot-over sweep tested the user's requested constraint: modify only test/probe losses, not production runtime.
  - Added sensor-pose-aware probe grid conversion; before this, foot-over losses could chase local scanner coordinates instead of world obstacle coordinates.
  - Added and tested loss-only variants `footover_v1-v3`, `gate_v4-v6`, `leg_v7-v8`, `wide_v9-v10`, and `cap_v11-v12`.
  - Best current result is `loss_low_small_footover_gate_v4`: `foot_over_low_small_success=2/2`, min lateral `0.105655m -> 0.037337m`, frame count mean `10`, and contact/penetration `0`.
  - The same best result fails continuity: `foot_accel_max_to_mean 4.545 -> 57.133`, so `semantic_task` remains `2/2`.
  - Harder smoothness, leg-consistency, wider gates, and explicit step/accel caps did not solve the tradeoff; they either lost foot-over success or worsened ratio/score.
  - Production planner code remains unchanged in this pass.
- 2026-05-25 production rolling25 low-small foot-over pass follows the user's correction that 300-step testing must be `25`-step MPC replans over a total 300-step rollout, not a single 300-frame plan.
  - Probe rolling logic now forces each MPC segment to horizon `25`, writes the selected frame to Isaac, refreshes scanner/state, and replans until `300` total frames are collected.
  - Production `low_small_foot_over_loss` now includes path/window hooks plus stronger height clearance; selected defaults are `clearance_m=0.065`, `z_weight=420`, `path_curve_weight=120`, `path_curve_z_weight=90`.
  - Production `plan_segment` now replaces non-finite optimizer rows with a grounded standstill fallback before returning, preventing NaN plans from being written back into Isaac and poisoning later replans.
  - True rolling25 low-small acceptance now passes: `semantic_task=0/2`, `small_overpass=2/2`, `foot_over=2/2`, stance/touchdown/penetration `0`, no non-finite diagnostics.
  - Remaining risk: `min_z_quadratic_r2` is only `0.325-0.397`, so the foot crosses continuously enough for the current gates but is not yet a perfect visual parabola; `playback_foot_error_max≈0.29m` remains a separate planned-vs-realized mismatch.

## Open Children

| Child | Status | Priority | Purpose | Primary Files |
| --- | --- | --- | --- | --- |
| T302h.1 | verify | P0 | Keep semantic-obstacle jitter/collision reproduction probe usable and compare future candidate directions against the same metrics | `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`, `Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py` |
| T302h.2 | todo | P0 | Run a longer multi-cycle near-obstacle sequence and, if needed, split one process per semantic class/command to avoid Isaac fixture reuse caveats | `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`, `notes/log/` |
| T302h.3 | verify | P0 | Test candidate directions inside the probe only, then rank by semantic collision rate plus T300f swing/root jitter metrics before production edits | `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py` |
| T302h.4 | todo | P0 | Convert the best test-only hypothesis into a robust acceptance gate: multi-cycle low-small, high-small, and large must pass as production baseline before any default loss change remains | `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`, `Go2Pvcnn/extension/batch_mpc_planner/config.py` |
| T302h.5 | todo | P0 | Test a structural loss direction in the probe only: class-conditioned high/large root/body margin plus low-small stance/touchdown foot exclusion with differentiable foot soft field | `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py` |
| T302h.6 | verify | P0 | Test a stronger mode/candidate gate after structural selector evidence: low-small crossing candidate + high/large avoidance candidate + policy/clearance-first selector before production changes | `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py` |
| T302h.7 | todo | P0 | Build a multi-cycle acceptance gate using clearance-aware policy: low-small linear crossing, high-small/large clearance avoidance, zero stance/root semantic contact, bounded jump/accel | `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py` |
| T302h.8 | todo | P0 | Split corrected task gate by command frame: low-small pure forward overpass, low-small mixed `vx/vy/yaw` path-frame overpass, high-small avoidance, large-forward avoidance | `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py` |
| T302h.9 | todo | P0 | Convert the passing test-only selector direction into a production design: local low-small overpass gate, high/large class selector, and first-frame foot handoff/blending without corrupting contact semantics | `Go2Pvcnn/extension/batch_mpc_planner/*`, `Go2Pvcnn/extension/viz/go2_foostep_planner.py` |
| T302h.10 | verify | P0 | Record loss-only evidence under the user's constraint and decide whether to continue loss-only large-forward tuning or revisit the rejected selector/handoff design | `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py` |
| T302h.11 | verify | P0 | Diagnose run/process sensitivity in combined nominal command shaping: same low-small/large directions can pass targeted runs but fail combined/rerun continuity gates | `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`, `tmp/t302h/` |
| T302h.12 | verify | P0 | Validate `nominal_cmd_shape_a_combined_v10` as the current best test-only direction across low-small pure/mixed, high-small, and large before deciding production scope | `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`, `tmp/t302h/` |
| T302h.13 | verify | P0 | Productionize v10 without selector/postprocess: internal high/large planning-command shaping plus low-small foot/stepcap losses, then verify real IsaacLab corrected task metrics | `Go2Pvcnn/extension/batch_mpc_planner/*`, `Go2Pvcnn/tests/test_batch_mpc_backend.py`, `tmp/t302h/` |
| T302h.14 | verify | P0 | Keep deterministic replan phase as the production default and broaden multi-cycle acceptance; separately investigate long-playback planned-foot vs realized-foot mismatch | `Go2Pvcnn/extension/batch_mpc_planner/config.py`, `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`, `tmp/t302h/` |
| T302h.15 | todo | P0 | Fix low-small crossing semantics so swing feet actually pass over the small obstacle, not around it, while preserving no contact/penetration and continuity | `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`, `Go2Pvcnn/extension/batch_mpc_planner/*`, `tmp/t302h/` |
| T302h.16 | todo | P0 | Continue loss-only design only if it encodes swing-window shape/timing directly; scalar smoothness/cap tuning has already failed to combine foot-over and continuity | `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py`, `tmp/t302h/` |
| T302h.17 | todo | P1 | Improve visual parabola quality beyond the passing continuity gate: raise `min_z_quadratic_r2` without reintroducing foot penetration or acceleration spikes | `Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py`, `Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py` |

## Closed Children Archive

- None yet.

## Related Logs

- [../log/2026-05-24-1110-mpc-semantic-obstacle-jitter-reproduction.md](../log/2026-05-24-1110-mpc-semantic-obstacle-jitter-reproduction.md)
- [../log/2026-05-24-1223-t302h-semantic-obstacle-variant-sweep.md](../log/2026-05-24-1223-t302h-semantic-obstacle-variant-sweep.md)
- [../log/2026-05-24-1327-t302h-continued-semantic-loss-direction-sweep.md](../log/2026-05-24-1327-t302h-continued-semantic-loss-direction-sweep.md)
- [../log/2026-05-24-1401-t302h-structural-selector-loss-sweep.md](../log/2026-05-24-1401-t302h-structural-selector-loss-sweep.md)
- [../log/2026-05-24-1501-t302h-clearance-jitter-selector-sweep.md](../log/2026-05-24-1501-t302h-clearance-jitter-selector-sweep.md)
- [../log/2026-05-24-1605-t302h-corrected-small-overpass-task-metric.md](../log/2026-05-24-1605-t302h-corrected-small-overpass-task-metric.md)
- [../log/2026-05-24-1715-t302h-local-overpass-and-post-blend-selector.md](../log/2026-05-24-1715-t302h-local-overpass-and-post-blend-selector.md)
- [../log/2026-05-24-1822-t302h-loss-only-direction-sweep.md](../log/2026-05-24-1822-t302h-loss-only-direction-sweep.md)
- [../log/2026-05-24-1903-t302h-nominal-command-shaping-sweep.md](../log/2026-05-24-1903-t302h-nominal-command-shaping-sweep.md)
- [../log/2026-05-24-1928-t302h-seeded-diagnostics-and-v10-sweep.md](../log/2026-05-24-1928-t302h-seeded-diagnostics-and-v10-sweep.md)
- [../log/2026-05-24-1948-t302h-production-v10-implementation.md](../log/2026-05-24-1948-t302h-production-v10-implementation.md)
- [../log/2026-05-24-2109-t302h-deterministic-replan-phase.md](../log/2026-05-24-2109-t302h-deterministic-replan-phase.md)
- [../log/2026-05-25-1000-t302h-low-small-foot-over-reproduction.md](../log/2026-05-25-1000-t302h-low-small-foot-over-reproduction.md)
- [../log/2026-05-25-1047-t302h-low-small-foot-over-loss-only-sweep.md](../log/2026-05-25-1047-t302h-low-small-foot-over-loss-only-sweep.md)
- [../log/2026-05-25-1222-t302h-rolling25-low-small-foot-over-production.md](../log/2026-05-25-1222-t302h-rolling25-low-small-foot-over-production.md)
- [../log/2026-05-22-1358-mpc-swing-trajectory-quality-reproduction.md](../log/2026-05-22-1358-mpc-swing-trajectory-quality-reproduction.md)
- [../log/2026-05-17-0804-t302-strict-collision-metric-tuning.md](../log/2026-05-17-0804-t302-strict-collision-metric-tuning.md)

## Git Refs

- Last Feature Commit: `pending`
- Last Verified Commit: `working tree, 2026-05-24 21:09 CST production v10 deterministic replan phase + real IsaacLab multi-cycle acceptance`
- Current Work Ref: `working tree`
- Key Files:
  - [../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py](../../Go2Pvcnn/tests/mpc_semantic_obstacle_jitter_probe.py)
  - [../../Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py](../../Go2Pvcnn/tests/test_mpc_semantic_obstacle_jitter_probe.py)
  - [../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py](../../Go2Pvcnn/tests/fixtures/viewer_runtime_diagnostics.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/terrain.py](../../Go2Pvcnn/extension/batch_mpc_planner/terrain.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py](../../Go2Pvcnn/extension/batch_mpc_planner/semantic_policy.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py](../../Go2Pvcnn/extension/batch_mpc_planner/losses/terrain_clearance.py)

## Next Step

- Use the new probe for test-only candidate directions, starting with directions that reduce semantic contact while preserving the accepted `boundary8+accel8` trajectory regularization.
- Broaden from one-cycle 300-step plans to multi-cycle near-obstacle replans to quantify the user's repeated “walks to semantic object then shakes” observation over time.
- Treat `body_stance_crossing` as the next hypothesis only; do not productionize it until it passes production-baseline probes for low-small, high-small, and large in the same verification pass.
- Stop blind scalar-only sweeps unless they are targeted controls. Next useful work is a test-only structural loss, then rerun the same low-small/large/high-small metrics.
- After the clearance/jitter selector sweep, prioritize a multi-cycle acceptance gate around `select_policy_class_clearance_jitter_margin`: old projection-line crossing is insufficient for large/high-small avoidance, so the acceptance metric should use clearance, root/stance semantic contact, foot penetration tolerance, jump, boundary, and acceleration together.
- After the corrected task-metric pass, keep `select_policy_class_straight_task_jitter_margin` as a test-only hypothesis only. Do not productionize until low-small mixed forward+lateral+yaw and large-forward avoidance pass the corrected task gate.
- After the local-overpass/post-blend pass, keep `select_policy_class_large_smooth_metric_margin` as the best test-only hypothesis only. The next production step should target first-frame foot handoff/blending and class-conditioned candidate selection; do not copy probe post-processing directly into production without a handoff design and multi-cycle acceptance.
- After the loss-only pass, `loss_low_small_cont_v2` is the best loss-only small-obstacle crossing direction. Large-forward should not be claimed solved by loss-only because the tested directions split into either clearance-clean/jittery or smooth/too-near. If continuing strictly loss-only, the next test should target a less conflicting high/large avoidance loss; otherwise the earlier selector/handoff evidence remains stronger but violates the user's current design constraint.
- After the nominal command-shaping pass, keep conservative command shaping as a useful high/large avoidance component, not a complete solution. Do not productionize combined v6-v8; first add diagnostics for optimizer initialization/per-loss breakdown around the low-small mixed and large-forward failing frames.
- After the seeded diagnostics/v10 pass, keep `nominal_cmd_shape_a_combined_v10` as the best current test-only direction. It is not production code yet, but it gives a concrete production design candidate: deterministic/phase-aware comparison, low-small command-conditioned loss path, and high/large conservative nominal-before command shaping. Before productionizing, decide whether low-small pure/mixed command-conditioned loss selection is acceptable and whether the remaining low `min_z_quadratic_r2` needs a shape-specific loss.
- After the production v10 implementation pass, the next useful work is broader acceptance rather than more test-only tuning: multi-cycle near-obstacle replans, T302/T300f regression sweep, and optional stricter swing-shape visual/R2 work. Do not reintroduce selector/postprocess unless a new metric failure requires it.
- After the deterministic replan phase pass, keep the default phase deterministic unless a future training experiment explicitly overrides it. Next useful work is either broader multi-cycle low-small/high-small/large acceptance or a separate planned-foot vs realized-foot mismatch investigation; do not conflate that playback mismatch with semantic avoidance failure.
- After the rolling25 production pass, use `tmp/t302h/production_true_rolling25_low_small_clearance065.jsonl` as the current low-small acceptance reference. Do not return to single 300-frame MPC plans for this issue. Next work should either broaden true rolling25 to high-small/large non-regression or target T302h.17 visual R2 improvement without breaking the accepted `semantic_task=0/2`.

## Node Details

### T302h.1 Reproduction Probe

- why-created: the previous T302 strict metrics showed `17/17` rows with collision ratios `0.0`, but the new user report describes runtime shaking and semantic-object contact near small/large objects. This is a metric-anomaly expansion and needs its own branch page.
- probe contract:
  - real IsaacLab only for runtime reproduction
  - start near semantic S4 anchors
  - 300-step horizon by default
  - preserve previous swing metrics: jump ratio, boundary ratio, z unimodal violation, quadratic R2
  - add root/foot acceleration metrics for shaking
  - add semantic object rates split by small/large and stance/swing/touchdown/root
  - output JSONL so future candidate directions can be compared mechanically
- first verification:
  - local helper pytest passed
  - py_compile passed
  - real `env_isaacsim` probe completed with `6` cycle rows and reproduced small/large semantic contact plus high root/foot acceleration spikes

### T302h.3 Test-Only Variant Sweeps

- why-created: user asked to try multiple loss directions in tests first, compare metric optimization, and only then modify real code.
- tested directions:
  - `semantic_strong`: broad semantic stance/touchdown/contact/body increase; rejected for side effects.
  - `contact_only_semantic`: stance/touchdown/contact-only; rejected because low-small/large contact worsened in the refined sweep.
  - `risk_crossing`: stronger high-obstacle risk plus low-small crossing; reduced some root-on metrics but kept stance/penetration side effects.
  - `high_body_margin`: stronger high/large body margin; improved stance but created root/body side effects and poor trajectory metrics.
  - `body_stance_crossing`: high/large body margin + low-small crossing + stance/touchdown rejection; best current hypothesis but not robust enough for production.
  - `body_stance_crossing_smooth`: same plus stronger smoothness; rejected after low-small stance and high-small crossing came back.
- production attempt result:
  - temporary config default change was made and real IsaacLab baseline probes were run.
  - verification failed (`low-small stance contact_sum=0.0690`; high-small crossed in `1/3` commands), so the config change was reverted in the same session.
- conclusion: current evidence supports a more structured loss/acceptance design, not a simple scalar default change.

### T302h.4/T302h.5 Continued Direction Search

- why-created: user asked to keep proposing/testing directions until metrics improve, without modifying running code.
- best scalar/config-only result:
  - `opt40_body_hard_contact_progress` reached zero contact on low-small and large in one single-cycle sweep.
  - low-small: `contact_sum=0.0000`, `cross=2/3`, `jump=20.11`.
  - large: `contact_sum=0.0000`, `min_dist=0.350`, `jump=13.60`.
  - high-small remained unstable: one later run had `min_dist=0.075`, `cross=1/3`.
- rejected follow-ups:
  - `risk` improved high/large but regressed low-small stance.
  - `long_swing` and `support_touchdown` did not robustly remove low-small stance and often worsened continuity.
  - `foot_soft` helped high-small but reintroduced low-small stance contact.
- next hypothesis:
  - existing cfg terms are not class-conditioned enough.
  - test a custom structural loss in the probe only, with high/large root-body avoidance separated from low-small foot-only stance/touchdown exclusion.

### T302h.6 Structural Selector Follow-Up

- why-created: structural loss and smoothness improved contact/jitter but stayed run-variable; selector improved robustness but still left one low-small/large policy violation.
- tested directions:
  - `struct_lowfoot_highbody`: useful low-small crossing/contact signal, but high-small/large side effects and root-on cases remain.
  - `struct_lowfoot_largebody_gentle`: first direction to show a clean `policy_violation=0/6` low-small/large run, but jitter was high and rerun stability was weak.
  - `struct_lowfoot_largebody_gentle_smooth`: lowers continuity spikes but does not make policy robust alone.
  - `select_policy_pool`: best current structural direction; low-small/large `policy_violation=1/6`, high-small `policy_violation=0/3`, no stance/root-on semantic contact in the focused selector sweeps.
- conclusion: a candidate/mode selector is more promising than a single scalar loss, but the current candidate set is not strong enough for production.

### T302h.7 Clearance-Aware Acceptance Gate

- why-created: follow-up selector tests showed `crossed_obstacle_along_command` can mark a clean large-obstacle lateral bypass as a policy violation. This metric anomaly needs a corrected acceptance gate before production changes.
- tested directions:
  - `select_policy_class_hardcross_margin`: rejected; stronger crossing candidate regressed low/large policy to `2/6`.
  - `select_policy_class_jitter_margin`: useful; reduced foot acceleration and jump in one low/large comparison, but not robust enough alone.
  - `select_policy_class_risk_jitter_margin` / `priority_jitter`: useful for avoidance pools but still sensitive to metric definition and run variability.
  - `select_policy_class_clearance_jitter_margin`: best current test-only direction; targeted large and high-small sweeps both reached `clearance_policy=0/3` with no stance/root semantic contact.
- next gate:
  - low-small linear commands must cross.
  - high-small and large must keep clearance margin and zero root/stance semantic contact.
  - foot penetration should distinguish repeated contact from one tiny swing-frame sample.
  - continuity bounds should include `foot_accel_max_to_mean`, `root_accel_max_to_mean`, `worst_max_to_median_step`, and `worst_boundary_to_median_step`.

### T302h.8 Corrected Task Gate

- why-created: the user clarified that small-obstacle crossing must mean root follows the commanded path over the object, not simply reaching the backside. This exposed two metric bugs: `crossed_obstacle_along_command` alone was too weak, and `root_on_semantic_rate` was too strict for low-small overpass.
- tested directions:
  - `straight_low_small_task`: probe-only straight-path/root-clearance loss.
  - `straight_smooth_low_small_task`: same plus stronger foot trajectory regularization; rejected for high foot acceleration in the focused forward case.
  - `select_policy_class_straight_task_jitter_margin`: task-first selector with straight crossing candidates for low-small and risk/avoidance candidates for high-small/large.
- results:
  - low-small pure forward: succeeds with `small_overpass_success=1`, zero stance/touchdown/penetration, and bounded trajectory metrics.
  - high-small: succeeds in the selector sweep with `semantic_task_violation_count=0/3` and `large_avoid_success_count=3/3`.
  - large: improves score and foot/root acceleration versus clearance selector, but large forward remains open because margin deficit and jump spike remain.
  - low-small mixed `vx=0.50, vy=0.25, yaw=1.00`: remains open because current path-frame gate treats commanded lateral/yaw motion as excessive drift or continuity risk.

### T302h.9 Local Overpass And Post-Blend Selector

- why-created: the user clarified that low-small crossing is local passage over the obstacle, while large/high-small must avoid; earlier gates still failed mixed low-small and large forward.
- tested directions:
  - body-frame command-path metrics: useful diagnostic, but full-horizon max drift is too strict for long mixed-yaw plans.
  - `ever_crossed_obstacle_along_command`: fixes cases where the robot crosses the obstacle then later turns back.
  - local overpass gate: low-small succeeds when the trajectory locally passes through the obstacle lane with no stance/touchdown/penetration and continuity clean.
  - `path_tube_low_small_task`: rejected as optimizer loss because it worsened mixed path drift/root acceleration.
  - `post_blend_body_hard_contact_only`: validates that large-forward jump is a frame-0 handoff problem; selected large-forward reaches `large_avoid_success=1`.
- final evidence:
  - default low-small/large combined sweep: selector `semantic_task=0/6`, low-small linear `2/2`, large `3/3`, contact/continuity violations `0`, max jump `15.290`; baseline `semantic_task=3/6`, max jump `105.334`.
  - high-small sweep at `0.46m`: selector `semantic_task=0/3`, large/high avoidance `3/3`, small crossing `0`, max jump `5.305`.
- conclusion: test-only metrics now encode the user's requirement. Production implementation should focus on class-conditioned selection plus first-frame foot handoff/blending, followed by multi-cycle acceptance.
