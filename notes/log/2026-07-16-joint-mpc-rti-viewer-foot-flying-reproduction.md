# Joint MPC RTI Viewer Foot-Flying Reproduction

- Purpose: reproduce the reported viewer behavior where feet fly, do not stay grounded, and behave almost identically for zero and forward commands.
- Stage: `joint_mpc_rti` IsaacLab input adapter -> rolling RTI manager -> viewer direct playback.
- Related todo: [T302v.3](../todo/T302v-joint-mpc-rti-gpu.md).
- Baseline Ref: `f4bafa0` plus the current diagnostic probe.
- Candidate Ref: diagnosis only; no production fix applied.
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/integration/isaaclab_adapter.py`, `Go2Pvcnn/extension/viz/go2_foostep_planner.py`, `Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_viewer_reproduction_probe.py`.

## Procedure

Run a real one-environment `TeacherElevationTrajectoryMpcSemanticEnvCfg_VIEWER` on GPU 2 with the `joint_mpc_rti` backend. For each cycle, use the same viewer path:

```text
state_from_env
-> manager.refresh_from_env(force=True)
-> latest trajectory
-> direct playback frame x1
-> scene write/render/update
-> read actual robot joints and feet
```

Test `8` rolling cycles for both commands:

- standstill: `[0.0, 0.0, 0.0]`
- forward: `[0.3, 0.0, 0.0]`

Command:

```bash
PYTHONUNBUFFERED=1 MPC_TEST_DEVICE=cuda:2 JOINT_MPC_VIEWER_REPRO_CYCLES=8 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_viewer_reproduction_probe.py
```

## Joint-Order Evidence

Isaac robot order:

```text
FL_hip, FR_hip, RL_hip, RR_hip,
FL_thigh, FR_thigh, RL_thigh, RR_thigh,
FL_calf, FR_calf, RL_calf, RR_calf
```

Planner order:

```text
FL_hip, FL_thigh, FL_calf,
FR_hip, FR_thigh, FR_calf,
RL_hip, RL_thigh, RL_calf,
RR_hip, RR_thigh, RR_calf
```

The viewer explicitly converts between these orders during playback/readback. `state_from_env()` directly consumes `robot.data.joint_pos` and `joint_vel` without robot-to-planner conversion.

## Metrics

| Metric | Standstill mean / max | Forward mean / max |
| --- | --- | --- |
| adapter joint-order error | `2.464795 / 2.531080 rad` | `2.464795 / 2.531080 rad` |
| per-cycle joint step max | `2.478560 / 2.530938 rad` | `2.478560 / 2.530938 rad` |
| per-cycle foot step max | `0.590084 / 0.729393 m` | `0.590617 / 0.730515 m` |
| stance ground-gap max | `0.262314 / 0.452360 m` | `0.262314 / 0.452359 m` |
| swing ground-gap max | `0.283795 / 0.539331 m` | `0.283796 / 0.539331 m` |
| actual vs planner joint max | `0 / 0 rad` | `0 / 0 rad` |
| actual vs planner foot max | `5.36e-7 / 9.58e-7 m` | `4.55e-7 / 7.45e-7 m` |

## Result

- Reproduced: yes.
- Primary root cause confirmed: Isaac robot-order joint state is fed directly into a planner-order state vector. After viewer playback writes planner-order output through the correct planner-to-robot permutation, the next RTI cycle reads it back without the inverse permutation. This creates an approximately `2.5 rad` state discontinuity and approximately `0.59 m` foot jump every cycle.
- Viewer playback is faithful: actual joints exactly match the requested planner frame and actual feet agree at micron scale. The visualizer and Isaac writeback are not the source of the large jumps.
- Command independence is confirmed: standstill and forward metrics are nearly identical because the ordering discontinuity dominates command effects.

## Secondary Issue

The primary ordering bug does not explain every lower-amplitude behavior. A separate pure-tensor rolling probe showed that fixed trot phase continues at zero command and can produce stance-ground drift even without the catastrophic ordering mismatch. After the input order is fixed, standstill scheduling/contact-grounding must be evaluated as a separate child issue rather than assumed solved.

## Follow-up

Before production changes, add failing adapter contract tests for both `joint_pos` and `joint_vel`, then apply one robot-to-planner conversion at the IsaacLab input boundary and rerun this exact probe. Only after the large discontinuity is removed should the zero-command fixed-trot issue be tuned.
