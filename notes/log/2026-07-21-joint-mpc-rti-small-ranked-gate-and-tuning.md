# Joint MPC RTI Small Ranked Gate And Tuning

## Purpose

Close the Task 14 ranked small-obstacle behavior gate with production defaults, while preserving the completed flat behavior and the cold-once/warm-only lifecycle.

## Stage

- Planner: `extension/joint_mpc_rti`
- Plan: Task 14, Steps 1-3
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md)

## Inputs

- Ranked cells: zero, signed forward, signed lateral, and signed pure-yaw commands
- Terrain: native sphere small-obstacle field, phase `0`, offset `0`
- Horizon: H30
- Rolling steps: small `160`, flat regression `144`
- Device: `cuda:1`

## Diagnosis And Tuning

The initial production baseline crossed safely for all four translation commands, with zero per-part collision, penetration, touchdown-on-small, stance-on-small, and airborne touchdown. Its remaining failures were root roll/pitch rate and late pure-yaw instability.

Single-variable scans showed:

- `posture_roll_pitch=100` fixed the 32-step rate but did not prevent late 160-step yaw instability.
- `posture_roll_pitch=1000` made all translation cells pass, but pure yaw still saturated joint/trust behavior.
- `root_orientation_trust=0.05` still failed both yaw signs.
- `root_orientation_trust=0.02` fixed both yaw signs, but required stronger posture stabilization for lateral contact quality.
- `posture_roll_pitch=10000` with `root_orientation_trust=0.02` passed all seven complete 160-step ranked cells.

Only these two approved existing parameters changed. No recovery, projection, candidate filter, semantic branch, new loss, or hard constraint was added.

## Verification

Focused test:

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest \
  Go2Pvcnn/tests/joint_mpc_rti/test_small_acceptance.py -q
```

Result: `7 passed in 5.35s`.

Production-default ranked small:

```bash
PYTHONPATH=Go2Pvcnn /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/joint_mpc_rti/run_monitored_joint_mpc.py \
  --timeout-seconds 600 --heartbeat-seconds 10 --gpu-index 1 -- \
  /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python \
  Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py \
  --stage small --ranked-cells 7 --steps 160 --device cuda:1
```

Result: `7/7` cells and the small gate passed in `73.161s`.

- strict crossing: `1.0` for every applicable translation cell
- root roll/pitch rate max: `0.018632 rad/s`
- root yaw-rate error max: `0.168273 rad/s`
- joint safe margin min: `0.295501 rad`
- joint step max: `0.233204 rad`
- stance XY slip max: `0.0001222 m`
- swing clearance min: `0.004138 m`
- all part collision rates, maximum penetration, touchdown-on-small, stance-on-small, and airborne touchdown: `0`
- unexpected cold restart count: `0`

Production-default ranked flat regression: `7/7` cells and the flat gate passed in `62.189s`; joint safe margin min was `0.331372 rad`, joint step max `0.229110 rad`, stance XY slip max `0.0000569 m`, and swing clearance min `0.004099 m`.

## Result

Task 14 ranked behavior and approved tuning are green. The formal small matrix and real viewer crossing are not yet claimed.

## Follow-Up

The full Cartesian selector expands to `19 x 5 x 24 x 13 = 29,640` cells. Add deterministic resumable test sharding and a complete merge gate before starting that run; coverage and per-cell thresholds must remain unchanged.

## Git Refs

- Baseline Ref: `41f1b18`
- Candidate Ref: `41f1b18` plus current Task 14 working tree
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/config.py`, `Go2Pvcnn/tests/joint_mpc_rti/run_joint_acceptance.py`, `Go2Pvcnn/tests/joint_mpc_rti/test_small_acceptance.py`
