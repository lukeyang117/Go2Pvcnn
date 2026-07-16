# Joint MPC RTI Viewer Grounding Fix

- Purpose: fix the reproduced foot-flying behavior, preserve zero-command fixed-trot leg motion without root drift, and verify stance contact for varied command directions and magnitudes.
- Stage: Isaac joint adapter, rolling RTI stance state/loss/linearization, viewer direct playback.
- Related todo: [T302v.3](../todo/T302v-joint-mpc-rti-gpu.md).
- Baseline Ref: `7be3523`.
- Candidate Ref: `2f6f09d`.
- Key Files: `integration/joint_order.py`, `integration/isaaclab_adapter.py`, `planner.py`, `losses/contact.py`, `runtime/cuda_graph.py`, `joint_mpc_rti_viewer_reproduction_probe.py`.

## Changes

- Added one shared name-based joint-order boundary. Isaac `joint_pos` and `joint_vel` now convert from robot order to planner leg order; viewer playback/readback uses the same helper.
- Added one persistent world-space stance anchor per leg to `JointMpcRtiSolverState`; touchdown updates the anchor and consecutive stance carries it across rolling MPC calls and CUDA Graph replay.
- Wired existing stance XY and ground-Z residuals into the RTI joint linearization.
- Changed stance/touchdown contact height from terrain point height to Go2 foot-center height using the existing physical collision radius `0.022m`.
- Kept fixed trot active at zero command. No stand gate, output projection, foot snapping, specified crossing leg, or semantic hard gate was added.
- Raised existing `max_nominal_joint_velocity` from `5` to `9rad/s`, the minimum tested value that lets the four-step swing return to touchdown without velocity clipping.
- Used dense per-leg `3x3` Jacobians instead of sparse `[3x12]` blocks for stance gradient/curvature computation.

## TDD And Regression

- Joint-order RED: `10/12` position entries mismatched, greatest sentinel difference `6.0`; GREEN adapter/viewer focused `34 passed`, backend `14 passed`.
- Rolling RED: old nine-command x1 loop reached `0.0974m` stance height error; GREEN contact-surface residual `<=0.012m`, zero root XY/yaw drift, finite state, and no terrain penetration.
- Final joint suite: `99 passed`.
- Old MPC compatibility subset: `193 passed`.
- Pycompile and `git diff --check`: exit `0` before the feature commit.

## Real Isaac Viewer Acceptance

Command matrix, `8` rolling cycles each:

```text
standstill [0,0,0]
forward slow/fast [0.1,0,0], [0.4,0,0]
backward [-0.25,0,0]
lateral left/right [0,+/-0.25,0]
yaw [0,0,0.5]
mixed [0.2,0.15,0.3], [0.35,-0.2,-0.35]
```

Results:

- overall acceptance: pass.
- adapter joint position/velocity order error: `0` for all cases.
- max stance contact-surface residual `|foot_center_z - terrain_z - 0.022|`: `0.010303m`.
- max joint step: `0.183284rad`, down from reproduced `2.530938rad`.
- max viewer actual-vs-planner foot error: `9.65e-7m`.
- standstill root XY drift: `2.09e-11m`.
- standstill root yaw drift: `2.45e-12rad`.
- moving command root displacement/yaw follows direction and magnitude over the eight x1 steps.

Command:

```bash
PYTHONUNBUFFERED=1 MPC_TEST_DEVICE=cuda:2 JOINT_MPC_VIEWER_REPRO_CYCLES=8 JOINT_MPC_VIEWER_REPRO_OUTPUT=/tmp/joint_mpc_viewer_grounding_verified.json /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_viewer_reproduction_probe.py
```

## Performance Status

The prior uncontended accepted ref remains `4bad2e0`: `1024 x H16 x 1000` synchronous exact EDT + MPC in `4469.05ms`.

No uncontended GPU was available for the current candidate. All four cards were occupied by long-running external training/evaluation jobs. Current samples varied materially on the same GPU:

- GPU2 `1000` steps: `5629.56ms`, then `8990.98ms` and `8907.50ms` as external load changed.
- GPU2 best `200`-step sample: mean `5.118ms/refresh`.
- GPU0/GPU3 `200`-step samples: mean `5.418/5.670ms`.
- field mean stayed around `1.68-1.75ms`; MPC time varied with contention.
- nonfinite count remained `0`; peak allocated `1038.5MiB`.

Result: functional viewer/stance acceptance passes. The current candidate's five-second `1024 x 1000` performance acceptance is not claimed and remains an explicit uncontended-GPU recheck.
