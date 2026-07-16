# Joint MPC RTI Small-Obstacle Crossing Implementation

- Purpose: implement and verify the approved signed-field, stance-swing-stance, full-leg small-obstacle crossing design without hard behavior gates.
- Stage: `extension/joint_mpc_rti` terrain fields, geometry, continuous merit/GGN losses, rolling x1 behavior, and viewer integration.
- Related todo: [T302v.4](../todo/T302v-joint-mpc-rti-gpu.md).
- Baseline Ref: `b99cda0` for viewer behavior; feature commits `af65eaf`, `136857c`, and `471b304`.
- Candidate Ref: `625768f`.
- Key Files: `config.py`, `losses/semantic.py`, `losses/rollout_objective.py`, `planner.py`, `terrain/csrc/work_efficient_edt_cuda.cu`, `small_obstacle_crossing_probe.py`.

## Implemented Contracts

- Small and large semantic channels use signed boundary distance: outside positive, occupied interior negative, half-cell corrected, finite for empty and full channels.
- `H=16` with `half_cycle_steps=8` covers one per-leg stance-swing-stance event.
- Fixed foot, calf, and thigh geometry uses physical radii and analytic point Jacobians.
- Separate foot/calf/thigh small-object residuals contribute to final merit and GGN/LQ joint directions. Continuous proximity and height weighting use the single elevation map plus signed-distance boundary reconstruction.
- No shape gate, specified crossing leg, fixed avoidance side, foot snapping, projection, repair, or other hard behavior gate was added.
- The rolling acceptance counts only published `x1`, rebuilds the robot-centered `151 x 151` field synchronously, and preserves world obstacle coordinates.

## Verification

CUDA and PyTorch probe on physical GPU3:

```text
torch 2.7.0+cu128
cuda available true, CUDA 12.8
device NVIDIA GeForce RTX 4090
```

Signed terrain correctness:

```bash
CUDA_VISIBLE_DEVICES=3 /mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m pytest -q Go2Pvcnn/tests/joint_mpc_rti/test_terrain_fields.py
```

Result: `23 passed`. This compiles the bbox signed-EDT CUDA candidate and covers CPU/CUDA parity, signs, half-cell correction, degenerate channels, independent 1024-map batches, and dynamic cache batches.

Full joint MPC and legacy regression:

```text
Go2Pvcnn/tests/joint_mpc_rti: 110 passed
legacy batch MPC/reward/viewer subset: 213 passed
git diff --check: exit 0
```

Native crossing matrix, five shapes x three speeds:

```text
strict cross: 254/254 = 100%
minimum per shape-speed cross rate: 100%
foot/calf/thigh/base collision frames: 0/0/0/0
foot/calf/thigh/base maximum penetration: 0/0/0/0m
stance-on-small frames: 0
invalid count: 0
```

Real IsaacLab viewer, nine commands x eight rolling cycles:

```text
passed: true
joint position/velocity order error max: 0
stance ground gap max: 0.010098m
joint step max: 0.183805rad
actual-vs-planner foot error max: 5.38e-7m
standstill root XY drift: 2.09e-11m
standstill root yaw drift: 2.45e-12rad
```

## Performance Boundary

All four RTX 4090 cards were occupied by external workloads during this verification. Per user direction, performance was deferred. The current signed-field candidate does not inherit the earlier unsigned-field `4469.05ms` result and is not declared accepted or rejected from contested measurements.

Open acceptance: on an idle GPU, synchronous signed field + MPC for `1024 x H16 x 1000` must remain below `5s`, advance the field version by `1000`, and produce zero nonfinite values. Metrics and collision requirements must not be relaxed to reach it.
