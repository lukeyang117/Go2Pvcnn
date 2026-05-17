# AI：batched planner — time complexity only (N only if serial)

## Navigation

- paired human: [../human/human-13-batched-planner-swing-stance-ik-complexity.md](../human/human-13-batched-planner-swing-stance-ik-complexity.md)
- master index: [../index.md](../index.md)

## Rule

- **Time complexity** only. No separate “tensor work” table.
- Include **N** in Big-O **only** when code has **Python serial over envs** (e.g. `for idx in range(self.batch_size)`, `.item()` per env in a loop).
- Otherwise **omit N**; keep **T**, **K** if relevant.

## One-shot `batched_generate_trajectory` (`trajectory.py:121`)

| Block | Order | N in formula? |
| --- | --- | --- |
| gait schedule | **O(T)** | no |
| foothold + spiral prep | **O(K)** | no |
| touchdown eval | **O(1)** | no |
| `max_height_along_segment` ×4 | **O(N)** | yes — `terrain.py:266-287` |
| swing targets | **O(N·T)** | yes — `swing.py:149-150`, `104-115` |
| integrate base (×2) | **O(T)** | no |
| `batched_estimate_terrain` | **O(T)** | no — `terrain_estimator.py:127-134` |
| `batched_solve_base_trajectory` | **O(T)** | no — `base_solver.py:134-136` |
| IK/FK + tail | **O(T)** | no — `ik.py` |

**Sum:** **O(N·T + N + T + K)**; dominant **O(N·T)** for large N,T.

**N = 1:** **O(T + K)**.

## T302 MPC Collision/Semantic Addendum

Active MPC path: `Go2Pvcnn/extension/batch_mpc_planner`.

- `kinematics.py` now exposes FK leg points: foot, knee, and shank world samples.
- `terrain_clearance.py` owns T302 height-field body/leg collision losses, stance semantic obstacle loss, and all-scanner obstacle risk scaling.
- `registry.py` wires body collision, leg collision, stance semantic, and obstacle risk scaling into total loss; risk diagnostics are exported as non-loss breakdown tensors.
- `tracking.py` accepts per-env linear/yaw scale tensors.
- `planner.py` copies loss diagnostics into `cost_breakdown`, so collision/risk diagnostics are available even when full `loss_breakdown` is not emitted.
- Headless acceptance lives in `Go2Pvcnn/tests/test_mpc_body_leg_collision_headless.py`; evidence is logged in [../log/2026-05-16-2309-t302-mpc-body-leg-collision-implementation.md](../log/2026-05-16-2309-t302-mpc-body-leg-collision-implementation.md).
