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
