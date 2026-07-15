# MPX Reference Reading

- Purpose: map `raw/mpx` to the joint RTI performance design.
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md).
- Reference: MPX `0bddce8`, solver submodule `f397c9a`.
- Chain: `examples/mjx_quad.py` / `examples/multi_env.py` -> `utils/mpc_wrapper.py` -> `jax_ocp_solvers/optimizers.py` -> `primal_tvlqr.py`.
- Findings: multiple-shooting SQP, not MPPI/CEM; `jit(vmap)` environment batching; `associative_scan` temporal solve/rollout; parallel alpha evaluation; shifted warm start.
- Applied impact: fixed compiled program, packed field queries, rollout reuse, diagonal/block Riccati, CUDA Graph replay.
- Result: mapping complete; `raw/mpx` remains reference-only.
