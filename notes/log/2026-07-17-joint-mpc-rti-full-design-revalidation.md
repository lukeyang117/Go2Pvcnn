# Joint MPC RTI Full Design Revalidation

- Purpose: rerun every implemented acceptance in the 2026-07-15 GPU RTI design and its 2026-07-16 continuations, then continue fixing failed gates.
- Stage: `Go2Pvcnn/extension/joint_mpc_rti` signed field, fixed-shape GGN/LQ RTI, rolling `x1`, native-shape behavior, viewer integration, and 1024-env synchronous performance.
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md).
- Baseline Ref: `d954671`.
- Candidate Ref: `9e71ac1`.

## Functional Revalidation

- Initial joint suite: `117 passed`; final joint suite after adding compiled fixed-shape LQ/query/rollout paths and executable compiled/eager parity tests: `120 passed in 65.00s`.
- Legacy MPC/reward/viewer suite: `213 passed`.
- Native strict cross: `185-186/185-186` depending on the repeated deterministic run, overall and minimum shape-speed rate `100%`; foot/calf/thigh/base collision frames and maximum penetration all `0`; stance-on-small `0`; invalid `0`.
- Native stop: `65/65`, maximum consecutive zero-support `1`, maximum total zero-support `2`, root drift `0`, collision/stance-on-small/invalid `0`.
- Real nine-command Isaac viewer: `passed=true`, joint order error `0`, stance gap max about `0.01145m`, joint step max `0.18537rad`, actual/planner foot error below `1e-6m`, standstill XY/yaw drift numerical zero.

## Performance Investigation

The first single-cell formal probe measured `7.13-7.23s` for 1000 synchronous signed field + MPC refreshes. Compiling the fixed-shape LQ construction, terrain/semantic linearizations, query geometry, desired-control/stance-anchor helpers, and rollout geometry reduced MPC from about `4.26ms` to `2.55-2.85ms` without changing any loss or geometry contract.

An initial fused signed-EDT candidate reached `4.768s` only because the probe placed one occupied cell in each channel. Independent code review identified that the inside-distance kernel still scanned the occupied bbox per occupied cell. A density sweep on idle GPU3 showed field-only time growing from about `1.20ms` for one cell to `4.95ms` at `41x41` and `9.72ms` at `51x51`; therefore the `4.768s` result was rejected as probe-specialized.

The formal probe was strengthened to use an `11x11` small footprint and a `41x41` large footprint. The final idle-GPU3 1000-step result is total `7402.55ms`, mean `7.4025ms`, P50/P95/P99 `7.3984/7.4742/7.5162ms`, max `7.7005ms`, field mean/P95 `4.5884/4.6071ms`, MPC mean/P95 `2.8416/2.9093ms`, version `+1000`, nonfinite `0`, and peak `1127.79MiB`.

Four exact alternatives were measured and rejected:

1. complementary occupied/free FH transforms: representative full refresh about `7.60ms`;
2. parallel CUDA streams/chunks: no overlap benefit and higher memory;
3. fixed brute-force horizontal min reduction: representative full refresh about `8.47ms`;
4. compact occupied-list plus warp bbox reduction: representative full refresh about `9.36ms`.

The rejected CUDA experiments were removed; the branch retains the original numerically verified signed EDT and the independently useful compiled MPC path.

## Result

- Functional, collision, rolling, environment-count, old-backend, and real-viewer gates pass.
- The strengthened realistic multi-cell `1024 x H16 x 1000 <=5s` synchronous exact signed-field + MPC gate does **not** pass.
- This is now an architecture/budget conflict rather than an unresolved local kernel launch issue. A new exact batched EDT architecture (for example a true batched PBA-style implementation), a revised update-frequency contract, or a revised performance workload/threshold must be agreed before further implementation.
- Real 1024-env Isaac physics plus RayCaster generation remains a separate unmeasured boundary.

## Git Refs

- Last Verified Functional Ref: `9e71ac1`.
- Performance Status: blocked by realistic multi-cell exact signed EDT throughput; do not cite the earlier single-cell `4.768s` as acceptance.
