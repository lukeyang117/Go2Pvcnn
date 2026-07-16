# Joint MPC RTI Synchronous CUDA Exact EDT

- Purpose: replace the B1024 tensor Jump Flood bottleneck and verify one exact distance-field refresh per MPC call.
- Stage: `joint_mpc_rti` terrain field, query, line search and full-refresh runtime.
- Related todo: [T302v](../todo/T302v-joint-mpc-rti-gpu.md).
- Baseline Ref: `643a172`; Candidate Ref: `4bad2e0`.
- Key Files: `Go2Pvcnn/extension/joint_mpc_rti/terrain/cuda_edt.py`, `terrain/csrc/work_efficient_edt_cuda.cu`, `terrain/field_cache.py`, `terrain/query.py`, `planner.py`.

## Implementation

- Native lazy-built C++/CUDA extension for fixed `151x151`, small/large exact EDT.
- Dual-channel vertical scan reads semantic ids once and stores exact squared vertical distances in fixed int16 workspace.
- Eight warps per image build/evaluate horizontal parabola envelopes; final distances are float32 metres.
- Height/valid/metadata publish and semantic EDT run concurrently inside the same refresh, then explicitly join before MPC.
- Distance gradients are analytic derivatives of bilinear exact-distance interpolation at query points; no per-frame full gradient maps.
- Line-search candidate batches gather original env field rows and no longer duplicate complete per-env maps.
- `latest_field()` performs one current-row refresh even without a new callback; no stale-field path.

## Verification

- Exact parity command: `CUDA_VISIBLE_DEVICES=2 .../env_isaacsim/bin/python -m pytest Go2Pvcnn/tests/joint_mpc_rti/test_terrain_fields.py -q`.
- Inputs: single/boundary/rectangle/random sparse masks, empty maps, B1024 independent two-channel maps, full/partial cache transitions.
- Result: `14 passed`; SciPy squared-distance tolerance `atol=1e-4`; empty maps return finite grid-diagonal distance; scanner semantic storage is not mutated by later partial updates.

## Performance Acceptance

- Command: `CUDA_VISIBLE_DEVICES=2 .../env_isaacsim/bin/python Go2Pvcnn/tests/joint_mpc_rti/joint_mpc_rti_full_refresh_probe.py --num-envs 1024 --horizon 16 --steps 1000 --warmup 100`.
- Conditions: RTX 4090 GPU2 observed idle, fixed scanner buffers, unique per-env 151x151 storage, exact small+large EDT every iteration, H16 one-RTI CUDA Graph, full boundary CUDA events only; first extension build/compile/capture excluded.
- Metrics: total `4469.0463 ms`; mean `4.4690 ms`; P50 `4.4677 ms`; P95 `4.4923 ms`; P99 `4.4995 ms`; max `4.6551 ms`; field diagnostic mean `1.7516 ms`; MPC diagnostic mean `2.7763 ms`; field version `+1000`; nonfinite `0`; peak `858.99 MiB`.
- Result: pass for total `<=5s`, mean `<=5ms`, P95 `<=10ms`, max `<=20ms`, version and finite-output gates.
- Contention note: GPU0 had two unrelated compute processes and produced `8537 ms`; that run is environment evidence, not accepted performance. GPU2 acceptance used the same code and thresholds without lowering batch, resolution, horizon or update count.
- Later final-session GPU2 rerun started at `54%` unrelated utilization / `13.5GiB` used and measured `8370.70ms`; field stayed `1.705ms` while MPC rose to `6.558ms`. All four GPUs were occupied, so this is recorded as contended evidence rather than replacing the earlier idle-GPU acceptance.
- Configured-batch probes: B1 mean `1.278ms`, B40 `1.568ms`, B512 short-run P50 `4.702ms`; every run had exact EDT, correct version increments and nonfinite `0`. B1024 uses the formal 1000-step acceptance above.
- Follow-up: real IsaacLab physics and RayCaster ray generation are outside this planner acceptance boundary and remain separately measurable.
