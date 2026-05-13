# MPC Dir15/Dir19 Production Grounding

- Time: 2026-05-13 12:53 CST
- Purpose: move the best current drift/touchdown design from test-layer variants into production MPC planner code, per user request to stop more sweeps and inspect visually next.
- Stage: `extension/batch_mpc_planner` production planner/runtime/viewer path.
- Related todo: [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)
- Baseline Ref: `e90e3a4`
- Candidate Ref: working tree with `dir15 + dir19` production foothold memory and terrain grounding
- Key Files:
  - [../../Go2Pvcnn/extension/batch_mpc_planner/types.py](../../Go2Pvcnn/extension/batch_mpc_planner/types.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py](../../Go2Pvcnn/extension/batch_mpc_planner/nominal.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/planner.py](../../Go2Pvcnn/extension/batch_mpc_planner/planner.py)
  - [../../Go2Pvcnn/extension/batch_mpc_planner/manager.py](../../Go2Pvcnn/extension/batch_mpc_planner/manager.py)
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)

## Design Implemented

This pass combines the accepted `dir15` and `dir19` direction into the production MPC path:

- manager-owned `MpcFootholdMemory`
  - persistent body-frame footprint seed
  - world-frame stance anchor
  - previous contact state
  - previous yaw dominance and yaw-entry step counter
- soft command-regime gates
  - linear-dominant commands use persistent body-frame footprint seed
  - yaw-dominant commands use contact-gated stance anchor replacement
- yaw-entry ramp from `dir19`
  - entering yaw-dominant control ramps anchor influence over several replans
  - this targets the mixed-to-yaw foot-step spike exposed by the mixed/sequence sweeps
- terrain grounding
  - planned contact foot z is sampled from `MpcPlannerTerrain.height_map + world_x/y_range`
  - contact foot z is grounded before touchdown extraction, IK solve, diagnostics, and result emission
- viewer direct path parity
  - direct `go2_foostep_planner.py --planner-backend mpc` replans maintain the same lightweight foothold memory instead of bypassing the production behavior

## Fresh Verification

Commands:

```bash
source /mnt/mydisk/lhy/anaconda3/etc/profile.d/conda.sh
conda activate env_isaacsim
python -m py_compile Go2Pvcnn/extension/batch_mpc_planner/config.py Go2Pvcnn/extension/batch_mpc_planner/types.py Go2Pvcnn/extension/batch_mpc_planner/nominal.py Go2Pvcnn/extension/batch_mpc_planner/planner.py Go2Pvcnn/extension/batch_mpc_planner/manager.py Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/test_mpc_runtime_headless.py Go2Pvcnn/tests/mpc_touchdown_sequence_probe.py
python -m pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q
MPC_TEST_DEVICE=cuda:2 python -m pytest Go2Pvcnn/tests/test_mpc_runtime_headless.py::test_mpc_runtime_plan_case_headless_smoke -q -s
```

Results:

- `py_compile`: exit `0`
- focused backend suite: `12 passed in 3.46s`
- IsaacLab headless MPC smoke on `cuda:2`: exit `0`, test passed

The IsaacLab smoke log still emitted PhysX GPU kernel warnings from the simulator layer, but pytest exited successfully and the MPC manager attached.

## Conclusion

The production MPC path now contains the tested memory/anchor direction and explicit terrain-grounded contact output. This directly targets:

- `touchdown_ground_gap_mean`
- `touchdown_event_ground_gap_mean`
- `touchdown_airborne_ratio`
- long replan foot/root drift under forward/backward/lateral/yaw commands

## Remaining Risk

- No new long-run sweep was run after the production edit because the user explicitly chose visual inspection as the next step.
- Terrain grounding is currently an output correction after optimization decode, not an optimizer loss term. If visual behavior shows IK discontinuity or contact snapping, the next follow-up should move terrain height into the nominal/loss path as well.
- True 4096-env long counter extraction remains a separate Isaac runtime stability issue.
