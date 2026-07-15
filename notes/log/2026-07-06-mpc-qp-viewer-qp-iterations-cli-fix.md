# MPC QP Viewer QP Iterations CLI Fix

## Purpose

Fix the viewer command documented for `mpc_qp`: `go2_foostep_planner.py --planner-backend mpc_qp --qp-iterations 1` failed at argparse with `unrecognized arguments: --qp-iterations 1`.

## Stage

MPC-QP backend / viewer CLI / command-guide runtime compatibility.

## Related Todo

[T302v MPC QP safety-constrained backend plan](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Root Cause

The `mpc_qp` runtime config already had `runtime.qp_iterations`, and probe/perf scripts already exposed `--qp-iterations`, but `Go2Pvcnn/extension/viz/go2_foostep_planner.py` only exposed `--planner-backend mpc_qp`. The viewer parser rejected the documented command before IsaacLab startup.

## Change

- Added `--qp-iterations` to the viewer argparse surface.
- Added `_apply_planner_runtime_cli_overrides()` so viewer CLI overrides update both the shared horizon/dt fields and the `mpc_qp`-specific `mpc_qp_planner_cfg.runtime.qp_iterations`.
- Added a regression test proving the viewer parser accepts `--planner-backend mpc_qp --qp-iterations 3` and applies the value to the QP config.

## Commands

```bash
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py::test_mpc_qp_viewer_argparse_accepts_and_applies_qp_iterations -q
pytest Go2Pvcnn/tests/test_mpc_qp_backend.py -q
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python -m py_compile Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/test_mpc_qp_backend.py
git diff --check -- Go2Pvcnn/extension/viz/go2_foostep_planner.py Go2Pvcnn/tests/test_mpc_qp_backend.py
/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/extension/viz/go2_foostep_planner.py --help | rg -n "qp-iterations|planner-backend"
```

## Result

Pass.

- RED reproduced: parser raised `SystemExit: 2` with `unrecognized arguments: --qp-iterations 3`.
- GREEN single regression: `1 passed`.
- Focused QP suite: `21 passed`.
- Pycompile: exit `0`.
- Diff check: exit `0`.
- `env_isaacsim` viewer help now lists `--planner-backend {mpc,mpc_qp}` and `--qp-iterations QP_ITERATIONS`.

## Conclusion

The documented `mpc_qp` viewer command should now parse and propagate `--qp-iterations`. A full IsaacLab viewer launch was not rerun in this Codex pass; the fix is at the pre-launch CLI/config boundary that produced the reported failure.

## Follow-Up

If a later viewer launch fails after argparse, treat it as a separate runtime issue because this pass only covered CLI parsing and config propagation.

## Git Refs

- Baseline Ref: current worktree before this fix
- Candidate Ref: current worktree after this fix
- Key Files:
  - [../../Go2Pvcnn/extension/viz/go2_foostep_planner.py](../../Go2Pvcnn/extension/viz/go2_foostep_planner.py)
  - [../../Go2Pvcnn/tests/test_mpc_qp_backend.py](../../Go2Pvcnn/tests/test_mpc_qp_backend.py)
