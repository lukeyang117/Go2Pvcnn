# Train CLI MPC Backend Option Fix

- timestamp: 2026-05-11 13:59 CST
- todo: [T300/T300d](../todo/T300-unified-dense-mpc-backend.md#t300d-subagent-driven-implementation-and-test-execution-for-extensionbatch_mpc_planner)
- result: pass

## Purpose

Fix `train.py` CLI parsing so user can run `--planner-backend mpc` for `teacher_elevation_trajectory`.

## Stage

Training entrypoint CLI compatibility.

## Changes

- Updated [../../Go2Pvcnn/scripts/train.py](../../Go2Pvcnn/scripts/train.py):
  - `--planner-backend` choices changed from `["together", "legacy"]` to `["together", "legacy", "mpc"]`
  - help text updated to include `mpc`

## Verification

- `python -m py_compile Go2Pvcnn/scripts/train.py` -> pass
- `timeout -s INT -k 5s 20s bash -lc '/mnt/mydisk/lhy/anaconda3/envs/env_isaacsim/bin/python Go2Pvcnn/scripts/train.py --help | rg "planner-backend|\\{together,legacy,mpc\\}" -n'` -> usage now shows `{together,legacy,mpc}`

## Conclusion

`train.py` no longer rejects `--planner-backend mpc` at argparse stage.

## Git Refs

- Baseline Ref: working tree on top of `130c635`
- Candidate Ref: working tree with `train.py` planner-backend choice update
- Key Files:
  - [../../Go2Pvcnn/scripts/train.py](../../Go2Pvcnn/scripts/train.py)
