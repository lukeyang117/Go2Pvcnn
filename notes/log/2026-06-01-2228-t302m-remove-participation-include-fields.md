# T302m Remove MPC Participation Include Fields

- Purpose: remove unused MPC participation whitelist parameters.
- Stage: teacher elevation MPC semantic cleanup / MPC participation contract.
- Related todo: [T302m](../todo/T302m-teacher-elevation-mpc-semantic-cleanup-plan.md)

## Command / Procedure

```bash
pytest Go2Pvcnn/tests/test_mpc_rl_participation.py -q
pytest Go2Pvcnn/tests/test_batch_mpc_backend.py -q
rg -n "include_terrain_cols|include_terrain_names|include_terrain_rows|mpc_reference_include" Go2Pvcnn notes -g '*.py' -g '*.md'
```

## Input Conditions

- Local working tree after MPC config unification.
- User requested deleting `include_terrain_cols`, `include_terrain_names`, and `include_terrain_rows`.

## Key Metrics

- `test_mpc_rl_participation.py`: `4 passed`
- `test_batch_mpc_backend.py`: `129 passed, 1 warning`
- Source scan for removed include names: no matches

## Result

Pass. `MpcReferenceParticipationCfg` now only supports `enabled`, `exclude_pairs`, and `selection_mode`.

## Conclusion

MPC reference participation is now blacklist-only: all envs are eligible by default, and selected terrain+difficulty combinations are removed through `exclude_pairs`.

## Follow-up

- If future training needs a whitelist again, add it only after confirming the interface with the user.
- No IsaacLab runtime smoke was run for this local config-contract cleanup.

## Git Refs

- Baseline Ref: `d6f77d7`
- Candidate Ref: working tree at 2026-06-01 22:28
- Key Files:
  - [Go2Pvcnn/extension/batch_mpc_planner/participation.py](../../Go2Pvcnn/extension/batch_mpc_planner/participation.py)
  - [Go2Pvcnn/extension/batch_mpc_planner/config.py](../../Go2Pvcnn/extension/batch_mpc_planner/config.py)
  - [Go2Pvcnn/tests/test_mpc_rl_participation.py](../../Go2Pvcnn/tests/test_mpc_rl_participation.py)
