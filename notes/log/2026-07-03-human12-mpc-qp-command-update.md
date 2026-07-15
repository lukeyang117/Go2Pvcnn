# Human 12 MPC QP Command Update

## Purpose

Update the human command guide with current `mpc_qp` opt-in usage, strict low-small crossing probe commands, 1024/1024 perf smoke command, parameter notes, and final verification evidence.

## Stage

Documentation / MPC-QP command guide.

## Related Todo

[T302v MPC QP safety-constrained backend plan](../todo/T302v-mpc-qp-safety-constrained-backend-plan.md)

## Command

```bash
rg -n "mpc_qp|qp-iterations|strict low-small|2026-07-03-mpc-qp|planner-backend mpc_qp|1024 RL env" notes/human/human-12-batched-planner-train-viewer-commands.md
git diff --check -- notes/human/human-12-batched-planner-train-viewer-commands.md
```

## Result

Pass.

- `human-12` now documents `mpc_qp` as explicit opt-in while keeping default/formal training on `mpc`.
- Added viewer interactive/headless `mpc_qp` commands.
- Added GPU1 strict crossing left/right lane probe commands.
- Added GPU1 1024 RL env / 1024 MPC env perf smoke command.
- Added `--planner-backend mpc_qp` and `--qp-iterations` parameter explanations.
- Added final 2026-07-03 `mpc_qp` verification metrics and linked the strict crossing final log.
- `rg` confirmed the expected command and evidence anchors.
- `git diff --check` exit `0`.

## Conclusion

The command guide is aligned with the current `mpc_qp` backend state. No IsaacLab run was executed in this pass because the change was documentation-only and reused already recorded GPU1 evidence.

## Follow-Up

Keep future `mpc_qp` tuning evidence under T302v logs and update this command guide when the backend changes from projected safety-QP/repair to a different solver contract.

## Git Refs

- Baseline Ref: current worktree before this documentation edit
- Candidate Ref: current worktree after this documentation edit
- Key Files:
  - [../human/human-12-batched-planner-train-viewer-commands.md](../human/human-12-batched-planner-train-viewer-commands.md)
  - [2026-07-03-mpc-qp-strict-contact-crossing-final.md](2026-07-03-mpc-qp-strict-contact-crossing-final.md)
