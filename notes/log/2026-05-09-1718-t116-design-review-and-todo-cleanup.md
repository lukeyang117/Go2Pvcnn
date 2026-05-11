# T116 Design Review And Todo Cleanup

## Summary

- stage: design review / todo cleanup
- todo: [T100/T116](../todo/T100-batched-together-planner-gpu-migration.md#t116-k5-mode-first-small-obstacle-crossing-rewrite)
- result: design blockers incorporated into spec and active todo dashboard cleaned around T116
- code implementation: not changed
- tests: not run; this is a design/notes pass

## Procedure

- Spawned design review subagents for:
  - user requirement coverage
  - GPU implementation feasibility
  - state hard constraints
  - test matrix and runtime acceptance
  - todo cleanup relevance
- Main agent integrated review output into:
  - [../../docs/superpowers/specs/2026-05-09-k5-mode-first-cross-small-design.md](../../docs/superpowers/specs/2026-05-09-k5-mode-first-cross-small-design.md)
  - [../todo.md](../todo.md)
  - [../todo/T100-batched-together-planner-gpu-migration.md](../todo/T100-batched-together-planner-gpu-migration.md)

## Key Review Findings Incorporated

- `small` obstacle crossing must be tested for four velocity directions:
  - forward
  - backward
  - lateral-left
  - lateral-right
- `CROSS_SMALL` schedule must be command-relative, not body-front-only.
- `CROSS_SMALL` exit to `CRUISE` requires root/body and all four leg anchors beyond the small back edge.
- `cross_small_success` must be an iff over per-leg touchdown, grounding, foot path, root path, body, thigh, and calf clearance.
- `APPROACH_SMALL` requires explicit no-premature-crossing inequalities.
- `BYPASS_OBSTACLE` cannot count center/zero-speed as successful bypass when safe non-center bypass exists.
- Small obstacle geometry must use fixed-grid GPU reductions, not `nonzero`, `argwhere`, `masked_select`, CPU sync, or env loops.
- `50`-step horizon affects manager/cache/reward/env/viewer consumers, not only the core planner files.
- Active tests asserting old `K=3`, `35` frames, or `FRONT_CROSS -> REAR_FOLLOW -> CLEAR` must be deleted or rewritten as non-authoritative historical checks.
- Runtime timeout command must preserve pytest exit code:

```text
timeout -s INT -k 20s <bounded-seconds> bash -lc '<env_isaacsim python pytest ...>; code=$?; echo EXIT_CODE:$code; exit $code'
```

## Todo Cleanup

- Root dashboard now focuses on T116 as the only active together-planner architecture front.
- T113/T114/T115 were demoted to historical baselines and evidence, not active architecture targets.
- T111/T109/T110/T103/T107/T200-related items were removed from active root todo visibility unless directly relevant to T116.
- T100 branch `Open Children` now keeps only T116 active.
- T100 branch `Related Logs` is grouped into active T116 evidence, historical baselines, and viewer/runtime peripheral evidence.
- Root dashboard `Recent Logs` was shortened to T116 plus compact historical baseline/background rows only.
- T100 branch long T111/T110/T109 runtime nodes were compressed into a short historical peripheral note; detailed evidence remains reachable through logs.

## Verification

- Notes were edited only.
- No implementation tests were run.
- Readback confirmed root dashboard points to T116 as the only active front/open leaf.
- Grep confirmed stale "patch spec before todo leaves" and "pending user review" language was removed from active todo notes.
- Remaining old-mode terms are intentional historical/spec-cleanup references, not active next steps.

## Next

- Convert the patched T116 design into implementation todo leaves.
- Implement through subagent-driven development, with main-agent review and final-code-state deterministic/guardrail/runtime verification.
