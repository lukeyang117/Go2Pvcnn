# inspire skill design

## Purpose

- Record the approved design direction for a project-local `/inspire` skill that enforces discussion-first requirement analysis, design gating, todo-first execution breakdown, and subagent-only implementation/testing.

## Stage

- repository tooling / local skill design

## Related Todo

- [T001/T001a](../todo/T001-inspire-skill-design.md#t001a-inspire-design-spec-review-gate)

## Command / Procedure

- Read repository constraints and memory entrypoints:
  - [../index.md](../index.md)
  - [../../.codex/RULES.md](../../.codex/RULES.md)
  - [../todo.md](../todo.md)
  - [../todo/T000-notes-workflow.md](../todo/T000-notes-workflow.md)
  - [index.md](index.md)
- Read the relevant skill references:
  - [../../.agents/skills/brainstorming/SKILL.md](../../.agents/skills/brainstorming/SKILL.md)
  - [../../.agents/skills/skill-forge/SKILL.md](../../.agents/skills/skill-forge/SKILL.md)
  - [../../.agents/skills/writing-skills/SKILL.md](../../.agents/skills/writing-skills/SKILL.md)
  - [../../.agents/skills/systematic-debugging/SKILL.md](../../.agents/skills/systematic-debugging/SKILL.md)
  - external reference: `code-flow-guide` at `/mnt/mydisk/lhy/.codex/skills/code-flow-guide/SKILL.md`
- Brainstormed the skill design in conversation and confirmed:
  - manual `/inspire` trigger only
  - discussion-first behavior
  - explicit `生成设计` gate
  - design generation through `brainstorming`
  - no standalone plan document
  - explicit `写 todo` gate after design
  - subagent-only code changes and testing
  - primary-agent ownership of orchestration, notes, logs, and review
- Wrote the design spec at [../../docs/superpowers/specs/2026-05-07-inspire-skill-design.md](../../docs/superpowers/specs/2026-05-07-inspire-skill-design.md).
- Ran three focused subagent reviews against the spec:
  - requirement coverage review
  - acceptance-indicator review
  - final blocker-only review
- Updated the spec after each review to tighten:
  - brainstorming workflow entry
  - todo-first/no-plan boundaries
  - design-state memory sync
  - approval semantics
  - design/todo log requirements
  - primary-agent no-code-edit rule during execution
- Updated repository memory:
  - [../todo.md](../todo.md)
  - [../todo/T001-inspire-skill-design.md](../todo/T001-inspire-skill-design.md)
  - [index.md](index.md)

## Input Conditions

- Workspace: `/mnt/mydisk/lhy/testPvcnnWithIsaacsim`
- Baseline ref: `cf7e9cf`
- Current work ref: `working tree on top of cf7e9cf; unrelated planner/viewer/plugin dirt present`
- Design output:
  - [../../docs/superpowers/specs/2026-05-07-inspire-skill-design.md](../../docs/superpowers/specs/2026-05-07-inspire-skill-design.md)

## Key Metrics

- Design sections written: `16`
- Focused subagent review rounds: `3`
- Final blocker-only review result: `approved`
- Official memory surfaces updated: `3`
  - [../todo.md](../todo.md)
  - [../todo/T001-inspire-skill-design.md](../todo/T001-inspire-skill-design.md)
  - [index.md](index.md)

## Result

- Pass

## Conclusion

- The `/inspire` design is now written, self-reviewed, and confirmed by a final blocker-only subagent review as ready for user review.
- No implementation or todo-writing for the skill itself has started yet; the next gate is user review of the written spec.

## Follow-up

- Ask the user to review [../../docs/superpowers/specs/2026-05-07-inspire-skill-design.md](../../docs/superpowers/specs/2026-05-07-inspire-skill-design.md).
- If the user approves the written spec, proceed to implement the skill and its supporting files under `.agents/skills/inspire/`.

## Git Refs

- Baseline Ref: `cf7e9cf`
- Candidate Ref: `working tree on top of cf7e9cf (2026-05-07 12:58 +0800); inspire design spec and notes update`
- Key Files:
  - [../../docs/superpowers/specs/2026-05-07-inspire-skill-design.md](../../docs/superpowers/specs/2026-05-07-inspire-skill-design.md)
  - [../todo.md](../todo.md)
  - [../todo/T001-inspire-skill-design.md](../todo/T001-inspire-skill-design.md)
  - [index.md](index.md)
