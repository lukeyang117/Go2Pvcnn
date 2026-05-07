# T001 Inspire Skill Design

## Current State

- Project-local `/inspire` skill design spec is written at [../../docs/superpowers/specs/2026-05-07-inspire-skill-design.md](../../docs/superpowers/specs/2026-05-07-inspire-skill-design.md).
- The design fixes the main workflow as:
  - manual `/inspire` trigger
  - discussion first
  - explicit `生成设计` gate
  - `brainstorming`-governed design generation
  - subagent design review
  - design-state memory sync
  - explicit `写 todo` gate
  - todo-first execution breakdown
  - subagent-only code changes and testing
- Three narrow subagent review passes were used to harden requirement coverage and acceptance indicators.
- Final blocker-only review recommends passing the design.
- The user has approved the written spec and authorized implementation.
- Implementation should still follow the spec:
  - todo-first, no standalone plan doc
  - primary agent owns orchestration, notes, logs, and review
  - subagents own skill-file edits and validation runs
- First implementation pass landed the skill package structure under `.agents/skills/inspire/`.
- Review found two remaining implementation gaps:
  - design gate must require actual `brainstorming` workflow use, not only a declaration
  - design template must hard-require constraints, alternative approaches with trade-offs, and recommended design
- Final blocker-only review found one remaining memory-sync gap:
  - design-stage sync must explicitly include the relevant branch page, not only dashboard/log memory
- Follow-up subagent patches closed all three issues.
- Final blocker-only implementation review found no remaining blockers.

## Open Children

- none

## Closed Children Archive

- T001a done: user reviewed and approved the written design spec.
- T001b done: first-pass `SKILL.md` implementation landed.
- T001c done: first-pass `references/` package landed.
- T001d done: final validation and blocker-only review passed.
- T001e done: follow-up design-gate and branch-page-sync patches landed.

## Related Logs

- [2026-05-07-1258-inspire-skill-design.md](../log/2026-05-07-1258-inspire-skill-design.md)
- [2026-05-07-1404-inspire-skill-implementation.md](../log/2026-05-07-1404-inspire-skill-implementation.md)

## Git Refs

- Last Feature Commit: `pending (skill implementation stage)`
- Last Verified Commit: `working tree verification at 2026-05-07 14:04 +0800`
- Current Work Ref: `working tree on top of b94722f (2026-05-07 14:04 +0800); inspire skill implementation and notes update; unrelated planner/viewer/plugin dirt present`
- Key Files:
  - [../../docs/superpowers/specs/2026-05-07-inspire-skill-design.md](../../docs/superpowers/specs/2026-05-07-inspire-skill-design.md)
  - [../../.agents/skills/inspire/SKILL.md](../../.agents/skills/inspire/SKILL.md)
  - [../../.agents/skills/inspire/references/intake-modes.md](../../.agents/skills/inspire/references/intake-modes.md)
  - [../../.agents/skills/inspire/references/analysis-lenses.md](../../.agents/skills/inspire/references/analysis-lenses.md)
  - [../../.agents/skills/inspire/references/design-template.md](../../.agents/skills/inspire/references/design-template.md)
  - [../../.agents/skills/inspire/references/design-review-checklist.md](../../.agents/skills/inspire/references/design-review-checklist.md)
  - [../../.agents/skills/inspire/references/todo-write-contract.md](../../.agents/skills/inspire/references/todo-write-contract.md)
  - [../../.agents/skills/inspire/references/delegation-contract.md](../../.agents/skills/inspire/references/delegation-contract.md)
  - [../../notes/todo.md](../../notes/todo.md)
  - [../../notes/log/index.md](../../notes/log/index.md)
  - [2026-05-07-1258-inspire-skill-design.md](../log/2026-05-07-1258-inspire-skill-design.md)
  - [2026-05-07-1404-inspire-skill-implementation.md](../log/2026-05-07-1404-inspire-skill-implementation.md)

## Next Step

- Use `/inspire` on a real requirement when you want to pressure-test the dialogue experience in practice.

## Node Details

### T001a inspire design spec review gate

- status: done
- why-created:
  - the user approved the design section-by-section in conversation
  - the written spec still needed explicit repository memory sync and requirement-focused review before implementation
- evidence:
  - [2026-05-07-1258-inspire-skill-design.md](../log/2026-05-07-1258-inspire-skill-design.md)
- outcome:
  - the user approved the written spec and authorized implementation

### T001b inspire core skill file

- status: done
- why-created:
  - the implementation needs one concise `SKILL.md` that enforces the `/inspire` trigger, discussion freeze rule, explicit transition gates, `brainstorming` handoff for design, and subagent-only execution boundaries
- design-mapping:
  - [../../docs/superpowers/specs/2026-05-07-inspire-skill-design.md](../../docs/superpowers/specs/2026-05-07-inspire-skill-design.md) sections 5, 6, 8, 9, 11, 12, 14
- acceptance:
  - `/inspire` manual trigger only
  - discussion path only offers `继续分析 / 生成设计 / 结束`
  - no implicit approval from direct user phrasing
  - no standalone plan path
  - implementation phase forbids primary-agent code edits

### T001c inspire support references

- status: done
- why-created:
  - the skill should keep `SKILL.md` focused and load detailed templates/contracts from `references/`
- design-mapping:
  - [../../docs/superpowers/specs/2026-05-07-inspire-skill-design.md](../../docs/superpowers/specs/2026-05-07-inspire-skill-design.md) section 13
- acceptance:
  - reference files exist for intake modes, analysis lenses, design template, design review checklist, todo-write contract, and delegation contract
  - the files match the approved design boundaries and do not reintroduce standalone plan behavior

### T001d inspire validation

- status: done
- why-created:
  - the skill needs at least a minimum validation pass after implementation so it is not only documented but also checked against the approved behavior
- design-mapping:
  - [../../docs/superpowers/specs/2026-05-07-inspire-skill-design.md](../../docs/superpowers/specs/2026-05-07-inspire-skill-design.md) section 14
- acceptance:
  - skill files exist in the expected location
  - wording and workflow still reflect the approved spec
  - no `writing-plans` / standalone implementation-plan path remains
  - the validation result is written to an official log

### T001e inspire design-gate patch

- status: done
- why-created:
  - review of the first implementation pass found that the design gate still allowed a weak interpretation of `brainstorming` entry
  - the design template still treated some brainstorming-aligned content as soft guidance instead of hard required content
- evidence:
  - review finding: actual `brainstorming` workflow use must be a hard requirement
  - review finding: `constraints`, `alternative approaches with trade-offs`, and `recommended design` must be hard-required in the template
  - final blocker-only review: design-stage memory sync must explicitly include the relevant branch page, and the checklist must enforce it
- design-mapping:
  - [../../docs/superpowers/specs/2026-05-07-inspire-skill-design.md](../../docs/superpowers/specs/2026-05-07-inspire-skill-design.md) sections 9, 13, 14
- acceptance:
  - `SKILL.md` makes actual `brainstorming` workflow use a hard requirement, not only a declaration
  - `references/design-template.md` makes `constraints`, `alternative approaches with trade-offs`, and `recommended design` explicit required content
  - design-stage memory sync explicitly includes both `notes/todo.md` and the relevant branch page
  - `references/design-review-checklist.md` checks that design-state memory sync includes the branch page
  - a fresh review confirms the design-gate mismatch is closed
