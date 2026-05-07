---
name: inspire
description: Use when the user explicitly invokes /inspire to analyze a requirement, bug, image, or reference mapping before implementation begins
---

# Inspire

Use only on an explicit `/inspire ...` trigger. Do not auto-trigger from exploratory language alone.

## Core Rules

- Discussion comes first. Until the user explicitly chooses `生成设计`, do not edit code, write todo nodes, write a design file, or start implementation.
- During discussion, only offer: `继续分析`, `生成设计`, `结束`.
- Do not treat directional phrasing like "应该这样做" as approval.
- Do not jump from discussion directly to todo. The required sequence is discussion -> design -> optional `写 todo`.
- When the user chooses `生成设计`, actually use the `brainstorming` workflow as the parent process, but override its default exit: no standalone implementation plan path.
- Every design draft requires a focused subagent review before it is ready for user review.
- Execution breakdown is todo-first only. Never create a standalone implementation plan document.
- During execution, the primary agent owns orchestration, approvals, notes, logs, and review. Subagents own code edits, testing, and verification.
- During the execution phase governed by this skill, the primary agent must not directly edit code.

## Workflow

```text
/inspire <content>
-> classify intake mode
-> apply multiple analysis lenses
-> surface evidence gaps / ambiguity
-> ask one focused follow-up question or offer the allowed next-step choices
-> wait

If user chooses `生成设计`
-> invoke and use the actual brainstorming workflow
-> write design using references/design-template.md
-> dispatch required subagent review using references/design-review-checklist.md
-> integrate review
-> sync design-state memory in both `notes/todo.md` and the relevant branch page under `notes/todo/`, then write the official design-stage log
-> stop and wait

If user chooses `写 todo` after an approved design
-> write todo tree using references/todo-write-contract.md
-> write official todo-stage log
-> stop and wait

If user approves implementation of a specific todo leaf
-> delegate to subagent with references/delegation-contract.md
-> subagent edits code + runs tests
-> primary agent reviews results, updates todo/log, and handles follow-up delegation
```

## Intake

Classify the request into one primary mode:

- `image-analysis`
- `bug-log-analysis`
- `reference-mapping`
- `generic-demand`

Use [references/intake-modes.md](references/intake-modes.md) for the first-pass structure and [references/analysis-lenses.md](references/analysis-lenses.md) for reusable lenses.

## Design Gate

When the user chooses `生成设计`:

1. Explicitly state that the workflow is entering `brainstorming`.
2. Actually invoke and use the `brainstorming` workflow rather than only imitating its style.
3. Write the design around the required inspire sections in [references/design-template.md](references/design-template.md).
4. Run the required subagent review using [references/design-review-checklist.md](references/design-review-checklist.md).
5. If review-driven changes materially affect requirements, transitions, responsibilities, or acceptance indicators, run review again.
6. Sync design-state memory in both `notes/todo.md` and the relevant branch page under `notes/todo/`, then write the official design-stage log.
7. Only after review integration and the required memory sync may the design be presented as ready for user review.

After the design path, the available choices become:

- `修改 design`
- `写 todo`
- `结束`

## Todo And Execution Gate

- `写 todo` is valid only after a reviewed design exists.
- Convert the approved design into todo nodes directly; do not create a separate plan doc.
- Use [references/todo-write-contract.md](references/todo-write-contract.md) for dashboard, branch-page, and log expectations.
- For implementation of a specific leaf, delegate with [references/delegation-contract.md](references/delegation-contract.md).
- If results or metrics look strange, record the anomaly, create a concrete follow-up leaf, and dispatch another subagent instead of editing code directly.
