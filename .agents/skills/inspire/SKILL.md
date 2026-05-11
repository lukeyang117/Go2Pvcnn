---
name: inspire
description: Use when the user explicitly invokes /inspire to analyze a requirement, bug, image, or reference mapping before implementation begins
---

# Inspire

Use only on an explicit `/inspire ...` trigger. Do not auto-trigger from exploratory language alone.

## Core Rules

- Discussion comes first. Until the user explicitly chooses `生成设计`, do not edit code, write todo nodes, write a design file, or start implementation.
- During discussion, reply in clear sections or bullets. After each round, say what the next discussion part will cover before asking the next question or offering choices.
- During discussion, only offer: `继续分析`, `生成设计`, `结束`.
- Do not treat directional phrasing like "应该这样做" as approval to enter design while discussion mode is still active.
- Once a stage boundary has been reached, treat explicit advancement phrasing such as `写 todo`, `开始实现`, `继续`, or `下一阶段` as approval to enter the next stage immediately. Do not ask for the same stage approval twice.
- Do not jump from discussion directly to todo. The required sequence is discussion -> design -> optional `写 todo` -> implementation.
- When the user chooses `生成设计`, actually use the `brainstorming` workflow as the parent process, but override its default exit: no standalone implementation plan path.
- Every design draft requires a focused subagent review before it is ready for user review.
- Execution breakdown is todo-first only. Never create a standalone implementation plan document.
- After the user approves moving past the todo stage, do not require any further user approval. The primary agent should choose the next ready leaf or ready leaf batch, delegate it, and keep orchestration moving until the requested implementation scope is complete or objectively blocked.
- During execution, the primary agent owns orchestration, notes, logs, review, ready-leaf sequencing, and anomaly follow-up. Subagents own code edits, testing, and verification.
- During execution, the primary agent stays live: wait for subagent results, update todo/log state, and continue the next ready delegation instead of handing control back to the user between normal leaves.
- During the execution phase governed by this skill, the primary agent must not directly edit code.

## Workflow

```text
/inspire <content>
-> classify intake mode
-> apply multiple analysis lenses
-> surface evidence gaps / ambiguity
-> say what the next discussion part will cover
-> ask one focused follow-up question or offer the allowed next-step choices
-> wait

If user chooses `生成设计`
-> invoke and use the actual brainstorming workflow
-> write design using references/design-template.md
-> dispatch required subagent review using references/design-review-checklist.md
-> integrate review
-> sync design-state memory in both `notes/todo.md` and the relevant branch page under `notes/todo/`, then write the official design-stage log
-> stop and wait

If user chooses `写 todo` or gives explicit next-stage approval after an approved design
-> write todo tree using references/todo-write-contract.md
-> write official todo-stage log
-> present the recommended execution order / first ready leaf
-> stop and wait

If user chooses `开始实现`, `继续`, `下一阶段`, or otherwise explicitly advances after todo
-> choose the highest-priority ready leaf or ready leaf batch
-> delegate to subagent with references/delegation-contract.md
-> subagent edits code + runs tests
-> primary agent reviews results, updates todo/log, and handles follow-up delegation
-> if another ready leaf remains and execution is still viable, continue automatically
-> stop only on stage completion or objective blocker
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

If the user says `继续` or `下一阶段` at this boundary, interpret it as approval to start the todo stage immediately.

## Todo And Execution Gate

- `写 todo` or explicit next-stage language after a reviewed design is sufficient to enter the todo stage. Do not re-ask for the same approval.
- Convert the approved design into todo nodes directly; do not create a separate plan doc.
- Use [references/todo-write-contract.md](references/todo-write-contract.md) for dashboard, branch-page, and log expectations.
- After the todo stage is complete, explicit advancement to implementation is stage-level approval, not leaf-level approval.
- Once implementation starts, the primary agent should choose and sequence ready leaves without asking for any further approval.
- For implementation of the approved execution stage, delegate with [references/delegation-contract.md](references/delegation-contract.md).
- If results or metrics look strange, record the anomaly, create a concrete follow-up leaf, and dispatch another subagent instead of editing code directly.
- Keep control flow alive between normal leaves: wait for the subagent result and continue until complete or objectively blocked.
