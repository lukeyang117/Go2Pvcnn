# Create Todo Log Systerm

This standalone bootstrap specification creates a todo/log agent-memory system for a new repository. It does not depend on any external repository Markdown files.

The format is intentionally aligned with `$compact-todo`: root dashboard, branch memory, evidence logs, and archives.

## Bootstrap Inputs

Determine these values:

- `PROJECT_NAME`: repository or product name.
- `PROJECT_FOCUS`: active architecture or main workflow.
- `PRIMARY_CODE_ENTRY_POINTS`: key files or commands future agents should inspect first.
- `CURRENT_GIT_BASE`: `git rev-parse --short HEAD`, or `untracked`.
- `BOOTSTRAP_TIME`: local timestamp in `YYYY-MM-DD-HHMM`.
- `BOOTSTRAP_DATE_TIME`: local timestamp in `YYYY-MM-DD HH:MM`.
- `USE_COMPACT_TODO`: `yes` if `$compact-todo` will also be installed; otherwise `no`.

Use `TODO(<name>)` when a value is unknown.

## Target File Tree

Create:

```text
AGENTS.md
.codex/RULES.md
notes/
├── index.md
├── todo.md
├── todo/
│   ├── README.md
│   ├── T000-notes-workflow.md
│   └── archive/
└── log/
    ├── index.md
    ├── BOOTSTRAP_TIME-notes-workflow-bootstrap.md
    └── archive/
```

Optional:

```text
notes/
├── ai/
│   └── README.md
└── human/
    └── README.md
```

## AGENTS.md

Create or update `AGENTS.md`:

- If `AGENTS.md` already exists and has repository-specific constraints, preserve them and add the todo/log pre-read requirement instead of replacing the file with the generic template below.
- If the repository already has a stronger mandatory note entrypoint than `notes/todo.md`, keep that entrypoint.

```md
# AGENTS.md instructions for this repository

Before any analysis, code reading, code changes, test execution, debugging, or completion claims in this repository, read and follow:

- `.codex/RULES.md`

The `.codex/RULES.md` file is a hard repository constraint and takes priority over default workflow preferences unless the user explicitly overrides it in the current conversation.

If `.codex/RULES.md` requires additional notes, todo branch pages, logs, or stage-specific documents to be read before edits, complete that pre-read first and then continue with the task.
```

## .codex/RULES.md

Create `.codex/RULES.md`:

- If `.codex/RULES.md` already exists, merge these todo/log workflow rules into the existing file and preserve any stricter local rules, stage-specific pre-reads, path boundaries, and link constraints.

```md
# PROJECT_NAME Agent Rules

## Project Focus

PROJECT_FOCUS

Primary code entry points:

- PRIMARY_CODE_ENTRY_POINT_1
- PRIMARY_CODE_ENTRY_POINT_2
- PRIMARY_CODE_ENTRY_POINT_3

Primary notes entry points:

- `notes/index.md`
- `notes/todo.md`
- `notes/todo/`
- `notes/log/index.md`

---

## Hard Constraints

- Treat the documented active architecture as the default working architecture unless the user explicitly asks to work on legacy or experimental paths.
- Do not silently route new work into legacy paths.
- Any code change must first be understood in the context of the stage, module, or workflow it belongs to.
- Do not make cross-stage or cross-module edits casually. Modify the smallest coherent part of the system and its direct upstream/downstream contracts.
- Do not claim completion without checking whether code behavior, verification, and notes are aligned.
- Keep `notes/todo.md` as a dashboard. Use branch pages for detailed memory and logs for evidence.
- When `notes/todo.md`, branch pages, or `notes/log/index.md` grow too large, use `$compact-todo` if available. The system must remain usable without it.

---

## Mandatory Pre-Read Before Work

Before every analysis session, debugging session, code reading pass, code change, test run, or final completion claim, read:

1. `notes/todo.md`
2. relevant branch pages under `notes/todo/` for active/open nodes
3. relevant logs from `notes/log/index.md`
4. `notes/index.md`
5. relevant files under `notes/ai/` or `notes/human/` when the task touches documented stages, contracts, or user-facing behavior

The purpose is to continue the current investigation instead of restarting from scratch.

---

## Required Change Workflow

For every meaningful investigation update, test, code change, or behavior change:

1. Read the mandatory notes.
2. Identify the affected stage, module, or workflow.
3. Confirm current input, method, output, and downstream consumers before editing.
4. Modify only the smallest coherent scope.
5. Run or describe appropriate verification.
6. Update `notes/todo.md` dashboard when focus, active fronts, open leaves, or recent logs changed.
7. Update the relevant branch page under `notes/todo/`.
8. Update `notes/log/index.md`.
9. Create one per-test or per-verification log under `notes/log/`.
10. Align `notes/ai/` and `notes/human/` when stage behavior, contracts, inputs, or outputs changed.
11. Report what changed in code, verification, and notes.

---

## Todo And Test Log Rule

Required files:

- `notes/todo.md`
- `notes/todo/`
- `notes/todo/archive/`
- `notes/log/index.md`
- `notes/log/archive/`
- one per-test Markdown file under `notes/log/`

### Root Dashboard Requirement

`notes/todo.md` is a dashboard, not a full database.

It must keep:

- `Start Here`
- `Active Fronts`
- `Root Map`
- `Open Leaves`
- `Branch Pages`
- `Recent Logs`

It must not expand every closed child node. Closed detail belongs in branch pages or archives.

### Tree-Structured Todo Requirement

Agents must maintain root problem nodes, child problem nodes, explicit relationships, and links from important nodes into branch pages.

For each new issue, classify it as:

- new root node
- child of an existing node
- sibling of an existing node
- related by cause, blocking, or dependency

Do not append isolated flat todos.

### Dynamic-Issue Discovery Requirement

If a new issue is discovered during testing, debugging, or code reading:

1. create a node in `notes/todo.md` if it is open/active, or in the relevant branch page if closed/immediate context only
2. attach it to the correct parent/sibling/related position
3. explain why it was created in the branch page
4. link the triggering test log

New issues must not live only inside a test log.

### Metric-Anomaly Expansion Requirement

If metrics look strange, conclusions look strange, or results conflict with expectations, create a concrete follow-up node. Avoid vague placeholders.

### Branch Memory Requirement

Important or evolving issues must have branch pages under `notes/todo/`.

Branch pages should include:

- `Current State`
- `Open Children`
- `Closed Children Archive`
- `Related Logs`
- `Git Refs`
- `Next Step`
- `Node Details`

### Log Requirement

Every distinct test or verification pass must create a Markdown file under `notes/log/`.

Each log records:

- purpose
- stage
- related todo
- command/procedure
- input conditions
- key metrics
- result
- conclusion
- follow-up
- git refs

### Git Traceability Requirement

Track:

- `Last Feature Commit`
- `Last Verified Commit`
- `Current Work Ref`
- `Key Files`

For logs, track:

- `Baseline Ref`
- `Candidate Ref`
- `Key Files`

Do not overwrite module refs merely because newer unrelated commits exist.

### Relative-Link Requirement

All links must be relative.

---

## Final Response Rule

When closing a task, include:

- which stage/module/workflow changed
- what contracts changed, if any
- what metrics or constraints were checked
- whether todo/log/notes were aligned
- what remains unverified
```

## notes/index.md

Create `notes/index.md`:

```md
# Notes Index

This directory is persistent memory for agent work.

## Entry Points

- [Investigation Dashboard](todo.md)
- [Todo Branch Directory](todo/README.md)
- [Test Log Index](log/index.md)

## Optional Areas

- `notes/ai/`: agent-facing architecture, stage, and contract notes.
- `notes/human/`: human-facing explanations and decisions.

## Working Rule

Before analysis, code reading, debugging, tests, code changes, or completion claims, read [todo.md](todo.md), relevant branch pages, and relevant logs.
```

## notes/todo.md

Create `notes/todo.md`:

```md
# Investigation Dashboard

This page is the fast-start dashboard for agent work. It is not a full database. Detailed memory lives in [todo/](todo/); evidence lives in [log/](log/).

## Start Here

- Current focus: none yet
- Read next:
  - [T000 notes workflow](todo/T000-notes-workflow.md)
  - [bootstrap log](log/BOOTSTRAP_TIME-notes-workflow-bootstrap.md)
- Avoid redoing:
  - Do not recreate the memory system; update it incrementally.
- Current git base: `CURRENT_GIT_BASE`

## Status Legend

- `todo`: not started
- `doing`: under investigation
- `blocked`: waiting on a condition
- `verify`: changed or hypothesized, awaiting verification
- `done`: completed and closed
- `drop`: abandoned direction

## Active Fronts

| Leaf | Why Active Or Next | Suggested Action |
| --- | --- | --- |
| none | No active project issue yet. | Add a real root node when work begins. |

## Root Map

| Root | Status | Stage | Branch | Current | Refs |
| --- | --- | --- | --- | --- | --- |
| T000 | done | notes workflow | [T000](todo/T000-notes-workflow.md) | memory system bootstrapped | feature `CURRENT_GIT_BASE`; verified `CURRENT_GIT_BASE` |

## Open Leaves

| Leaf | Parent | Status | Priority | Why Active | Next Read |
| --- | --- | --- | --- | --- | --- |
| none | none | done | P0 | No open issue yet. | Add a real leaf when work begins. |

## Branch Pages

- [todo/README.md](todo/README.md)
- [T000-notes-workflow.md](todo/T000-notes-workflow.md)

## Recent Logs

| Time | Topic | Result | Todo | File |
| --- | --- | --- | --- | --- |
| BOOTSTRAP_DATE_TIME | notes workflow bootstrap | pass | [T000](todo/T000-notes-workflow.md) | [BOOTSTRAP_TIME-notes-workflow-bootstrap.md](log/BOOTSTRAP_TIME-notes-workflow-bootstrap.md) |

## Maintenance

- Keep this page short: dashboard only.
- Put detailed background and conclusions in branch pages.
- Put metrics and command output in per-test logs.
- Use `$compact-todo` when this page, branch pages, or log index grow too large.
```

## notes/todo/README.md

Create `notes/todo/README.md`:

````md
# Todo Branch Directory

This directory stores detailed branch memory split out from [../todo.md](../todo.md).

## Layout Contract

- `todo.md` is the dashboard.
- `notes/todo/` stores branch memory.
- `notes/todo/archive/` stores cold closed memory.
- `notes/log/` stores evidence.

## Branch Page Shape

Branch pages should include:

- `Current State`
- `Open Children`
- `Closed Children Archive`
- `Related Logs`
- `Git Refs`
- `Next Step`
- `Node Details`

## Relationship Types

- `child-of`
- `caused-by`
- `blocks`
- `related-to`

## Root Map Row Template

```text
| T100 | doing | module or flow | [T100](T100-topic.md) | active child T101 | feature `abc1234`; verified `pending` |
```

## Open Leaf Row Template

```text
| T101 | T100 | doing | P0 | why active | [branch](T100-topic.md) + [log](../log/YYYY-MM-DD-HHMM-topic.md) |
```

## Branch Page Template

```text
# T100 Problem Theme

## Current State

- Short current summary.

## Open Children

- [T101](../todo.md#open-leaves): next check; see details in this branch page.

## Closed Children Archive

- T102 done: one-line outcome.

## Related Logs

- [YYYY-MM-DD-HHMM-topic.md](../log/YYYY-MM-DD-HHMM-topic.md)

## Git Refs

- Last Feature Commit: `abc1234`
- Last Verified Commit: `abc1234`
- Current Work Ref: `working tree on top of abc1234 (YYYY-MM-DD HH:MM)`
- Key Files:
  - [path/to/file.py](../../path/to/file.py)

## Next Step

- Next action.

## Node Details

### T101 title

- why-created:
- hypothesis:
- evidence:
```
````

## notes/todo/T000-notes-workflow.md

Create `notes/todo/T000-notes-workflow.md`:

```md
# T000 notes workflow

## Current State

- Todo/log memory system bootstrapped.
- Root dashboard, branch memory, log index, and archive folders exist.
- Hard constraints are installed through `AGENTS.md` and `.codex/RULES.md`.

## Root Dashboard

- [notes/todo.md](../todo.md)

## Open Children

- none

## Closed Children Archive

- T000 bootstrap: memory system created.

## Related Logs

- [BOOTSTRAP_TIME-notes-workflow-bootstrap.md](../log/BOOTSTRAP_TIME-notes-workflow-bootstrap.md)

## Git Refs

- Last Feature Commit: `CURRENT_GIT_BASE`
- Last Verified Commit: `CURRENT_GIT_BASE`
- Current Work Ref: `working tree on top of CURRENT_GIT_BASE (BOOTSTRAP_TIME); bootstrap notes memory system`
- Key Files:
  - [AGENTS.md](../../AGENTS.md)
  - [.codex/RULES.md](../../.codex/RULES.md)
  - [notes/index.md](../index.md)
  - [notes/todo.md](../todo.md)
  - [notes/todo/README.md](README.md)
  - [notes/log/index.md](../log/index.md)

## Next Step

- Add the first real root node when project work begins.

## Node Details

### T000 bootstrap

- why-created: establish persistent agent memory and evidence trail.
- evidence: bootstrap log.
```

## notes/log/index.md

Create `notes/log/index.md`:

````md
# Test Log Index

This page indexes verification evidence. Keep it short enough to scan.

## Recent Logs

| Time | Topic | Stage | Result | Key Metrics | Todo | File |
| --- | --- | --- | --- | --- | --- | --- |
| BOOTSTRAP_DATE_TIME | notes workflow bootstrap | notes workflow | pass | dashboard memory system created | [T000](../todo.md#root-map) | [BOOTSTRAP_TIME-notes-workflow-bootstrap.md](BOOTSTRAP_TIME-notes-workflow-bootstrap.md) |

## Topic Log Index

- T000 notes workflow:
  - [BOOTSTRAP_TIME-notes-workflow-bootstrap.md](BOOTSTRAP_TIME-notes-workflow-bootstrap.md)

## Archived Logs

- none yet

## How To Add A New Entry

1. Create one log file under `notes/log/`.
2. Add a row to `Recent Logs`.
3. Add or update the topic group in `Topic Log Index`.
4. Update `notes/todo.md` and relevant branch pages.
5. If `Recent Logs` grows too long, move older rows to `notes/log/archive/`.

## Copyable Row Template

```text
| YYYY-MM-DD HH:MM | topic | stage | pass/fail/partial | metric summary | [T001](../todo.md#open-leaves) | [YYYY-MM-DD-HHMM-topic.md](YYYY-MM-DD-HHMM-topic.md) |
```
````

## notes/log/BOOTSTRAP_TIME-notes-workflow-bootstrap.md

Create `notes/log/BOOTSTRAP_TIME-notes-workflow-bootstrap.md`:

```md
# Notes Workflow Bootstrap

## Purpose

Create the initial todo/log memory system and hard repository rules.

## Stage

- notes workflow

## Related Todo

- [T000 notes workflow](../todo.md#root-map)

## Procedure

- Created `AGENTS.md`, `.codex/RULES.md`, notes dashboard, branch directory, T000 branch, log index, archive folders, and this bootstrap log.

## Input Conditions

- Project: `PROJECT_NAME`
- Git base: `CURRENT_GIT_BASE`
- Bootstrap time: `BOOTSTRAP_TIME`

## Key Metrics

- Required files created.
- Root dashboard links to T000 and bootstrap log.
- Log index links to bootstrap log.
- Branch page links back to dashboard and log.

## Result

- pass

## Conclusion

- The repository now has persistent todo/log memory with hard pre-read and update rules.

## Follow-Up

- Add the first real root node when project work begins.

## Git Refs

- Base HEAD at test time: `CURRENT_GIT_BASE`
- Baseline Ref: `none`
- Candidate Ref: `working tree on top of CURRENT_GIT_BASE (BOOTSTRAP_TIME); bootstrap notes memory system`
- Key Files:
  - [AGENTS.md](../../AGENTS.md)
  - [.codex/RULES.md](../../.codex/RULES.md)
  - [notes/index.md](../index.md)
  - [notes/todo.md](../todo.md)
  - [notes/todo/README.md](../todo/README.md)
  - [notes/todo/T000-notes-workflow.md](../todo/T000-notes-workflow.md)
  - [notes/log/index.md](index.md)
```

## Optional Directories

`notes/ai/README.md`:

```md
# AI Notes

Agent-facing architecture, stage, contract, and data-flow notes.
```

`notes/human/README.md`:

```md
# Human Notes

Human-facing explanations, decisions, workflows, and summaries.
```

## Verification

Run:

```bash
find notes -maxdepth 3 -type f -o -type d | sort
rg -n "## Start Here|## Root Map|## Open Leaves|## Topic Log Index|T000|Git Refs" AGENTS.md .codex/RULES.md notes
```

Check:

- `notes/todo.md` is a dashboard, not a full tree.
- No separate `Node Index` exists.
- Branch pages and logs link with relative paths.
- Archive directories exist.
- If `AGENTS.md`, `.codex/RULES.md`, or `notes/` already existed, stronger repository-specific constraints were preserved.
- All placeholders are replaced:
  - `PROJECT_NAME`
  - `PROJECT_FOCUS`
  - `PRIMARY_CODE_ENTRY_POINT_1`
  - `PRIMARY_CODE_ENTRY_POINT_2`
  - `PRIMARY_CODE_ENTRY_POINT_3`
  - `CURRENT_GIT_BASE`
  - `BOOTSTRAP_TIME`
  - `BOOTSTRAP_DATE_TIME`
  - `USE_COMPACT_TODO`
