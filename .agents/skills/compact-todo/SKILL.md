---
name: compact-todo
description: Use when repository memory or `Go2Pvcnn/tests` need a direction-driven grooming pass because the active mainline has shifted, notes grew hard to navigate, or tests/logs may be stale relative to current code
---

# Compact Todo

## Goal

Run `compact-todo` as a direction-driven grooming session, not a one-shot formatting pass.

The skill must help the next agent answer:

- what future code direction matters now
- which note nodes are still active or only background
- which open nodes need to split into deeper pages
- which notes/logs should be kept, archived, merged, or deleted
- which tests still serve the future mainline and which ones should be rewritten or removed

The system being groomed includes:

- `notes/todo.md`
- `notes/todo/`
- `notes/todo/archive/`
- `notes/log/index.md`
- `notes/log/archive/`
- `Go2Pvcnn/tests/`

Destructive actions stay user-approved:

- archive
- delete
- merge
- test deletion

## Shared Layout

The repository memory layout still follows the same base structure:

```text
notes/
├── index.md
├── todo.md
├── todo/
│   ├── README.md
│   ├── T000-notes-workflow.md
│   └── archive/
└── log/
    ├── index.md
    └── archive/
```

Per-test logs stay directly under `notes/log/` until archived.

## Pre-Read

Before changing files, read:

1. repository rules such as `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, or `.codex/RULES.md`
2. `notes/todo.md`
3. `notes/log/index.md`
4. `notes/todo/README.md`
5. branch pages for active/open nodes
6. recent logs linked from active/open nodes

If the repository has stricter todo/log rules, follow them.

## Root Dashboard Contract

`notes/todo.md` remains a dashboard rather than a full database.

It must keep:

- `Start Here`
- `Status Legend`
- `Active Fronts`
- `Root Map`
- `Open Leaves`
- `Branch Pages`
- `Recent Logs`
- `Maintenance`

Keep the dashboard roughly `60-120` lines when possible.
Treat `notes/todo.md` as the top Obsidian index for active roots and fronts.
Closed detail belongs in branch pages or archives, not in the root dashboard.

## Branch Page Contract

Branch pages remain the durable memory surface.

Expected sections:

- `Current State`
- `Open Children`
- `Closed Children Archive`
- `Related Logs`
- `Git Refs`
- `Next Step`
- `Node Details`

Rules:

- `Current State` stays short and operational.
- `Open Children` lists active leaves and next checks.
- `Closed Children Archive` stores one-line closed summaries.
- `Node Details` stores long `why-created`, evidence, hypotheses, and decisions.
- Detailed metrics belong in per-test logs; branch pages summarize and link.
- `Open Children` and `Closed Children Archive` stay as subtree indexes, not freeform prose.
- Keep child ids and destination links visible so Obsidian navigation survives compaction.

## Tree Preservation And Obsidian Index Contract

Treat `notes/todo.md`, branch pages under `notes/todo/`, and archive indexes as Obsidian navigation surfaces, not disposable summaries.

When compacting child nodes:

- preserve a visible parent -> child tree in markdown lists or tables
- keep every compacted child named by node id, status, and destination link
- keep the parent page as the subtree index even when child details move to another page
- keep an upward link path `child -> parent page anchor` and a downward path `root/parent -> child page or archive entry`

Compression means moving detail out of the hot path, not flattening the tree into prose.

Forbidden flattening shortcuts:

- do not replace several named children with one unlabeled sentence
- do not leave a child page reachable only from backlinks or search
- do not remove the last index link to a child page, archive entry, or related log chain

## Direction Intake

Start every `compact-todo` session by asking for the future code direction in plain language.

Requirements:

- Do not require file paths up front.
- Treat the user's direction statement as the main relevance signal.
- If the user provides code paths anyway, use them as stronger supporting signals.
- If the direction is still too vague after one pass, ask a short clarifying question before restructuring anything.

## Whole-Tree Scan

After direction intake, scan the full memory tree:

- `notes/todo.md`
- branch pages under `notes/todo/`
- `notes/todo/archive/`
- `notes/log/index.md`
- `notes/log/archive/`

Also scan the full test tree:

- `Go2Pvcnn/tests/`

Build a session-local inventory of:

- active/open nodes
- recent evidence
- cold history
- oversized nodes
- mixed-topic nodes
- stale or off-mainline tests

Do not make destructive edits during the scan.

## Existing Task Statuses Stay Intact

The existing todo execution statuses remain unchanged:

- `todo`
- `doing`
- `blocked`
- `verify`
- `done`
- `drop`
- any repository-local extension already in use

These describe task progress, not grooming priority.

## Memory State Refresh

Refresh memory state on every compact session.

Use one primary memory classification:

- `active`
- `context`
- `cold`
- `ambiguous`

And one optional structural overlay:

- `split-needed`

Interpretation:

- `active`
  - strongly relevant to the future direction
  - open or recently evidenced
- `context`
  - still useful background, but not current mainline
- `cold`
  - low relevance and mostly historical
- `ambiguous`
  - not safe to classify without asking the user
- `split-needed`
  - still important, but structurally too large or mixed to keep in one page

In conversation it is valid to say a node is both `active` and `split-needed`.
Treat `split-needed` as a structural overlay, not a replacement for the primary class.

## Memory Classification Signals

Judge each node from several signals:

- future-direction keywords from the user
- overlap with node title, `Current State`, `Node Details`, and related logs
- direct links from active leaves or active fronts
- presence of open children
- recent logs still being appended
- whether the node is already superseded by newer design or execution fronts
- whether a single page is now carrying multiple future directions

When signals conflict:

- prefer `ambiguous`
- ask the user
- default to keeping the node instead of deleting or archiving it

## Safe Automatic Reshaping

Before asking the user for destructive decisions, the skill may automatically:

- refresh dashboard emphasis for `active` and `context` nodes
- reorder summaries for readability
- tighten parent/child index lines while preserving relationship visibility
- create deeper branch pages for oversized open nodes
- shrink parent pages into short summaries plus links

The skill must not automatically:

- archive a node
- delete a node
- merge nodes
- delete tests
- rewrite tests

## Auto-Split Rules For Open Nodes

Open nodes may be auto-split only when the action is clearly non-destructive.

Trigger `split-needed` if any one of these is true:

- effective child count is greater than `3`
- one or more major sections are bloated
- multiple future directions are mixed inside one page

Allowed auto-split actions:

- create a deeper child page under `notes/todo/`
- move detailed open-child material into that page
- keep the child visible in the parent `Open Children` or `Closed Children Archive`
- leave the parent page as a short summary with links and relationships

Recommended child-page naming:

```text
notes/todo/T116-k5-mode-first-cross-small.md
notes/todo/T002-interactive-compact-todo-redesign.md
```

Forbidden auto-split shortcuts:

- do not discard content without a destination
- do not silently collapse evidence into prose with no links
- do not flatten several named children into one generic history sentence
- do not move history into archive without user approval

## Global Decision Queue

Do not interrupt one item at a time when archive/delete/merge decisions are needed.

Build one global-first queue with grouped decisions:

- `archive candidates`
- `delete candidates`
- `merge candidates`
- `ambiguous nodes`
- `test rewrite/delete candidates`

For each candidate, include:

- affected file or node
- proposed action
- why it was flagged
- evidence supporting the flag

Valid user actions:

- keep
- archive
- merge
- delete
- defer

## Test Asset Review

Every `compact-todo` session must scan the full `Go2Pvcnn/tests/` tree.

Review tests against:

- the user's future mainline direction
- current implementation code
- related `notes/todo` and `notes/log` evidence
- nearby fixtures/helpers in the same module family

Use the following transient test states:

- `keep`
- `rewrite-candidate`
- `remove-candidate`
- `ambiguous`

These are session judgments only, not persistent annotations.

## Test Candidate Types

Surface tests when they appear to be:

- `interface-lagging`
  - still asserting old names, old parameters, old horizons, or removed concepts
- `mainline-divergent`
  - still guarding an architecture path the future mainline is moving away from
- `low-value-duplicate`
  - mostly duplicated by stronger, more current coverage
- `ambiguous`
  - not safe to classify without user input

If a test looks stale but may still be the only guard for a live contract, prefer `rewrite-candidate` over `remove-candidate`.

## Mixed Test Grouping Format

Present test review in mixed groups:

1. group by module family first
   - `batched_together_*`
   - `viewer_*`
   - `semantic_*`
   - other repository test families
2. within each module group, label each candidate with:
   - `interface-lagging`
   - `mainline-divergent`
   - `remove-candidate`
   - `ambiguous`

The user must be able to decide in batches:

- keep
- rewrite
- delete
- defer

If the user decides a whole batch of tests is no longer valuable, support batch deletion after explicit approval.

Never delete tests automatically.

## Apply User Decisions

Once the grouped decisions are reviewed, apply only the approved actions:

- archive approved note/log items
- delete approved note/log items
- merge approved nodes
- rewrite approved tests
- delete approved tests
- leave deferred items untouched but documented

## Log Index Expectations

`notes/log/index.md` still keeps:

- `Recent Logs`
- `Topic Log Index`
- `Archived Logs`
- `How To Add A New Entry`

Rules:

- keep only recent operational rows in `Recent Logs`
- group older important evidence by todo/topic in `Topic Log Index`
- move old chronological rows into `notes/log/archive/YYYY-MM.md` or `notes/log/archive/<topic>.md`
- do not delete per-test logs unless the user explicitly asks

## Modes

Choose the lightest mode that solves the problem:

- `daily compact`
  - refresh dashboard focus and recent evidence
- `root compact`
  - repair an oversized root dashboard
- `branch compact`
  - shorten one branch page while preserving memory
- `log compact`
  - trim `notes/log/index.md` and move old rows into archive indexes
- `archive compact`
  - move confirmed cold history into archive files while preserving traceability
- `test grooming`
  - review stale or off-mainline tests together with notes/log evidence
- `full grooming session`
  - direction intake + whole-tree scan + test review + decision queue

## Session Close

At the end of a meaningful compact session:

- update relevant note surfaces
- update archive indexes if used
- update touched branch pages
- update `notes/log/index.md`
- write one official compact-session log

If the repository requires memory-system logs, do not skip the log step.

## Validation

Before claiming completion:

- `notes/todo.md` is still a dashboard
- `notes/todo.md` still works as the top Obsidian index entry
- root page does not expand every closed child
- compacted child nodes still appear in a visible `root -> parent -> child/archive` path
- every open leaf still has a branch link and relevant log path
- every root theme has a branch page or explicit archive
- parent pages keep named child entries instead of flattening them into prose
- each new child page has a parent-page link path
- archive entries remain traceable from active memory
- no obvious orphan note pages were created
- no page is reachable only through backlinks or search
- relative links still resolve
- test candidate bundles are grouped by module first, then candidate type
- the compact session produced one official log

Useful checks:

```bash
wc -l notes/todo.md notes/log/index.md notes/todo/*.md
rg -n "## Start Here|## Active Fronts|## Open Leaves|## Topic Log Index" notes
rg -n "Go2Pvcnn/tests|interface-lagging|mainline-divergent|remove-candidate|split-needed" .agents/skills/compact-todo/SKILL.md
rg -n "Obsidian|parent -> child|parent-page link|backlinks|orphan" .agents/skills/compact-todo/SKILL.md notes/todo/README.md
```

## kinematic_footsteps

For `/home/lhy/kinematic_footsteps`, read `references/kinematic-todo-schema.md` before editing its todo/log memory files.
