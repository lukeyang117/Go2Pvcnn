# Compact Todo Interactive Memory And Test Grooming Design

## Metadata

- **Date**: 2026-05-09
- **Topic**: redesign project-local `compact-todo` into a direction-driven memory and test grooming workflow
- **Status**: Draft for spec review
- **Primary Target**: `.agents/skills/compact-todo/SKILL.md`
- **Repository Surfaces**:
  - `notes/todo.md`
  - `notes/todo/`
  - `notes/todo/archive/`
  - `notes/log/index.md`
  - `notes/log/archive/`
  - `Go2Pvcnn/tests/`

## 1. Problem Statement

The current `compact-todo` skill is a static compaction guide. It helps compress `notes/todo.md`, branch pages, and `notes/log/index.md`, but it still assumes that the agent already knows what matters and mainly needs formatting rules.

The user wants a more exploratory workflow:

1. start from a plain-language description of the future code direction
2. scan the whole repository memory tree rather than only the root dashboard
3. refresh the status of memory nodes based on that future direction
4. automatically split oversized open nodes into deeper branch pages
5. discuss archive, delete, and merge decisions in batches instead of one interruption per node
6. inspect `Go2Pvcnn/tests/` alongside logs and current code, then surface stale or off-mainline tests for bulk review

The redesigned skill should behave less like a formatter and more like a repository memory steward that helps the user keep notes and tests aligned with the evolving implementation mainline.

## 2. Goals

### In Scope

- Turn `compact-todo` into an interactive grooming session rather than a one-shot compression checklist.
- Use the user's spoken future-direction description as the main relevance signal.
- Scan the whole repository memory system:
  - `notes/todo.md`
  - `notes/todo/*.md`
  - `notes/todo/archive/*`
  - `notes/log/index.md`
  - `notes/log/archive/*`
- Refresh memory classifications on every compact session.
- Preserve the existing todo execution statuses while adding a separate memory-grooming interpretation layer.
- Auto-split open nodes into deeper `.md` pages when structure is no longer readable.
- Batch archive/delete/merge proposals and ask the user for decisions after the global scan.
- Scan the entire `Go2Pvcnn/tests/` tree every compact session.
- Compare tests against:
  - future mainline direction
  - current implementation
  - related notes/log evidence
- Surface stale tests in batch groups and let the user choose whether to keep, rewrite, or delete them.
- Keep repository-relative links and evidence traceability intact.
- Write an official notes/log entry for each meaningful compact session.

### Out Of Scope

- Automatically deleting notes, logs, archive entries, or tests without explicit user approval.
- Adding persistent YAML metadata to todo nodes or test files.
- Replacing the existing todo execution statuses such as `todo`, `doing`, `verify`, or `done`.
- Performing the actual skill implementation in this design stage.
- Extending the workflow to non-`Go2Pvcnn/tests/` test trees in this first revision.

## 3. User Requirements Captured From Discussion

The design must preserve these confirmed requirements:

1. The user describes the future code direction in plain language; the skill should explore from that description without requiring code paths up front.
2. The scope is the whole repository memory system, not just `notes/todo.md`.
3. Open nodes may be auto-split into new deeper `.md` pages.
4. Auto-split triggers include:
   - more than `3` effective child nodes
   - one or more bloated sections
   - mixed future directions in a single node
5. Archive, delete, and merge decisions must always be surfaced to the user before execution.
6. If the skill cannot judge a node confidently, it should ask the user and default to keeping the node.
7. The interaction style should be global first, then local decision review.
8. Every compact session should refresh memory state rather than relying on stale previous judgments.
9. The full `Go2Pvcnn/tests/` tree should be scanned every compact session.
10. Test candidates should be judged by both:
    - interface lag against current code
    - loss of relevance to the future mainline direction
11. Test review should be grouped in a mixed structure:
    - first by module family
    - then labeled by candidate type such as `interface-lagging`, `mainline-divergent`, or `remove-candidate`
12. If the user decides a batch of tests no longer matters, the workflow should support batch deletion after explicit approval.

## 4. Recommended Design

The recommended design is a direction-driven memory steward with two linked workstreams:

1. `memory grooming`
   - classify and reshape `notes/todo`, `notes/log`, and archives
2. `test grooming`
   - compare `Go2Pvcnn/tests/` against current code, future direction, and log evidence

This is heavier than a simple compactor, but it best matches the user's desired workflow:

- start from future intent rather than current file size alone
- let the agent explore freely before proposing actions
- keep destructive decisions user-controlled
- gradually turn the notes tree into a deeper, cleaner hierarchy
- remove or rewrite tests that no longer serve the live architecture

## 5. Architecture Overview

The redesigned skill should act like a small workflow engine with these logical components:

1. `Direction Intake`
   - asks for the future code direction in plain language
2. `Whole-Tree Memory Scanner`
   - reads todo, log, and archive surfaces
3. `Memory State Refresher`
   - classifies node relevance and structural pressure
4. `Split Planner`
   - decides whether an open node should become a deeper page
5. `Decision Queue Builder`
   - batches archive/delete/merge/ambiguous items for user review
6. `Test Asset Reviewer`
   - scans `Go2Pvcnn/tests/` against code and notes/log evidence
7. `Change Applier`
   - applies safe restructures and user-approved actions
8. `Validation And Log Writer`
   - checks links/sections and records the compact session

### Data Flow

```text
future direction
-> whole-tree scan
-> memory classification + structural pressure review
-> auto-safe restructures for open nodes
-> decision queue for destructive or ambiguous actions
-> user decisions
-> note/test changes
-> validation
-> official compact log
```

## 6. Session Workflow

Each `compact-todo` session should follow the same high-level sequence.

### 6.1 Direction Intake

- Ask the user what future code direction matters next.
- Do not require file paths unless the user chooses to provide them.
- Treat the direction statement as the primary relevance signal for both notes and tests.

### 6.2 Whole-Tree Scan

- Scan:
  - `notes/todo.md`
  - branch pages under `notes/todo/`
  - `notes/todo/archive/`
  - `notes/log/index.md`
  - `notes/log/archive/`
  - `Go2Pvcnn/tests/`
- Build a session-local inventory of:
  - active/open nodes
  - recent evidence
  - cold history
  - oversized nodes
  - stale or off-mainline tests

### 6.3 Safe Automatic Reshaping

Before asking the user for destructive decisions, the skill may automatically:

- refresh dashboard emphasis for active/context nodes
- create deeper branch pages for oversized open nodes
- shrink parent pages to summaries plus links
- reorganize non-destructive sections for readability

The skill must not automatically:

- archive a node
- delete a node
- merge nodes
- delete or rewrite tests

### 6.4 Global Decision Queue

After the scan and safe reshaping pass, the skill should present a batched decision list:

- archive candidates
- delete candidates
- merge candidates
- ambiguous notes/log nodes
- rewrite/delete test candidates

This queue should be global first, then local within each group.

### 6.5 Apply User Decisions

Once the user responds, the skill applies:

- approved archive moves
- approved note deletions
- approved node merges
- approved test rewrites or deletions
- deferred items left untouched but documented

### 6.6 Close The Session

At the end of the session, the skill should:

- update relevant note surfaces
- update archive indexes if used
- update any touched branch pages
- write one official compact-session log

## 7. State Model

The design needs two separate status systems so task execution state does not get confused with grooming state.

### 7.1 Existing Task Statuses Stay Intact

The existing todo execution statuses remain unchanged:

- `todo`
- `doing`
- `blocked`
- `verify`
- `done`
- `drop`
- any repository-local extensions already in use

These describe task progress, not whether a note or test still belongs in the active memory surface.

### 7.2 Memory Grooming State

To avoid contradictions, the memory-grooming layer should use:

- one primary classification:
  - `active`
  - `context`
  - `cold`
  - `ambiguous`
- plus one optional structural overlay:
  - `split-needed`

This resolves an otherwise awkward conflict: a node can be both highly relevant and structurally oversized. In conversation, the skill may still say "this node is active and split-needed," but internally that should be modeled as:

- primary class: `active`
- structural overlay: `split-needed`

### 7.3 Test Grooming State

Test review uses a separate session-local state:

- `keep`
- `rewrite-candidate`
- `remove-candidate`
- `ambiguous`

These states are transient session judgments, not persistent file annotations.

## 8. Memory Classification Rules

Each node should be judged from several signals.

### 8.1 Relevance Signals

- future-direction keywords from the user
- matching module or stage names in node titles
- overlap with `Current State`, `Node Details`, and related logs
- direct links from active leaves or active fronts

### 8.2 Activity Signals

- open children still present
- recent logs still being appended
- explicit mention in current dashboard or branch summaries

### 8.3 Structural Pressure Signals

- more than `3` effective child nodes
- bloated `Node Details`, `Closed Children Archive`, `Related Logs`, or `Current State`
- one node covering multiple future directions

### 8.4 Cooling Signals

- node is completed
- low or no recent evidence
- little connection to the described future direction
- superseded by newer nodes or newer design logs

### 8.5 Ambiguity Signals

- conflicting evidence between active links and cold content
- unclear whether a node is background context or abandoned direction
- unclear whether a test is stale or still a useful guardrail

## 9. Auto-Split Rules For Open Nodes

Open nodes may be split automatically only when the action is clearly non-destructive.

### 9.1 Split Triggers

Any one of these can trigger `split-needed`:

- effective child count greater than `3`
- a major section is visibly bloated
- multiple future directions are mixed together in one page

### 9.2 Allowed Auto-Split Actions

The skill may automatically:

- create a deeper child page under `notes/todo/`
- move detailed open-child material into that page
- leave the parent page as a short summary with links and relationships

### 9.3 Naming Rule For New Child Pages

When a child node graduates into its own page, the new file should use the node id as the leading token:

```text
notes/todo/T116-k5-mode-first-cross-small.md
notes/todo/T002-interactive-compact-todo-redesign.md
```

This keeps page discovery compatible with the current branch-page style while allowing deeper nodes to become first-class pages.

### 9.4 Forbidden Auto-Split Shortcuts

Auto-splitting must not be used as a hidden delete/archive path:

- do not discard old content without a destination
- do not silently collapse evidence into prose with no links
- do not move closed history into archive without user approval

## 10. Decision Queue For Notes And Logs

The skill should present user decisions in grouped batches instead of interrupting per node.

### 10.1 Queue Categories

- `archive candidates`
- `delete candidates`
- `merge candidates`
- `ambiguous nodes`

### 10.2 Per-Candidate Evidence

Each candidate should include at least:

- what file or node is affected
- why it was flagged
- what evidence supports the flag
- what action is proposed

Example reasoning:

- low relevance to the future direction
- superseded by a newer branch or design chain
- duplicated summary already preserved elsewhere
- old log chain now kept only for historical evidence

### 10.3 User Actions

For each group or subgroup, the user should be able to choose:

- keep
- archive
- merge
- delete
- defer

## 11. Test Asset Review

Every compact session should scan the full `Go2Pvcnn/tests/` tree.

### 11.1 Test Review Inputs

The test reviewer should compare each test candidate against:

- future mainline direction from the user
- current implementation code
- linked todo and log evidence
- nearby module and fixture structure

### 11.2 Test Candidate Types

A test may be surfaced when it appears:

- `interface-lagging`
  - still asserting old names, old parameters, old horizons, or removed concepts
- `mainline-divergent`
  - still guarding an architecture path that the future mainline is moving away from
- `low-value-duplicate`
  - largely redundant with stronger, more current tests
- `ambiguous`
  - insufficient evidence to classify safely

The first two are required categories from the user's requests. The third is optional but useful when obvious.

### 11.3 Mixed Grouping Format

The output should be organized:

1. by module family first
   - `batched_together_*`
   - `viewer_*`
   - `semantic_*`
   - other repository test families
2. by candidate label inside the module group
   - `interface-lagging`
   - `mainline-divergent`
   - `remove-candidate`
   - `ambiguous`

### 11.4 Test Decision Actions

Tests must never be deleted automatically. The user chooses, in batches when desired:

- keep
- rewrite
- delete
- defer

If the user decides a whole batch is no longer valuable, the workflow should support batch deletion after explicit approval.

## 12. File Impacts

Implementation should primarily update:

- `.agents/skills/compact-todo/SKILL.md`

Optional only if the main skill becomes unwieldy:

- `.agents/skills/compact-todo/references/interactive-session-pattern.md`

Likely related note updates during implementation:

- `notes/todo/README.md`
  - if child-page naming or deeper branch-page rules need to be documented

## 13. Guardrails And Error Handling

- If relevance is unclear, prefer `ambiguous` and ask the user.
- If a proposed archive/delete/merge would orphan links or evidence trails, stop and surface the risk.
- If a test looks stale but still appears to be the only guard for a live contract, prefer `rewrite-candidate` over `remove-candidate`.
- If a note is structurally bloated but still central to the future direction, split it instead of archiving it.
- Never treat absence from the current mainline as enough evidence by itself to delete logs or tests.

## 14. Validation

Before claiming a compact session complete, the workflow should verify:

- `notes/todo.md` is still a dashboard
- each new child page has a parent-page link path
- no obvious orphan note pages were created
- relative links remain repository-relative
- archive entries remain traceable from active memory
- the compact session produced one official log
- test candidate bundles are grouped by module first, then candidate type

Useful checks after implementation:

```bash
wc -l notes/todo.md notes/log/index.md notes/todo/*.md
rg -n "## Start Here|## Active Fronts|## Open Leaves|## Topic Log Index" notes
rg -n "Go2Pvcnn/tests|interface-lagging|mainline-divergent|remove-candidate" .agents/skills/compact-todo
```

## 15. Acceptance Criteria

The design is successful when the implemented skill can do all of the following in one session:

1. ask for a future code direction in plain language
2. scan the whole notes memory tree
3. refresh node relevance judgments
4. auto-split oversized open nodes without deleting information
5. present archive/delete/merge decisions in global-first batches
6. scan the full `Go2Pvcnn/tests/` tree
7. batch stale-test decisions by module family and candidate type
8. default to asking the user when confidence is low
9. leave a clean official compact-session log

## 16. Next Step

After the user reviews and approves this written spec, the next step is to invoke the `writing-plans` workflow and produce an implementation plan for the `compact-todo` redesign.
