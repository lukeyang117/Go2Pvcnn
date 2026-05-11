# Compact Todo Interactive Memory And Test Grooming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign `compact-todo` so it runs as a direction-driven memory and test grooming session instead of a static notes compaction checklist.

**Architecture:** Keep the implementation centered in `.agents/skills/compact-todo/SKILL.md` and preserve the existing repository memory layout. Replace the old static sections with a session workflow that scans the whole notes tree, classifies nodes, auto-splits only safe open-node structure, and batches destructive decisions. Extend the skill contract to include a full `Go2Pvcnn/tests/` review pass grouped by module family and candidate type.

**Tech Stack:** Markdown skill authoring, repository notes workflow, skill-TDD pressure scenarios, ripgrep-based verification.

---

## File Structure

- Modify: `.agents/skills/compact-todo/SKILL.md`
  - Replace the static compaction workflow with the approved interactive memory/test grooming workflow.
- Optionally modify later only if implementation proves necessary: `.agents/skills/compact-todo/references/kinematic-todo-schema.md`
  - Only if the main skill needs a clarified cross-reference for the kinematic repository case.
- Modify: `notes/todo.md`
  - Keep T002 active-front state aligned after implementation.
- Modify: `notes/todo/T002-compact-todo-interactive-memory-and-test-grooming.md`
  - Record implementation progress and verification state.
- Modify: `notes/log/index.md`
  - Add an implementation log row.
- Create: `notes/log/2026-05-09-<time>-compact-todo-implementation.md`
  - Record baseline, implementation, and verification for the skill update.

## Task 1: Write Skill-TDD Baseline Scenarios

**Files:**
- Modify: `notes/todo/T002-compact-todo-interactive-memory-and-test-grooming.md`
- Test: `.agents/skills/compact-todo/SKILL.md`

- [ ] **Step 1: Write the failing baseline scenarios in the T002 branch page**

Add a short checklist under `T002a` documenting the pre-implementation failures:

```md
- baseline-failures:
  - current skill does not ask for future code direction
  - current skill does not scan `notes/todo/archive` or `notes/log/archive`
  - current skill does not define memory-grooming states
  - current skill does not review `Go2Pvcnn/tests/`
  - current skill does not batch test deletion/rewrite candidates
```

- [ ] **Step 2: Run baseline grep checks to verify the current skill fails**

Run:

```bash
rg -n "Direction Intake|Go2Pvcnn/tests|interface-lagging|mainline-divergent|split-needed|Decision Queue" .agents/skills/compact-todo/SKILL.md
```

Expected: missing matches for most or all of the new workflow terms.

- [ ] **Step 3: Commit baseline note only if it improves traceability**

```bash
git add notes/todo/T002-compact-todo-interactive-memory-and-test-grooming.md
git commit -m "docs: record compact-todo baseline gaps"
```

Skip this commit if the baseline note will be immediately superseded by the implementation update.

## Task 2: Rewrite The Skill Frontmatter And Goal Sections

**Files:**
- Modify: `.agents/skills/compact-todo/SKILL.md`
- Test: `.agents/skills/compact-todo/SKILL.md`

- [ ] **Step 1: Write the failing assertion list for the frontmatter and overview**

Required outcomes:

```text
- description says when to use the skill, not how it works
- overview names both memory grooming and test grooming
- old static “compress and reorganize” framing is no longer the primary framing
```

- [ ] **Step 2: Run a focused readback check to verify the current text is outdated**

Run:

```bash
sed -n '1,80p' .agents/skills/compact-todo/SKILL.md
```

Expected: the current description still centers static compaction instead of the new interactive session model.

- [ ] **Step 3: Write the minimal implementation**

Update the frontmatter and opening sections so they explicitly say:

```md
description: Use when repository memory or test assets need a direction-driven grooming pass ...
```

And revise the opening body to state:

- the skill starts from future code direction
- the scope includes the whole notes tree
- the workflow includes `Go2Pvcnn/tests/`
- destructive actions remain user-approved

- [ ] **Step 4: Run readback verification**

Run:

```bash
sed -n '1,120p' .agents/skills/compact-todo/SKILL.md
```

Expected: the description and overview now match the approved design.

- [ ] **Step 5: Commit**

```bash
git add .agents/skills/compact-todo/SKILL.md
git commit -m "docs: reframe compact-todo as interactive grooming skill"
```

## Task 3: Add The Interactive Session Workflow

**Files:**
- Modify: `.agents/skills/compact-todo/SKILL.md`
- Test: `.agents/skills/compact-todo/SKILL.md`

- [ ] **Step 1: Write the failing assertion list for the session workflow**

Required sections:

```text
Direction Intake
Whole-Tree Scan
Memory State Refresh
Safe Automatic Reshaping
Global Decision Queue
Apply User Decisions
Close The Session
```

- [ ] **Step 2: Run grep to verify these workflow sections are absent or incomplete**

Run:

```bash
rg -n "Direction Intake|Whole-Tree Scan|Memory State Refresh|Safe Automatic Reshaping|Global Decision Queue|Apply User Decisions|Close The Session" .agents/skills/compact-todo/SKILL.md
```

Expected: no complete match set.

- [ ] **Step 3: Write the minimal implementation**

Replace the old `Modes` + `Workflow` core with a new ordered session workflow that:

- begins with future-direction intake
- scans `notes/todo.md`, branch pages, archives, and log indexes
- allows only non-destructive automatic reshaping
- defers archive/delete/merge actions into a batch decision queue
- ends with notes/log sync

- [ ] **Step 4: Run readback verification**

Run:

```bash
sed -n '120,260p' .agents/skills/compact-todo/SKILL.md
```

Expected: the new session flow appears in the correct order and no longer reads like the original static checklist.

- [ ] **Step 5: Commit**

```bash
git add .agents/skills/compact-todo/SKILL.md
git commit -m "docs: add interactive compact-todo session workflow"
```

## Task 4: Add Memory And Test State Models

**Files:**
- Modify: `.agents/skills/compact-todo/SKILL.md`
- Test: `.agents/skills/compact-todo/SKILL.md`

- [ ] **Step 1: Write the failing assertion list for state definitions**

Required definitions:

```text
memory primary classes: active/context/cold/ambiguous
memory overlay: split-needed
test states: keep/rewrite-candidate/remove-candidate/ambiguous
```

- [ ] **Step 2: Run grep to verify the current skill lacks these state definitions**

Run:

```bash
rg -n "active|context|cold|ambiguous|split-needed|rewrite-candidate|remove-candidate" .agents/skills/compact-todo/SKILL.md
```

Expected: missing or partial coverage.

- [ ] **Step 3: Write the minimal implementation**

Add sections that define:

- preserved todo execution statuses
- memory grooming classes and overlay
- transient test grooming states

Also explain that these are session judgments, not persistent YAML tags.

- [ ] **Step 4: Run readback verification**

Run:

```bash
sed -n '260,360p' .agents/skills/compact-todo/SKILL.md
```

Expected: memory and test state sections are present and consistent with the spec.

- [ ] **Step 5: Commit**

```bash
git add .agents/skills/compact-todo/SKILL.md
git commit -m "docs: add compact-todo memory and test state model"
```

## Task 5: Add Auto-Split And Decision Queue Rules

**Files:**
- Modify: `.agents/skills/compact-todo/SKILL.md`
- Test: `.agents/skills/compact-todo/SKILL.md`

- [ ] **Step 1: Write the failing assertion list for split and decision behavior**

Required rules:

```text
split triggers: >3 effective children, bloated sections, mixed directions
allowed auto-split actions for open nodes only
forbidden silent archive/delete shortcuts
batched archive/delete/merge/ambiguous queues
```

- [ ] **Step 2: Run grep to verify the current skill lacks the full rule set**

Run:

```bash
rg -n "split-needed|effective child|bloated sections|archive candidates|delete candidates|merge candidates|ambiguous nodes" .agents/skills/compact-todo/SKILL.md
```

Expected: missing or incomplete rule coverage.

- [ ] **Step 3: Write the minimal implementation**

Add sections that:

- define the three split triggers
- restrict auto-split to non-destructive restructuring
- require user confirmation for archive/delete/merge
- require per-candidate evidence in the batch queue

- [ ] **Step 4: Run readback verification**

Run:

```bash
sed -n '360,460p' .agents/skills/compact-todo/SKILL.md
```

Expected: split rules and decision queue rules are both present and non-contradictory.

- [ ] **Step 5: Commit**

```bash
git add .agents/skills/compact-todo/SKILL.md
git commit -m "docs: add compact-todo split and decision queue rules"
```

## Task 6: Add Full Test-Asset Review Rules

**Files:**
- Modify: `.agents/skills/compact-todo/SKILL.md`
- Test: `.agents/skills/compact-todo/SKILL.md`

- [ ] **Step 1: Write the failing assertion list for test review**

Required content:

```text
scan full Go2Pvcnn/tests tree
compare against future direction, current code, todo/log evidence
module-first grouping
candidate labels include interface-lagging and mainline-divergent
support batch keep/rewrite/delete/defer
```

- [ ] **Step 2: Run grep to verify the current skill does not define the test workflow**

Run:

```bash
rg -n "Go2Pvcnn/tests|interface-lagging|mainline-divergent|module family|rewrite|defer|batch deletion" .agents/skills/compact-todo/SKILL.md
```

Expected: missing matches before the rewrite.

- [ ] **Step 3: Write the minimal implementation**

Add a dedicated `Test Asset Review` section that:

- scans the full `Go2Pvcnn/tests/` tree
- compares tests to code and notes/log evidence
- groups output by module family first, then candidate type
- supports keep/rewrite/delete/defer decisions
- forbids automatic test deletion

- [ ] **Step 4: Run readback verification**

Run:

```bash
sed -n '460,560p' .agents/skills/compact-todo/SKILL.md
```

Expected: full test-asset review rules are present and include the user's batch-cleanup requirement.

- [ ] **Step 5: Commit**

```bash
git add .agents/skills/compact-todo/SKILL.md
git commit -m "docs: add compact-todo test asset review rules"
```

## Task 7: Update Validation, Notes, And Final Verification

**Files:**
- Modify: `.agents/skills/compact-todo/SKILL.md`
- Modify: `notes/todo.md`
- Modify: `notes/todo/T002-compact-todo-interactive-memory-and-test-grooming.md`
- Modify: `notes/log/index.md`
- Create: `notes/log/2026-05-09-<time>-compact-todo-implementation.md`

- [ ] **Step 1: Write the failing assertion list for final validation**

Required coverage:

```text
dashboard still short
new child pages linked
archive remains traceable
test bundles grouped by module then candidate type
implementation log created
```

- [ ] **Step 2: Run the full verification commands**

Run:

```bash
rg -n "Direction Intake|Whole-Tree Scan|Memory State|Test Asset Review|Go2Pvcnn/tests|interface-lagging|mainline-divergent|remove-candidate" .agents/skills/compact-todo/SKILL.md
sed -n '1,260p' .agents/skills/compact-todo/SKILL.md
```

Expected: all required sections and terms are present and coherent.

- [ ] **Step 3: Write the implementation log and sync notes**

Update:

- `notes/todo.md`
- `notes/todo/T002-compact-todo-interactive-memory-and-test-grooming.md`
- `notes/log/index.md`
- one new implementation log under `notes/log/`

The log must record:

- baseline gaps
- skill sections added/replaced
- verification commands run
- any remaining unverified pressure scenarios

- [ ] **Step 4: Run final note-link verification**

Run:

```bash
rg -n "T002|compact-todo interactive memory and test grooming" notes/todo.md notes/todo/T002-compact-todo-interactive-memory-and-test-grooming.md notes/log/index.md notes/log
```

Expected: T002 references resolve across dashboard, branch page, and logs.

- [ ] **Step 5: Commit**

```bash
git add .agents/skills/compact-todo/SKILL.md notes/todo.md notes/todo/T002-compact-todo-interactive-memory-and-test-grooming.md notes/log/index.md notes/log/2026-05-09-*-compact-todo-implementation.md
git commit -m "docs: implement interactive compact-todo redesign"
```
