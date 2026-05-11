---
name: create-todo-log-systerm
description: "Create a self-contained todo/log agent-memory system for a new repository using the same dashboard-oriented layout maintained by compact-todo: notes/todo.md as Start Here dashboard, notes/todo branch pages as detailed memory, notes/log index as recent/topic log index, archive folders for cold history, git traceability fields, dynamic issue expansion rules, and hard repository constraints. Use when bootstrapping a new project with a complete agent memory workflow without depending on any existing repository Markdown files."
---

# Create Todo Log Systerm

IRON LAW: Use this skill to bootstrap a new repository memory system, not to overwrite stronger repository-specific rules. If `AGENTS.md`, `.codex/RULES.md`, or an established `notes/` workflow already exist, merge carefully and preserve stricter local constraints.

## Goal

Bootstrap a complete `notes/` based agent memory system in a new repository. The generated system must be immediately useful for agents:

- `notes/todo.md` is a concise dashboard.
- `notes/todo/` stores branch memory.
- `notes/log/` stores per-test evidence.
- `archive/` folders store cold closed history.
- `.codex/RULES.md` and `AGENTS.md` make the workflow a hard constraint.

Use the bundled standalone template:

- `assets/create-todo-log-systerm.md`

The asset is self-contained. Do not require the target project to copy Markdown from another repository.

## Relationship With compact-todo

This skill creates the initial structure. `$compact-todo` maintains the same structure later.

If both skills are installed, the format should stay aligned:

- same target file tree
- same `notes/todo.md` sections
- same branch page sections
- same log index sections
- same archive folders

## Workflow

1. Read `assets/create-todo-log-systerm.md`.
2. In the target repository, create the directories and files specified by the template.
3. Replace placeholders such as project name, active architecture, key files, git base, and bootstrap time.
4. Create or update `AGENTS.md` and `.codex/RULES.md`.
   - If the repository already has stronger or more specific rules, merge the todo/log workflow into those files instead of replacing repository-specific constraints.
   - Keep any existing mandatory pre-read order, module-specific reading guides, path constraints, and note-linking conventions.
5. Create `notes/index.md`, `notes/todo.md`, `notes/todo/README.md`, `notes/todo/T000-notes-workflow.md`, `notes/log/index.md`, and the bootstrap log.
6. Create `notes/todo/archive/` and `notes/log/archive/`.
7. Verify all relative links and required sections.

## Required Target Files

```text
AGENTS.md
.codex/RULES.md
notes/index.md
notes/todo.md
notes/todo/README.md
notes/todo/T000-notes-workflow.md
notes/todo/archive/
notes/log/index.md
notes/log/YYYY-MM-DD-HHMM-notes-workflow-bootstrap.md
notes/log/archive/
```

Add `notes/ai/` and `notes/human/` when the project needs stage-level or user-facing documentation alignment rules.

## Validation

Before claiming completion, check:

- `notes/todo.md` has `Start Here`, `Active Fronts`, `Root Map`, `Open Leaves`, `Branch Pages`, and `Recent Logs`.
- `notes/todo.md` does not contain a separate flat `Node Index`.
- `notes/todo/README.md` describes the branch-memory contract and templates.
- `notes/todo/T000-notes-workflow.md` links back to root and bootstrap log.
- `notes/log/index.md` has `Recent Logs`, `Topic Log Index`, `Archived Logs`, and row template.
- `notes/todo/archive/` and `notes/log/archive/` exist.
- `AGENTS.md` points agents to `.codex/RULES.md`.
- `.codex/RULES.md` requires pre-reading `notes/todo.md`, relevant branch pages, and relevant logs before work.
- Existing repository-specific constraints were preserved if the repository already had `AGENTS.md`, `.codex/RULES.md`, or `notes/`.
- All non-template links are relative and resolve.

Useful checks:

```bash
find notes -maxdepth 3 -type f -o -type d | sort
rg -n "## Start Here|## Root Map|## Open Leaves|## Topic Log Index|Git Refs|T000" AGENTS.md .codex/RULES.md notes
```
