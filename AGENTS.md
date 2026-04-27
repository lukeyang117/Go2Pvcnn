# Repository Agent Instructions

Before any analysis, code reading, code changes, test execution, training, evaluation, debugging, or completion claims in this repository, read and follow:

- [notes/index.md](notes/index.md)
- [.codex/RULES.md](.codex/RULES.md)

These files are hard repository constraints for work performed in `/home/lhy/testPvcnnWithIsaacsim` unless the user explicitly overrides them in the current conversation.

Repository-specific expectations:

- `notes/index.md` is the mandatory knowledge entrypoint for every new conversation.
- `notes/todo.md`, relevant branch pages under `notes/todo/`, and relevant logs under `notes/log/` are mandatory working memory before substantive work.
- Notes must keep repository-relative links so they render correctly from the server workspace and from the local Obsidian vault opened through `Z:`.
- `raw/` and `onlyReference/` are reference directories by default.
- Active code is primarily under `Go2Pvcnn/`, with root-level assets, docs, scripts, and helper files supporting that project.
- Tasks involving `Go2Pvcnn/extension/planner`, planner-based trajectory rewards, or `raw/kinematic_footsteps` sync must read the planner notes entry first.
- The minimum planner pre-read set is `notes/human/human-08-extension-planner-reading-guide.md` and `notes/human/human-09-extension-planner-mapping.md`, or the paired AI notes.
- If `.codex/RULES.md` requires additional notes, todo branch pages, logs, or stage-specific documents to be read before edits, complete that pre-read first and then continue with the task.

## Subagent Override

- If you are running as a subagent for a delegated task in this repository, do not automatically invoke skills or superpower workflows.
- For subagents here, the delegating prompt is the primary workflow contract.
- A subagent may use a skill only if the delegating agent or the end user explicitly asks for that skill.
- The goal for subagents in this repository is minimal startup overhead and direct task execution.

## Cursor Parity

Cursor loads `.cursor/rules/repository-constraints.mdc` with `alwaysApply: true`; it should mirror this file and `.codex/RULES.md` so agent work stays aligned across tools.
