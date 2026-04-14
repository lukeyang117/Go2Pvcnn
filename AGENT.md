# Repository Agent Instructions

Before any analysis, code reading, code changes, test execution, training, evaluation, or completion claims in this repository, read and follow:

- [notes/index.md](notes/index.md)
- [.codex](.codex)

These files are hard repository constraints for work performed in `/home/lhy/testPvcnnWithIsaacsim` unless the user explicitly overrides them in the current conversation.

Repository-specific expectations:

- `notes/index.md` is the mandatory knowledge entrypoint for every new conversation.
- Notes must keep repository-relative links so they render correctly from the server workspace and from the local Obsidian vault opened through `Z:`.
- `raw/` and `onlyReference/` are reference directories by default.
- Active code is primarily under `Go2Pvcnn/`, with root-level assets, docs, scripts, and helper files supporting that project.
- Tasks involving `Go2Pvcnn/extension/planner`, planner-based trajectory rewards, or `raw/kinematic_footsteps` sync must read the planner notes entry first.
- The minimum planner pre-read set is `notes/human/human-08-extension-planner-reading-guide.md` and `notes/human/human-09-extension-planner-mapping.md`, or the paired AI notes.

## Cursor parity

Cursor loads `.cursor/rules/repository-constraints.mdc` with `alwaysApply: true`; it mirrors this file and `.codex` so agent work stays aligned across tools.
