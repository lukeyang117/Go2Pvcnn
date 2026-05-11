# Local skill installation

## Purpose

- Install two reusable local skills from `lukeyang117/my_skill` into the current repository so they are available under `.agents/skills/`.

## Stage

- repository tooling / local skill availability

## Related Todo

- [T000](../todo/T000-notes-workflow.md)

## Command / Procedure

- Confirmed repository constraints by reading [notes/index.md](../index.md), [notes/todo.md](../todo.md), [notes/log/index.md](index.md), and [.codex/RULES.md](../../.codex/RULES.md).
- Inspected the remote repository layout and identified the two skill paths:
  - `skills/compact-todo`
  - `skills/create-todo-log-systerm`
- Installed both into the project-local skill directory with:

```bash
python /mnt/mydisk/lhy/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo lukeyang117/my_skill \
  --path skills/compact-todo skills/create-todo-log-systerm \
  --dest .agents/skills \
  --method download
```

## Input Conditions

- Workspace: `/mnt/mydisk/lhy/testPvcnnWithIsaacsim`
- Remote repo head: `ef07832` (`main`)
- Current work ref: `cf7e9cf`

## Key Metrics

- Install result: `2` skills installed
- Installed paths:
  - `.agents/skills/compact-todo`
  - `.agents/skills/create-todo-log-systerm`

## Result

- Pass

## Conclusion

- The requested two skills are now present in the local project under `.agents/skills/` and can be committed with the repository if desired.

## Follow-up

- If these skills should also be globally available across repositories, install the same paths again into `~/.codex/skills`.
- If the user wants them surfaced in repository docs, add them to a local skill inventory note later.

## Git Refs

- Baseline Ref: `cf7e9cf`
- Candidate Ref: `working tree on top of cf7e9cf`
- Key Files:
  - [.agents/skills/compact-todo/SKILL.md](../../.agents/skills/compact-todo/SKILL.md)
  - [.agents/skills/create-todo-log-systerm/SKILL.md](../../.agents/skills/create-todo-log-systerm/SKILL.md)
  - [notes/log/2026-05-06-1532-local-skill-installation.md](2026-05-06-1532-local-skill-installation.md)
