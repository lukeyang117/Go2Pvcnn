# create-todo-log skill hardening

## Purpose

- Review and harden the project-local `create-todo-log-systerm` skill so it does not overwrite stronger repository rules and does not generate broken relative links in its template output.

## Stage

- repository tooling / local skill maintenance

## Related Todo

- [T000](../todo/T000-notes-workflow.md)

## Command / Procedure

- Reviewed:
  - `.agents/skills/create-todo-log-systerm/SKILL.md`
  - `.agents/skills/create-todo-log-systerm/assets/create-todo-log-systerm.md`
  - `.agents/skills/create-todo-log-systerm/agents/openai.yaml`
  - `.agents/skills/compact-todo/SKILL.md`
  - `.codex/RULES.md`
- Patched `SKILL.md` to add an explicit hard constraint: bootstrap new repositories only, and preserve stricter existing repository-specific rules.
- Patched the asset template to:
  - preserve existing `AGENTS.md` / `.codex/RULES.md` constraints when present
  - preserve stronger existing note entrypoints
  - fix broken relative link examples in `notes/todo/README.md`
  - replace the invalid `../todo.md#t101-title` branch-page example with a stable dashboard anchor
  - add an explicit root-dashboard link to the generated `T000` branch page
  - require preservation checks during verification
- Ran targeted grep checks against the updated skill files.

## Input Conditions

- Workspace: `/mnt/mydisk/lhy/testPvcnnWithIsaacsim`
- Current work ref: `cf7e9cf`
- Skill under review:
  - `.agents/skills/create-todo-log-systerm/`

## Key Metrics

- Updated files: `2`
  - `.agents/skills/create-todo-log-systerm/SKILL.md`
  - `.agents/skills/create-todo-log-systerm/assets/create-todo-log-systerm.md`
- `SKILL.md` size after hardening: `87` lines
- Asset template size after hardening: `640` lines
- Verified presence of:
  - overwrite-protection wording
  - repository-specific constraint preservation checks
  - corrected relative link examples
  - root-dashboard link in `T000`

## Result

- Pass

## Conclusion

- The skill now better matches this repository's expectations for safe note/rule maintenance and no longer teaches broken relative link patterns in its generated templates.

## Follow-up

- Optional later cleanup: rename `create-todo-log-systerm` to `create-todo-log-system` across directory, frontmatter, asset file, and agent metadata if you want naming consistency.

## Git Refs

- Baseline Ref: `cf7e9cf`
- Candidate Ref: `working tree on top of cf7e9cf`
- Key Files:
  - [.agents/skills/create-todo-log-systerm/SKILL.md](../../.agents/skills/create-todo-log-systerm/SKILL.md)
  - [.agents/skills/create-todo-log-systerm/assets/create-todo-log-systerm.md](../../.agents/skills/create-todo-log-systerm/assets/create-todo-log-systerm.md)
  - [notes/log/2026-05-06-1542-create-todo-log-skill-hardening.md](2026-05-06-1542-create-todo-log-skill-hardening.md)
