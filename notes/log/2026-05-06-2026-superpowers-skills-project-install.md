# superpowers skills project install

## Purpose

- Make `superpowers` usable from the VS Code Codex workflow by installing its skill directories into the project-local `.agents/skills/` tree.

## Stage

- repository tooling / project-local skill availability

## Related Todo

- [T000](../todo/T000-notes-workflow.md)

## Command / Procedure

- Compared existing project-local skills under `.agents/skills/` against the upstream `plugins/superpowers/skills/` tree.
- Confirmed there were no directory-name conflicts with the incoming superpowers skills.
- Copied the entire `plugins/superpowers/skills/` tree into `.agents/skills/` using `rsync`.
- Verified that `brainstorming` and its companion files now exist directly under `.agents/skills/`.

## Input Conditions

- Workspace: `/mnt/mydisk/lhy/testPvcnnWithIsaacsim`
- Source plugin checkout: `plugins/superpowers`
- Target skill root: `.agents/skills`
- Current work ref: `cf7e9cf`

## Key Metrics

- Installed superpowers skills: `14`
- Example installed skill:
  - `.agents/skills/brainstorming/SKILL.md`
- Existing project skills preserved: yes
- Skill name conflicts detected: no

## Result

- Pass

## Conclusion

- The project now contains the `superpowers` skills in the same `.agents/skills/` location that VS Code Codex project workflows commonly scan for local skills.

## Follow-up

- If the current VS Code Codex chat session still does not list the new skills, start a fresh Codex chat in the same workspace so it reloads the project skill inventory.

## Git Refs

- Baseline Ref: `cf7e9cf`
- Candidate Ref: `working tree on top of cf7e9cf`
- Key Files:
  - [.agents/skills/brainstorming/SKILL.md](../../.agents/skills/brainstorming/SKILL.md)
  - [.agents/skills/using-superpowers/SKILL.md](../../.agents/skills/using-superpowers/SKILL.md)
  - [notes/log/2026-05-06-2026-superpowers-skills-project-install.md](2026-05-06-2026-superpowers-skills-project-install.md)
