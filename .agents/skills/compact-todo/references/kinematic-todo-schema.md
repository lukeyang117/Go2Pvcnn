# kinematic_footsteps Todo Compaction Schema

Use this reference when compacting `/home/lhy/kinematic_footsteps/notes/`.

## Required Inputs

Read:

- `.codex/RULES.md`
- `notes/todo.md`
- `notes/log/index.md`
- `notes/todo/README.md`
- active branch pages, currently often `notes/todo/T001-speed-command-alignment.md`
- latest logs for active leaves

## Target Root Dashboard

`notes/todo.md` should be a dashboard, not a full annotated tree.

Target sections:

```md
# Investigation Dashboard

## Start Here
## Status Legend
## Active Fronts
## Root Map
## Open Leaves
## Branch Pages
## Recent Logs
## Maintenance
```

Expected size: about 60-120 lines.

## What To Keep In Root

- current focus, next reads, avoid-redo notes
- active fronts
- root-level map: T000, T001, T200, T300, T400
- open leaves only: e.g. T104a when active
- branch page list
- recent 3-8 logs
- compact refs for active/root nodes

## What To Move Out Of Root

- all closed child details
- long `why-created`
- long `current conclusion`
- full `Key Files`
- old templates
- detailed metrics

Move these into branch pages or logs.

## Branch Page Shape

For active branches, use:

```md
## Current State
## Open Children
## Closed Children Archive
## Related Logs
## Git Refs
## Next Step
## Node Details
```

For this repository:

- T104/T104a/T104b details belong in `notes/todo/T001-speed-command-alignment.md`.
- T401/T402/T403/T407/T408 closed reachability/terrain/frame details belong in `notes/todo/T400-batch-planner-command-terrain-response.md`.
- notes workflow structure changes belong in `notes/todo/T000-notes-workflow.md`.

## Log Index Shape

When `notes/log/index.md` grows:

- keep recent 10-15 rows under `Recent Logs`
- create `Topic Log Index` grouped by T104, T401, T300, T000, etc.
- move older row blocks into `notes/log/archive/YYYY-MM.md`
- preserve per-test files under `notes/log/`

## Repository Constraints

- Preserve `.codex/RULES.md` semantics: agents must read root todo, relevant branches, and relevant logs before work.
- Preserve module-scoped git refs.
- Preserve relative links.
- If compaction changes the notes workflow, update T000 and create a notes workflow log.

## Suggested Validation

```bash
wc -l notes/todo.md notes/log/index.md notes/todo/*.md
rg -n "## Start Here|## Root Map|## Open Leaves|## Current State|## Closed Children Archive|## Topic Log Index" notes
rg -n "T104a|T104b|T401|T408|latest log|Refs|Git Refs" notes/todo.md notes/todo notes/log/index.md
```
