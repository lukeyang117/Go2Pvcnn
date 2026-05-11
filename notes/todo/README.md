# Todo Branch Directory

This directory stores detailed branch memory split out from [../todo.md](../todo.md).

## Layout Contract

- `todo.md` is the dashboard.
- `notes/todo/` stores branch memory.
- `notes/todo/archive/` stores cold closed memory.
- `notes/log/` stores evidence.

## Branch Page Shape

Branch pages should include:

- `Current State`
- `Open Children`
- `Closed Children Archive`
- `Related Logs`
- `Git Refs`
- `Next Step`
- `Node Details`

## Navigation Contract

- [../todo.md](../todo.md) is the root Obsidian index for active roots.
- Each branch page is a subtree index, not just a prose summary.
- When compacting children, keep child ids, statuses, and destination links visible in `Open Children` or `Closed Children Archive`.
- If a child gets its own page, keep a link back to the parent page anchor or owning branch page.

## Relationship Types

- `child-of`
- `caused-by`
- `blocks`
- `related-to`

## Root Map Row Template

```text
| T100 | doing | module or flow | [T100](T100-topic.md) | active child T101 | feature `abc1234`; verified `pending` |
```

## Open Leaf Row Template

```text
| T101 | T100 | doing | P0 | why active | [branch](T100-topic.md#t101-title) + [log](../log/YYYY-MM-DD-HHMM-topic.md) |
```

## Branch Page Template

```text
# T100 Problem Theme

## Current State

- Short current summary.

## Open Children

- [T101](../todo.md#open-leaves): next check.

## Closed Children Archive

- T102 done: one-line outcome.

## Related Logs

- [YYYY-MM-DD-HHMM-topic.md](../log/YYYY-MM-DD-HHMM-topic.md)

## Git Refs

- Last Feature Commit: `abc1234`
- Last Verified Commit: `abc1234`
- Current Work Ref: `working tree on top of abc1234 (YYYY-MM-DD HH:MM)`
- Key Files:
  - [path/to/file.py](../../path/to/file.py)

## Next Step

- Next action.

## Node Details

### T101 title

- why-created:
- hypothesis:
- evidence:
```
