# Todo Write Contract

Use this only after a reviewed design exists and the user explicitly chooses `写 todo`.

## Hard Rules

- Do not create a standalone implementation plan document.
- Update both `notes/todo.md` and the relevant branch page under `notes/todo/`.
- Write an official todo-stage log under `notes/log/`.
- Keep `notes/todo.md` as the dashboard and the branch page as durable memory.

## Todo Tree Requirements

Express the work as root, child, and leaf nodes where appropriate. Each important leaf should record:

- what the leaf is solving
- which design section it maps to
- which acceptance or test indicators apply
- whether it depends on another leaf
- current status

## Primary Agent Ownership

The primary agent owns:

- mapping approved design content into todo structure
- state updates in the dashboard and branch page
- official todo-stage logging
- deciding which leaf is ready for execution approval

The primary agent does not directly edit code during the later execution phase of this workflow.

## Completion Standard

Todo writing is not complete unless all three are updated:

- `notes/todo.md`
- the relevant branch page
- the official todo-stage log
