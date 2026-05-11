# superpowers plugin download

## Purpose

- Download the `obra/superpowers` Codex plugin repository into the current project as a repo-local plugin checkout.

## Stage

- repository tooling / local plugin availability

## Related Todo

- [T000](../todo/T000-notes-workflow.md)

## Command / Procedure

- Inspected the remote repository layout and confirmed it already ships a native Codex plugin manifest at `.codex-plugin/plugin.json`.
- Downloaded the repository and copied it into `plugins/superpowers/` without the upstream `.git/` directory.
- Confirmed the copied plugin contains the Codex manifest, skills, hooks, assets, and docs.

## Input Conditions

- Workspace: `/mnt/mydisk/lhy/testPvcnnWithIsaacsim`
- Source repo: `https://github.com/obra/superpowers`
- Current work ref: `cf7e9cf`

## Key Metrics

- Install mode: repo-local plugin checkout
- Target path: `plugins/superpowers`
- Manifest present: `plugins/superpowers/.codex-plugin/plugin.json`
- Marketplace entry added: no

## Result

- Pass

## Conclusion

- `superpowers` is now available in the project as a local plugin source tree under `plugins/superpowers/`.

## Follow-up

- If you want this plugin to appear in a local Codex marketplace list, create or update `.agents/plugins/marketplace.json` with a `superpowers` entry.
- If you want a lighter footprint, we can trim this checkout down to only the Codex-facing files instead of keeping the full upstream repository.

## Git Refs

- Baseline Ref: `cf7e9cf`
- Candidate Ref: `working tree on top of cf7e9cf`
- Key Files:
  - [plugins/superpowers/.codex-plugin/plugin.json](../../plugins/superpowers/.codex-plugin/plugin.json)
  - [plugins/superpowers/README.md](../../plugins/superpowers/README.md)
  - [notes/log/2026-05-06-1939-superpowers-plugin-download.md](2026-05-06-1939-superpowers-plugin-download.md)
