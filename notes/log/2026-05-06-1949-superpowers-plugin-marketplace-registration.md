# superpowers plugin marketplace registration

## Purpose

- Register the local `superpowers` plugin in the repository marketplace so it can be discovered and used directly by Codex from this project.

## Stage

- repository tooling / local plugin marketplace registration

## Related Todo

- [T000](../todo/T000-notes-workflow.md)

## Command / Procedure

- Used the local plugin path `plugins/superpowers/`.
- Created `.agents/plugins/marketplace.json` with a repo-local marketplace entry for `superpowers`.
- Detected that the scaffold command had overwritten `plugins/superpowers/.codex-plugin/plugin.json` with placeholders.
- Restored the real upstream `superpowers` Codex manifest and finalized the marketplace metadata with non-placeholder root values.

## Input Conditions

- Workspace: `/mnt/mydisk/lhy/testPvcnnWithIsaacsim`
- Current work ref: `cf7e9cf`
- Marketplace target: `.agents/plugins/marketplace.json`
- Plugin target: `plugins/superpowers`

## Key Metrics

- Marketplace file created: yes
- Registered plugins: `1`
- Registered plugin path: `./plugins/superpowers`
- Marketplace policy:
  - installation: `AVAILABLE`
  - authentication: `ON_INSTALL`
- Plugin manifest restored after scaffold overwrite: yes

## Result

- Pass

## Conclusion

- The repository now has a local plugin marketplace entry for `superpowers`, and the plugin manifest has been restored to the upstream working version.

## Follow-up

- Restart or refresh the Codex session if the plugin list is cached and `superpowers` does not appear immediately.

## Git Refs

- Baseline Ref: `cf7e9cf`
- Candidate Ref: `working tree on top of cf7e9cf`
- Key Files:
  - [.agents/plugins/marketplace.json](../../.agents/plugins/marketplace.json)
  - [plugins/superpowers/.codex-plugin/plugin.json](../../plugins/superpowers/.codex-plugin/plugin.json)
  - [notes/log/2026-05-06-1949-superpowers-plugin-marketplace-registration.md](2026-05-06-1949-superpowers-plugin-marketplace-registration.md)
