# 2026-05-06 22:48 Human-16 IsaacLab AppLauncher WebRTC Migration Guide

## Scope

- Todo: [T100/T111 viewer livestream black-screen / GLFW disconnect triage](../todo/T100-batched-together-planner-gpu-migration.md#t111-viewer-livestream-black-screen--glfw-disconnect-triage)
- Stage: human documentation for IsaacLab WebRTC migration
- Environment: repository notes update only

## Purpose

Record the IsaacLab `AppLauncher` WebRTC fix as a reusable migration guide under `notes/human/`, so future server moves do not require rediscovering the `livestream=2` duplicate-extension issue from raw logs.

## Change

- Added [../human/human-16-isaaclab-applauncher-webrtc-migration.md](../human/human-16-isaaclab-applauncher-webrtc-migration.md).
- The guide documents:
  - the black-screen / `nvstPushStreamData` symptom pattern
  - the exact IsaacLab file to patch
  - why `livestream=2` must not enable both `omni.kit.livestream.webrtc` and `omni.services.livestream.nvcf`
  - the target post-patch behavior
  - the migration checklist and recommended remote launch command
- Updated [../index.md](../index.md) to include `human-16`.
- Updated [../todo.md](../todo.md) `Start Here` to include the migration guide.

## Verification

- Readback confirmed `human-16` keeps repository-relative links.
- `python -m py_compile` is not applicable because this change is Markdown-only.
- `git diff --check` should stay clean for the touched notes files.

## Follow-up

- If a later migration needs browser-side or firewall-specific guidance, add a separate human note rather than overloading `human-16`.
