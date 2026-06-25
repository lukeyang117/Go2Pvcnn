# Codex logs_2.sqlite TRACE write guard

## Purpose

Investigate whether `/mnt/mydisk/lhy/.codex/logs_2.sqlite` was receiving high-frequency TRACE log writes, then stop the local disk churn without deleting retained rows.

## Stage

Repository tooling / local Codex SQLite log store.

## Related Todo

- [T000 notes workflow](../todo/T000-notes-workflow.md)

## Command / Procedure

- Located `logs_2.sqlite`, `logs_2.sqlite-wal`, and `logs_2.sqlite-shm` under `/mnt/mydisk/lhy/.codex`.
- Read `logs` schema, journal mode, retained row counts, level/target distribution, and filesystem/device metadata.
- Sampled `MAX(id)`, `sqlite_sequence.seq`, row count, WAL size, and WAL mtime before and after mitigation.
- Created online backup and file copies in `/mnt/mydisk/lhy/.codex/backup-logs-2-20260624-1957/`.
- Installed SQLite trigger:

```sql
CREATE TRIGGER codex_block_logs_insert
BEFORE INSERT ON logs
BEGIN
  SELECT RAISE(IGNORE);
END;
```

- Ran `PRAGMA wal_checkpoint(TRUNCATE);`.

## Input Conditions

- Database file: `/mnt/mydisk/lhy/.codex/logs_2.sqlite`
- Main DB size before/after: `208490496` bytes
- WAL before mitigation: about `4.45 MB`
- Journal mode: `wal`
- Device: `/dev/sdc1` mounted at `/mnt/mydisk`, `WD Game Drive`, rotational `1`, ext4 `rw,relatime`

## Key Metrics

- Pre-mitigation 10-second sample: `MAX(id)` grew from `309091253` to `309091800`, while `COUNT(*)` stayed `16114`; WAL mtime refreshed every sample.
- Retained levels before mitigation: `TRACE=13691`, `INFO=1395`, `DEBUG=974`, `WARN=54`.
- Retained min/max log time: `2026-06-15 16:00:26` to `2026-06-24 19:57:30`.
- `sqlite_sequence.logs` after trigger: `309098852`.
- 30-second post-mitigation sample: `seq=309098852`, `MAX(id)=309098852`, `COUNT(*)=16114`, `WAL=0 bytes`, no WAL mtime growth.
- Longer 40-second post-mitigation sample also stayed fixed at `309098852` and `WAL=0 bytes`.
- Kernel per-process I/O observed for active `codex app-server` processes: about `2.25 GB` and `40.98 GB` `write_bytes`, with the `/mnt/mydisk/lhy` app-server the likely local log writer.

## Result

Pass. The high-frequency insert path was confirmed and then blocked by `codex_block_logs_insert`. WAL was checkpointed/truncated to `0 bytes`; subsequent sampling showed both `MAX(id)` and WAL size no longer growing.

## Conclusion

The root cause was Codex app-server TRACE/DEBUG logging into SQLite WAL mode with a retention policy that kept table row count nearly constant while still advancing `AUTOINCREMENT` ids and writing to disk. The trigger is a local guard that drops future `logs` inserts silently.

## Follow-up

- If full local Codex logging is needed again, drop `codex_block_logs_insert`.
- Prefer an upstream config-level log-level fix when available, because the trigger intentionally suppresses all future `logs` table inserts.

## Git Refs

- Baseline Ref: `feea80f`
- Candidate Ref: `working tree`
- Key Files:
  - [notes/log/2026-06-24-codex-logs2-sqlite-trace-write-guard.md](2026-06-24-codex-logs2-sqlite-trace-write-guard.md)
  - [notes/todo/T000-notes-workflow.md](../todo/T000-notes-workflow.md)
  - [notes/log/index.md](index.md)
