"""Atomic per-condition result shards and resumable progress."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Iterable


class ResultWriter:
    def __init__(self, root: str | Path, run_manifest: dict[str, object]) -> None:
        self.root = Path(root)
        self.episodes = self.root / "episodes"
        self.episodes.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.root / "progress.json"
        self.manifest_path = self.root / "run_manifest.json"
        self._next_episode_index = len(tuple(self.episodes.glob("*.jsonl")))
        candidate = dict(run_manifest)
        candidate.setdefault("schema_version", 1)
        if self.manifest_path.exists():
            current = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            for key, value in candidate.items():
                if key in current and current[key] != value:
                    raise ValueError(f"run manifest hash mismatch for {key}")
        else:
            self._atomic_json(self.manifest_path, candidate)

    def write_episode(self, record: dict[str, object]) -> Path:
        condition_id = str(record.get("condition_id", "")).strip()
        if not condition_id:
            raise ValueError("episode record requires condition_id")
        safe_id = condition_id.replace("/", "_").replace("\\", "_")
        target = self.episodes / f"{safe_id}__episode_{self._next_episode_index:08d}.jsonl"
        self._next_episode_index += 1
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        self._atomic_text(target, payload)
        return target

    def write_progress(self, completed_condition_ids: Iterable[str]) -> None:
        rows = sorted({str(value) for value in completed_condition_ids})
        self._atomic_json(self.progress_path, {"completed_condition_ids": rows})

    def load_completed_condition_ids(self) -> set[str]:
        completed: set[str] = set()
        if self.progress_path.exists():
            value = json.loads(self.progress_path.read_text(encoding="utf-8"))
            completed.update(str(item) for item in value.get("completed_condition_ids", []))
        for path in self.episodes.glob("*.jsonl"):
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        item = json.loads(line)
                        if "condition_id" in item:
                            completed.add(str(item["condition_id"]))
            except (OSError, json.JSONDecodeError):
                continue
        return completed

    def write_summary(self, summary: dict[str, object]) -> None:
        self._atomic_json(self.root / "summary.json", summary)

    @staticmethod
    def _atomic_text(target: Path, content: str) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)

    @classmethod
    def _atomic_json(cls, target: Path, value: dict[str, object]) -> None:
        cls._atomic_text(target, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
