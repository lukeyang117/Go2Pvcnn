from __future__ import annotations

import json

from evaluation.result_writer import ResultWriter


def test_result_writer_resume_skips_completed_conditions(tmp_path) -> None:
    writer = ResultWriter(tmp_path, {"manifest_hash": "x"})
    writer.write_episode({"condition_id": "c0", "value": 1})
    writer.write_progress(["c0"])
    assert writer.load_completed_condition_ids() == {"c0"}
    assert json.loads(next((tmp_path / "episodes").glob("*.jsonl")).read_text())["condition_id"] == "c0"


def test_result_writer_rejects_hash_mismatch(tmp_path) -> None:
    ResultWriter(tmp_path, {"manifest_hash": "x"})
    try:
        ResultWriter(tmp_path, {"manifest_hash": "y"})
    except ValueError as exc:
        assert "hash" in str(exc)
    else:
        raise AssertionError("expected hash mismatch")


def test_result_writer_preserves_multiple_episodes_for_same_condition(tmp_path) -> None:
    writer = ResultWriter(tmp_path, {"manifest_hash": "x"})
    writer.write_episode({"condition_id": "same", "episode": 1})
    writer.write_episode({"condition_id": "same", "episode": 2})

    records = []
    for path in sorted((tmp_path / "episodes").glob("*.jsonl")):
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    assert [row["episode"] for row in records] == [1, 2]
