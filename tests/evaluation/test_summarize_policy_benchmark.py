from __future__ import annotations

import json

import pytest

from scripts.summarize_policy_benchmark import load_formal_run, summarize


def test_summary_rejects_smoke_results(tmp_path):
    root = tmp_path / "amp"
    root.mkdir()
    (root / "run_manifest.json").write_text(json.dumps({"is_smoke_test": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="smoke"):
        load_formal_run(root)


def test_summary_writes_paired_results(tmp_path):
    runs = {}
    for model, offset in (("amp", 0.0), ("ppo", 0.2)):
        root = tmp_path / model
        (root / "episodes").mkdir(parents=True)
        (root / "run_manifest.json").write_text(json.dumps({"is_smoke_test": False}), encoding="utf-8")
        (root / "episodes" / "c0.jsonl").write_text(json.dumps({"condition_id": "c0", "fall": False, "large_collision_episode": False, "small_collision_episode": False, "traversed_path_m": 1.0 + offset, "command_progress_m": 0.5 + offset}) + "\n", encoding="utf-8")
        runs[model] = load_formal_run(root)
    summary = summarize(runs, tmp_path / "out")
    assert (tmp_path / "out" / "summary.json").exists()
    assert summary["paired"][0]["paired_n"] == 1
