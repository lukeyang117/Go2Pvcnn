from __future__ import annotations

import pytest

from evaluation.manifest import build_conditions, read_manifest, write_manifest


def test_manifest_round_trip_and_duplicate_guard(tmp_path) -> None:
    conditions = build_conditions("large_runway", layout_count=2, seed=7)
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, conditions)
    assert read_manifest(path) == conditions
    path.write_text(path.read_text(encoding="utf-8") + path.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate condition_id"):
        read_manifest(path)


def test_manifest_suite_cardinality_and_counts() -> None:
    mixed = build_conditions("complex_mixed", layout_count=2, seed=1)
    large = build_conditions("large_runway", layout_count=2, seed=1)
    small = build_conditions("small_runway", layout_count=2, seed=1)
    assert len(mixed) == 240
    assert len(large) == 2 * 4 * 3
    assert len(small) == 2 * 4 * 3
    assert {row.large_obstacle_count for row in large} == {4, 6, 8, 10}
    assert {row.small_obstacle_count for row in small} == {40, 80, 120, 160}
