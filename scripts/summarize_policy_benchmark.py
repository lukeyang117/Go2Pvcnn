"""Aggregate formal policy benchmark runs into paired JSON/CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


def load_formal_run(root: str | Path) -> dict[str, object]:
    path = Path(root)
    manifest_path = path / "run_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"missing run_manifest.json: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("is_smoke_test", False)):
        raise ValueError("smoke results cannot be summarized as formal results")
    episodes = []
    for shard in sorted((path / "episodes").glob("*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                episodes.append(json.loads(line))
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return {"root": str(path), "manifest": manifest, "episodes": episodes, "summary": summary}


def _bootstrap(values: list[float], seed: int, samples: int = 2000) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    rng = random.Random(seed)
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, mean, mean
    means = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples)]
    means.sort()
    return mean, means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]


def _paired_table(runs: dict[str, dict[str, object]], metric: str) -> list[dict[str, object]]:
    by_model = {
        name: {str(row.get("condition_id")): float(row[metric]) for row in data["episodes"] if metric in row}
        for name, data in runs.items()
    }
    names = list(runs)
    rows = []
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            common = sorted(set(by_model[left]) & set(by_model[right]))
            diffs = [by_model[left][key] - by_model[right][key] for key in common]
            mean, low, high = _bootstrap(diffs, seed=20260903 + left_index)
            positives = sum(value > 0 for value in diffs)
            rows.append({"model_a": left, "model_b": right, "metric": metric, "paired_n": len(diffs), "mean_difference": mean, "bootstrap95_low": low, "bootstrap95_high": high, "positive_difference_count": positives, "negative_difference_count": sum(value < 0 for value in diffs)})
    return rows


def summarize(runs: dict[str, dict[str, object]], output: Path) -> dict[str, object]:
    metrics = ("large_collision_episode", "small_collision_episode", "fall", "traversed_path_m", "command_progress_m")
    summary: dict[str, object] = {"models": {}, "paired": []}
    for name, data in runs.items():
        rows = data["episodes"]
        model_stats = {"episodes": len(rows)}
        for metric in metrics:
            values = [float(bool(row[metric])) if isinstance(row.get(metric), bool) else float(row[metric]) for row in rows if metric in row]
            mean, low, high = _bootstrap(values, seed=20260903)
            model_stats[metric] = {"mean": mean, "bootstrap95_low": low, "bootstrap95_high": high}
        run_summary = data.get("summary", {})
        valid_mse_rows = run_summary.get("valid_mse", []) if isinstance(run_summary, dict) else []
        valid_windows = [row for row in rows if int(row.get("valid_window_count", 0)) > 0]
        model_stats["valid_window_episode_count"] = max(len(valid_windows), len(valid_mse_rows))
        summary["models"][name] = model_stats
        for metric in ("joint_position_mse", "joint_velocity_mse", "root_position_mse", "root_rotation_mse"):
            values = [float(row[metric]) for row in rows if metric in row]
            if not values:
                values = [float(row[metric]) for row in valid_mse_rows if metric in row]
            if values:
                mean, low, high = _bootstrap(values, seed=20260903)
                model_stats[metric] = {"mean": mean, "bootstrap95_low": low, "bootstrap95_high": high}
    for metric in metrics:
        summary["paired"].extend(_paired_table(runs, metric))
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "paired.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = summary["paired"]
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]) if rows else ["metric"])
        writer.writeheader()
        writer.writerows(rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize formal policy benchmark runs")
    parser.add_argument("--run", action="append", nargs=2, metavar=("MODEL", "ROOT"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = {name: load_formal_run(root) for name, root in args.run}
    summarize(runs, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
