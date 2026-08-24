"""Bounded M1 Isaac viewer smoke entry point.

Run this file through the Isaac Lab launcher, not the system Python:

    ./isaaclab.sh -p Go2Pvcnn/extension/viz/go2_foostep_planner.py \
      --robot m1 --planner-backend parallelism --terrain-col 1 \
      --scripted-command "0.1 0 0" --scripted-command-cycles 1 \
      --max-plan-cycles 1 --headless
"""

from __future__ import annotations

from pathlib import Path


def test_m1_smoke_entrypoint_and_local_usd_are_present():
    root = Path(__file__).resolve().parents[1]
    viewer = root / "extension/viz/go2_foostep_planner.py"
    usd = root.parents[1] / "m1/ZJ_V3_URDF_V1_0/ZJ_V3_URDF_V1_0.usd"
    assert viewer.exists()
    assert usd.exists()
    assert "--max-plan-cycles" in viewer.read_text()
