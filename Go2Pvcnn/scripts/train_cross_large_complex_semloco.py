#!/usr/bin/env python3
"""Train the isolated SemLoco baseline using the project's PPO runner."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from train import main  # noqa: E402


if __name__ == "__main__":
    sys.argv[1:1] = ["--experiment", "cross_large_complex_semloco"]
    raise SystemExit(main())
