#!/usr/bin/env python3
"""Play a SemLoco checkpoint with the existing Go2 play pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from play import main  # noqa: E402


if __name__ == "__main__":
    sys.argv[1:1] = ["--experiment", "cross_large_complex_semloco"]
    raise SystemExit(main())
