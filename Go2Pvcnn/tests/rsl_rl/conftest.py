from __future__ import annotations

import sys
from pathlib import Path


GO2PVCNN_ROOT = Path(__file__).resolve().parents[2]
RSL_RL_ROOT = GO2PVCNN_ROOT / "rsl_rl"
for path in (GO2PVCNN_ROOT, RSL_RL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
