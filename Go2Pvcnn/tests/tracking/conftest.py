from __future__ import annotations

import sys
from pathlib import Path

GO2PVCNN_ROOT = Path(__file__).resolve().parents[2]
if str(GO2PVCNN_ROOT) not in sys.path:
    sys.path.insert(0, str(GO2PVCNN_ROOT))
RSL_RL_ROOT = GO2PVCNN_ROOT / "rsl_rl"
if str(RSL_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RSL_RL_ROOT))
