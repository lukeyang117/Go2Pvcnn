"""Local pytest hooks for L3 planner benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_BENCH_DIR = Path(__file__).resolve().parent
_TESTS_DIR = _BENCH_DIR.parent
_GO2_ROOT = _TESTS_DIR.parent
_REPO_ROOT = _GO2_ROOT.parent

for _p in (str(_REPO_ROOT), str(_GO2_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--device",
        action="store",
        default="cpu",
        help="Torch device for planner scaling benchmarks (e.g. cpu, cuda:0).",
    )


@pytest.fixture
def bench_device(request: pytest.FixtureRequest) -> torch.device:
    return torch.device(str(request.config.getoption("--device")))
