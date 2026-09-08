"""PPO configuration entrypoint for the SemLoco baseline."""

from __future__ import annotations

from agent.train_cfg import get_train_cfg


def get_semloco_train_cfg() -> dict:
    return get_train_cfg("cross_large_complex_semloco")


__all__ = ["get_semloco_train_cfg"]
