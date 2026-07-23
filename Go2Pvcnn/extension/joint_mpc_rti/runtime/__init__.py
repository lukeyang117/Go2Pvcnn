"""Runtime lifecycle for rolling joint MPC RTI."""

from __future__ import annotations


def __getattr__(name: str):
    if name == "JointMpcRtiManager":
        from .manager import JointMpcRtiManager

        return JointMpcRtiManager
    raise AttributeError(name)


__all__ = ["JointMpcRtiManager"]
