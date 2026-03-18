"""Velocity command with curriculum support (ranges -> limit_ranges)."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    """Uniform velocity command with curriculum: ranges expand toward limit_ranges."""

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
