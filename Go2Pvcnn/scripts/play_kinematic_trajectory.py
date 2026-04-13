#!/usr/bin/env python3
"""Scaffold entrypoint for future pure-kinematic Isaac Lab playback."""

from __future__ import annotations

import argparse

from extension.planner.viz.kinematic_player import KinematicPlayerConfig, KinematicTrajectoryPlayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play planner trajectories in Isaac Lab (scaffold).")
    parser.add_argument("--terrain-name", type=str, default="default")
    parser.add_argument("--n-frames", type=int, default=50)
    parser.add_argument("--dt", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    player = KinematicTrajectoryPlayer(
        KinematicPlayerConfig(
            terrain_name=args.terrain_name,
            n_frames=args.n_frames,
            dt=args.dt,
        )
    )
    print(player.describe())
    print("Scaffold only: Isaac Lab kinematic playback logic has not been wired yet.")


if __name__ == "__main__":
    main()
