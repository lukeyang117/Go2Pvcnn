"""Rolling-viewer helpers for the joint MPC RTI backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class JointMpcRtiPlaybackFrame:
    frame_index: int
    trajectory: object


class JointMpcRtiViewerAdapter:
    """Expose the first future node while retaining the full horizon for drawing."""

    def __init__(self, trajectory) -> None:
        self._trajectory = trajectory

    @classmethod
    def for_test(cls, trajectory) -> "JointMpcRtiViewerAdapter":
        return cls(trajectory)

    def update_trajectory(self, trajectory) -> None:
        self._trajectory = trajectory

    def next_playback_frame(self) -> JointMpcRtiPlaybackFrame:
        return JointMpcRtiPlaybackFrame(frame_index=1, trajectory=self._trajectory)


__all__ = ["JointMpcRtiPlaybackFrame", "JointMpcRtiViewerAdapter"]
