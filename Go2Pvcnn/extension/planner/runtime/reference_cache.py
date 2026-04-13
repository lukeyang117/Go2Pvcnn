"""Reference trajectory cache scaffolding for planner-guided training."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _horizon_dim(root_pos_w: torch.Tensor) -> int:
    if root_pos_w.ndim == 2:
        return int(root_pos_w.shape[0])
    if root_pos_w.ndim == 3:
        return int(root_pos_w.shape[1])
    raise ValueError(f"root_pos_w must be (H,3) or (N,H,3), got {tuple(root_pos_w.shape)}")


def _num_envs_dim(root_pos_w: torch.Tensor) -> int | None:
    if root_pos_w.ndim == 2:
        return None
    if root_pos_w.ndim == 3:
        return int(root_pos_w.shape[0])
    return None


def expand_reference_cache_to_num_envs(cache: "ReferenceTrajectoryCache", num_envs: int) -> "ReferenceTrajectoryCache":
    """Broadcast an unbatched cache ``(H, ...)`` to ``(num_envs, H, ...)``."""
    if num_envs < 1:
        raise ValueError("num_envs must be positive")
    n0 = _num_envs_dim(cache.root_pos_w)  # type: ignore[arg-type]
    if n0 is not None:
        if n0 != num_envs:
            raise ValueError(f"cache already batched with N={n0}, cannot expand to {num_envs}")
        return cache

    def exp2(t: torch.Tensor | None) -> torch.Tensor | None:
        if t is None:
            return None
        return t.unsqueeze(0).expand(num_envs, *t.shape).clone()

    return ReferenceTrajectoryCache(
        root_pos_w=exp2(cache.root_pos_w),
        root_quat_w=exp2(cache.root_quat_w),
        joint_angles=exp2(cache.joint_angles),
        foot_pos_root=exp2(cache.foot_pos_root),
        contact_state=exp2(cache.contact_state),
        planned_touchdown_w=exp2(cache.planned_touchdown_w),
        phase_index=exp2(cache.phase_index),
        valid_mask=exp2(cache.valid_mask),
    )


@dataclass
class ReferenceTrajectoryCache:
    """Container for cached reference trajectory tensors.

    Layout:
    - Unbatched: ``root_pos_w`` is ``(horizon, 3)``, one trajectory shared implicitly.
    - Batched: ``root_pos_w`` is ``(num_envs, horizon, 3)`` for per-environment references.
    """

    root_pos_w: torch.Tensor | None = None
    root_quat_w: torch.Tensor | None = None
    joint_angles: torch.Tensor | None = None
    foot_pos_root: torch.Tensor | None = None
    contact_state: torch.Tensor | None = None
    planned_touchdown_w: torch.Tensor | None = None
    phase_index: torch.Tensor | None = None
    valid_mask: torch.Tensor | None = None

    def to(self, *args, **kwargs) -> "ReferenceTrajectoryCache":
        """Return a copy of the cache moved to the requested tensor device/dtype."""
        def _move(tensor: torch.Tensor | None) -> torch.Tensor | None:
            if tensor is None:
                return None
            return tensor.to(*args, **kwargs)

        return ReferenceTrajectoryCache(
            root_pos_w=_move(self.root_pos_w),
            root_quat_w=_move(self.root_quat_w),
            joint_angles=_move(self.joint_angles),
            foot_pos_root=_move(self.foot_pos_root),
            contact_state=_move(self.contact_state),
            planned_touchdown_w=_move(self.planned_touchdown_w),
            phase_index=_move(self.phase_index),
            valid_mask=_move(self.valid_mask),
        )

    def horizon_length(self) -> int | None:
        """Return the cached horizon length, if any tensor has been populated."""
        if self.root_pos_w is None:
            return None
        try:
            return _horizon_dim(self.root_pos_w)
        except ValueError:
            return None

    def shape_issues(self) -> tuple[str, ...]:
        """Return a tuple describing any structural problems in the cache."""
        issues: list[str] = []
        required = {
            "root_pos_w": self.root_pos_w,
            "root_quat_w": self.root_quat_w,
            "joint_angles": self.joint_angles,
            "foot_pos_root": self.foot_pos_root,
            "contact_state": self.contact_state,
            "planned_touchdown_w": self.planned_touchdown_w,
            "phase_index": self.phase_index,
            "valid_mask": self.valid_mask,
        }
        missing = [name for name, tensor in required.items() if tensor is None]
        if missing:
            return tuple(f"missing:{name}" for name in missing)

        assert self.root_pos_w is not None
        assert self.root_quat_w is not None
        assert self.joint_angles is not None
        assert self.foot_pos_root is not None
        assert self.contact_state is not None
        assert self.planned_touchdown_w is not None
        assert self.phase_index is not None
        assert self.valid_mask is not None

        rp = self.root_pos_w
        batched = rp.ndim == 3
        if rp.ndim not in (2, 3):
            issues.append(f"root_pos_w:ndim={rp.ndim}")
            return tuple(issues)
        if batched:
            n, horizon, last = rp.shape[0], rp.shape[1], rp.shape[2]
            if last != 3:
                issues.append(f"root_pos_w:last_dim={last}")
        else:
            n, horizon, last = None, rp.shape[0], rp.shape[1]
            if last != 3:
                issues.append(f"root_pos_w:last_dim={last}")

        def check(name: str, t: torch.Tensor, tail: tuple[int | None, ...]) -> None:
            """tail: expected trailing dims e.g. (4,3) or (3,) or (None,) for inner."""
            if batched:
                if t.ndim != len(tail) + 2:
                    issues.append(f"{name}:ndim={t.ndim}")
                    return
                if t.shape[0] != n or t.shape[1] != horizon:
                    issues.append(f"{name}:batch_horizon={t.shape[:2]}")
            else:
                if t.ndim != len(tail) + 1:
                    issues.append(f"{name}:ndim={t.ndim}")
                    return
                if t.shape[0] != horizon:
                    issues.append(f"{name}:horizon={t.shape[0]}")
            for i, exp_d in enumerate(tail):
                if exp_d is not None:
                    dim_idx = -len(tail) + i
                    if t.shape[dim_idx] != exp_d:
                        issues.append(f"{name}:dim{dim_idx}={t.shape[dim_idx]}")

        check("root_quat_w", self.root_quat_w, (4,))
        check("joint_angles", self.joint_angles, (12,))
        check("foot_pos_root", self.foot_pos_root, (4, 3))
        check("contact_state", self.contact_state, (4,))
        check("planned_touchdown_w", self.planned_touchdown_w, (4, 3))

        # phase / valid: (H,) or (N,H)
        pi = self.phase_index
        vm = self.valid_mask
        if batched:
            if pi.ndim != 2 or pi.shape != (n, horizon):
                issues.append(f"phase_index:shape={tuple(pi.shape)}")
            if vm.ndim != 2 or vm.shape != (n, horizon):
                issues.append(f"valid_mask:shape={tuple(vm.shape)}")
        else:
            if pi.ndim != 1 or pi.shape[0] != horizon:
                issues.append(f"phase_index:shape={tuple(pi.shape)}")
            if vm.ndim != 1 or vm.shape[0] != horizon:
                issues.append(f"valid_mask:shape={tuple(vm.shape)}")
        if self.valid_mask.dtype != torch.bool:
            issues.append(f"valid_mask:dtype={self.valid_mask.dtype}")

        return tuple(issues)

    def is_ready(self) -> bool:
        """Return True when the cache has a complete, shape-consistent trajectory."""
        return len(self.shape_issues()) == 0
