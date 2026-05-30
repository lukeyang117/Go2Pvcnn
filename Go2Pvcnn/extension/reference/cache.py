"""Reference trajectory cache scaffolding for planner-guided training.

Canonical cache ABI:
- Layout is batch-first and horizon-second.
- Floating tensors are stored as CPU ``float32`` when produced by planner
  conversion helpers.
- ``contact_state`` / ``valid_mask`` are boolean tensors.
- ``phase_index`` is ``int64``.
- ``ReferenceTrajectoryCache.to(...)`` may move the cache to another device or
  dtype for downstream use, but the canonical in-memory cache produced by the
  planner conversion path follows the rules above.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

CANONICAL_REFERENCE_CACHE_DEVICE = torch.device("cpu")
CANONICAL_REFERENCE_CACHE_FLOAT_DTYPE = torch.float32


def canonical_reference_cache_float_tensor(
    tensor: torch.Tensor | None,
    *,
    device: torch.device = CANONICAL_REFERENCE_CACHE_DEVICE,
    dtype: torch.dtype = CANONICAL_REFERENCE_CACHE_FLOAT_DTYPE,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    out = torch.as_tensor(tensor).to(device=device)
    if out.is_floating_point():
        out = out.to(dtype=dtype)
    return out.contiguous()


def canonical_reference_cache_bool_tensor(
    tensor: torch.Tensor | None,
    *,
    device: torch.device = CANONICAL_REFERENCE_CACHE_DEVICE,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return torch.as_tensor(tensor, dtype=torch.bool, device=device).contiguous()


def canonical_reference_cache_index_tensor(
    tensor: torch.Tensor | None,
    *,
    device: torch.device = CANONICAL_REFERENCE_CACHE_DEVICE,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return torch.as_tensor(tensor, dtype=torch.long, device=device).contiguous()


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
        foot_pos_w=exp2(cache.foot_pos_w),
        foot_pos_root=exp2(cache.foot_pos_root),
        contact_state=exp2(cache.contact_state),
        planned_touchdown_w=exp2(cache.planned_touchdown_w),
        phase_index=exp2(cache.phase_index),
        valid_mask=exp2(cache.valid_mask),
    )


def masked_write_reference_cache_rows(
    dst: "ReferenceTrajectoryCache",
    src: "ReferenceTrajectoryCache",
    env_ids: torch.Tensor,
) -> None:
    """Write ``src`` rows into ``dst`` at the given env indices.

    Both caches must be batched with a leading env dimension and share the same horizon.
    ``env_ids`` is a 1D long tensor selecting destination rows.
    """

    env_ids = torch.as_tensor(env_ids, dtype=torch.long)
    if env_ids.ndim != 1:
        raise ValueError(f"env_ids must be 1D, got {tuple(env_ids.shape)}")

    def _tensor(name: str) -> torch.Tensor:
        t = getattr(dst, name)
        if t is None:
            raise ValueError(f"dst missing {name}")
        return t

    def _src(name: str) -> torch.Tensor:
        t = getattr(src, name)
        if t is None:
            raise ValueError(f"src missing {name}")
        return t

    # Use the destination tensor devices for the index to satisfy index_copy_ requirements.
    for name in (
        "root_pos_w",
        "root_quat_w",
        "joint_angles",
        "foot_pos_w",
        "foot_pos_root",
        "contact_state",
        "planned_touchdown_w",
        "phase_index",
        "valid_mask",
    ):
        dst_t = _tensor(name)
        src_t = _src(name)
        if dst_t.ndim < 2 or src_t.ndim < 2:
            raise ValueError(f"{name} must be batched, got dst {tuple(dst_t.shape)} src {tuple(src_t.shape)}")
        if int(dst_t.shape[1]) != int(src_t.shape[1]):
            raise ValueError(f"{name} horizon mismatch: dst {int(dst_t.shape[1])} src {int(src_t.shape[1])}")
        if int(src_t.shape[0]) != int(env_ids.shape[0]):
            raise ValueError(f"{name} batch mismatch: src {int(src_t.shape[0])} ids {int(env_ids.shape[0])}")
        idx = env_ids.to(device=dst_t.device)
        dst_t.index_copy_(0, idx, src_t.to(device=dst_t.device))


def fill_reference_cache_standstill_rows(cache: "ReferenceTrajectoryCache", env_ids: torch.Tensor) -> None:
    """Overwrite selected env rows with a standstill (time-constant) trajectory.

    The standstill trajectory repeats the first cached frame across the horizon.
    This keeps reward-facing cache contracts intact even if replanning fails.
    """

    env_ids = torch.as_tensor(env_ids, dtype=torch.long)
    if env_ids.numel() == 0:
        return
    if env_ids.ndim != 1:
        raise ValueError(f"env_ids must be 1D, got {tuple(env_ids.shape)}")

    def _standstill(name: str) -> torch.Tensor:
        t = getattr(cache, name)
        if t is None:
            raise ValueError(f"cache missing {name}")
        if t.ndim < 2:
            raise ValueError(f"{name} must be batched, got {tuple(t.shape)}")
        h = int(t.shape[1])
        idx = env_ids.to(device=t.device)
        first = t.index_select(0, idx)[:, :1]
        return first.expand(first.shape[0], h, *first.shape[2:]).clone()

    for name in (
        "root_pos_w",
        "root_quat_w",
        "joint_angles",
        "foot_pos_w",
        "foot_pos_root",
        "contact_state",
        "planned_touchdown_w",
    ):
        t = getattr(cache, name)
        if t is None:
            continue
        idx = env_ids.to(device=t.device)
        t.index_copy_(0, idx, _standstill(name))

    # Preserve phase_index and valid_mask as-is; they are not interpreted as physical state.


@dataclass
class ReferenceTrajectoryCache:
    """Container for cached reference trajectory tensors."""

    root_pos_w: torch.Tensor | None = None
    root_quat_w: torch.Tensor | None = None
    joint_angles: torch.Tensor | None = None
    foot_pos_w: torch.Tensor | None = None
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
            foot_pos_w=_move(self.foot_pos_w),
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
        """Return issues that would make the cache unsafe to consume."""
        return self._issues(canonical=False)

    def canonical_issues(self) -> tuple[str, ...]:
        """Return issues that make the cache diverge from the canonical planner ABI."""
        return self._issues(canonical=True)

    def _issues(self, *, canonical: bool) -> tuple[str, ...]:
        issues: list[str] = []
        required = {
            "root_pos_w": self.root_pos_w,
            "root_quat_w": self.root_quat_w,
            "joint_angles": self.joint_angles,
            "foot_pos_w": self.foot_pos_w,
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
        assert self.foot_pos_w is not None
        assert self.foot_pos_root is not None
        assert self.contact_state is not None
        assert self.planned_touchdown_w is not None
        assert self.phase_index is not None
        assert self.valid_mask is not None

        rp = self.root_pos_w
        canonical_device = CANONICAL_REFERENCE_CACHE_DEVICE if canonical else rp.device
        batched = rp.ndim == 3
        if rp.ndim not in (2, 3):
            issues.append(f"root_pos_w:ndim={rp.ndim}")
            return tuple(issues)
        if not rp.is_floating_point():
            issues.append(f"root_pos_w:dtype={rp.dtype}")
        elif canonical and rp.dtype != CANONICAL_REFERENCE_CACHE_FLOAT_DTYPE:
            issues.append(f"root_pos_w:dtype={rp.dtype}")
        if rp.device != canonical_device:
            issues.append(f"root_pos_w:device={rp.device}")
        if batched:
            n, horizon, last = rp.shape[0], rp.shape[1], rp.shape[2]
            if last != 3:
                issues.append(f"root_pos_w:last_dim={last}")
        else:
            n, horizon, last = None, rp.shape[0], rp.shape[1]
            if last != 3:
                issues.append(f"root_pos_w:last_dim={last}")

        def check_float(name: str, t: torch.Tensor, tail: tuple[int | None, ...]) -> None:
            if not t.is_floating_point():
                issues.append(f"{name}:dtype={t.dtype}")
            elif canonical and t.dtype != CANONICAL_REFERENCE_CACHE_FLOAT_DTYPE:
                issues.append(f"{name}:dtype={t.dtype}")
            if t.device != canonical_device:
                issues.append(f"{name}:device={t.device}")
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

        def check_bool(name: str, t: torch.Tensor, tail: tuple[int | None, ...]) -> None:
            if t.dtype != torch.bool:
                issues.append(f"{name}:dtype={t.dtype}")
            if t.device != canonical_device:
                issues.append(f"{name}:device={t.device}")
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

        def check_long(name: str, t: torch.Tensor) -> None:
            if t.dtype != torch.long:
                issues.append(f"{name}:dtype={t.dtype}")
            if t.device != canonical_device:
                issues.append(f"{name}:device={t.device}")

        check_float("root_quat_w", self.root_quat_w, (4,))
        check_float("joint_angles", self.joint_angles, (12,))
        check_float("foot_pos_w", self.foot_pos_w, (4, 3))
        check_float("foot_pos_root", self.foot_pos_root, (4, 3))
        check_bool("contact_state", self.contact_state, (4,))
        check_float("planned_touchdown_w", self.planned_touchdown_w, (4, 3))

        pi = self.phase_index
        vm = self.valid_mask
        check_long("phase_index", pi)
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
        if vm.dtype != torch.bool:
            issues.append(f"valid_mask:dtype={vm.dtype}")
        if vm.device != canonical_device:
            issues.append(f"valid_mask:device={vm.device}")

        return tuple(issues)

    def is_ready(self) -> bool:
        """Return True when the cache is structurally safe for consumer use."""
        return len(self.shape_issues()) == 0

    def is_canonical(self) -> bool:
        """Return True when the cache still matches the canonical planner ABI."""
        return len(self.canonical_issues()) == 0


__all__ = [
    "CANONICAL_REFERENCE_CACHE_DEVICE",
    "CANONICAL_REFERENCE_CACHE_FLOAT_DTYPE",
    "ReferenceTrajectoryCache",
    "canonical_reference_cache_bool_tensor",
    "canonical_reference_cache_float_tensor",
    "canonical_reference_cache_index_tensor",
    "expand_reference_cache_to_num_envs",
    "fill_reference_cache_standstill_rows",
    "masked_write_reference_cache_rows",
]
