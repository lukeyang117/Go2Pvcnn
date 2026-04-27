"""Backend factory and attach helpers for trajectory managers."""

from __future__ import annotations

import functools

import torch


VALID_PLANNER_BACKENDS = ("together", "legacy")


def planner_backend_from_cfg(cfg) -> str:
    backend = str(getattr(cfg, "planner_backend", "together")).lower()
    if backend not in VALID_PLANNER_BACKENDS:
        valid = ", ".join(VALID_PLANNER_BACKENDS)
        raise ValueError(f"Invalid planner_backend={backend!r}; expected one of: {valid}")
    return backend


def create_trajectory_manager(cfg, *, device):
    backend = planner_backend_from_cfg(cfg)
    if backend == "legacy":
        from extension.batched_planner.manager import BatchedTrajectoryManager

        return BatchedTrajectoryManager(cfg, device=device)
    from extension.batched_together_planner.manager import TogetherTrajectoryManager

    return TogetherTrajectoryManager(cfg, device=device)


def _env_root(env):
    return getattr(env, "unwrapped", env)


def _command_name(cfg) -> str:
    return str(getattr(cfg, "reference_command_name", "base_velocity"))


def _command_term(command_manager, name: str):
    if hasattr(command_manager, "get_term"):
        try:
            return command_manager.get_term(name)
        except Exception:  # noqa: BLE001 - Isaac versions differ
            pass
    terms = getattr(command_manager, "_terms", None)
    if isinstance(terms, dict):
        return terms.get(name)
    return getattr(command_manager, name, None)


def _extract_env_mask(args, kwargs):
    if "env_ids" in kwargs:
        return kwargs["env_ids"]
    if "env_mask" in kwargs:
        return kwargs["env_mask"]
    if args:
        return args[0]
    return None


def _env_ids_to_mask(env_ids, *, num_envs: int, device):
    if env_ids is None:
        return None
    tensor = torch.as_tensor(env_ids, device=device)
    if tensor.dtype == torch.bool:
        return tensor
    mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
    mask[tensor.to(dtype=torch.long, device=device)] = True
    return mask


def _wrap_command_hook(term, attr_name: str, manager, root) -> bool:
    mark_command_changed = getattr(manager, "mark_command_changed", None)
    if mark_command_changed is None or not callable(mark_command_changed):
        return False
    original = getattr(term, attr_name, None)
    if original is None or not callable(original) or getattr(original, "_trajectory_hook_wrapped", False):
        return False

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        env_mask = _env_ids_to_mask(_extract_env_mask(args, kwargs), num_envs=root.num_envs, device=manager._device)
        mark_command_changed(env_mask)
        return result

    wrapped._trajectory_hook_wrapped = True
    setattr(term, attr_name, wrapped)
    return True


def _wrap_env_reset(root, manager) -> bool:
    original = getattr(root, "reset", None)
    if original is None or not callable(original) or getattr(original, "_trajectory_reset_hook_wrapped", False):
        return False

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        env_mask = _env_ids_to_mask(_extract_env_mask(args, kwargs), num_envs=root.num_envs, device=manager._device)
        if env_mask is None:
            env_mask = torch.ones(root.num_envs, dtype=torch.bool, device=manager._device)
        manager.reset_envs(env_mask)
        return original(*args, **kwargs)

    wrapped._trajectory_reset_hook_wrapped = True
    setattr(root, "reset", wrapped)
    return True


def install_trajectory_manager_hooks(env, cfg, manager) -> None:
    root = _env_root(env)
    command_manager = getattr(root, "command_manager", None)
    if command_manager is not None:
        term = _command_term(command_manager, _command_name(cfg))
        if term is not None:
            _wrap_command_hook(term, "_resample_command", manager, root)
            _wrap_command_hook(term, "reset", manager, root)
    _wrap_env_reset(root, manager)


def attach_trajectory_manager(env, cfg, *, device=None):
    sim = getattr(cfg, "sim", None)
    manager_device = device if device is not None else getattr(env, "device", getattr(sim, "device", "cpu"))
    manager = create_trajectory_manager(cfg, device=manager_device)
    root = _env_root(env)
    root._trajectory_manager = manager
    root._trajectory_reference_cache = None
    install_trajectory_manager_hooks(root, cfg, manager)
    return manager


def attach_trajectory_manager_if_enabled(env, cfg, *, experiment_name: str | None = None, device=None):
    if experiment_name is not None and experiment_name != "teacher_elevation_trajectory":
        return None
    if not getattr(cfg, "planner_owned_reference_cache", False):
        if experiment_name == "teacher_elevation_trajectory":
            raise RuntimeError("teacher_elevation_trajectory requires planner_owned_reference_cache=True")
        return None
    return attach_trajectory_manager(env, cfg, device=device)


__all__ = [
    "VALID_PLANNER_BACKENDS",
    "attach_trajectory_manager",
    "attach_trajectory_manager_if_enabled",
    "create_trajectory_manager",
    "install_trajectory_manager_hooks",
    "planner_backend_from_cfg",
]
