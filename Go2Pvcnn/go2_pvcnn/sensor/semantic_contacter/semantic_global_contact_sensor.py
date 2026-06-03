from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.sensors import ContactSensor


DEFAULT_SEMANTIC_CONTACT_BODY_NAMES: tuple[str, ...] = (
    "FL_foot",
    "FR_foot",
    "RL_foot",
    "RR_foot",
    "FL_calf",
    "FR_calf",
    "RL_calf",
    "RR_calf",
    "FL_thigh",
    "FR_thigh",
    "RL_thigh",
    "RR_thigh",
    "base",
)


def filter_semantic_leaf_obstacle_paths(paths: list[str], semantic_root: str) -> list[str]:
    prefix = semantic_root.rstrip("/") + "/"
    out: list[str] = []
    for path in sorted(str(path) for path in paths):
        if not path.startswith(prefix):
            continue
        rel = path[len(prefix) :]
        parts = rel.split("/")
        if (
            len(parts) == 3
            and parts[0].startswith("row_")
            and parts[1].startswith("col_")
            and parts[2].startswith("slot_")
        ):
            out.append(path)
    return out


def _semantic_root_from_filter_expr(expr: str) -> str:
    suffix = "/.*"
    if expr.endswith(suffix):
        return expr[: -len(suffix)]
    return expr.rstrip("/")


def resolve_contact_body_paths(
    *,
    parent_paths: Sequence[str],
    body_names: Sequence[str],
    has_contact_report,
) -> list[str]:
    sensor_paths: list[str] = []
    for parent_path in parent_paths:
        robot_path = str(parent_path).rstrip("/")
        for body_name in body_names:
            body_path = f"{robot_path}/{body_name}"
            if not has_contact_report(body_path):
                raise RuntimeError(
                    f"SemanticGlobalContactSensor could not find contact-reporting body '{body_path}'."
                )
            sensor_paths.append(body_path)
    return sensor_paths


class SemanticGlobalContactSensor(ContactSensor):
    """ContactSensor variant for global static semantic-course objects."""

    @property
    def body_names(self) -> list[str]:
        return list(getattr(self, "_body_names", DEFAULT_SEMANTIC_CONTACT_BODY_NAMES))

    @property
    def semantic_filter_paths(self) -> list[str]:
        return list(getattr(self, "_semantic_filter_paths", ()))

    @property
    def has_semantic_filters(self) -> bool:
        return bool(getattr(self, "_has_semantic_filters", True))

    def _initialize_impl(self):
        if self.cfg.track_pose:
            raise RuntimeError("SemanticGlobalContactSensor does not support track_pose=True.")
        if self.cfg.track_air_time:
            raise RuntimeError("SemanticGlobalContactSensor does not support track_air_time=True.")
        if self.cfg.history_length > 0:
            raise RuntimeError("SemanticGlobalContactSensor does not support history_length > 0.")
        if len(self.cfg.filter_prim_paths_expr) != 1:
            raise RuntimeError("SemanticGlobalContactSensor expects exactly one semantic filter root expression.")

        # Call SensorBase._initialize_impl(), bypassing ContactSensor._initialize_impl() because
        # the default filtered view cannot represent many global semantic objects per env body.
        super(ContactSensor, self)._initialize_impl()

        from isaaclab.sim import utils as sim_utils
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import PhysxSchema

        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        self._body_names = list(DEFAULT_SEMANTIC_CONTACT_BODY_NAMES)

        stage = sim_utils.stage_utils.get_current_stage()

        def _has_contact_report(body_path: str) -> bool:
            prim = stage.GetPrimAtPath(body_path)
            return prim is not None and prim.IsValid() and prim.HasAPI(PhysxSchema.PhysxContactReportAPI)

        sensor_paths = resolve_contact_body_paths(
            parent_paths=[parent_prim.GetPath().pathString for parent_prim in self._parent_prims],
            body_names=self._body_names,
            has_contact_report=_has_contact_report,
        )
        if not sensor_paths:
            raise RuntimeError(f"Sensor at path '{self.cfg.prim_path}' could not resolve robot body paths.")

        semantic_root = _semantic_root_from_filter_expr(str(self.cfg.filter_prim_paths_expr[0]))
        candidate_paths = sim_utils.find_matching_prim_paths(f"{semantic_root}/.*/.*/.*")
        filter_paths = filter_semantic_leaf_obstacle_paths(candidate_paths, semantic_root)
        self._semantic_filter_paths = filter_paths
        self._has_semantic_filters = len(filter_paths) > 0

        self._sensor_paths = list(sensor_paths)
        self._body_physx_view = self._physics_sim_view.create_rigid_body_view(sensor_paths)
        self._num_bodies = len(self._body_names)
        self._rebuild_contact_view(filter_paths)

        self._data.net_forces_w = torch.zeros(self._num_envs, self._num_bodies, 3, device=self._device)
        self._data.net_forces_w_history = self._data.net_forces_w.unsqueeze(1)
        self._data.force_matrix_w = torch.zeros(
            self._num_envs,
            self._num_bodies,
            len(filter_paths),
            3,
            device=self._device,
        )

    def _rebuild_contact_view(self, filter_paths: list[str]) -> None:
        self._semantic_filter_paths = list(filter_paths)
        self._has_semantic_filters = len(filter_paths) > 0
        self._contact_physx_view = None
        if not self._has_semantic_filters:
            return
        sensor_paths = list(getattr(self, "_sensor_paths", ()))
        if not sensor_paths:
            raise RuntimeError("SemanticGlobalContactSensor cannot rebuild contact view before sensor paths exist.")
        self._contact_physx_view = self._physics_sim_view.create_rigid_contact_view(
            sensor_paths,
            filter_patterns=[filter_paths] * len(sensor_paths),
        )
        expected_sensor_count = self._num_envs * self._num_bodies
        if int(self.contact_physx_view.sensor_count) != expected_sensor_count:
            raise RuntimeError(
                "SemanticGlobalContactSensor contact view sensor count mismatch."
                f"\n\tExpected: {expected_sensor_count}"
                f"\n\tActual: {self.contact_physx_view.sensor_count}"
            )
        if int(self.contact_physx_view.filter_count) != len(filter_paths):
            raise RuntimeError(
                "SemanticGlobalContactSensor contact view filter count mismatch."
                f"\n\tExpected: {len(filter_paths)}"
                f"\n\tActual: {self.contact_physx_view.filter_count}"
            )

    def refresh_semantic_filters(self) -> None:
        from isaaclab.sim import utils as sim_utils

        semantic_root = _semantic_root_from_filter_expr(str(self.cfg.filter_prim_paths_expr[0]))
        candidate_paths = sim_utils.find_matching_prim_paths(f"{semantic_root}/.*/.*/.*")
        filter_paths = filter_semantic_leaf_obstacle_paths(candidate_paths, semantic_root)
        self._rebuild_contact_view(filter_paths)
        self._data.net_forces_w = torch.zeros(self._num_envs, self._num_bodies, 3, device=self._device)
        self._data.net_forces_w_history = self._data.net_forces_w.unsqueeze(1)
        self._data.force_matrix_w = torch.zeros(
            self._num_envs,
            self._num_bodies,
            len(filter_paths),
            3,
            device=self._device,
        )

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        if len(env_ids) == self._num_envs:
            env_ids = slice(None)

        if not self.has_semantic_filters:
            self._data.net_forces_w[env_ids, :, :] = 0.0
            self._data.force_matrix_w[env_ids] = 0.0
            return

        net_forces_w = self.contact_physx_view.get_net_contact_forces(dt=self._sim_physics_dt)
        self._data.net_forces_w[env_ids, :, :] = net_forces_w.view(self._num_envs, self._num_bodies, 3)[env_ids]
        force_matrix_w = self.contact_physx_view.get_contact_force_matrix(dt=self._sim_physics_dt)
        force_matrix_w = force_matrix_w.view(
            self._num_envs,
            self._num_bodies,
            self.contact_physx_view.filter_count,
            3,
        )
        self._data.force_matrix_w[env_ids] = force_matrix_w[env_ids]
