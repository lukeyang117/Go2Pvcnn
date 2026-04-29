from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


GO2_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_RAY_CASTER_PATH = GO2_ROOT / "go2_pvcnn" / "sensor" / "semantic_raycaster" / "semantic_ray_caster.py"


class _FakePrim:
    def __init__(self, path: str, prim_type: str, children: list["_FakePrim"] | None = None):
        self._path = path
        self._prim_type = prim_type
        self._children = children or []

    def GetTypeName(self) -> str:
        return self._prim_type

    def GetChildren(self) -> list["_FakePrim"]:
        return list(self._children)

    def IsValid(self) -> bool:
        return True

    def GetPath(self) -> str:
        return self._path


class _FakeView:
    def __init__(self, pos_w: torch.Tensor, quat_w: torch.Tensor):
        self._pos_w = pos_w
        self._quat_w = quat_w
        self.count = pos_w.shape[0]

    def get_world_poses(self, env_ids):
        return self._pos_w[env_ids], self._quat_w[env_ids]


def _install_semantic_raycaster_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    def _register_module(name: str, module: ModuleType) -> ModuleType:
        monkeypatch.setitem(sys.modules, name, module)
        return module

    for package_name in (
        "go2_pvcnn",
        "go2_pvcnn.sensor",
        "go2_pvcnn.sensor.semantic_raycaster",
        "isaaclab",
        "isaaclab.sensors",
        "isaaclab.sensors.ray_caster",
        "isaaclab.terrains",
        "isaaclab.terrains.trimesh",
        "isaaclab.utils",
        "omni",
        "omni.physics",
        "omni.physics.tensors",
        "omni.physics.tensors.impl",
        "pxr",
    ):
        pkg = ModuleType(package_name)
        pkg.__path__ = []
        _register_module(package_name, pkg)

    data_module = ModuleType("go2_pvcnn.sensor.semantic_raycaster.semantic_ray_caster_data")

    class SemanticGridRayCasterData:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    data_module.SemanticGridRayCasterData = SemanticGridRayCasterData
    _register_module(data_module.__name__, data_module)

    omni_module = sys.modules["omni"]
    omni_module.usd = SimpleNamespace(get_world_transform_matrix=lambda _geom: np.eye(4, dtype=np.float32))

    physx_api = ModuleType("omni.physics.tensors.impl.api")
    physx_api.ArticulationView = type("ArticulationView", (), {})
    physx_api.RigidBodyView = type("RigidBodyView", (), {})
    _register_module(physx_api.__name__, physx_api)

    pxr_module = sys.modules["pxr"]
    pxr_module.Usd = SimpleNamespace(Prim=object)
    pxr_module.UsdGeom = SimpleNamespace(
        Mesh=lambda prim: prim,
        Plane=lambda prim: prim,
        Cube=lambda prim: prim,
        Sphere=lambda prim: prim,
        Cylinder=lambda prim: prim,
    )

    sim_module = ModuleType("isaaclab.sim")
    sim_module.find_first_matching_prim = lambda _path: None
    _register_module(sim_module.__name__, sim_module)

    ray_caster_module = ModuleType("isaaclab.sensors.ray_caster.ray_caster")

    class RayCaster:
        def _initialize_rays_impl(self):
            return None

    ray_caster_module.RayCaster = RayCaster
    _register_module(ray_caster_module.__name__, ray_caster_module)

    trimesh_utils = ModuleType("isaaclab.terrains.trimesh.utils")
    trimesh_utils.make_plane = lambda **_kwargs: SimpleNamespace(
        vertices=np.zeros((4, 3), dtype=np.float32),
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
    )
    _register_module(trimesh_utils.__name__, trimesh_utils)

    math_utils = ModuleType("isaaclab.utils.math")
    math_utils.convert_quat = lambda quat, to=None: quat
    math_utils.quat_apply = lambda quat, vec: vec
    math_utils.quat_apply_yaw = lambda quat, vec: vec
    _register_module(math_utils.__name__, math_utils)

    warp_utils = ModuleType("isaaclab.utils.warp")
    warp_utils.convert_to_warp_mesh = lambda points, triangles, device: {
        "points": points,
        "triangles": triangles,
        "device": device,
    }
    warp_utils.raycast_mesh = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("patch raycast_mesh"))
    _register_module(warp_utils.__name__, warp_utils)


def _load_semantic_raycaster_module(monkeypatch: pytest.MonkeyPatch):
    _install_semantic_raycaster_stubs(monkeypatch)
    module_name = "tests._semantic_ray_caster_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, SEMANTIC_RAY_CASTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_recursive_root_collection_skips_unsupported_descendants(monkeypatch: pytest.MonkeyPatch):
    module = _load_semantic_raycaster_module(monkeypatch)
    root = _FakePrim(
        "/World/semantic_course/small",
        "Xform",
        children=[
            _FakePrim(
                "/World/semantic_course/small/row_00",
                "Xform",
                children=[
                    _FakePrim("/World/semantic_course/small/row_00/slot_00", "Mesh"),
                    _FakePrim("/World/semantic_course/small/row_00/ignored_leaf", "Capsule"),
                ],
            ),
            _FakePrim(
                "/World/semantic_course/small/row_01",
                "Scope",
                children=[
                    _FakePrim(
                        "/World/semantic_course/small/row_01/group",
                        "Xform",
                        children=[_FakePrim("/World/semantic_course/small/row_01/group/slot_01", "Cube")],
                    ),
                    _FakePrim("/World/semantic_course/small/row_01/slot_02", "Mesh"),
                ],
            ),
        ],
    )
    monkeypatch.setattr(module.sim_utils, "find_first_matching_prim", lambda path: root if path == root.GetPath() else None)

    def _fake_geometry_to_world_trimesh(prim, geom_type):
        path_len = len(str(prim.GetPath()))
        points = np.full((3, 3), path_len, dtype=np.float32)
        triangles = np.array([[0, 1, 2]], dtype=np.int32)
        return points, triangles

    monkeypatch.setattr(module, "_geometry_prim_to_world_trimesh", _fake_geometry_to_world_trimesh)

    meshes = module._usd_prim_to_world_trimeshes("/World/semantic_course/small")

    assert [(path, geom_type) for path, geom_type, *_ in meshes] == [
        ("/World/semantic_course/small/row_00/slot_00", "Mesh"),
        ("/World/semantic_course/small/row_01/group/slot_01", "Cube"),
        ("/World/semantic_course/small/row_01/slot_02", "Mesh"),
    ]


def test_initialize_warp_meshes_preserves_root_semantic_ids_and_allows_empty_roots(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_semantic_raycaster_module(monkeypatch)
    sensor = object.__new__(module.SemanticGridRayCaster)
    sensor.cfg = SimpleNamespace(
        mesh_prim_paths=[
            "/World/ground",
            "/World/semantic_course/small",
            "/World/semantic_course/large",
        ],
        mesh_semantic_ids={
            "/World/ground": 0,
            "/World/semantic_course/small": 1,
            "/World/semantic_course/large": 2,
        },
    )
    sensor.device = "cpu"
    sensor._semantic_dbg_remaining = 0

    mesh_map = {
        "/World/ground": [
            (
                "/World/ground/terrain_mesh",
                "Mesh",
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
                np.array([[0, 1, 2]], dtype=np.int32),
            )
        ],
        "/World/semantic_course/small": [],
        "/World/semantic_course/large": [
            (
                "/World/semantic_course/large/row_00/slot_00",
                "Cube",
                np.array([[0.0, 0.0, 0.4], [1.0, 0.0, 0.4], [0.0, 1.0, 0.4]], dtype=np.float32),
                np.array([[0, 1, 2]], dtype=np.int32),
            ),
            (
                "/World/semantic_course/large/row_00/slot_01",
                "Cube",
                np.array([[0.0, 0.0, 0.7], [1.0, 0.0, 0.7], [0.0, 1.0, 0.7]], dtype=np.float32),
                np.array([[0, 1, 2]], dtype=np.int32),
            ),
        ],
    }
    monkeypatch.setattr(module, "_usd_prim_to_world_trimeshes", lambda path: mesh_map[path])

    converted = {}

    def _fake_convert_to_warp_mesh(points, triangles, device):
        converted["points"] = points
        converted["triangles"] = triangles
        converted["device"] = device
        return "warp-mesh"

    monkeypatch.setattr(module, "convert_to_warp_mesh", _fake_convert_to_warp_mesh)

    sensor._initialize_warp_meshes()

    assert sensor._combined_wp_mesh == "warp-mesh"
    assert sensor._face_semantic_ids.tolist() == [0, 2, 2]
    assert converted["points"].shape == (9, 3)
    assert converted["triangles"].shape == (3, 3)


def test_initialize_rays_impl_builds_151_square_maps(monkeypatch: pytest.MonkeyPatch):
    module = _load_semantic_raycaster_module(monkeypatch)
    monkeypatch.setattr(module.RayCaster, "_initialize_rays_impl", lambda self: None, raising=False)

    sensor = object.__new__(module.SemanticGridRayCaster)
    sensor.cfg = SimpleNamespace(pattern_cfg=SimpleNamespace(size=[1.5, 1.5], resolution=0.01))
    sensor._device = "cpu"
    sensor._view = SimpleNamespace(count=2)
    sensor.num_rays = 151 * 151
    sensor._data = SimpleNamespace()
    sensor._semantic_dbg_remaining = 0

    sensor._initialize_rays_impl()

    assert sensor._grid_nx == 151
    assert sensor._grid_ny == 151
    assert sensor.num_rays == 151 * 151
    assert sensor._data.elevation_map.shape == (2, 151, 151)
    assert sensor._data.semantic_map.shape == (2, 151, 151)


def test_update_buffers_handles_invalid_face_ids_and_keeps_flatten_alignment(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_semantic_raycaster_module(monkeypatch)
    sensor = object.__new__(module.SemanticGridRayCaster)
    sensor.cfg = SimpleNamespace(max_distance=5.0, height_scan_offset=0.5, attach_yaw_only=True)
    sensor._combined_wp_mesh = object()
    sensor._face_semantic_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    sensor.device = "cpu"
    sensor.drift = torch.zeros((1, 3), dtype=torch.float32)
    sensor.num_rays = 4
    sensor.ray_starts = torch.zeros((1, 4, 3), dtype=torch.float32)
    sensor.ray_directions = torch.zeros((1, 4, 3), dtype=torch.float32)
    sensor._grid_nx = 2
    sensor._grid_ny = 2
    sensor._view = _FakeView(
        pos_w=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    )
    sensor._data = SimpleNamespace(
        pos_w=torch.zeros((1, 3), dtype=torch.float32),
        quat_w=torch.zeros((1, 4), dtype=torch.float32),
        ray_hits_w=torch.zeros((1, 4, 3), dtype=torch.float32),
        elevation_map=torch.zeros((1, 2, 2), dtype=torch.float32),
        semantic_map=torch.zeros((1, 2, 2), dtype=torch.float32),
    )
    sensor._semantic_dbg_remaining = 0

    ray_hits = torch.tensor(
        [[[0.0, 0.0, 0.10], [0.0, 0.0, 0.20], [0.0, 0.0, 0.30], [0.0, 0.0, 0.40]]],
        dtype=torch.float32,
    )
    face_ids = torch.tensor([[2, 1, -1, 7]], dtype=torch.int64)

    def _fake_raycast_mesh(*args, **kwargs):
        return ray_hits, None, None, face_ids

    monkeypatch.setattr(module, "raycast_mesh", _fake_raycast_mesh)
    monkeypatch.setattr(module, "quat_apply_yaw", lambda quat, vec: vec)

    sensor._update_buffers_impl([0])

    assert torch.allclose(
        sensor._data.elevation_map[0],
        torch.tensor([[0.4, 0.3], [0.2, 0.1]], dtype=torch.float32),
    )
    assert torch.equal(
        sensor._data.semantic_map[0],
        torch.tensor([[2.0, 1.0], [0.0, 0.0]], dtype=torch.float32),
    )
    assert torch.allclose(sensor._data.ray_hits_w[0], ray_hits[0])
