import ast
import re
import unittest
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
GO2PVCNN_ROOT = REPO_ROOT / "Go2Pvcnn"


@dataclass(frozen=True)
class ScanFile:
    path: Path
    reason: str


TOGETHER_PLANNER_DIR = GO2PVCNN_ROOT / "extension" / "batched_together_planner"

REQUIRED_TRAINING_PATH_FILES = (
    ScanFile(
        GO2PVCNN_ROOT / "extension" / "trajectory_manager_factory.py",
        "backend factory / train attach wiring for the together planner",
    ),
    ScanFile(
        GO2PVCNN_ROOT / "go2_pvcnn" / "tasks" / "teacher_elevation_trajectory_env_cfg.py",
        "teacher_elevation_trajectory together env cfg wiring",
    ),
    ScanFile(
        GO2PVCNN_ROOT / "extension" / "mdp" / "rewards_reference.py",
        "reward/cache gather path consumed by together training",
    ),
)

# Every production file under extension/batched_together_planner must be listed
# here or in ALLOWED_UNMANIFESTED_TOGETHER_FILES with a concrete reason.
TOGETHER_PLANNER_MANIFEST: tuple[ScanFile, ...] = (
    ScanFile(TOGETHER_PLANNER_DIR / "__init__.py", "together planner package exports"),
    ScanFile(TOGETHER_PLANNER_DIR / "adapter.py", "IsaacLab env-to-planner tensor adapter"),
    ScanFile(TOGETHER_PLANNER_DIR / "config.py", "together planner runtime configuration"),
    ScanFile(TOGETHER_PLANNER_DIR / "costs.py", "batched planner objective/cost path"),
    ScanFile(TOGETHER_PLANNER_DIR / "kinematics.py", "batched kinematic rollout path"),
    ScanFile(TOGETHER_PLANNER_DIR / "manager.py", "training manager/cache refresh path"),
    ScanFile(TOGETHER_PLANNER_DIR / "optimizer.py", "batched optimizer path"),
    ScanFile(TOGETHER_PLANNER_DIR / "parameterization.py", "batched candidate parameterization path"),
    ScanFile(TOGETHER_PLANNER_DIR / "planner.py", "top-level together planner execution path"),
    ScanFile(TOGETHER_PLANNER_DIR / "schedule.py", "batched gait/schedule path"),
    ScanFile(TOGETHER_PLANNER_DIR / "terrain.py", "batched terrain query path"),
    ScanFile(TOGETHER_PLANNER_DIR / "types.py", "together planner tensor dataclasses/types"),
)

ALLOWED_UNMANIFESTED_TOGETHER_FILES: dict[Path, str] = {}

EXCLUDED_PATH_PARTS = {
    ("extension", "viz"),
    ("extension", "batched_planner"),
    ("tests",),
    ("benchmarks",),
    ("raw",),
    ("onlyReference",),
}

FORBIDDEN_IMPORT_ROOTS = {
    "numpy",
    "scipy",
    "pandas",
    "sklearn",
    "cv2",
}

FORBIDDEN_CALL_ATTRS = {
    "cpu",
    "item",
    "numpy",
    "tolist",
    "nonzero",
    "index_select",
    "index_copy_",
    "masked_select",
    "chunk",
    "split",
}

FORBIDDEN_TORCH_CALLS = {
    "torch.equal",
    "torch.allclose",
    "torch.cuda.synchronize",
    "torch.linalg.svd",
    "torch.svd",
    "torch.split",
    "torch.chunk",
}

FORBIDDEN_LOOP_NODES = (
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)

TENSORISH_NAME_RE = re.compile(
    r"(tensor|tensors|cache|frame|frames|ids|idx|indices|mask|masks|episode|"
    r"command|commands|root|quat|joint|foot|feet|contact|touchdown|terrain|ray|hits|"
    r"height|cost|state|states|candidate|candidates|batch)",
    re.IGNORECASE,
)

SCALAR_METADATA_NAME_RE = re.compile(
    r"(^|_)(count|size|shape|ndim|dim|horizon|steps?|frames?|cap|num|n)$|"
    r"^(batch_size|num_envs|ray_count|side|max_phase|horizon_steps)$",
    re.IGNORECASE,
)

DYNAMIC_SUBBATCH_FUNCTIONS = {
    "batched_generate_trajectory",
    "generate_trajectory",
    "plan_segment",
    "refresh_from_env",
}

DYNAMIC_SUBBATCH_KEYWORDS = {
    "env_ids",
    "env_idx",
    "env_indices",
    "candidate_ids",
    "candidate_idx",
    "candidate_indices",
    "mask",
    "masks",
}


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _has_path_parts(path: Path, parts: tuple[str, ...]) -> bool:
    rel_parts = path.relative_to(REPO_ROOT).parts
    if len(parts) == 1:
        return parts[0] in rel_parts
    return any(rel_parts[i : i + len(parts)] == parts for i in range(len(rel_parts) - len(parts) + 1))


def _is_excluded(path: Path) -> bool:
    return any(_has_path_parts(path, parts) for parts in EXCLUDED_PATH_PARTS)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _call_name(node.value)
        return f"{owner}.{node.attr}" if owner else node.attr
    return ""


def _node_label(node: ast.AST) -> str:
    return f"L{getattr(node, 'lineno', '?')}:C{getattr(node, 'col_offset', '?')}"


def _is_config_expr(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in {"cfg", "config", "planner_cfg"}
    if isinstance(node, ast.Attribute):
        return node.attr in {"cfg", "_cfg", "config", "planner_cfg"} or _is_config_expr(node.value)
    if isinstance(node, ast.Subscript):
        return _is_config_expr(node.value)
    return False


def _is_shape_expr(node: ast.AST) -> bool:
    if isinstance(node, ast.Attribute):
        return node.attr in {"shape", "ndim", "device", "dtype"} or _is_shape_expr(node.value)
    if isinstance(node, ast.Subscript):
        return _is_shape_expr(node.value)
    return False


def _is_tensorish_expr(node: ast.AST) -> bool:
    if _is_shape_expr(node) or _is_config_expr(node):
        return False
    if isinstance(node, ast.Name):
        if SCALAR_METADATA_NAME_RE.search(node.id):
            return False
        return bool(TENSORISH_NAME_RE.search(node.id))
    if isinstance(node, ast.Attribute):
        return bool(TENSORISH_NAME_RE.search(node.attr)) or _is_tensorish_expr(node.value)
    if isinstance(node, ast.Subscript):
        return _is_tensorish_expr(node.value) or _is_tensorish_expr(node.slice)
    if isinstance(node, ast.Call):
        name = _call_name(node.func)
        if name in {"getattr", "len", "range"}:
            return False
        return name.startswith("torch.") or any(_is_tensorish_expr(arg) for arg in node.args)
    if isinstance(node, ast.BinOp):
        return _is_tensorish_expr(node.left) or _is_tensorish_expr(node.right)
    if isinstance(node, ast.UnaryOp):
        return _is_tensorish_expr(node.operand)
    if isinstance(node, ast.Compare):
        return _is_tensorish_expr(node.left) or any(_is_tensorish_expr(c) for c in node.comparators)
    return False


def _is_subbatch_expr(node: ast.AST) -> bool:
    if isinstance(node, ast.Subscript):
        return _is_tensorish_expr(node.value) and _is_tensorish_expr(node.slice)
    if isinstance(node, ast.Call):
        return _call_name(node.func) in {"torch.where", "torch.nonzero"} or any(_is_subbatch_expr(arg) for arg in node.args)
    return False


def _scan_files() -> list[ScanFile]:
    files = list(REQUIRED_TRAINING_PATH_FILES)
    files.extend(TOGETHER_PLANNER_MANIFEST)
    for path, reason in ALLOWED_UNMANIFESTED_TOGETHER_FILES.items():
        files.append(ScanFile(path, f"allowlisted: {reason}"))
    return [scan_file for scan_file in files if not _is_excluded(scan_file.path)]


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=_rel(path))


class BatchedTogetherGuardrailsTest(unittest.TestCase):
    def test_required_training_path_files_exist_and_are_not_excluded(self):
        problems: list[str] = []
        for scan_file in REQUIRED_TRAINING_PATH_FILES:
            if _is_excluded(scan_file.path):
                problems.append(f"{_rel(scan_file.path)} is unexpectedly excluded")
            if not scan_file.path.is_file():
                problems.append(f"{_rel(scan_file.path)} is missing ({scan_file.reason})")
        self.assertEqual(problems, [])

    def test_batched_together_planner_files_are_manifested_or_allowlisted(self):
        discovered = set(TOGETHER_PLANNER_DIR.rglob("*.py"))
        manifested = {scan_file.path for scan_file in TOGETHER_PLANNER_MANIFEST}
        allowlisted = set(ALLOWED_UNMANIFESTED_TOGETHER_FILES)
        extra = sorted(discovered - manifested - allowlisted)
        missing = sorted(path for path in manifested if not path.is_file())
        bad_allowlist = sorted(path for path, reason in ALLOWED_UNMANIFESTED_TOGETHER_FILES.items() if not reason.strip())

        problems = []
        problems.extend(f"unmanifested together training file: {_rel(path)}" for path in extra)
        problems.extend(f"manifested together file is missing: {_rel(path)}" for path in missing)
        problems.extend(f"allowlisted together file lacks reason: {_rel(path)}" for path in bad_allowlist)
        self.assertEqual(problems, [])

    def test_training_path_does_not_import_cpu_numeric_packages(self):
        violations: list[str] = []
        for scan_file in _scan_files():
            if not scan_file.path.is_file():
                continue
            tree = _parse(scan_file.path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root = alias.name.split(".", 1)[0]
                        if root in FORBIDDEN_IMPORT_ROOTS:
                            violations.append(f"{_rel(scan_file.path)}:{node.lineno} forbidden import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    root = (node.module or "").split(".", 1)[0]
                    if root in FORBIDDEN_IMPORT_ROOTS:
                        violations.append(f"{_rel(scan_file.path)}:{node.lineno} forbidden from-import {node.module}")
                elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "np":
                    violations.append(f"{_rel(scan_file.path)}:{node.lineno} forbidden np.{node.attr}")
        self.assertEqual(violations, [])

    def test_training_path_has_no_python_loops_or_comprehensions(self):
        violations: list[str] = []
        for scan_file in _scan_files():
            if not scan_file.path.is_file():
                continue
            tree = _parse(scan_file.path)
            for node in ast.walk(tree):
                if isinstance(node, FORBIDDEN_LOOP_NODES):
                    violations.append(f"{_rel(scan_file.path)}:{node.lineno} forbidden {type(node).__name__}")
        self.assertEqual(violations, [])

    def test_training_path_has_no_cpu_sync_or_dynamic_indexing_calls(self):
        violations: list[str] = []
        for scan_file in _scan_files():
            if not scan_file.path.is_file():
                continue
            tree = _parse(scan_file.path)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                name = _call_name(node.func)
                attr = node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if name in FORBIDDEN_TORCH_CALLS or attr in FORBIDDEN_CALL_ATTRS:
                    violations.append(f"{_rel(scan_file.path)}:{node.lineno} forbidden call {name or attr}")
                if name in {"bool", "int", "float"} and node.args and _is_tensorish_expr(node.args[0]):
                    violations.append(
                        f"{_rel(scan_file.path)}:{node.lineno} forbidden tensor-derived {name}(...)"
                    )
                if name in DYNAMIC_SUBBATCH_FUNCTIONS:
                    if any(_is_subbatch_expr(arg) for arg in node.args):
                        violations.append(
                            f"{_rel(scan_file.path)}:{node.lineno} dynamic subbatch positional call to {name}"
                        )
                    for keyword in node.keywords:
                        if keyword.arg in DYNAMIC_SUBBATCH_KEYWORDS or _is_subbatch_expr(keyword.value):
                            violations.append(
                                f"{_rel(scan_file.path)}:{node.lineno} dynamic subbatch keyword call to {name}"
                            )
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
