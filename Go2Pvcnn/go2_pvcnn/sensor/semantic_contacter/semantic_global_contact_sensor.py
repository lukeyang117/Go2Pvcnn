from __future__ import annotations

from isaaclab.sensors import ContactSensor


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


class SemanticGlobalContactSensor(ContactSensor):
    """ContactSensor variant for global static semantic-course objects."""
