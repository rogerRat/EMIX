"""Best-effort extraction of RoboTwin object randomization ranges."""

from __future__ import annotations

import ast
import importlib
import logging
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _literal(node: ast.AST) -> Any:
    """Return a literal task-manifest expression without importing SAPIEN."""
    return ast.literal_eval(node)


def _load_manifest_ranges_without_simulator(path: Path, root: Path) -> list[dict[str, Any]]:
    """Read EMIX's declarative left-task manifest when SAPIEN is unavailable."""
    task_source = path.read_text(encoding="utf-8")
    manifest_match = re.search(r'get_manifest\(["\']([^"\']+)["\']\)', task_source)
    if not manifest_match:
        # Camera variants are thin wrappers; follow their implementation import.
        impl_match = re.search(r"from \.\.([A-Za-z0-9_]+) import", task_source)
        if impl_match:
            impl_path = path.parent.parent / f"{impl_match.group(1)}.py"
            if impl_path.is_file():
                return _load_manifest_ranges_without_simulator(impl_path, root)
        raise ValueError("task has no declarative EMIX object manifest")
    semantic_task = manifest_match.group(1)
    manifest_path = root / "envs" / "left" / "left_task_manifests.py"
    manifest_tree = ast.parse(manifest_path.read_text(encoding="utf-8"), str(manifest_path))
    for node in ast.walk(manifest_tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "_register":
            continue
        if not node.args or not isinstance(node.args[0], ast.Call):
            continue
        manifest_call = node.args[0]
        if not manifest_call.args or not isinstance(manifest_call.args[0], ast.Constant):
            continue
        if manifest_call.args[0].value != semantic_task:
            continue
        records = []
        specs_node = manifest_call.args[1]
        for spec in specs_node.elts:
            if not isinstance(spec, ast.Call) or len(spec.args) < 4:
                continue
            records.append({
                "name": str(_literal(spec.args[0])),
                "x": tuple(float(v) for v in _literal(spec.args[1])),
                "y": tuple(float(v) for v in _literal(spec.args[2])),
                "yaw": tuple(float(v) for v in _literal(spec.args[3])),
            })
        if records:
            return records
    raise ValueError(f"manifest not found for task {semantic_task!r}")


def load_object_ranges(task_path: str | Path) -> list[dict[str, Any]]:
    """Load non-robot actor ranges from a RoboTwin task module.

    Tasks following the EMIX manifest convention expose ``manifest.actor_specs``
    on their implementation class.  The returned records contain x/y bounds,
    a representative z value (the table plane), and yaw bounds.  This function
    deliberately avoids constructing a simulator scene.
    """
    path = Path(task_path)
    if path.suffix != ".py":
        path = path.with_suffix(".py")
    if not path.is_file():
        raise FileNotFoundError(f"RoboTwin task file not found: {path}")
    root = next((p for p in path.parents if (p / "envs").is_dir()), path.parent)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # Import as ``envs...`` rather than executing the file directly: task
    # modules use package-relative imports (for example ``..utils``).
    module_name = ".".join(path.relative_to(root).with_suffix("").parts)
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name not in {"sapien", "sapien.core"}:
            raise
        return _load_manifest_ranges_without_simulator(path, root)
    candidates = [v for v in vars(module).values() if isinstance(v, type)]
    for cls in candidates:
        try:
            obj = cls()
        except Exception:
            continue
        manifest = getattr(obj, "manifest", None)
        specs = getattr(manifest, "actor_specs", None)
        if not specs:
            continue
        records = []
        for name, actor in specs.items():
            workspace = getattr(actor, "workspace", None)
            if workspace is None:
                continue
            x = tuple(float(v) for v in workspace.x)
            y = tuple(float(v) for v in workspace.y)
            yaw = tuple(float(v) for v in workspace.yaw_rad)
            records.append({"name": str(name), "x": x, "y": y, "yaw": yaw})
        if records:
            return records
    raise ValueError("task does not expose manifest.actor_specs")


def attach_object_ranges(viz_data: dict[str, Any], task_path: str | Path) -> dict[str, Any]:
    """Attach parsed object ranges, logging and continuing on failure."""
    try:
        viz_data["robotwin_object_ranges"] = load_object_ranges(task_path)
        logger.info("Loaded %d RoboTwin object ranges from %s", len(viz_data["robotwin_object_ranges"]), task_path)
    except Exception as exc:
        logger.warning("Could not load RoboTwin object ranges from %s: %s", task_path, exc)
        viz_data["robotwin_object_ranges"] = []
    return viz_data
