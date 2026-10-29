#!/usr/bin/env python3
"""Prepare a hard-linked LeRobot copy with reviewed object names and segments."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import h5py


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def prepare_task(source_root: Path, segment_root: Path, robotwin_task_root: Path,
                 output_root: Path, recipe: dict):
    task = source_root.name
    if task not in recipe["tasks"] or task not in recipe["object_actor_ids"]:
        raise ValueError(f"task {task!r} is missing from the reviewed recipe")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite prepared dataset: {output_root}")
    shutil.copytree(source_root, output_root, copy_function=os.link)

    role_ids = {role: int(actor_id)
                for role, actor_id in recipe["object_actor_ids"][task].items()}
    if set(role_ids) != set(recipe["tasks"][task]):
        raise ValueError(f"object actor IDs do not match operation roles for {task}")
    sidecars = sorted((output_root / "annotations").glob("episode_*.instances.h5"),
                      key=lambda p: int(re.search(r"episode_(\d+)", p.name).group(1)))
    if not sidecars:
        raise FileNotFoundError(f"no instance sidecars in {output_root}/annotations")
    role_names = set(role_ids)
    for destination in sidecars:
        match = re.search(r"episode_(\d+)", destination.name)
        episode = int(match.group(1))
        label_path = segment_root / f"episode{episode}.json"
        object_path = robotwin_task_root / "instance_objects" / f"episode{episode}.json"
        if not label_path.is_file() or not object_path.is_file():
            raise FileNotFoundError(f"missing labels or renderer object map for episode {episode}")
        raw_map = json.loads(object_path.read_text(encoding="utf-8"))
        role_by_id = {str(actor_id): role for role, actor_id in role_ids.items()}
        renamed = {}
        for actor_id, role in role_by_id.items():
            if actor_id not in raw_map:
                raise ValueError(f"renderer actor ID {actor_id} absent for {task} episode {episode}")
            renamed[raw_map[actor_id]] = role

        label_payload = json.loads(label_path.read_text(encoding="utf-8"))
        segments = label_payload.get("segments") if isinstance(label_payload, dict) else label_payload
        if not isinstance(segments, list):
            raise ValueError(f"invalid segment annotation file: {label_path}")

        temporary = destination.with_name(destination.name + ".tmp")
        shutil.copy2(destination, temporary)
        try:
            with h5py.File(temporary, "r+") as handle:
                objects_value = _text(handle.attrs["objects"])
                old_objects = json.loads(objects_value)
                new_objects = [renamed.get(name, name) for name in old_objects]
                if len(set(new_objects)) != len(new_objects):
                    raise ValueError(f"object-name collision in {task} episode {episode}")
                missing_roles = role_names - set(new_objects)
                if missing_roles:
                    raise ValueError(
                        f"object masks missing canonical roles {sorted(missing_roles)} "
                        f"in {task} episode {episode}"
                    )
                if "patch_target" in new_objects:
                    raise ValueError(f"historical patch_target remains in {task} episode {episode}")
                frame_count = len(handle["frame_ids"])
                if len(segments) != frame_count:
                    raise ValueError(
                        f"segment/frame count mismatch in {task} episode {episode}: "
                        f"{len(segments)} != {frame_count}"
                    )
                for frame, segment in enumerate(segments):
                    if segment is None:
                        continue
                    if (not isinstance(segment, dict)
                            or set(segment) != {"id", "object"}
                            or not segment["id"] or segment["object"] not in role_names):
                        raise ValueError(
                            f"invalid segment label in {task} episode {episode}, frame {frame}"
                        )
                    if segment["object"] not in new_objects:
                        raise ValueError(
                            f"segment object {segment['object']} has no mask in {task} episode {episode}"
                        )
                if "segment_id" in handle or "segment_object" in handle:
                    raise ValueError(f"prepared labels already exist in {destination}")
                handle.attrs["objects"] = json.dumps(new_objects, ensure_ascii=False)
                # This sidecar now follows the current instance-annotation
                # contract, including required per-frame segment arrays.
                handle.attrs["schema_version"] = 2
                string_type = h5py.string_dtype(encoding="utf-8")
                handle.create_dataset(
                    "segment_id",
                    data=[segment["id"] if segment else "" for segment in segments],
                    dtype=string_type,
                )
                handle.create_dataset(
                    "segment_object",
                    data=[segment["object"] if segment else "" for segment in segments],
                    dtype=string_type,
                )
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
    return len(sidecars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--segment-root", required=True, type=Path)
    parser.add_argument("--robotwin-task-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    count = prepare_task(args.source_root, args.segment_root, args.robotwin_task_root,
                         args.output_root, recipe)
    print(json.dumps({"task": args.source_root.name, "episodes": count,
                      "output": str(args.output_root)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
