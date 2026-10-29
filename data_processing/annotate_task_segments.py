#!/usr/bin/env python3
"""Create per-frame PICK/PLACE segments from task recipes and gripper commands.

Task code supplies the object order. The action gripper signal supplies each
PICK/PLACE end boundary: the first sample of the new constant endpoint plateau.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pyarrow.parquet as pq


def _endpoint_value(value: float) -> str | None:
    if math.isclose(value, 0.0, abs_tol=1e-6):
        return "closed"
    if math.isclose(value, 1.0, abs_tol=1e-6):
        return "open"
    return None


def episode_events(rows, expected_picks: int):
    rows.sort(key=lambda row: int(row["frame_index"]))
    frames = [int(row["frame_index"]) for row in rows]
    if frames != list(range(len(frames))):
        raise ValueError("frame_index must be contiguous and start at zero")
    values = [float(row["action"][-1]) for row in rows]
    if not values or _endpoint_value(values[0]) != "open":
        raise ValueError("episode does not start with an open gripper")

    events = []
    for index in range(1, len(values) - 1):
        endpoint = _endpoint_value(values[index])
        if endpoint is None or _endpoint_value(values[index - 1]) == endpoint:
            continue
        if values[index + 1] != values[index]:
            continue
        # The signal has reached its command endpoint and remains unchanged.
        events.append(("PICK" if endpoint == "closed" else "PLACE", index))

    expected = [event for _ in range(expected_picks) for event in ("PICK", "PLACE")]
    actual = [event for event, _ in events]
    if actual != expected:
        raise ValueError(f"expected gripper events {expected}; found {events}")
    if _endpoint_value(values[-1]) != "open":
        raise ValueError("episode does not finish with an open gripper")
    return len(frames), events


def label_episode(frame_count: int, events: list[tuple[str, int]], objects: list[str]):
    labels = [None] * frame_count
    stage_start = 0
    for step, object_name in enumerate(objects, start=1):
        pick_end = events[2 * (step - 1)][1]
        place_end = events[2 * (step - 1) + 1][1]
        if not (stage_start <= pick_end < place_end < frame_count):
            raise ValueError(f"non-monotonic or out-of-range event at step {step}")

        pick_label = {"id": f"step_{step:02d}_pick", "object": object_name}
        place_label = {"id": f"step_{step:02d}_place", "object": object_name}
        # Include the endpoint sample in the stage whose gripper motion ended
        # there. The next stage begins at the following frame.
        for frame in range(stage_start, pick_end + 1):
            labels[frame] = pick_label
        for frame in range(pick_end + 1, place_end + 1):
            labels[frame] = place_label
        stage_start = place_end + 1
    # Final return after the last PLACE is intentionally unlabeled.
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    recipe = json.loads(args.recipe.read_text(encoding="utf-8"))
    task_name = args.dataset_root.name
    if task_name not in recipe["tasks"]:
        raise ValueError(f"task {task_name!r} is not in the recipe")
    parquet_paths = sorted(args.dataset_root.glob("data/chunk-*/file-*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"no LeRobot parquet files under {args.dataset_root}/data")
    episode_rows = {}
    for path in parquet_paths:
        table = pq.read_table(path, columns=["action", "episode_index", "frame_index"])
        for row in table.to_pylist():
            episode_rows.setdefault(int(row["episode_index"]), []).append(row)

    objects = recipe["tasks"][task_name]
    prepared = {}
    for episode_index, rows in sorted(episode_rows.items()):
        frame_count, events = episode_events(rows, len(objects))
        prepared[episode_index] = (
            frame_count, events, label_episode(frame_count, events, objects)
        )

    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty annotation directory: {output}")
    for episode_index, (_, _, labels) in prepared.items():
        path = output / f"episode{episode_index}.json"
        path.write_text(json.dumps({"segments": labels}, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    frame_counts = [item[0] for item in prepared.values()]
    example_events = prepared[min(prepared)][1]
    print(json.dumps({
        "task": task_name, "episodes": len(prepared),
        "frames_per_episode": [min(frame_counts), max(frame_counts)],
        "example_episode": min(prepared),
        "example_event_end_frames": example_events,
        "output": str(output)
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
