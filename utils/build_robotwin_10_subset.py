#!/usr/bin/env python3
"""Build a reproducible local 10-task RoboTwin subset and its normalization stats.

The subset is a symlink view over the processed 50-task dataset. For each paper
task, it keeps up to 50 valid clean episodes and exactly 200 valid randomized
episodes, ordered by numeric episode id. Source data is never modified.
"""

import argparse
import json
import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import h5py
import numpy as np


TASKS = (
    "adjust_bottle",
    "grab_roller",
    "hanging_mug",
    "move_stapler_pad",
    "open_microwave",
    "press_stapler",
    "scan_object",
    "stack_blocks_two",
    "stamp_seal",
    "turn_switch",
)


def episode_number(path: Path) -> int:
    match = re.fullmatch(r"episode(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Unexpected episode filename: {path.name}")
    return int(match.group(1))


def read_episode(path: Path):
    with h5py.File(path, "r") as root:
        action = root["joint_action/vector"][:]
        left_pose = root["endpose/left_endpose"][:]
        left_grip = root["endpose/left_gripper"][:]
        right_pose = root["endpose/right_endpose"][:]
        right_grip = root["endpose/right_gripper"][:]

    if left_grip.ndim == 1:
        left_grip = left_grip[:, None]
    if right_grip.ndim == 1:
        right_grip = right_grip[:, None]
    state = np.concatenate(
        [left_pose, left_grip, right_pose, right_grip], axis=1)
    return action, state


def has_action_outlier(action: np.ndarray) -> bool:
    joint_dims = [idx for idx in range(action.shape[1]) if idx not in (6, 13)]
    return bool(np.any(np.abs(action[:, joint_dims]) > np.pi))


def select_valid(data_dir: Path, limit: int):
    selected = []
    skipped_outliers = []
    candidates = sorted(data_dir.glob("*.hdf5"), key=episode_number)
    for path in candidates:
        action, _ = read_episode(path)
        if has_action_outlier(action):
            skipped_outliers.append(path)
            continue
        selected.append(path)
        if len(selected) == limit:
            break
    return selected, skipped_outliers


def update_range(current_min, current_max, values):
    value_min = values.min(axis=0)
    value_max = values.max(axis=0)
    if current_min is None:
        return value_min, value_max
    return np.minimum(current_min, value_min), np.maximum(current_max, value_max)


def calculate_stats(selection):
    global_action_min = global_action_max = None
    global_state_min = global_state_max = None
    per_task = {}

    for task in TASKS:
        task_action_min = task_action_max = None
        task_state_min = task_state_max = None
        task_paths = selection[task]["demo_clean"] + selection[task]["demo_randomized"]
        for path in task_paths:
            action, state = read_episode(path)
            task_action_min, task_action_max = update_range(
                task_action_min, task_action_max, action)
            task_state_min, task_state_max = update_range(
                task_state_min, task_state_max, state)

        global_action_min, global_action_max = update_range(
            global_action_min, global_action_max,
            np.stack([task_action_min, task_action_max]))
        global_state_min, global_state_max = update_range(
            global_state_min, global_state_max,
            np.stack([task_state_min, task_state_max]))
        per_task[task] = {
            "file_count": len(task_paths),
            "valid_files": len(task_paths),
            "skipped_outlier_files": 0,
            "action": {
                "min": task_action_min.tolist(),
                "max": task_action_max.tolist(),
                "dim": int(task_action_min.shape[0]),
            },
            "state": {
                "min": task_state_min.tolist(),
                "max": task_state_max.tolist(),
                "dim": int(task_state_min.shape[0]),
            },
        }

    total_files = sum(v["file_count"] for v in per_task.values())
    return {
        "robotwin2": {
            "meta": {
                "data_mode": "both",
                "total_files": total_files,
                "valid_files": total_files,
                "skipped_outlier_files": 0,
                "num_tasks": len(TASKS),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "source": "local deterministic 10-task/200-randomized subset",
            },
            "action": {
                "min": global_action_min.tolist(),
                "max": global_action_max.tolist(),
                "dim": int(global_action_min.shape[0]),
            },
            "state": {
                "min": global_state_min.tolist(),
                "max": global_state_max.tolist(),
                "dim": int(global_state_min.shape[0]),
            },
            "per_task": per_task,
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument("--clean-per-task", type=int, default=50)
    parser.add_argument("--randomized-per-task", type=int, default=200)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    assets_dir = args.assets_dir.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {source}")
    if output.exists() or assets_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output} or {assets_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    assets_dir.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    assets_tmp = Path(tempfile.mkdtemp(prefix=f".{assets_dir.name}.tmp-", dir=assets_dir.parent))

    try:
        selection = {}
        manifest = {
            "source_dataset": str(source),
            "tasks": {},
            "selection": "numeric episode order after action-outlier filtering",
            "action_outlier_threshold": "abs(joint action) <= pi; gripper dims 6/13 excluded",
        }
        total_outliers = 0

        for task in TASKS:
            selection[task] = {}
            manifest["tasks"][task] = {}
            for split, limit in (
                ("demo_clean", args.clean_per_task),
                ("demo_randomized", args.randomized_per_task),
            ):
                data_dir = source / task / split / "data"
                selected, skipped = select_valid(data_dir, limit)
                if split == "demo_randomized" and len(selected) != limit:
                    raise ValueError(
                        f"{task}/{split}: required {limit}, found {len(selected)} valid episodes")
                if not selected:
                    raise ValueError(f"{task}/{split}: no valid episodes found")

                selection[task][split] = selected
                total_outliers += len(skipped)
                link_dir = output_tmp / task / split / "data"
                link_dir.mkdir(parents=True, exist_ok=True)
                for source_path in selected:
                    (link_dir / source_path.name).symlink_to(source_path.resolve())

                manifest["tasks"][task][split] = {
                    "count": len(selected),
                    "episodes": [path.stem for path in selected],
                    "outliers_skipped_before_limit": [path.stem for path in skipped],
                }

        manifest["total_episodes"] = sum(
            len(paths)
            for task_selection in selection.values()
            for paths in task_selection.values())
        manifest["outliers_skipped_before_limits"] = total_outliers
        with open(output_tmp / "subset_meta.json", "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)

        stats = calculate_stats(selection)
        with open(assets_tmp / "stat-local-200-10.json", "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=4)

        os.rename(output_tmp, output)
        os.rename(assets_tmp, assets_dir)
    except Exception:
        shutil.rmtree(output_tmp, ignore_errors=True)
        shutil.rmtree(assets_tmp, ignore_errors=True)
        raise

    print(f"Created subset: {output}")
    print(f"Created stats: {assets_dir / 'stat-local-200-10.json'}")
    print(f"Episodes: {manifest['total_episodes']} | tasks: {len(TASKS)}")


if __name__ == "__main__":
    main()
