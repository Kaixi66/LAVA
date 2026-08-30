from types import SimpleNamespace

import numpy as np
import torch

from probe_lava_action_layers import (
    balanced_validation_indices,
    make_order_permutations,
    parse_layer_names,
    validation_metrics,
)


def test_all_action_layers_parse_in_order():
    value = "1,2,3,4,5,6,7,8,9,10,11,12,final"
    assert parse_layer_names(value, 12) == value.split(",")


def test_balanced_validation_selection_covers_every_task_and_mode():
    metadata = []
    cursor = 0
    for task_index in range(10):
        for mode in ("clean", "randomized"):
            metadata.append({
                "global_start": cursor,
                "global_end": cursor + 10,
                "task_name": f"task_{task_index}",
                "hdf5_path": f"/data/task_{task_index}/demo_{mode}/episode.hdf5",
            })
            cursor += 10
    dataset = SimpleNamespace(
        episode_metadata=metadata,
        _ep_end_bounds=np.asarray([meta["global_end"] for meta in metadata]),
    )
    selected, summary = balanced_validation_indices(
        dataset, list(range(cursor)), sample_count=100, seed=42)
    assert len(selected) == len(set(selected)) == 100
    assert summary["tasks"] == 10
    assert summary["strata"] == 20
    assert summary["min_frames_per_stratum"] == 5
    assert summary["max_frames_per_stratum"] == 5
    assert summary["min_frames_per_task"] == 10
    assert summary["max_frames_per_task"] == 10


def test_common_order_permutations_cover_scale_boundaries():
    torch.manual_seed(42)
    permutations = make_order_permutations(torch.tensor([1, 2, 3, 8]))
    assert permutations[0] is None
    assert permutations[1].tolist() == [1, 0]
    for length, permutation in zip((3, 8), permutations[2:]):
        identity = torch.arange(length)
        assert torch.equal(permutation.sort().values, identity)
        assert torch.all(permutation != identity)


def test_validation_metrics_report_balanced_task_coverage():
    torch.manual_seed(1)
    action = torch.randn(20, 8)
    world = action + 0.05 * torch.randn(20, 8)
    shuffled = torch.randn(20, 8)
    valid = torch.ones(20, dtype=torch.bool)
    tasks = [f"task_{index // 2}" for index in range(20)]
    metrics = validation_metrics(
        action, world, shuffled, valid, tasks,
        scales=torch.tensor([2, 4] * 10), temperature=0.07)
    assert metrics["unique_tasks"] == 10
    assert metrics["min_samples_per_task"] == 2
    assert metrics["max_samples_per_task"] == 2
    assert np.isfinite(metrics["task_shortcut_gap"])
