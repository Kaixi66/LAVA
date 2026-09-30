import math

import pytest
import torch

from probe_lava_negatives import capture_signatures, compare_negatives
from test_lava_film import small_model, inputs


def test_replay_matches_production_and_keeps_probability_mass_normalized():
    model = small_model("none")
    kwargs = inputs((4, 4, 4))
    kwargs.pop("lava_context")
    kwargs["order_negative"] = True
    capture = capture_signatures(model, lambda: model.compute_lava_loss(**kwargs))
    rows = compare_negatives(capture, world_tasks=("a", "b"))
    assert len(rows) == 12
    baseline = [r for r in rows if r["variant"] == "all_sum"]
    assert sum(r["loss"] for r in baseline) / 3 == pytest.approx(capture["reference_loss"], abs=3e-4)
    for row in rows:
        assert sum(row[k] for k in ("p_positive", "p_batch", "p_local", "p_far", "p_order")) == pytest.approx(1., abs=1e-6)
    averaged = [r for r in rows if r["variant"] == "all_mean"]
    assert all(b["p_batch"] <= a["p_batch"] for a, b in zip(baseline, averaged))


def test_equal_candidates_counting_and_empty_cross_task_set():
    values = torch.tensor([[1., 0.]]).expand(3, -1)
    capture = dict(action_signatures=values, world_signatures=values,
                   temporal_negative_signatures=values, far_negative_signatures=values,
                   block_swap_signatures=[], derangement_signatures=[],
                   block_action_indices=[], derangement_action_indices=[],
                   interval_scales=torch.ones(3, dtype=torch.long), selected_task_names=["a"] * 3,
                   reference_loss=math.log(5))
    rows = compare_negatives(capture)
    for row in rows:
        expected = {"all_sum": math.log(5), "all_mean": math.log(4),
                    "cross_sum": math.log(3), "cross_mean": math.log(3)}[row["variant"]]
        assert row["loss"] == pytest.approx(expected, abs=1e-6)
