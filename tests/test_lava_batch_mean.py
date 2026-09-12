import math
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from test_lava import _small_v5_model


@pytest.mark.parametrize("count", [1, 2, 4])
def test_equal_paths_mean_mass_keeps_same_task_candidates(count):
    model = _small_v5_model(False, signature_normalization="per_level_unit").eval()
    model.lava_batch_negative_reduction = "mean"
    hidden = torch.randn(1, 8, 32).repeat(count, 1, 1).requires_grad_()
    paths = [torch.randn(1, 5, 16)] * count
    loss, info = model.compute_lava_loss(
        action_hidden=hidden, world_feature_differences=paths,
        temporal_negative_feature_differences=paths,
        far_negative_feature_differences=paths, negative_mode="mixed_batch",
        batch_indices=torch.arange(count), interval_starts=torch.zeros(count, dtype=torch.long),
        interval_scales=torch.ones(count, dtype=torch.long), task_names=["a"] * count)
    assert loss.item() == pytest.approx(math.log(3 if count == 1 else 4), abs=1e-5)
    assert info["same_task_batch_candidate_count"] == count - 1
    loss.backward()
    assert torch.isfinite(hidden.grad).all()


@pytest.mark.parametrize("bf16", [False, True])
def test_mean_changes_only_batch_logits_and_matches_full_parameter_gradients(bf16):
    torch.manual_seed(6203)
    model = _small_v5_model(False, signature_normalization="per_level_unit").eval()
    hidden = torch.randn(4, 8, 32, requires_grad=True)
    scales = torch.tensor([2, 2, 2, 4])
    kwargs = dict(
        action_hidden=hidden, negative_mode="mixed_batch",
        world_feature_differences=[torch.randn(int(s), 5, 16) for s in scales],
        temporal_negative_feature_differences=[torch.randn(int(s), 5, 16) for s in scales],
        far_negative_feature_differences=[torch.randn(int(s), 5, 16) for s in scales],
        batch_indices=torch.arange(4), interval_starts=torch.zeros(4, dtype=torch.long),
        interval_scales=scales, task_names=["a", "a", "b", "c"], order_negative=True)
    ce = F.cross_entropy
    captured = []

    def record(logits, *args, **kw):
        captured.append(logits)
        return ce(logits, *args, **kw)

    rng = torch.get_rng_state()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        with patch("models.vla_model_fm.F.cross_entropy", side_effect=record):
            model.compute_lava_loss(**kwargs)
        original = captured[-1]
        expected = original.clone()
        # Three scale-2 anchors each have two batch candidates. The scale-4
        # anchor has none: keep all its batch logits masked, no NaN/log(0).
        expected[:, 1:5] = expected[:, 1:5] - torch.tensor([2., 2., 2., 1.]).log()[:, None]
        reference = ce(expected, torch.zeros(4, dtype=torch.long))
        torch.set_rng_state(rng)
        model.lava_batch_negative_reduction = "mean"
        with patch("models.vla_model_fm.F.cross_entropy", side_effect=record):
            actual, info = model.compute_lava_loss(**kwargs)
    torch.testing.assert_close(captured[-1], expected, atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=1e-5)
    assert info["batch_candidate_count"] == 1.5
    assert info["same_task_batch_candidate_count"] == .5
    params = (hidden, *model.lava_world_encoder.parameters(), *model.lava_action_projector.parameters())
    for measured, target in zip(torch.autograd.grad(actual, params), torch.autograd.grad(reference, params)):
        assert torch.isfinite(measured).all()
        torch.testing.assert_close(measured, target, atol=3e-4 if bf16 else 1e-5, rtol=.02 if bf16 else 1e-4)
