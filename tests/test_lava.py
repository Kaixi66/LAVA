import csv
import math
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataloader.dataset import (
    LAVABatchScaleBatchSampler, RobotWinTaskDataset, collate_fn)
from models.model_runner import VLAWrapper
from models.vla_model_fm import (
    EMALogSignatureCalibrator,
    EMAActionDistanceCalibrator,
    VLAModel,
    _raw_logsignature_levels,
    normalized_logsignature,
    sample_contiguous_block_swap_permutation,
    sample_full_shuffle_permutation,
    action_path_distance,
    action_similarity_weight,
    weighted_count_balanced_family_logit,
)


class _ScaleEchoDataset(torch.utils.data.Dataset):
    def __init__(self, length=160):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        index, scale = item
        return torch.tensor([index, scale], dtype=torch.long)


def test_v5_gripper_aware_action_descriptor():
    normalized = torch.zeros(4, 14)
    raw_open = torch.zeros(4, 14)
    raw_closed = raw_open.clone()
    raw_closed[:, [6, 13]] = 1.0
    arm, state, change, combined = action_path_distance(
        normalized, raw_open, normalized, raw_closed)
    assert arm == 0
    assert state > 0
    assert change == 0
    assert combined == 0.5 * state

    raw_changing = raw_open.clone()
    raw_changing[:, [6, 13]] = torch.tensor([0.0, 0.0, 1.0, 1.0])[:, None]
    _, _, change, _ = action_path_distance(
        normalized, raw_open, normalized, raw_changing)
    assert change > 0

    values = action_path_distance(
        normalized, raw_open, normalized.clone(), raw_open.clone())
    assert all(value == 0 for value in values)


def test_v5_family_balanced_beta_initialization_ema_and_checkpoint():
    calibrator = EMAActionDistanceCalibrator(scales=(4,), momentum=0.99)
    cross = torch.ones(100) * 9.0
    local = torch.tensor([3.0])
    far = torch.tensor([6.0])
    first = calibrator.update(4, (cross, local, far))
    assert torch.allclose(first, torch.tensor(6.0))
    assert torch.allclose(calibrator.beta(4), torch.tensor(6.0))
    second = calibrator.update(4, (torch.ones(10000) * 12.0,
                                   torch.tensor([3.0]), torch.tensor([6.0])))
    assert torch.allclose(second, torch.tensor(6.01), atol=1e-6)
    missing = EMAActionDistanceCalibrator(scales=(4,), momentum=0.99)
    assert torch.allclose(
        missing.update(4, (torch.tensor([]), local, far)), torch.tensor(4.5))
    restored = EMAActionDistanceCalibrator(scales=(4,))
    restored.load_state_dict(calibrator.state_dict(), strict=True)
    assert torch.allclose(restored.beta(4), calibrator.beta(4))


def test_v5_weighted_count_balancing_uses_candidate_count_not_weight_sum():
    temperature = 0.2
    values = torch.tensor([[0.4, 0.2, -torch.inf]])
    weights = torch.tensor([[0.25, 0.5, 1.0]])
    actual, counts = weighted_count_balanced_family_logit(
        values, weights, temperature)
    expected = temperature * (
        torch.logsumexp(values[0, :2] / temperature + weights[0, :2].log(), dim=0)
        - torch.tensor(2.0).log())
    assert counts.tolist() == [2]
    assert torch.allclose(actual[0], expected)
    assert torch.allclose(
        action_similarity_weight(torch.tensor(0.0), torch.tensor(1.0), 0.1),
        torch.tensor(0.1))
    assert action_similarity_weight(
        torch.tensor(100.0), torch.tensor(1.0), 0.1) > 0.999

    # CUDA autocast produces BF16 similarities while action-distance gates stay
    # FP32. The weighted family reduction must safely promote the mixed dtypes.
    mixed_values = values.to(torch.bfloat16)
    mixed_result, mixed_counts = weighted_count_balanced_family_logit(
        mixed_values, weights.float(), temperature)
    assert mixed_result.dtype == torch.float32
    assert torch.isfinite(mixed_result).all()
    assert torch.equal(mixed_counts, counts)


def test_v5_batch_uniform_scale_sampler_multiworker_and_resume():
    dataset = _ScaleEchoDataset()
    sampler = LAVABatchScaleBatchSampler(
        len(dataset), batch_size=8, scales=(1, 2, 4, 8, 16), seed=17)
    loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=sampler, num_workers=4, prefetch_factor=2)
    seen = []
    for batch in loader:
        assert batch[:, 1].unique().numel() == 1
        seen.append(int(batch[0, 1]))
    assert len(seen) == 20
    for start in range(0, 20, 5):
        assert sorted(seen[start:start + 5]) == [1, 2, 4, 8, 16]

    left = LAVABatchScaleBatchSampler(40, 4, (1, 2, 4, 8, 16), seed=9)
    iterator = iter(left)
    for _ in range(3):
        next(iterator)
    state = left.state_dict()
    expected = [next(iterator) for _ in range(4)]
    right = LAVABatchScaleBatchSampler(40, 4, (1, 2, 4, 8, 16), seed=999)
    right.load_state_dict(state)
    actual_iterator = iter(right)
    actual = [next(actual_iterator) for _ in range(4)]
    assert actual == expected


def _small_v5_model(weighting):
    return VLAModel(
        action_dim=14, proprio_dim=2, hidden_dim=32, num_heads=4, depth=1,
        action_len=8, proprio_len=1, num_registers=0,
        dino_feat_dims=(16,), vlm_num_queries=2, adapter_depth=1,
        use_future_feat=False, use_lava=True, lava_dino_feat_dim=16,
        lava_residual_dim=4, lava_qformer_hidden_dim=16,
        lava_qformer_num_queries=1, lava_qformer_num_layers=1,
        lava_qformer_num_heads=4, lava_logsig_depth=2,
        lava_scales=(1, 2, 4, 8, 16),
        lava_action_similarity_weighting=weighting,
    )


def test_v5_conditional_state_dict_and_mixed_bf16_backward():
    v4 = _small_v5_model(False)
    assert not any("action_distance_calibrator" in key for key in v4.state_dict())
    v4.load_state_dict(v4.state_dict(), strict=True)

    model = _small_v5_model(True)
    assert any("action_distance_calibrator" in key for key in model.state_dict())
    action_hidden = torch.randn(4, 8, 32, requires_grad=True)
    normalized_actions = torch.rand(4, 8, 14) * 2 - 1
    raw_actions = (normalized_actions + 1) / 2
    scales = torch.tensor([2, 2, 2, 2])
    positive = [torch.randn(2, 5, 16) for _ in range(4)]
    local = [torch.randn(2, 5, 16) for _ in range(4)]
    far = [torch.randn(2, 5, 16) for _ in range(4)]
    local_raw = [torch.rand(3, 14) for _ in range(4)]
    far_raw = [torch.rand(3, 14) for _ in range(4)]
    local_norm = [value * 2 - 1 for value in local_raw]
    far_norm = [value * 2 - 1 for value in far_raw]
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        loss, diagnostics = model.compute_lava_loss(
            action_hidden=action_hidden,
            world_feature_differences=positive,
            temporal_negative_feature_differences=local,
            far_negative_feature_differences=far,
            negative_mode="mixed",
            batch_indices=torch.arange(4),
            interval_starts=torch.tensor([0, 1, 2, 3]),
            interval_scales=scales,
            temperature=0.07,
            order_negative=True,
            task_names=["a", "b", "a", "b"],
            normalized_actions=normalized_actions,
            raw_actions=raw_actions,
            temporal_negative_normalized_actions=local_norm,
            temporal_negative_raw_actions=local_raw,
            far_negative_normalized_actions=far_norm,
            far_negative_raw_actions=far_raw,
        )
    assert torch.isfinite(loss)
    assert diagnostics["arm_action_distance"] >= 0
    assert diagnostics["gripper_state_distance"] >= 0
    assert diagnostics["gripper_change_distance"] >= 0
    assert 0.1 <= diagnostics["local_neg_weight_mean"] <= 1.0
    assert math.isfinite(diagnostics["action_distance_beta_s2"])
    assert diagnostics["raw_candidate_acc"] >= 0
    assert diagnostics["weighted_candidate_acc"] >= 0
    assert len(diagnostics["_action_similarity_audit"]) == 4
    loss.backward()
    assert action_hidden.grad is not None
    assert model.lava_world_encoder.output_proj.weight.grad is not None
from train import (
    LossLogger,
    build_lava_gradient_parameter_groups,
    compute_lava_weight,
    compute_shared_gradient_diagnostics,
    should_run_lava_grad_diagnostics,
)


def test_lava_weight_schedule_is_step_based_and_resume_safe():
    peak = 0.01
    minimum = 0.003
    warmup_steps = 8
    decay_start_step = 60

    assert compute_lava_weight(
        1, peak, warmup_steps, "step", minimum, decay_start_step
    ) == (0.00125, 0)
    assert compute_lava_weight(
        8, peak, warmup_steps, "step", minimum, decay_start_step
    ) == (peak, 0)
    assert compute_lava_weight(
        9, peak, warmup_steps, "step", minimum, decay_start_step
    ) == (peak, 1)
    assert compute_lava_weight(
        60, peak, warmup_steps, "step", minimum, decay_start_step
    ) == (peak, 1)
    assert compute_lava_weight(
        61, peak, warmup_steps, "step", minimum, decay_start_step
    ) == (minimum, 2)
    # Recomputing from a restored global step gives the same phase and weight.
    assert compute_lava_weight(
        61, peak, warmup_steps, "step", minimum, decay_start_step
    ) == compute_lava_weight(
        61, peak, warmup_steps, "step", minimum, decay_start_step
    )


def test_lava_weight_cosine_warms_then_decays_to_zero_resume_safely():
    peak = 0.01
    minimum = 0.0
    warmup_steps = 5
    total_steps = 100
    decay_start_step = warmup_steps

    weight_1, phase_1 = compute_lava_weight(
        1, peak, warmup_steps, "cosine", minimum,
        decay_start_step, total_steps)
    weight_5, phase_5 = compute_lava_weight(
        5, peak, warmup_steps, "cosine", minimum,
        decay_start_step, total_steps)
    weight_mid, phase_mid = compute_lava_weight(
        53, peak, warmup_steps, "cosine", minimum,
        decay_start_step, total_steps)
    weight_100, phase_100 = compute_lava_weight(
        100, peak, warmup_steps, "cosine", minimum,
        decay_start_step, total_steps)

    assert math.isclose(weight_1, 0.002)
    assert phase_1 == 0
    assert math.isclose(weight_5, peak)
    assert phase_5 == 0
    assert 0.0 < weight_mid < peak
    assert phase_mid == 2
    assert math.isclose(weight_100, minimum, abs_tol=1e-12)
    assert phase_100 == 2

    # The schedule is a pure function of restored global_step.
    assert compute_lava_weight(
        77, peak, warmup_steps, "cosine", minimum,
        decay_start_step, total_steps
    ) == compute_lava_weight(
        77, peak, warmup_steps, "cosine", minimum,
        decay_start_step, total_steps
    )


def test_depth_two_logsignature_dimension_and_order():
    increments = torch.randn(4, 33)
    signature = normalized_logsignature(increments, depth=2)
    assert signature.shape == (561,)

    simple_path = torch.tensor([
        [1.0, 0.0, 0.25],
        [0.0, 1.0, 0.25],
        [-0.5, 0.0, 0.25],
        [0.0, 0.5, 0.25],
    ])
    reversed_path = simple_path.flip(0)
    assert torch.allclose(
        normalized_logsignature(simple_path, depth=1),
        normalized_logsignature(reversed_path, depth=1),
    )
    assert not torch.allclose(
        normalized_logsignature(simple_path, depth=2),
        normalized_logsignature(reversed_path, depth=2),
    )


def test_full_shuffle_moves_every_position():
    torch.manual_seed(5)
    assert sample_full_shuffle_permutation(2).tolist() == [1, 0]
    for length in (3, 4, 8, 16):
        identity = torch.arange(length)
        for _ in range(20):
            permutation = sample_full_shuffle_permutation(length)
            assert torch.equal(permutation.sort().values, identity)
            assert torch.all(permutation != identity)

    increments = torch.tensor([
        [1.0, 0.0, 0.25],
        [0.0, 1.0, 0.25],
        [-0.5, 0.0, 0.25],
        [0.0, 0.5, 0.25],
    ])
    permutation = sample_full_shuffle_permutation(len(increments))
    shuffled = increments[permutation]
    assert torch.allclose(
        normalized_logsignature(increments, depth=1),
        normalized_logsignature(shuffled, depth=1),
    )
    assert not torch.allclose(
        normalized_logsignature(increments, depth=2),
        normalized_logsignature(shuffled, depth=2),
    )


def test_contiguous_block_swap_is_non_identity_and_preserves_all_positions():
    torch.manual_seed(9)
    for length in (2, 3, 4, 8, 16):
        identity = torch.arange(length)
        for _ in range(20):
            permutation = sample_contiguous_block_swap_permutation(length)
            assert torch.equal(permutation.sort().values, identity)
            assert not torch.equal(permutation, identity)


def test_ema_calibration_preserves_weak_second_order_energy_and_checkpoints():
    calibrator = EMALogSignatureCalibrator(
        scales=(4,), momentum=0.99, level2_weight=0.5)
    curved = torch.tensor([
        [1.0, 0.0, 0.25], [0.0, 1.0, 0.25],
        [-1.0, 0.0, 0.25], [0.0, -1.0, 0.25],
    ])
    almost_straight = torch.tensor([
        [1.0, 0.0000, 0.25], [1.0, 0.0001, 0.25],
        [1.0, 0.0002, 0.25], [1.0, 0.0003, 0.25],
    ])
    curved_levels = _raw_logsignature_levels(curved)
    straight_levels = _raw_logsignature_levels(almost_straight)
    calibrator.update(
        "world", [curved_levels[0], straight_levels[0]],
        [curved_levels[1], straight_levels[1]], [4, 4])
    curved_sig, curved_stats = calibrator(
        *curved_levels, "world", 4, depth=2)
    straight_sig, straight_stats = calibrator(
        *straight_levels, "world", 4, depth=2)
    assert torch.allclose(curved_sig.norm(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(straight_sig.norm(), torch.tensor(1.0), atol=1e-6)
    assert straight_stats["level2_energy_fraction"] < 1e-4
    assert curved_stats["level2_energy_fraction"] > straight_stats[
        "level2_energy_fraction"]

    restored = EMALogSignatureCalibrator(scales=(4,))
    restored.load_state_dict(calibrator.state_dict())
    assert torch.equal(restored.ema_initialized, calibrator.ema_initialized)
    assert torch.allclose(restored.ema_squared_norm, calibrator.ema_squared_norm)


def test_raw_logsignature_norms_expose_tiny_second_order_level():
    near_straight = torch.tensor([
        [1.0, 0.0000, 0.25],
        [1.0, 0.0010, 0.25],
        [1.0, 0.0021, 0.25],
        [1.0, 0.0030, 0.25],
    ])
    signature, raw_norms = normalized_logsignature(
        near_straight, depth=2, return_raw_norms=True)
    ratio = raw_norms["level2_raw_norm"] / raw_norms["level1_raw_norm"]
    assert 0.0 < ratio < 0.01
    # The legacy helper demonstrates why V4 no longer uses per-level,
    # per-sample unit normalization inside the model.
    assert signature[near_straight.shape[1]:].norm() > 0.5


def test_lava_interval_stays_inside_episode_and_action_chunk():
    dataset = RobotWinTaskDataset.__new__(RobotWinTaskDataset)
    dataset.use_lava = True
    dataset.lava_sample_ratio = 1.0
    dataset.lava_scales = (1, 2, 4, 8, 16)
    dataset.lava_scale_sampling = "uniform"
    dataset.lava_scale_probs = None
    dataset.lava_sampling_balance = "none"
    dataset.chunk_size = 32

    random.seed(7)
    for _ in range(100):
        start, scale = dataset._sample_lava_interval(local_anchor_idx=93, total_frames=100)
        assert scale in (1, 2, 4)
        assert start >= 0
        assert start + scale <= 6
        assert start + scale <= 31

    assert dataset._sample_lava_interval(local_anchor_idx=99, total_frames=100) is None


def test_episode_local_pair_is_same_scale_non_overlapping_and_radius_capped():
    dataset = RobotWinTaskDataset.__new__(RobotWinTaskDataset)
    dataset.use_lava = True
    dataset.lava_sample_ratio = 1.0
    dataset.lava_scales = (16,)
    dataset.lava_scale_sampling = "uniform"
    dataset.lava_scale_probs = None
    dataset.lava_sampling_balance = "none"
    dataset.lava_negative_window_multiplier = 4
    dataset.lava_negative_window_max = 32
    dataset.chunk_size = 32

    random.seed(17)
    for _ in range(100):
        pair = dataset._sample_lava_contrastive_pair(
            local_anchor_idx=100, total_frames=300)
        assert pair is not None and not pair["dropped"]
        assert pair["scale"] == 16
        positive_start = pair["positive_abs_start"]
        negative_start = pair["negative_abs_start"]
        assert (negative_start + 16 <= positive_start
                or positive_start + 16 <= negative_start)
        assert pair["negative_distance"] <= 32
        assert not pair["used_global_fallback"]
        assert 100 <= positive_start <= 115
        assert 0 <= negative_start <= 283


def test_mixed_pair_adds_far_negative_outside_local_radius():
    dataset = RobotWinTaskDataset.__new__(RobotWinTaskDataset)
    dataset.use_lava = True
    dataset.lava_sample_ratio = 1.0
    dataset.lava_scales = (8,)
    dataset.lava_scale_sampling = "uniform"
    dataset.lava_scale_probs = None
    dataset.lava_sampling_balance = "none"
    dataset.lava_negative_mode = "mixed"
    dataset.lava_negative_window_multiplier = 4
    dataset.lava_negative_window_max = 32
    dataset.chunk_size = 32

    random.seed(21)
    pair = dataset._sample_lava_contrastive_pair(
        local_anchor_idx=100, total_frames=300)
    assert pair is not None and not pair["dropped"]
    assert pair["negative_distance"] <= 32
    assert pair["far_negative_distance"] > 32


def test_task_episode_balancing_equalizes_expected_lava_samples():
    dataset = RobotWinTaskDataset.__new__(RobotWinTaskDataset)
    dataset.use_lava = True
    dataset.lava_sample_ratio = 0.25
    dataset.lava_scales = (1, 2, 4, 8, 16)
    dataset.lava_sampling_balance = "task_episode"
    dataset.episode_metadata = [
        {"task_name": "task_a", "length": 101},
        {"task_name": "task_a", "length": 201},
        {"task_name": "task_b", "length": 51},
        {"task_name": "task_b", "length": 251},
    ]

    dataset._build_lava_sampling_probabilities()

    eligible = np.asarray([100.0, 200.0, 50.0, 250.0])
    expected = eligible * dataset._lava_episode_probabilities
    # 25% of 600 eligible anchors gives 150 paths: 75 per task and 37.5
    # per episode, even though frame counts differ by 5x.
    assert np.allclose(expected, [37.5, 37.5, 37.5, 37.5])
    assert math.isclose(expected[:2].sum(), expected[2:].sum())
    assert math.isclose(expected.sum(), 0.25 * eligible.sum())


def test_task_episode_balancing_waterfills_short_episodes():
    dataset = RobotWinTaskDataset.__new__(RobotWinTaskDataset)
    dataset.use_lava = True
    dataset.lava_sample_ratio = 0.5
    dataset.lava_scales = (1,)
    dataset.lava_sampling_balance = "task_episode"
    dataset.episode_metadata = [
        {"task_name": "short_long", "length": 11},
        {"task_name": "short_long", "length": 291},
        {"task_name": "other", "length": 151},
        {"task_name": "other", "length": 151},
    ]

    dataset._build_lava_sampling_probabilities()

    eligible = np.asarray([10.0, 290.0, 150.0, 150.0])
    expected = eligible * dataset._lava_episode_probabilities
    assert np.allclose(expected, [10.0, 140.0, 75.0, 75.0])
    assert np.all(dataset._lava_episode_probabilities <= 1.0)


def test_collate_keeps_variable_length_paths_and_batch_indices():
    common = {
        "state": torch.zeros(1, 2),
        "action_sequence": torch.zeros(8, 2),
        "pixel_values": torch.zeros(3, 16, 16),
        "state_mask": torch.ones(1, dtype=torch.bool),
        "action_mask": torch.ones(8, dtype=torch.bool),
        "task_name": "task_a",
    }
    sample_zero = dict(common)
    sample_one = dict(common)
    sample_one.update({
        "evolution_pixel_values": torch.zeros(5, 3, 16, 16),
        "temporal_negative_pixel_values": torch.ones(5, 3, 16, 16),
        "far_negative_pixel_values": torch.full((5, 3, 16, 16), 2.0),
        "evolution_start": 2,
        "evolution_scale": 4,
        "temporal_negative_distance": 10,
        "far_negative_distance": 40,
        "temporal_negative_local_fallback": False,
    })
    batch = collate_fn([sample_zero, sample_one])
    assert batch["evolution_batch_indices"].tolist() == [1]
    assert batch["evolution_starts"].tolist() == [2]
    assert batch["evolution_scales"].tolist() == [4]
    assert batch["evolution_pixel_values"][0].shape[0] == 5
    assert batch["temporal_negative_pixel_values"][0].shape[0] == 5
    assert batch["far_negative_pixel_values"][0].shape[0] == 5
    assert batch["temporal_negative_distances"].tolist() == [10]
    assert batch["far_negative_distances"].tolist() == [40]
    assert batch["temporal_negative_local_fallbacks"].tolist() == [False]
    assert batch["task_name"] == ["task_a", "task_a"]


def test_lava_loss_handles_scale_one_and_backpropagates():
    model = VLAModel(
        action_dim=2,
        proprio_dim=2,
        hidden_dim=32,
        num_heads=4,
        depth=1,
        action_len=8,
        proprio_len=1,
        num_registers=0,
        dino_feat_dims=(16,),
        vlm_num_queries=2,
        adapter_depth=1,
        use_future_feat=False,
        use_lava=True,
        lava_dino_feat_dim=16,
        lava_residual_dim=4,
        lava_qformer_hidden_dim=16,
        lava_qformer_num_queries=1,
        lava_qformer_num_layers=1,
        lava_qformer_num_heads=4,
        lava_logsig_depth=2,
    )
    action_hidden = torch.randn(3, 8, 32, requires_grad=True)
    world_differences = [
        torch.randn(1, 5, 16),
        torch.randn(2, 5, 16),
        torch.randn(4, 5, 16),
    ]
    # Exercise the same mixed-precision path used on H100. In particular,
    # order-margin diagnostics must safely combine BF16 logits and FP32 values.
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        loss, diagnostics = model.compute_lava_loss(
            action_hidden=action_hidden,
            world_feature_differences=world_differences,
            batch_indices=torch.tensor([0, 1, 2]),
            interval_starts=torch.tensor([0, 1, 2]),
            interval_scales=torch.tensor([1, 2, 4]),
            temperature=0.07,
            order_negative=True,
            task_names=["task_a", "task_a", "task_b"],
        )
    assert torch.isfinite(loss)
    assert {
        "pos_sim", "negative_sim", "shuffle_sim", "order_margin",
        "retrieval_acc", "action_pair_sim", "world_pair_sim",
        "lava_sample_count", "lava_order_negative_count",
        "raw_change_norm", "world_residual_std", "action_residual_std",
        "loss_s1", "loss_s4", "order_margin_s4", "loss_t0_025",
        "same_task_negative_sim", "cross_task_negative_sim", "task_shortcut_gap",
        "action_logsig_l2_l1_ratio", "world_logsig_l2_l1_ratio",
        "raw_change_norm_cv", "input_norm_change_norm_cv",
        "raw_input_norm_norm_corr", "raw_world_residual_norm_corr",
        "lava_coverage_pos_1_8", "lava_coverage_pos_9_16",
        "lava_coverage_pos_17_24", "lava_coverage_pos_25_31",
        "lava_position_mean", "lava_position_min", "lava_position_max",
        "lava_executed_horizon_ratio",
    }.issubset(diagnostics)
    assert diagnostics["lava_order_negative_count"] == 2
    assert math.isfinite(diagnostics["same_task_negative_sim"])
    assert math.isfinite(diagnostics["cross_task_negative_sim"])
    assert math.isfinite(diagnostics["task_shortcut_gap"])
    with torch.no_grad():
        _, same_only = model.compute_lava_loss(
            action_hidden=action_hidden.detach(),
            world_feature_differences=world_differences,
            batch_indices=torch.tensor([0, 1, 2]),
            interval_starts=torch.tensor([0, 1, 2]),
            interval_scales=torch.tensor([1, 2, 4]),
            task_names=["task_a", "task_a", "task_a"],
        )
        _, cross_only = model.compute_lava_loss(
            action_hidden=action_hidden.detach(),
            world_feature_differences=world_differences,
            batch_indices=torch.tensor([0, 1, 2]),
            interval_starts=torch.tensor([0, 1, 2]),
            interval_scales=torch.tensor([1, 2, 4]),
            task_names=["task_a", "task_b", "task_c"],
        )
    assert math.isfinite(same_only["same_task_negative_sim"])
    assert math.isnan(same_only["cross_task_negative_sim"])
    assert math.isnan(same_only["task_shortcut_gap"])
    assert math.isnan(cross_only["same_task_negative_sim"])
    assert math.isfinite(cross_only["cross_task_negative_sim"])
    assert math.isnan(cross_only["task_shortcut_gap"])
    loss.backward()
    assert action_hidden.grad is not None
    assert model.lava_world_encoder.output_proj.weight.grad is not None
    assert model.lava_action_projector[-1].weight.grad is not None
    assert math.isclose(
        sum(diagnostics[f"lava_coverage_pos_{bucket}"]
            for bucket in ("1_8", "9_16", "17_24", "25_31")),
        1.0, abs_tol=1e-6)
    assert 0.0 <= diagnostics["lava_executed_horizon_ratio"] <= 1.0


def test_episode_local_candidate_loss_and_no_cross_sample_competition():
    torch.manual_seed(23)
    model = VLAModel(
        action_dim=2,
        proprio_dim=2,
        hidden_dim=32,
        num_heads=4,
        depth=1,
        action_len=8,
        proprio_len=1,
        num_registers=0,
        dino_feat_dims=(16,),
        vlm_num_queries=2,
        adapter_depth=1,
        use_future_feat=False,
        use_lava=True,
        lava_dino_feat_dim=16,
        lava_residual_dim=4,
        lava_qformer_hidden_dim=16,
        lava_qformer_num_queries=1,
        lava_qformer_num_layers=1,
        lava_qformer_num_heads=4,
        lava_logsig_depth=2,
    )
    action_hidden = torch.randn(2, 8, 32, requires_grad=True)
    positive = [torch.randn(1, 5, 16), torch.randn(4, 5, 16)]
    temporal = [torch.randn(1, 5, 16), torch.randn(4, 5, 16)]

    torch.manual_seed(31)
    loss, diagnostics = model.compute_lava_loss(
        action_hidden=action_hidden,
        world_feature_differences=positive,
        temporal_negative_feature_differences=temporal,
        negative_mode="episode_local",
        batch_indices=torch.tensor([0, 1]),
        interval_starts=torch.tensor([0, 2]),
        interval_scales=torch.tensor([1, 4]),
        temperature=0.07,
        order_negative=True,
    )
    assert torch.isfinite(loss)
    assert diagnostics["lava_order_negative_count"] == 1
    assert all(math.isfinite(diagnostics[key]) for key in (
        "temporal_negative_sim", "temporal_margin", "temporal_acc",
        "order_acc", "candidate_acc", "positive_temporal_world_sim"))
    assert math.isnan(diagnostics["same_task_negative_sim"])
    assert math.isnan(diagnostics["cross_task_negative_sim"])
    scale_one_loss = diagnostics["loss_s1"]

    # Changing another sample's positive and negative paths cannot enter the
    # first sample's paired denominator. Distinct scales expose its exact loss.
    changed_positive = [positive[0], torch.randn_like(positive[1]) * 20]
    changed_temporal = [temporal[0], torch.randn_like(temporal[1]) * 20]
    torch.manual_seed(31)
    _, changed = model.compute_lava_loss(
        action_hidden=action_hidden,
        world_feature_differences=changed_positive,
        temporal_negative_feature_differences=changed_temporal,
        negative_mode="episode_local",
        batch_indices=torch.tensor([0, 1]),
        interval_starts=torch.tensor([0, 2]),
        interval_scales=torch.tensor([1, 4]),
        temperature=0.07,
        order_negative=True,
    )
    assert math.isclose(scale_one_loss, changed["loss_s1"], abs_tol=1e-6)

    loss.backward()
    assert action_hidden.grad is not None
    assert model.lava_world_encoder.output_proj.weight.grad is not None
    assert model.lava_action_projector[-1].weight.grad is not None


def test_mixed_negative_families_and_execution_diagnostics_backpropagate():
    torch.manual_seed(43)
    model = VLAModel(
        action_dim=2, proprio_dim=2, hidden_dim=32, num_heads=4, depth=1,
        action_len=8, proprio_len=1, num_registers=0,
        dino_feat_dims=(16,), vlm_num_queries=2, adapter_depth=1,
        use_future_feat=False, use_lava=True, lava_dino_feat_dim=16,
        lava_residual_dim=4, lava_qformer_hidden_dim=16,
        lava_qformer_num_queries=1, lava_qformer_num_layers=1,
        lava_qformer_num_heads=4, lava_logsig_depth=2,
        lava_scales=(1, 2, 4, 8, 16),
    )
    action_hidden = torch.randn(4, 8, 32, requires_grad=True)
    scales = [2, 2, 4, 4]
    positive = [torch.randn(scale, 5, 16) for scale in scales]
    local = [torch.randn(scale, 5, 16) for scale in scales]
    far = [torch.randn(scale, 5, 16) for scale in scales]
    loss, diagnostics = model.compute_lava_loss(
        action_hidden=action_hidden,
        world_feature_differences=positive,
        temporal_negative_feature_differences=local,
        far_negative_feature_differences=far,
        negative_mode="mixed",
        batch_indices=torch.arange(4),
        interval_starts=torch.tensor([0, 1, 0, 2]),
        interval_scales=torch.tensor(scales),
        temperature=0.07,
        order_negative=True,
        task_names=["task_a", "task_b", "task_a", "task_b"],
        action_execution_horizon=4,
    )
    assert torch.isfinite(loss)
    assert diagnostics["cross_task_candidate_count"] == 1.0
    assert math.isclose(diagnostics["order_candidate_count"], 1.5)
    assert math.isclose(
        sum(diagnostics[key] for key in (
            "hardest_cross_task_fraction", "hardest_local_fraction",
            "hardest_far_fraction", "hardest_order_fraction")),
        1.0, abs_tol=1e-6)
    assert all(math.isfinite(diagnostics[key]) for key in (
        "cross_task_margin", "far_margin", "temporal_margin",
        "block_swap_margin", "derangement_margin", "candidate_acc",
        "action_logsig_l2_energy_fraction",
        "world_logsig_l2_energy_fraction",
        "loss_lava_executed", "loss_lava_tail",
        "candidate_acc_executed", "candidate_acc_tail"))
    loss.backward()
    assert action_hidden.grad is not None
    assert model.lava_world_encoder.output_proj.weight.grad is not None
    assert model.lava_action_projector[-1].weight.grad is not None


def test_shared_gradient_diagnostics_are_correct_and_do_not_write_grads():
    shared = nn.Parameter(torch.tensor([1.0, 2.0]))
    base_only = nn.Parameter(torch.tensor([3.0]))
    lava_only = nn.Parameter(torch.tensor([4.0]))
    loss_base = shared.square().sum() + base_only.square().sum()
    loss_lava = -shared.square().sum() + lava_only.square().sum()

    diagnostics = compute_shared_gradient_diagnostics(
        loss_base, loss_lava, [shared, base_only, lava_only], lava_weight=0.01)
    assert math.isclose(diagnostics["grad_cos_shared"], -1.0, abs_tol=1e-6)
    assert math.isclose(diagnostics["weighted_grad_ratio"], 0.01, rel_tol=1e-6)
    assert all(parameter.grad is None for parameter in (shared, base_only, lava_only))

    total = loss_base + 0.01 * loss_lava
    total.backward()
    assert torch.allclose(shared.grad, 1.98 * shared.detach())

    same_direction = nn.Parameter(torch.tensor([1.0, 2.0]))
    same_stats = compute_shared_gradient_diagnostics(
        same_direction.square().sum(),
        2.0 * same_direction.square().sum(),
        [same_direction], lava_weight=0.01)
    assert math.isclose(same_stats["grad_cos_shared"], 1.0, abs_tol=1e-6)

    orthogonal = nn.Parameter(torch.tensor([1.0, 1.0]))
    orthogonal_stats = compute_shared_gradient_diagnostics(
        orthogonal[0].square(), orthogonal[1].square(),
        [orthogonal], lava_weight=0.01)
    assert math.isclose(
        orthogonal_stats["grad_cos_shared"], 0.0, abs_tol=1e-6)

    grouped_shared = nn.Parameter(torch.tensor([1.0, 2.0]))
    grouped_base_only = nn.Parameter(torch.tensor([3.0]))
    grouped_lava_only = nn.Parameter(torch.tensor([4.0]))
    grouped_base_loss = grouped_shared.square().sum() + grouped_base_only.square().sum()
    grouped_lava_loss = -grouped_shared.square().sum() + grouped_lava_only.square().sum()
    grouped = compute_shared_gradient_diagnostics(
        grouped_base_loss, grouped_lava_loss,
        [grouped_shared, grouped_base_only, grouped_lava_only], lava_weight=0.01,
        parameter_groups={"pair": [grouped_shared], "base_only": [grouped_base_only]})
    assert math.isclose(grouped["grad_cos_shared"], -1.0, abs_tol=1e-6)
    assert math.isclose(grouped["grad_cos_pair"], -1.0, abs_tol=1e-6)
    assert grouped["grad_norm_base_base_only"] > 0
    assert grouped["grad_norm_lava_base_only"] == 0
    assert math.isnan(grouped["grad_cos_base_only"])


def test_gradient_diagnostic_schedule_and_nan_csv_defaults():
    assert not should_run_lava_grad_diagnostics(199, True, 200)
    assert should_run_lava_grad_diagnostics(200, True, 200)
    assert not should_run_lava_grad_diagnostics(200, False, 200)
    assert not should_run_lava_grad_diagnostics(200, True, 0)

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = LossLogger(tmpdir)
        logger.log(1, 1, 1, 0.5, {})
        with Path(logger.log_file).open() as handle:
            row = next(csv.DictReader(handle))
        assert math.isnan(float(row["Grad_Cos_Shared"]))
        assert math.isnan(float(row["Weighted_Grad_Ratio"]))
        assert math.isnan(float(row["Grad_Cos_B9_10"]))
        assert float(row["Lambda_LAVA_Ratio"]) == 0.0
        assert int(row["LAVA_Weight_Phase"]) == 0


def test_wrapper_train_keeps_frozen_dino_in_eval_mode():
    wrapper = VLAWrapper.__new__(VLAWrapper)
    nn.Module.__init__(wrapper)
    wrapper.vision_encoder = nn.Sequential(nn.Dropout(0.5), nn.Linear(2, 2))
    wrapper.action_model = nn.Linear(2, 2)
    for parameter in wrapper.vision_encoder.parameters():
        parameter.requires_grad = False

    wrapper.train()
    assert wrapper.action_model.training
    assert not wrapper.vision_encoder.training
    assert not any(parameter.requires_grad for parameter in wrapper.vision_encoder.parameters())


def test_policy_forward_does_not_execute_lava_branch():
    model = VLAModel(
        action_dim=2,
        proprio_dim=2,
        hidden_dim=32,
        num_heads=4,
        depth=1,
        action_len=4,
        proprio_len=1,
        num_registers=0,
        dino_feat_dims=(16,),
        vlm_num_queries=2,
        adapter_depth=1,
        use_future_feat=False,
        use_lava=True,
        lava_dino_feat_dim=16,
        lava_residual_dim=4,
        lava_qformer_hidden_dim=16,
        lava_qformer_num_layers=1,
        lava_qformer_num_heads=4,
    ).eval()
    branch_calls = {"world": 0, "action": 0}
    hooks = [
        model.lava_world_encoder.register_forward_hook(
            lambda *_: branch_calls.__setitem__("world", branch_calls["world"] + 1)),
        model.lava_action_projector.register_forward_hook(
            lambda *_: branch_calls.__setitem__("action", branch_calls["action"] + 1)),
    ]
    with torch.no_grad():
        outputs = model(
            t=torch.rand(2),
            noisy_actions=torch.randn(2, 4, 2),
            qpos_history=torch.randn(2, 1, 2),
            dino_features_list=[torch.randn(2, 5, 16)],
        )
    for hook in hooks:
        hook.remove()
    assert outputs["final_pred"].shape == (2, 4, 2)
    assert branch_calls == {"world": 0, "action": 0}


def _small_tap_model(action_target_layer):
    return VLAModel(
        action_dim=2,
        proprio_dim=2,
        hidden_dim=32,
        num_heads=4,
        depth=12,
        action_len=4,
        proprio_len=1,
        num_registers=0,
        dino_feat_dims=(16,),
        vlm_num_queries=2,
        adapter_depth=1,
        use_future_feat=False,
        use_lava=True,
        lava_dino_feat_dim=16,
        lava_residual_dim=4,
        lava_qformer_hidden_dim=16,
        lava_qformer_num_layers=1,
        lava_qformer_num_heads=4,
        lava_action_target_layer=action_target_layer,
    )


def test_block6_tap_preserves_full_policy_path_and_stops_lava_gradient():
    torch.manual_seed(11)
    layer6_model = _small_tap_model(6)
    final_model = _small_tap_model("final")
    final_model.load_state_dict(layer6_model.state_dict(), strict=True)
    inputs = {
        "t": torch.rand(2),
        "noisy_actions": torch.randn(2, 4, 2),
        "qpos_history": torch.randn(2, 1, 2),
        "dino_features_list": [torch.randn(2, 5, 16)],
    }
    block_outputs = {}
    handle = layer6_model.blocks[5].register_forward_hook(
        lambda _module, _inputs, output: block_outputs.__setitem__("block6", output))
    layer6_output = layer6_model(**inputs)
    handle.remove()
    final_output = final_model(**inputs)

    assert torch.allclose(
        layer6_output["action_hidden"], block_outputs["block6"][:, :4])
    assert torch.allclose(
        layer6_output["final_pred"], final_output["final_pred"], atol=1e-6)
    assert not torch.allclose(
        layer6_output["action_hidden"], layer6_output["final_action_hidden"])

    layer6_output["action_hidden"].square().mean().backward()
    assert any(parameter.grad is not None for parameter in layer6_model.blocks[5].parameters())
    assert all(parameter.grad is None for parameter in layer6_model.blocks[6].parameters())
    assert all(parameter.grad is None for parameter in layer6_model.blocks[11].parameters())


def test_layer_gradient_groups_match_twelve_block_architecture():
    model = _small_tap_model(6)
    groups = build_lava_gradient_parameter_groups(model)
    assert all(groups[name] for name in ("input", "b1_4", "b5_8", "b9_10", "b11_12"))
    block11_ids = {id(parameter) for parameter in model.blocks[10].parameters()}
    assert block11_ids.issubset({id(parameter) for parameter in groups["b11_12"]})
