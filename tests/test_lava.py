import csv
import math
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataloader.dataset import RobotWinTaskDataset, collate_fn
from models.model_runner import VLAWrapper
from models.vla_model_fm import (
    VLAModel,
    normalized_logsignature,
    sample_full_shuffle_permutation,
)
from train import (
    LossLogger,
    build_lava_gradient_parameter_groups,
    compute_shared_gradient_diagnostics,
    should_run_lava_grad_diagnostics,
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
    # Existing per-level normalization remains unchanged and can magnify a
    # small-but-nonzero raw second level to a substantial signature block.
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
        "evolution_start": 2,
        "evolution_scale": 4,
    })
    batch = collate_fn([sample_zero, sample_one])
    assert batch["evolution_batch_indices"].tolist() == [1]
    assert batch["evolution_starts"].tolist() == [2]
    assert batch["evolution_scales"].tolist() == [4]
    assert batch["evolution_pixel_values"][0].shape[0] == 5
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
