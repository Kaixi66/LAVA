import random

import torch
import torch.nn as nn

from dataloader.dataset import RobotWinTaskDataset, collate_fn
from models.model_runner import VLAWrapper
from models.vla_model_fm import VLAModel, normalized_logsignature


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


def test_lava_interval_stays_inside_episode_and_action_chunk():
    dataset = RobotWinTaskDataset.__new__(RobotWinTaskDataset)
    dataset.use_lava = True
    dataset.lava_sample_ratio = 1.0
    dataset.lava_scales = (1, 2, 4, 8, 16)
    dataset.lava_scale_sampling = "uniform"
    dataset.lava_scale_probs = None
    dataset.chunk_size = 32

    random.seed(7)
    for _ in range(100):
        start, scale = dataset._sample_lava_interval(local_anchor_idx=93, total_frames=100)
        assert scale in (1, 2, 4)
        assert start >= 0
        assert start + scale <= 6
        assert start + scale <= 31

    assert dataset._sample_lava_interval(local_anchor_idx=99, total_frames=100) is None


def test_collate_keeps_variable_length_paths_and_batch_indices():
    common = {
        "state": torch.zeros(1, 2),
        "action_sequence": torch.zeros(8, 2),
        "pixel_values": torch.zeros(3, 16, 16),
        "state_mask": torch.ones(1, dtype=torch.bool),
        "action_mask": torch.ones(8, dtype=torch.bool),
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
    world_differences = [torch.randn(1, 5, 16), torch.randn(3, 5, 16)]
    # Exercise the same mixed-precision path used on H100. In particular,
    # order-margin diagnostics must safely combine BF16 logits and FP32 values.
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        loss, diagnostics = model.compute_lava_loss(
            action_hidden=action_hidden,
            world_feature_differences=world_differences,
            batch_indices=torch.tensor([0, 2]),
            interval_starts=torch.tensor([0, 2]),
            interval_scales=torch.tensor([1, 3]),
            temperature=0.07,
            order_negative=True,
        )
    assert torch.isfinite(loss)
    assert {
        "pos_sim", "negative_sim", "shuffle_sim", "order_margin",
        "retrieval_acc", "action_pair_sim", "world_pair_sim",
        "lava_sample_count", "lava_order_negative_count",
        "raw_change_norm", "world_residual_std", "action_residual_std",
        "loss_s1", "loss_s4", "order_margin_s4", "loss_t0_025",
    }.issubset(diagnostics)
    loss.backward()
    assert action_hidden.grad is not None
    assert model.lava_world_encoder.output_proj.weight.grad is not None
    assert model.lava_action_projector[-1].weight.grad is not None


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
