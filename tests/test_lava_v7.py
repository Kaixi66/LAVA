import torch
import torch.nn.functional as F
import yaml

from dataloader.dataset import collate_fn
from test_lava_dataset_pipeline import episode_root, make_dataset
from test_lava_film import inputs, small_model


def test_v7_config_changes_only_condition_negatives_and_run_name():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    old = yaml.safe_load((root / "configs/robotwin_lava_v64.yaml").read_text())
    new = yaml.safe_load((root / "configs/robotwin_lava_v7.yaml").read_text())
    assert old["model"]["lava"]["world_condition"]["enabled"] is True
    assert new["model"]["lava"]["world_condition"] == {"enabled": False}
    assert new["training"]["lava_negative_mode"] == "batch"
    assert new["training"]["lava_scale_sampling"] == "batch_uniform"
    assert new["training"]["lava_order_negative"] is False
    for key in ("lava_num_same_episode_negatives", "lava_num_cross_episode_negatives",
                "lava_negative_exclusion_multiplier"):
        old["training"].pop(key)
        assert key not in new["training"]
    old["model"]["lava"]["world_condition"] = {"enabled": False}
    old["training"]["lava_negative_mode"] = "batch"
    old["training"]["checkpoint_tag"] = new["training"]["checkpoint_tag"]
    assert new == old


def test_v7_batch_dataset_loads_only_positive_paths(episode_root):
    dataset = make_dataset(episode_root, mode="batch")
    samples = [dataset[(index, 4)] for index in (0, 1, 2)]
    for sample in samples:
        assert sample["evolution_pixel_values"].shape[0] == 5
        assert "same_episode_negative_pixel_values" not in sample
        assert "temporal_negative_pixel_values" not in sample
        assert "far_negative_pixel_values" not in sample
    batch = collate_fn(samples)
    assert len(batch["evolution_pixel_values"]) == 3
    assert batch["evolution_scales"].tolist() == [4, 4, 4]
    assert batch["temporal_negative_pixel_values"] is None
    assert batch["far_negative_pixel_values"] is None


def test_v7_unconditioned_world_is_encoded_once_and_uses_full_matrix(monkeypatch):
    model = small_model("none")
    model.lava_signature_normalization = "graded_soft"
    model.lava_signature_score = "neg_l2"
    assert model.lava_world_encoder.use_query_film is False
    kwargs = inputs((4, 4, 4))
    kwargs["negative_mode"] = "batch"
    kwargs.pop("temporal_negative_feature_paths")
    kwargs.pop("far_negative_feature_paths")
    kwargs.pop("lava_context")
    seen = []
    handle = model.lava_world_encoder.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].shape[0]))
    captured = []
    original = F.cross_entropy

    def capture(logits, labels, *args, **kwargs):
        captured.append((logits.detach().clone(), labels.detach().clone()))
        return original(logits, labels, *args, **kwargs)

    monkeypatch.setattr(F, "cross_entropy", capture)
    loss, info = model.compute_lava_loss(**kwargs)
    handle.remove()
    assert seen == [15]
    assert len(captured) == 1
    logits, labels = captured[0]
    assert logits.shape == (3, 3)
    torch.testing.assert_close(labels, torch.arange(3))
    assert torch.isfinite(logits).all()
    assert info["lava_sample_count"] == 3
    loss.backward()
    assert kwargs["action_hidden"].grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.lava_world_encoder.parameters()
               if p.grad is not None)
