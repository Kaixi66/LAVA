from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.model_runner import VLAWrapper
from models.vla_model_fm import VLAModel, WorldResidualEncoder, calc_flow_matching_loss


def small_model(condition="query_film"):
    return VLAModel(
        action_dim=14, proprio_dim=2, hidden_dim=32, num_heads=4, depth=1,
        action_len=17, proprio_len=1, num_registers=0,
        dino_feat_dims=(16,), vlm_num_queries=2, adapter_depth=1,
        use_future_feat=False, use_lava=True, lava_dino_feat_dim=16,
        lava_world_encoding="state_delta", lava_time_channel=True,
        lava_world_condition=condition, lava_residual_dim=128, lava_query_dim=16,
        lava_qformer_hidden_dim=32, lava_qformer_num_queries=8,
        lava_qformer_num_layers=2, lava_qformer_num_heads=4,
        lava_signature_normalization="per_level_unit")


def inputs(scales=(2, 2, 2)):
    return dict(
        action_hidden=torch.randn(4, 17, 32, requires_grad=True),
        world_feature_differences=None,
        batch_indices=torch.tensor([3, 1, 2]),
        interval_starts=torch.tensor([8, 2, 0]),
        interval_scales=torch.tensor(scales),
        world_feature_paths=[torch.randn(s + 1, 5, 16) for s in scales],
        temporal_negative_feature_paths=[torch.randn(s + 1, 5, 16) for s in scales],
        far_negative_feature_paths=[torch.randn(s + 1, 5, 16) for s in scales],
        lava_context=torch.randn(4, 16), negative_mode="mixed_batch",
        task_names=["unused", "a", "a", "b"], order_negative=False)


def test_zero_init_preserves_all_existing_parameters_and_loss():
    torch.manual_seed(62)
    legacy = small_model("none")
    torch.manual_seed(62)
    film = small_model()
    for name, value in legacy.state_dict().items():
        torch.testing.assert_close(value, film.state_dict()[name], rtol=0, atol=0)
    kwargs = inputs()
    actual, info = film.compute_lava_loss(**kwargs)
    kwargs.pop("lava_context")
    expected, _ = legacy.compute_lava_loss(**kwargs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    assert info["film_gamma_rms"] == info["film_beta_rms"] == 0
    assert info["film_context_count"] == 3
    actual.backward()
    assert film.lava_world_encoder.film.weight.grad.norm() > 0
    assert torch.isfinite(film.lava_world_encoder.film.weight.grad).all()


@pytest.mark.parametrize("bf16", [False, True])
def test_nonzero_film_changes_states_and_preserves_telescoping(bf16):
    torch.manual_seed(63)
    encoder = WorldResidualEncoder(16, 128, hidden_dim=32, num_queries=8,
                                   query_dim=16, use_query_film=True)
    nn.init.normal_(encoder.film.weight, std=0.02)
    frames = torch.randn(17, 5, 16)
    frames[4] = frames[3]
    context = torch.randn(1, 16).expand(17, -1)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        a = encoder(frames, context)
        b = encoder(frames, context + torch.randn(1, 16))
    assert a.shape == (17, 128) and a.dtype == torch.float32
    assert not torch.allclose(a, b)
    delta = a[1:] - a[:-1]
    assert delta[3].abs().max() < 1e-6
    for length in (1, 2, 4, 8, 16):
        torch.testing.assert_close(delta[:length].sum(0), a[length] - a[0],
                                   atol=2e-6, rtol=1e-5)


def test_context_mapping_uses_policy_anchor_for_all_frames_and_families():
    model = small_model()
    kwargs = inputs((1, 2, 4))
    observed = []
    handle = model.lava_world_encoder.register_forward_pre_hook(
        lambda module, args, kw: observed.append((args[0].detach(), kw["context"].detach())),
        with_kwargs=True)
    with torch.no_grad():
        model.compute_lava_loss(**kwargs)
    handle.remove()
    contexts = kwargs["lava_context"][kwargs["batch_indices"]]
    counts = [2, 3, 5]
    expected = torch.cat([c[None].expand(n, -1) for _ in range(3)
                          for c, n in zip(contexts, counts)])
    torch.testing.assert_close(observed[0][1], expected)
    assert len(observed) == 4
    for (frames, context), anchor in zip(observed[1:], contexts):
        torch.testing.assert_close(frames, torch.cat(kwargs["world_feature_paths"]))
        torch.testing.assert_close(context, anchor[None].expand(sum(counts), -1))


def test_candidate_context_is_row_anchor_not_candidate_owner(monkeypatch):
    import models.vla_model_fm as module
    model = small_model()
    nn.init.normal_(model.lava_world_encoder.film.weight, std=0.1)
    kwargs = inputs()
    recorded = []
    original = module.F.cross_entropy

    def capture(logits, *args, **kw):
        recorded.append(logits.detach().clone())
        return original(logits, *args, **kw)

    monkeypatch.setattr(module.F, "cross_entropy", capture)
    with torch.no_grad():
        model.compute_lava_loss(**kwargs)
        kwargs["lava_context"][1] += torch.randn(16) * 3
        model.compute_lava_loss(**kwargs)
    # Context for policy batch index 1 belongs only to anchor row 1.
    torch.testing.assert_close(recorded[0][0], recorded[1][0], rtol=0, atol=0)
    torch.testing.assert_close(recorded[0][2], recorded[1][2], rtol=0, atol=0)
    finite = torch.isfinite(recorded[0][1])
    assert not torch.allclose(recorded[0][1][finite], recorded[1][1][finite])


def test_checkpointed_rows_backward_and_serialization():
    model = small_model()
    nn.init.normal_(model.lava_world_encoder.film.weight, std=0.01)
    kwargs = inputs((4, 4, 4))
    kwargs["order_negative"] = True
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, info = model.compute_lava_loss(**kwargs)
    assert info["state_endpoint_error"] < 2e-6
    loss.backward()
    for parameter in model.lava_world_encoder.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert model.lava_world_encoder.context_norm.weight.grad.norm() > 0
    assert kwargs["action_hidden"].grad.norm() > 0
    assert kwargs["action_hidden"].grad[:, 0].count_nonzero() == 0
    restored = small_model()
    restored.load_state_dict(model.state_dict(), strict=True)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name])


@pytest.mark.parametrize("multicamera", [False, True])
def test_current_context_reuses_dino_and_excludes_cls_registers(multicamera):
    class Vision(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, pixel_values, **kwargs):
            assert not torch.is_grad_enabled()
            self.calls += 1
            values = pixel_values[:, 0, 0, 0]
            patches = values[:, None, None].expand(-1, 3, 16)
            special = torch.full((len(values), 2, 16), 10000.)
            return SimpleNamespace(hidden_states=[torch.cat((special, patches), dim=1)])

    wrapper = VLAWrapper.__new__(VLAWrapper)
    nn.Module.__init__(wrapper)
    wrapper.device, wrapper.dtype = "cpu", torch.float32
    wrapper.vision_encode_batch_size = 3
    wrapper.lava_target_layer = 0
    wrapper.feat_layers = [0]
    wrapper.num_register_tokens = 1
    wrapper.include_cls_register = True
    wrapper.vision_encoder = Vision()
    n = 8 if multicamera else 4
    pixels = torch.arange(n).float()[:, None, None, None].expand(-1, 3, 2, 2)
    if multicamera:
        pixels = pixels.reshape(4, 2, 3, 2, 2)
    features, context = wrapper.get_vision_features(pixels, return_lava_context=True)
    expected = torch.arange(4).float() * (2 if multicamera else 1)
    torch.testing.assert_close(context, expected[:, None].expand(4, 16))
    assert wrapper.vision_encoder.calls == (n + 2) // 3
    assert not context.requires_grad and features[0].shape[0] == 4


def test_flow_loss_passes_context_and_missing_context_fails():
    model = small_model()
    kwargs = inputs()
    context = kwargs.pop("lava_context")
    with pytest.raises(ValueError, match="current-observation context"):
        model.compute_lava_loss(**kwargs)
    loss, info = calc_flow_matching_loss(
        model, x1=torch.randn(4, 17, 14), dino_features_list=[torch.randn(4, 5, 16)],
        qpos_history=torch.randn(4, 1, 2), use_lava=True, lambda_lava=.01,
        lava_context=context, world_feature_paths=kwargs["world_feature_paths"],
        temporal_negative_feature_paths=kwargs["temporal_negative_feature_paths"],
        far_negative_feature_paths=kwargs["far_negative_feature_paths"],
        lava_batch_indices=kwargs["batch_indices"], lava_interval_starts=kwargs["interval_starts"],
        lava_interval_scales=kwargs["interval_scales"], lava_negative_mode="mixed_batch",
        task_names=kwargs["task_names"])
    assert info["film_context_count"] == 3 and torch.isfinite(loss)
    loss.backward()
    assert model.lava_world_encoder.film.weight.grad.norm() > 0
