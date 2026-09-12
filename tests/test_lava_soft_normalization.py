import copy
import io
import math

import pytest
import torch

from models.vla_model_fm import EMASoftLogSignatureNormalizer, VLAModel
from test_lava_film import inputs


def small_soft_model():
    return VLAModel(
        action_dim=14, proprio_dim=2, hidden_dim=32, num_heads=4, depth=1,
        action_len=17, proprio_len=1, num_registers=0,
        dino_feat_dims=(16,), vlm_num_queries=2, adapter_depth=1,
        use_future_feat=False, use_lava=True, lava_dino_feat_dim=16,
        lava_world_encoding="state_delta", lava_time_channel=False,
        lava_world_condition="query_film", lava_residual_dim=128, lava_query_dim=16,
        lava_qformer_hidden_dim=32, lava_qformer_num_queries=8,
        lava_qformer_num_layers=2, lava_qformer_num_heads=4,
        lava_signature_normalization="ema_rms_soft")


@pytest.mark.parametrize("scale", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("size", [0.0, 1e-10])
def test_zero_first_batch_and_tiny_paths_do_not_amplify_gradients(scale, size):
    normalizer = EMASoftLogSignatureNormalizer()
    one = torch.full((128,), size, requires_grad=True)
    two = torch.full((8128,), size, requires_grad=True)
    normalizer.update("world", [one * 0], [two * 0], [scale])
    signature, info = normalizer(one, two, "world", scale)
    assert signature.shape == (8256,)
    assert signature.norm() <= math.sqrt(8256) * size * 1.01
    assert info["level1_ema_rms"] == pytest.approx(math.sqrt(.99))
    if scale == 1:
        assert torch.count_nonzero(signature[128:]) == 0
        assert not normalizer.ema_initialized[1, 1, 0]
    signature.sum().backward()
    assert torch.isfinite(one.grad).all() and one.grad.abs().max() < 1.01
    if scale > 1:
        assert torch.isfinite(two.grad).all() and two.grad.abs().max() < 1.01


def test_shared_population_reference_preserves_weakness_and_fixed_level_weight():
    normalizer = EMASoftLogSignatureNormalizer(scales=(2,), momentum=.5)
    one, two = torch.tensor([3., 4.]), torch.tensor([6., 8., 0.])
    normalizer.update("world", [one], [two], [2])
    before = normalizer.ema_squared_norm.clone()
    strong, _ = normalizer(one, two, "world", 2)
    weak, _ = normalizer(one * 1e-6, two * 1e-6, "world", 2)
    normalizer(one * 1e6, two * 1e6, "world", 2)
    assert torch.equal(before, normalizer.ema_squared_norm)
    assert weak.norm() < 2e-6 and strong.norm() < 1
    assert strong.norm() > weak.norm() * 1e5
    expected = torch.cat((one / math.sqrt(25 + 13),
                          two / math.sqrt(100 + 50.5))) / math.sqrt(2)
    torch.testing.assert_close(strong, expected)


def test_fp32_statistics_survive_bf16_casts_and_checkpoint_round_trip():
    normalizer = EMASoftLogSignatureNormalizer()
    normalizer.ema_squared_norm.fill_(1.234567)
    expected = normalizer.ema_squared_norm.clone()
    normalizer.bfloat16()
    assert normalizer.ema_squared_norm.dtype == torch.float32
    assert torch.equal(expected, normalizer.ema_squared_norm)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        one, two = torch.randn(8), torch.randn(28)
        normalizer.update("action", [one], [two], [2])
        signature, _ = normalizer(one, two, "action", 2)
    assert signature.dtype == torch.float32
    stream = io.BytesIO()
    torch.save(normalizer.state_dict(), stream)
    stream.seek(0)
    restored = EMASoftLogSignatureNormalizer().bfloat16()
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    torch.testing.assert_close(restored(one, two, "action", 2)[0], signature, rtol=0, atol=0)


@pytest.mark.parametrize("scale", [1, 2, 4, 8, 16])
def test_static_paths_keep_negative_mass_and_finite_bf16_backward(scale):
    torch.manual_seed(6301)
    model = small_soft_model().bfloat16().train()
    kwargs = inputs((scale,) * 3)
    kwargs["action_hidden"] = kwargs["action_hidden"].detach().bfloat16().requires_grad_()
    kwargs["lava_context"] = kwargs["lava_context"].bfloat16()
    kwargs["interval_starts"].zero_()
    frame = torch.randn(1, 5, 16)
    for family in ("world_feature_paths", "temporal_negative_feature_paths", "far_negative_feature_paths"):
        kwargs[family] = [frame.repeat(scale + 1, 1, 1) for _ in range(3)]
    kwargs["order_negative"] = True
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, info = model.compute_lava_loss(**kwargs)
    assert loss.detach().item() == pytest.approx(math.log(6 if scale >= 4 else 5), abs=2e-5)
    assert info["lava_sample_count"] == info["film_context_count"] == 3
    assert info["batch_candidate_count"] == 2
    assert info["lava_order_negative_count"] == (3 if scale >= 4 else 0)
    (.01 * loss).backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.float().norm() < .1


def test_checkpoint_recomputation_uses_forward_snapshot_and_does_not_update_ema():
    torch.manual_seed(6302)
    model = small_soft_model()
    reference = copy.deepcopy(model)
    kwargs = inputs()
    reference_kwargs = {key: value.detach().clone().requires_grad_(value.requires_grad)
                        if isinstance(value, torch.Tensor) else value
                        for key, value in kwargs.items()}
    actual, _ = model.compute_lava_loss(**kwargs)
    expected, _ = reference.compute_lava_loss(**reference_kwargs)
    after_first = model.lava_signature_calibrator.ema_squared_norm.clone()
    with torch.no_grad():
        model.compute_lava_loss(**inputs())
    after_second = model.lava_signature_calibrator.ema_squared_norm.clone()
    assert not torch.equal(after_first, after_second)
    actual.backward()
    expected.backward()
    assert torch.equal(after_second, model.lava_signature_calibrator.ema_squared_norm)
    torch.testing.assert_close(kwargs["action_hidden"].grad, reference_kwargs["action_hidden"].grad)
    for (name, parameter), (_, target) in zip(model.named_parameters(), reference.named_parameters()):
        if target.grad is not None:
            torch.testing.assert_close(parameter.grad, target.grad, msg=name, atol=2e-6, rtol=2e-5)


def test_invalid_population_energy_fails_instead_of_silently_clamping():
    normalizer = EMASoftLogSignatureNormalizer()
    with pytest.raises(FloatingPointError, match="Degenerate EMA"):
        normalizer.update("world", [torch.full((8,), float("nan"))], [torch.zeros(28)], [2])
