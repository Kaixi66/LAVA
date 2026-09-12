import copy
import math
from pathlib import Path
import subprocess
import types

import pytest
import torch
from omegaconf import OmegaConf

import models.vla_model_fm as module
from models.vla_model_fm import graded_soft_logsignature, lava_signature_score
from test_lava_film import inputs
from test_lava_soft_normalization import small_soft_model


@pytest.mark.parametrize('scale', [1, 2, 4, 8, 16])
@pytest.mark.parametrize('amplitude', [0., 1e-8, 1.])
def test_graded_formula_finite_backward_and_structural_level(scale, amplitude):
    d = (torch.randn(128) * amplitude).requires_grad_()
    a = (torch.randn(8128) * amplitude).requires_grad_()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        z, _ = graded_soft_logsignature(d, a, scale)
    effective_a = a if scale > 1 else torch.zeros_like(a)
    s = (1 + d.square().sum().square() + effective_a.square().sum()).pow(.25)
    expected = torch.cat((d / s, effective_a / s.square()))
    torch.testing.assert_close(z, expected)
    assert z.shape == (8256,) and z.dtype == torch.float32
    if scale == 1:
        assert not z[128:].count_nonzero()
    z.sum().backward()
    assert torch.isfinite(d.grad).all()
    if scale > 1:
        assert torch.isfinite(a.grad).all()
    if amplitude == 0:
        assert not z.count_nonzero()
        torch.testing.assert_close(d.grad, torch.ones_like(d))


def test_graded_depth_one_ignores_area_and_weak_area_stays_weak():
    d = torch.randn(3, 128)
    a = torch.randn(3, 8128)
    z, _ = graded_soft_logsignature(d, a, 4, depth=1, rho=2.)
    other, _ = graded_soft_logsignature(d, a * 1e8, 4, depth=1, rho=2.)
    assert z.shape == (3, 128)
    torch.testing.assert_close(z, other, rtol=0, atol=0)
    strong, _ = graded_soft_logsignature(d, a * 1e-2, 4)
    weak, _ = graded_soft_logsignature(d, a * 1e-6, 4)
    assert weak[:, 128:].norm() < strong[:, 128:].norm() * .001
    repeated, _ = graded_soft_logsignature(d, a * 1e-6, 4)
    torch.testing.assert_close(weak, repeated, rtol=0, atol=0)


@pytest.mark.parametrize('rho', [0., -1., float('nan'), float('inf')])
def test_graded_rejects_invalid_rho(rho):
    with pytest.raises(ValueError, match='rho'):
        graded_soft_logsignature(torch.ones(2), torch.ones(1), 2, rho=rho)


@pytest.mark.parametrize('scale', [1, 2, 4, 8, 16])
def test_neg_l2_broadcast_and_exact_match_wins_under_autocast(scale):
    d, area = module._raw_logsignature_levels(torch.randn(scale, 128))
    a = graded_soft_logsignature(d, area, scale)[0][None].repeat(3, 1)
    w = torch.randn(3, 4, 8256)
    w[:, 0] = a
    w[:, 1] = a * 1.2
    w[:, 1, 0] += .001
    with torch.autocast('cpu', dtype=torch.bfloat16):
        scores = lava_signature_score(a[:, None], w, 'neg_l2')
    torch.testing.assert_close(scores, -(a[:, None].float() - w.float()).square().sum(-1))
    assert scores.dtype == torch.float32
    assert torch.equal(scores.argmax(-1), torch.zeros(3, dtype=torch.long))
    assert torch.equal(scores[:, 0], torch.zeros(3))
    assert (scores[:, 1:] < 0).all()
    # Demonstrate the norm-related failure of dot scoring this change targets.
    assert (lava_signature_score(a, w[:, 1]) > lava_signature_score(a, a)).all()


@pytest.mark.parametrize('normalization', ['graded_soft', 'ema_rms_soft'])
@pytest.mark.parametrize('scales', [(1, 1, 1), (2, 4, 8), (16, 16, 16)])
def test_all_candidate_scores_and_checkpoint_gradients(normalization, scales, monkeypatch):
    torch.manual_seed(6309)
    model = small_soft_model().bfloat16()
    model.lava_signature_normalization = normalization
    model.lava_signature_score = 'neg_l2'
    if normalization == 'graded_soft':
        model.lava_signature_calibrator = None
    reference = copy.deepcopy(model)
    kw = inputs(scales)
    kw['action_hidden'] = kw['action_hidden'].detach().bfloat16().requires_grad_()
    kw['lava_context'] = kw['lava_context'].bfloat16()
    kw['interval_starts'].zero_()
    kw['order_negative'] = True
    captured = []
    original = module.lava_signature_score
    def score(a, w, score_type='dot'):
        assert score_type == 'neg_l2'
        value = original(a, w, score_type)
        torch.testing.assert_close(value, -(a.float() - w.float()).square().sum(-1))
        captured.append(value.detach())
        return value
    monkeypatch.setattr(module, 'lava_signature_score', score)
    rng = torch.get_rng_state()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss, info = model.compute_lava_loss(**kw)
    assert len(captured) >= 4  # positive, row candidates, local, far (plus any order)
    buffers = copy.deepcopy(model.state_dict())
    loss.backward()
    for name, value in buffers.items():
        if 'ema_' in name:
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    monkeypatch.setattr(module, 'checkpoint', lambda fn, *args, **kwargs: fn(*args))
    kw2 = copy.deepcopy({k:v.detach().clone().requires_grad_(v.requires_grad) if torch.is_tensor(v) else v for k,v in kw.items()})
    torch.set_rng_state(rng)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        expected, _ = reference.compute_lava_loss(**kw2)
    expected.backward()
    torch.testing.assert_close(loss, expected)
    for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
        if q.grad is not None:
            assert torch.isfinite(p.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, atol=2e-4, rtol=.02, msg=name)
    assert model.lava_action_projector[0].weight.grad.norm() > 0
    assert model.lava_world_encoder.film.weight.grad.norm() > 0


@pytest.mark.parametrize('bf16', [False, True])
def test_disabled_options_match_frozen_v63_loss_and_gradients(bf16):
    """Compare against the actual published baseline, not a second new-code path."""
    old = types.ModuleType('published_v63')
    repo = Path(__file__).resolve().parents[1]
    code = subprocess.check_output(['git', 'show', '7bcfc1e:models/vla_model_fm.py'], cwd=repo, text=True)
    exec(compile(code, 'published_v63.py', 'exec'), old.__dict__)
    torch.manual_seed(6390)
    current = small_soft_model()
    reference = copy.deepcopy(current)
    # Same architecture and state; dispatch all old VLAModel methods and old normalizer.
    reference.__class__ = old.VLAModel
    reference.lava_signature_calibrator.__class__ = old.EMASoftLogSignatureNormalizer
    kw = inputs((4, 4, 4)); kw['order_negative'] = True
    kw2 = copy.deepcopy(kw)
    rng = torch.get_rng_state()
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
        actual, _ = current.compute_lava_loss(**kw)
    actual.backward()
    torch.set_rng_state(rng)
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=bf16):
        expected, _ = reference.compute_lava_loss(**kw2)
    expected.backward()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for (name, a), (_, b) in zip(current.named_parameters(), reference.named_parameters()):
        if b.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)


def test_independent_ablations_and_combined_config():
    root = Path(__file__).parents[1] / 'configs'
    variants = [('norm_only','graded_soft','dot','mixed_batch',True),
                ('score_only','ema_rms_soft','neg_l2','mixed_batch',True),
                ('geometry','graded_soft','neg_l2','mixed_batch',True),
                ('negatives','ema_rms_soft','dot','episode_balanced',False),
                ('order_off','ema_rms_soft','dot','mixed_batch',False),
                ('combined','graded_soft','neg_l2','episode_balanced',False)]
    for name, norm, score, negatives, order in variants:
        cfg = OmegaConf.load(root / 'ablations' / f'robotwin_lava_v63_{name}.yaml')
        assert cfg.model.lava.signature_normalization.type == norm
        assert cfg.model.lava.signature_score == score
        assert cfg.training.lava_negative_mode == negatives
        assert cfg.training.lava_order_negative == order
        assert cfg.training.lava_temperature == .07 and cfg.training.lambda_lava == .01
        assert cfg.training.epochs == 12 and cfg.training.batch_size == 128


def test_factory_graded_options_and_unsupported_combinations():
    from models.model_runner import ModelFactory
    config = OmegaConf.load(Path(__file__).parents[1] / 'configs/robotwin_lava_v63.yaml')
    config.model.action_expert.hidden_size = 32
    config.model.action_expert.depth = 1
    config.model.action_expert.num_heads = 4
    config.model.future_feat.enabled = False
    model = ModelFactory.create_action_model(config, 16, 1)
    assert model.lava_signature_calibrator is None
    assert model.lava_signature_score == 'neg_l2' and model.lava_signature_rho == 1.
    assert not any('ema_' in key for key in model.state_dict())
    restored = ModelFactory.create_action_model(config, 16, 1)
    restored.load_state_dict(model.state_dict(), strict=True)
    for field, value in [('time_channel', True), ('world_encoding', 'feature_delta')]:
        invalid = copy.deepcopy(config)
        invalid.model.lava[field] = value
        with pytest.raises(ValueError, match='state_delta'):
            ModelFactory.create_action_model(invalid, 16, 1)
    config.model.lava.signature_normalization.rho = float('nan')
    with pytest.raises(ValueError, match='rho'):
        ModelFactory.create_action_model(config, 16, 1)
