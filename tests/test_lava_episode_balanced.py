import copy
import math
import random
import sys

import pytest
import torch
import torch.nn.functional as F

import models.vla_model_fm as module
from dataloader.dataset import collate_fn
from models.vla_model_fm import sample_cross_episode_indices
from utils.lava_health import LAVAHealthMonitor
from test_lava_dataset_pipeline import episode_root, make_dataset
from test_lava_film import inputs
from test_lava_soft_normalization import small_soft_model


@pytest.mark.parametrize('scale', [1, 2, 4, 8, 16])
@pytest.mark.parametrize('position', [0, 24, 47])
def test_same_episode_sampling_boundaries_and_shortages(episode_root, scale, position):
    dataset = make_dataset(episode_root, mode='episode_balanced')
    position = min(position, 64 - scale - 1)
    legal = {n for n in range(64 - scale) if abs(n - position) >= 2 * scale}
    for _ in range(4):
        starts = dataset._sample_same_episode_starts(position, scale, 64)
        assert len(starts) == len(set(starts)) == min(4, len(legal))
        assert set(starts) <= legal
    if scale == 16 and position == 24:
        assert starts == []


@pytest.mark.parametrize('scale', [1, 2, 4, 8, 16])
def test_hdf5_ragged_paths_use_absolute_positive_start(episode_root, scale, monkeypatch):
    dataset = make_dataset(episode_root, mode='episode_balanced')
    monkeypatch.setattr(dataset, '_sample_lava_interval', lambda *a, **kw: (7, scale))
    sample = dataset[(0, scale)]
    assert sample['evolution_absolute_start'] == sample['evolution_observation_start'] + 7
    assert sample['evolution_episode_uid'].endswith('/adjust_bottle/demo_clean/data/episode0.hdf5')
    assert len(sample['same_episode_negative_pixel_values']) == 4
    for start, path in zip(sample['same_episode_negative_starts'], sample['same_episode_negative_pixel_values']):
        assert abs(start - sample['evolution_absolute_start']) >= 2 * scale
        assert path.shape == (scale + 1, 3, 16, 16)
    assert 'temporal_negative_pixel_values' not in sample
    assert 'far_negative_pixel_values' not in sample
    # A no-candidate positive still survives collate beside a full candidate row.
    empty = copy.deepcopy(sample)
    empty['same_episode_negative_starts'] = []
    empty['same_episode_negative_pixel_values'] = []
    batch = collate_fn([empty, sample])
    assert len(batch['evolution_pixel_values']) == 2
    assert list(map(len, batch['same_episode_negative_pixel_values'])) == [0, 4]
    assert batch['temporal_negative_pixel_values'] is None and batch['far_negative_pixel_values'] is None


def test_cross_episode_unique_physical_paths_and_scale():
    uids = ['/task_a/clean/episode0', '/task_a/clean/episode0',
            '/task_a/clean/episode1', '/task_b/clean/episode0',
            '/task_b/clean/episode0', '/task_b/clean/episode0', '/task_c/random/episode0']
    starts = [0, 16, 0, 0, 0, 12, 0]
    scales = [4, 4, 4, 4, 4, 4, 8]
    rows = sample_cross_episode_indices(uids, starts, scales)
    assert set(rows[0]) == {2, 3, 5}  # same task allowed, duplicate path omitted
    assert rows[6] == []
    for i, row in enumerate(rows):
        keys = [(uids[j], starts[j], scales[j]) for j in row]
        assert len(keys) == len(set(keys)) <= 4
        assert all(uids[j] != uids[i] and scales[j] == scales[i] for j in row)
    many = [f'/task/episode{i}' for i in range(9)]
    assert all(len(row) == 4 for row in sample_cross_episode_indices(many, [0]*9, [16]*9))


def balanced_inputs(scales=(4, 4, 4), same_counts=(4, 4, 4)):
    kw = inputs(scales)
    kw.pop('temporal_negative_feature_paths')
    kw.pop('far_negative_feature_paths')
    kw['interval_starts'].zero_()
    kw['negative_mode'] = 'episode_balanced'
    kw['same_episode_negative_features'] = [
        [torch.randn(scale + 1, 5, 16) for _ in range(n)] for scale, n in zip(scales, same_counts)]
    kw['lava_episode_uids'] = ['/task_a/episode0', '/task_a/episode1', '/task_b/episode0']
    kw['lava_absolute_starts'] = [10, 20, 30]
    return kw


def capture_objective(model, kwargs):
    captured = {}
    code = model._episode_balanced_objective.__func__.__code__
    previous = sys.getprofile()
    def profile(frame, event, arg):
        if event == 'return' and frame.f_code is code:
            for key in ('logits','valid','cross_indices','same_counts','cross_counts'):
                captured[key] = frame.f_locals[key]
    try:
        sys.setprofile(profile)
        loss, info = model.compute_lava_loss(**kwargs)
    finally:
        sys.setprofile(previous)
    return loss, info, captured


@pytest.mark.parametrize('scale', [1, 2, 4, 8, 16])
@pytest.mark.parametrize('normalization,score', [('ema_rms_soft','dot'),('graded_soft','neg_l2')])
def test_exact_ce_candidates_context_and_backward(scale, normalization, score, monkeypatch):
    torch.manual_seed(6355)
    model = small_soft_model().bfloat16()
    model.lava_signature_normalization = normalization
    model.lava_signature_score = score
    if normalization == 'graded_soft':
        model.lava_signature_calibrator = None
    kw = balanced_inputs((scale,) * 3, (0, 2, 4))
    kw['action_hidden'] = kw['action_hidden'].detach().bfloat16().requires_grad_()
    kw['lava_context'] = kw['lava_context'].bfloat16()
    contexts = []
    handle = model.lava_world_encoder.register_forward_pre_hook(
        lambda m, args, kwargs: contexts.append(kwargs['context'].detach()), with_kwargs=True)
    calls = []
    original = module.sample_cross_episode_indices
    def sampler(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(module, 'sample_cross_episode_indices', sampler)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss, info, capture = capture_objective(model, kw)
    assert capture['same_counts'] == [0, 2, 4] and capture['cross_counts'] == [2, 2, 2]
    assert torch.isfinite(capture['logits']).sum(1).tolist() == [3, 5, 7]
    expected = F.cross_entropy(capture['logits'].float() / .07, torch.zeros(3, dtype=torch.long))
    torch.testing.assert_close(loss, expected)
    assert info['lava_order_negative_count'] == 0
    assert info['lava_sample_count'] == info['lava_scored_anchor_count'] == 3
    assert len(contexts) == 4
    for i, context in enumerate(contexts[1:]):
        expected_context = kw['lava_context'][kw['batch_indices'][i]]
        torch.testing.assert_close(context, expected_context[None].expand(len(context), -1))
    before = copy.deepcopy(model.state_dict())
    (.01 * loss).backward()
    assert len(calls) == 1  # activation recomputation did not sample again
    for name, value in before.items():
        if 'ema_' in name:
            torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    handle.remove()
    for parameter in [model.lava_action_projector[0].weight, model.lava_world_encoder.film.weight,
                      model.lava_world_encoder.output_proj.weight]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.norm() > 0


@pytest.mark.parametrize('some_valid', [False, True])
def test_missing_candidates_preserve_positives_ema_and_health(some_valid):
    model = small_soft_model()
    kw = balanced_inputs((16,16,16), (0, int(some_valid), 0))
    kw['lava_episode_uids'] = ['/same/episode'] * 3
    loss, info, captured = capture_objective(model, kw)
    assert info['lava_sampled_anchor_count'] == info['lava_encoded_anchor_count'] == 3
    assert info['lava_scored_anchor_count'] == int(some_valid)
    assert info['lava_no_negative_anchor_count'] == 3 - int(some_valid)
    # All legal positives update the modality/scale references, including skipped rows.
    assert model.lava_signature_calibrator.ema_initialized[:, :, -1].all()
    if some_valid:
        expected = F.cross_entropy(captured['logits'][1:2] / .07, torch.zeros(1, dtype=torch.long))
        torch.testing.assert_close(loss, expected)
    else:
        assert loss.item() == 0 and loss.requires_grad
    info.update(loss_lava=loss.item(), _loss_lava_tensor=loss)
    monitor = LAVAHealthMonitor(True)
    monitor.check({'evolution_pixel_values': [1,2,3]}, info)
    assert monitor.no_candidate_batches == int(not some_valid)
    loss.backward()
    assert model.lava_action_projector[0].weight.grad is not None
    assert model.lava_world_encoder.film.weight.grad is not None
    bad = dict(info, lava_scored_anchor_count=3)
    if not some_valid:
        with pytest.raises(RuntimeError, match='accounting'):
            monitor.check({'evolution_pixel_values': [1,2,3]}, bad)


@pytest.mark.parametrize('normalization', ['ema_rms_soft','graded_soft'])
def test_balanced_checkpoint_disabled_matches_gradients(normalization, monkeypatch):
    torch.manual_seed(6371)
    model = small_soft_model()
    model.lava_signature_normalization = normalization
    model.lava_signature_score = 'neg_l2'
    if normalization == 'graded_soft': model.lava_signature_calibrator = None
    reference = copy.deepcopy(model)
    kw = balanced_inputs((2, 4, 16), (1,2,3)); kw2 = copy.deepcopy(kw)
    rng = torch.get_rng_state()
    loss, _ = model.compute_lava_loss(**kw)
    loss.backward()
    monkeypatch.setattr(module, 'checkpoint', lambda fn, *args, **kwargs: fn(*args))
    torch.set_rng_state(rng)
    expected, _ = reference.compute_lava_loss(**kw2)
    expected.backward()
    torch.testing.assert_close(loss, expected, rtol=0, atol=0)
    for (name,p),(_,q) in zip(model.named_parameters(),reference.named_parameters()):
        if q.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-5,atol=2e-6,msg=name)


def test_balanced_rejects_old_order_and_paired_candidates():
    model = small_soft_model()
    kw = balanced_inputs()
    with pytest.raises(ValueError, match='order'):
        model.compute_lava_loss(**dict(kw,order_negative=True))
    with pytest.raises(ValueError, match='paired'):
        model.compute_lava_loss(**dict(kw,temporal_negative_feature_paths=kw['world_feature_paths']))


def test_full_four_plus_four_has_exactly_nine_logits():
    model = small_soft_model()
    kw = balanced_inputs((2,)*9, (4,)*9)
    kw.update(action_hidden=torch.randn(9,17,32,requires_grad=True),
              batch_indices=torch.arange(9), interval_starts=torch.zeros(9,dtype=torch.long),
              lava_context=torch.randn(9,16), task_names=['same_task']*9,
              lava_episode_uids=[f'/same_task/episode{i}' for i in range(9)],
              lava_absolute_starts=[0]*9)
    loss, info, captured = capture_objective(model,kw)
    assert captured['logits'].shape == (9,9) and torch.isfinite(captured['logits']).all()
    assert info['same_episode_negative_count'] == info['cross_episode_negative_count'] == 4
    torch.testing.assert_close(loss,F.cross_entropy(captured['logits']/.07,torch.zeros(9,dtype=torch.long)))


def test_zero_candidates_leave_full_policy_flow_and_future_loss_unchanged():
    from models.vla_model_fm import calc_flow_matching_loss
    from torch import nn
    model = small_soft_model()
    model.future_feat_decoder = nn.Linear(32,16)
    kw = balanced_inputs((2,2,2),(0,0,0))
    common = dict(model=model,x1=torch.randn(4,17,14),dino_features_list=[torch.randn(4,5,16)],
                  qpos_history=torch.randn(4,1,2),use_future_feat=True,
                  future_feat_target=torch.randn(4,6,16),lambda_future_feat=.5)
    # Decoder gets the one proprio token and two adapted vision queries.
    with torch.no_grad():
        shape = model(torch.ones(4),noisy_actions=common['x1'],dino_features_list=common['dino_features_list'],
                      qpos_history=common['qpos_history'])['cond_tokens'].shape
    common['future_feat_target'] = torch.randn(shape[0],shape[1],16)
    rng = torch.get_rng_state()
    base, before = calc_flow_matching_loss(**common,use_lava=False)
    torch.set_rng_state(rng)
    actual, after = calc_flow_matching_loss(**common,use_lava=True,lambda_lava=.01,
        lava_negative_mode='episode_balanced',lava_order_negative=False,
        world_feature_paths=kw['world_feature_paths'],same_episode_negative_features=[[],[],[]],
        lava_episode_uids=['/same/episode']*3,lava_absolute_starts=kw['lava_absolute_starts'],
        lava_batch_indices=kw['batch_indices'],lava_interval_starts=kw['interval_starts'],
        lava_interval_scales=kw['interval_scales'],lava_context=kw['lava_context'])
    assert after['lava_scored_anchor_count'] == 0 and after['lava_sample_count'] == 3
    torch.testing.assert_close(actual,base,rtol=0,atol=0)
    assert after['loss_mse'] == before['loss_mse'] and after['loss_future_feat'] == before['loss_future_feat']
    actual.backward()
    assert model.output_proj.weight.grad.norm() > 0
    assert model.future_feat_decoder.weight.grad.norm() > 0
