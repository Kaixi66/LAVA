import copy
import numpy as np
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


@pytest.mark.parametrize('scale', [1, 2, 4, 8, 16])
def test_one_same_episode_family_diagnostics_match_logits(scale):
    torch.manual_seed(711)
    model = small_soft_model()
    tasks = ['task_a'] * 5 + ['task_b'] * 4
    kw = balanced_inputs((scale,) * 9, (1,) * 9)
    kw.update(action_hidden=torch.randn(9, 17, 32, requires_grad=True),
              batch_indices=torch.arange(9), interval_starts=torch.zeros(9, dtype=torch.long),
              lava_context=torch.randn(9, 16), task_names=tasks,
              lava_episode_uids=[f'/{task}/episode{i}' for i, task in enumerate(tasks)],
              lava_absolute_starts=[0] * 9)
    loss, info, captured = capture_objective(model, kw)
    logits = captured['logits'].float()
    assert logits.shape == (9, 6) and torch.isfinite(logits).all()
    p = torch.softmax(logits / .07, dim=1)
    torch.testing.assert_close(info['same_episode_mass_fraction'],
                               (p[:, 1].sum() / p[:, 1:].sum()).item(), rtol=1e-5, atol=1e-6)
    assert info['same_episode_mass_uniform'] == pytest.approx(0.2)
    hardest = (logits[:, 1] > logits[:, 2:].max(1).values).float().mean().item()
    assert info['hardest_is_same_episode'] == pytest.approx(hardest)
    assert info[f'hardest_is_same_episode_s{scale}'] == pytest.approx(hardest)
    assert info['same_episode_acc'] == pytest.approx((logits[:, 0] > logits[:, 1]).float().mean().item())
    assert info['cross_episode_acc'] == pytest.approx(
        (logits[:, 0] > logits[:, 2:].max(1).values).float().mean().item())
    same_task = [[tasks[j] == tasks[i] for j in row] for i, row in enumerate(captured['cross_indices'])]
    fraction = sum(map(sum, same_task)) / sum(map(len, same_task))
    assert info['same_task_cross_fraction'] == pytest.approx(fraction)
    for key in ('positive_probability', 'same_task_cross_acc', 'cross_task_cross_acc',
                'same_episode_probability_per_candidate', 'cross_episode_probability_per_candidate'):
        assert math.isfinite(info[key])
    (.01 * loss).backward()
    assert kw['action_hidden'].grad is not None and kw['action_hidden'].grad.norm() > 0


def v712_model(reuse):
    from models.vla_model_fm import VLAModel
    return VLAModel(
        action_dim=14, proprio_dim=2, hidden_dim=32, num_heads=4, depth=1,
        action_len=17, proprio_len=1, num_registers=0,
        dino_feat_dims=(16,), vlm_num_queries=2, adapter_depth=1,
        use_future_feat=False, use_lava=True, lava_dino_feat_dim=16,
        lava_world_encoding="state_delta", lava_time_channel=True,
        lava_world_condition="none", lava_residual_dim=128, lava_query_dim=16,
        lava_qformer_hidden_dim=32, lava_qformer_num_queries=8,
        lava_qformer_num_layers=2, lava_qformer_num_heads=4,
        lava_signature_normalization="graded_soft", lava_signature_score="neg_l2",
        lava_reuse_world_signatures=reuse)


@pytest.mark.parametrize('scale', [1, 2, 4, 8, 16])
def test_reused_world_signatures_match_per_anchor_encoding(scale):
    torch.manual_seed(712)
    slow, fast = v712_model(False), v712_model(True)
    fast.load_state_dict(slow.state_dict())
    tasks = ['task_a'] * 5 + ['task_b'] * 4
    kw = balanced_inputs((scale,) * 9, (1, 1, 1, 1, 0, 1, 1, 1, 1))
    kw.pop('lava_context')
    kw.update(batch_indices=torch.arange(9), interval_starts=torch.zeros(9, dtype=torch.long),
              task_names=tasks, lava_absolute_starts=[0] * 9,
              lava_episode_uids=[f'/{task}/episode{i}' for i, task in enumerate(tasks)])
    hidden = torch.randn(9, 17, 32)
    results = []
    for model in (slow, fast):
        calls = []
        handle = model.lava_world_encoder.register_forward_hook(lambda *args: calls.append(1))
        kw['action_hidden'] = hidden.clone().requires_grad_()
        torch.manual_seed(7)  # identical cross-episode candidate draws
        loss, info, captured = capture_objective(model, kw)
        loss.backward()
        handle.remove()
        results.append((loss, info, captured, kw['action_hidden'].grad, len(calls),
                        {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}))
    (l0, i0, c0, g0, n0, p0), (l1, i1, c1, g1, n1, p1) = results
    assert c0['cross_indices'] == c1['cross_indices']
    torch.testing.assert_close(c0['logits'], c1['logits'], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(l0, l1, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(g0, g1, rtol=1e-4, atol=1e-6)
    assert set(p0) == set(p1) and any('lava_world_encoder' in n for n in p1)
    for name in p0:
        torch.testing.assert_close(p0[name], p1[name], rtol=1e-4, atol=1e-5)  # summation order only
    # Old path: positives + per-anchor rows, each row recomputed once more by checkpointing.
    assert n1 == 2 and n0 == 1 + 2 * 9
    assert i1['episode_balanced_world_path_count'] == 9 + 8
    assert i0['episode_balanced_world_path_count'] == 9 + 8 + sum(c0['cross_counts'])


def test_reuse_rejects_query_film():
    from models.vla_model_fm import VLAModel
    with pytest.raises(ValueError, match='requires world_condition disabled'):
        VLAModel(action_dim=14, proprio_dim=2, hidden_dim=32, num_heads=4, depth=1,
                 action_len=17, proprio_len=1, num_registers=0, dino_feat_dims=(16,),
                 vlm_num_queries=2, adapter_depth=1, use_future_feat=False, use_lava=True,
                 lava_dino_feat_dim=16, lava_world_encoding="state_delta",
                 lava_world_condition="query_film", lava_residual_dim=128, lava_query_dim=16,
                 lava_qformer_hidden_dim=32, lava_qformer_num_queries=8,
                 lava_reuse_world_signatures=True)


def near_filtered_dataset(root):
    from dataloader.dataset import RobotWinTaskDataset
    return RobotWinTaskDataset(root,
        indices_config=dict(state_indices=[0], action_indices=list(range(32)), camera_indices=[0]),
        camera_names=['head_camera'], image_size=(16, 16), task_set='all',
        use_lava=True, lava_sample_ratio=1., lava_negative_mode='episode_balanced',
        lava_num_same_episode_negatives=1, lava_same_episode_sampling='near_filtered')


@pytest.mark.parametrize('scale', [1, 2, 4, 8, 16])
def test_near_filtered_prefers_nearest_behaviourally_different_start(episode_root, scale):
    dataset = near_filtered_dataset(episode_root)
    path = dataset.episode_metadata[0]['hdf5_path']
    rng = np.random.default_rng(scale)
    dataset._episode_action_cache[path] = rng.normal(size=(128, 14)).astype('float32')
    position = 60
    for _ in range(10):
        starts, info = dataset._sample_near_filtered_start(position, scale, 128, path)
        assert len(starts) == 1 and info['band'] == 0
        assert 2 * scale <= abs(starts[0] - position) < 4 * scale
        assert info['action_rel'] >= 0.25
        assert info['action_rel'] == pytest.approx(
            dataset._action_relation(dataset._episode_action_cache[path], position, starts[0], scale))


@pytest.mark.parametrize('scale', [1, 2, 4])
def test_near_filtered_rejects_identical_motion_and_falls_back(episode_root, scale):
    dataset = near_filtered_dataset(episode_root)
    path = dataset.episode_metadata[0]['hdf5_path']
    actions = np.random.default_rng(0).normal(size=(128, 14)).astype('float32')
    actions[:96] = np.arange(96, dtype='float32')[:, None] * 0.1  # constant velocity: identical motion
    dataset._episode_action_cache[path] = actions
    position = 30  # near and mid bands lie wholly in the constant-velocity segment
    for _ in range(10):
        starts, info = dataset._sample_near_filtered_start(position, scale, 128, path)
        assert info['band'] == 2 and info['rejected'] > 0
        assert abs(starts[0] - position) >= 8 * scale and info['action_rel'] >= 0.25
    dataset._episode_action_cache[path] = np.arange(128, dtype='float32')[:, None].repeat(14, 1)
    starts, info = dataset._sample_near_filtered_start(position, scale, 128, path)
    assert starts == [] and info['band'] == -1 and info['rejected'] == info['checked'] > 0


def test_near_filtered_sample_and_collate_carry_sampling_info(episode_root):
    dataset = near_filtered_dataset(episode_root)
    samples = [dataset[(i, 2)] for i in range(3)]
    for sample in samples:
        assert len(sample['same_episode_negative_starts']) <= 1
        assert set(sample['same_episode_sampling_info']) == {'band', 'action_rel', 'rejected', 'checked'}
    batch = collate_fn(samples)
    assert len(batch['same_episode_sampling_info']) == 3
    uniform = make_dataset(episode_root, mode='episode_balanced')
    assert 'same_episode_sampling_info' not in uniform[(0, 2)]
    assert collate_fn([uniform[(0, 2)]])['same_episode_sampling_info'] == [None]


def test_near_filtered_requires_exactly_one_negative(episode_root):
    from dataloader.dataset import RobotWinTaskDataset
    with pytest.raises(ValueError, match='exactly one'):
        RobotWinTaskDataset(episode_root,
            indices_config=dict(state_indices=[0], action_indices=list(range(32)), camera_indices=[0]),
            camera_names=['head_camera'], image_size=(16, 16), task_set='all', use_lava=True,
            lava_negative_mode='episode_balanced', lava_same_episode_sampling='near_filtered')


def test_band_accuracy_diagnostics_match_logits():
    torch.manual_seed(713)
    model = v712_model(True)
    tasks = ['task_a'] * 5 + ['task_b'] * 4
    kw = balanced_inputs((4,) * 9, (1,) * 9)
    kw.pop('lava_context')
    ratios = [[2.5], [3.0], [5.0], [6.0], [7.5], [9.0], [20.0], [2.0], [40.0]]
    kw.update(action_hidden=torch.randn(9, 17, 32), batch_indices=torch.arange(9),
              interval_starts=torch.zeros(9, dtype=torch.long), task_names=tasks,
              lava_absolute_starts=[0] * 9, same_episode_distance_ratios=ratios,
              lava_episode_uids=[f'/{task}/episode{i}' for i, task in enumerate(tasks)])
    _, info, captured = capture_objective(model, kw)
    logits = captured['logits'].float()
    wins = (logits[:, 0] > logits[:, 1]).float()
    for name, rows in (('near', [0, 1, 7]), ('mid', [2, 3, 4]), ('far', [5, 6, 8])):
        assert info[f'same_episode_acc_{name}'] == pytest.approx(wins[rows].mean().item())
        assert info[f'same_episode_band_fraction_{name}'] == pytest.approx(1 / 3)
