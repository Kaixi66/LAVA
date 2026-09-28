import copy
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from dataloader.lava_paired import LAVAPairedBatchSampler, paired_candidate_indices
from dataloader.dataset import collate_fn
from test_lava_dataset_pipeline import episode_root, make_dataset
from test_lava_film import small_model


def fake_dataset():
    return FakeDataset()


class FakeDataset:
    lava_sample_ratio = .125
    chunk_size = 32
    valid_indices = np.arange(12 * 256)
    episode_metadata = [dict(global_start=i * 256, length=256,
        task_name=str(i % 3), hdf5_path=f'/episode/{i}') for i in range(12)]

    def __len__(self):
        return len(self.valid_indices)


def test_sampler_pairs_all_scales_and_resume():
    sampler = LAVAPairedBatchSampler(fake_dataset(), 128, [1, 2, 4, 8, 16], seed=42)
    iterator = iter(sampler)
    for _ in range(5):
        batch = next(iterator)
        paired = [(idx, scale, rel) for idx, scale, rel in batch if rel is not None]
        assert len(paired) == 16
        assert len({scale for _, scale, _ in batch}) == 1
        uids = [idx // 256 for idx, _, _ in paired]
        starts = [idx % 256 + rel for idx, _, rel in paired]
        scales = [scale for _, scale, _ in paired]
        candidates = paired_candidate_indices(uids, starts, scales, 'cpu')
        assert candidates.shape == (16, 6)
        assert (candidates >= 0).all()
        for i, row in enumerate(candidates.tolist()):
            assert row[0] == i and uids[row[1]] == uids[i]
            assert all(uids[j] != uids[i] for j in row[2:])
            assert len(set(row)) == 6
    state = copy.deepcopy(sampler.state_dict())
    expected = next(iterator)
    resumed = LAVAPairedBatchSampler(fake_dataset(), 128, [1, 2, 4, 8, 16])
    resumed.load_state_dict(state)
    assert next(iter(resumed)) == expected


def test_dataset_forced_positive_only(episode_root):
    ds = make_dataset(episode_root, mode='paired_batch')
    samples = [ds[(0, 4, 2)], ds[(20, 4, 3)], ds[(40, 4, None)]]
    batch = collate_fn(samples)
    assert batch['evolution_absolute_start'] == [2, 23]
    assert batch['evolution_batch_indices'].tolist() == [0, 1]
    assert len(batch['evolution_pixel_values']) == 2
    assert batch['same_episode_negative_pixel_values'] == [[], []]
    assert batch['temporal_negative_pixel_values'] is None
    assert batch['far_negative_pixel_values'] is None
    assert batch['action_sequence'].shape[0] == 3


@pytest.mark.parametrize('scale', [1, 2, 4, 8, 16])
def test_paired_loss_manual_formula_and_single_encoding(monkeypatch, scale):
    model = small_model('none')
    model.lava_signature_normalization = 'graded_soft'
    model.lava_signature_score = 'neg_l2'
    hidden = torch.randn(8, 17, 32, requires_grad=True)
    calls, captured = [], []
    hook = model.lava_world_encoder.register_forward_pre_hook(
        lambda module, args: calls.append(args[0].shape[0]))
    original = F.cross_entropy
    def capture(logits, labels, **kw):
        captured.append(logits)
        assert not labels.any()
        return original(logits, labels, **kw)
    monkeypatch.setattr(F, 'cross_entropy', capture)
    loss, info = model.compute_lava_loss(action_hidden=hidden,
        world_feature_differences=None, batch_indices=torch.arange(8),
        interval_starts=torch.zeros(8, dtype=torch.long),
        interval_scales=torch.full((8,), scale),
        world_feature_paths=[torch.randn(scale + 1, 5, 16) for _ in range(8)],
        negative_mode='paired_batch', order_negative=False,
        lava_episode_uids=['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
        lava_absolute_starts=[0, 40] * 4)
    assert calls == [8 * (scale + 1)]
    assert captured[0].shape == (8, 6)
    expected = (torch.logsumexp(captured[0], 1) - captured[0][:, 0]).mean()
    torch.testing.assert_close(loss, expected)
    assert info['paired_hard_count'] == 1 and info['paired_cross_count'] == 4
    assert abs(sum(info[f'paired_{kind}_probability'] for kind in ('hard', 'cross', 'positive')) - 1) < 1e-5
    loss.backward()
    assert torch.isfinite(hidden.grad).all() and hidden.grad.norm() > 0
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.lava_world_encoder.parameters())
    hook.remove()


def test_invalid_partners_rejected_and_cross_shortage():
    with pytest.raises(ValueError):
        paired_candidate_indices(['a', 'b'], [0, 40], [4, 4], 'cpu')
    with pytest.raises(ValueError):
        paired_candidate_indices(['a', 'a'], [0, 4], [4, 4], 'cpu')
    result = paired_candidate_indices(['a', 'a'], [0, 40], [4, 4], 'cpu')
    assert result.tolist() == [[0, 1, -1, -1, -1, -1], [1, 0, -1, -1, -1, -1]]


def test_config_keeps_v7_representation():
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parents[1]
    old = yaml.safe_load((root / 'configs/robotwin_lava_v7.yaml').read_text())
    new = yaml.safe_load((root / 'configs/robotwin_lava_v71.yaml').read_text())
    old['training']['lava_negative_mode'] = 'paired_batch'
    old['training']['checkpoint_tag'] = new['training']['checkpoint_tag']
    assert old == new


def test_worker_collation_keeps_pairs(episode_root):
    ds = make_dataset(episode_root, mode='paired_batch')
    ds.lava_sample_ratio = .5
    sampler = LAVAPairedBatchSampler(ds, 4, [1, 2, 4, 8, 16], seed=7)
    loader = torch.utils.data.DataLoader(ds, batch_sampler=sampler,
        num_workers=2, collate_fn=collate_fn)
    for index, batch in enumerate(loader):
        assert len(batch['evolution_pixel_values']) == 2
        result = paired_candidate_indices(batch['evolution_episode_uid'],
            batch['evolution_absolute_start'], batch['evolution_scales'], 'cpu')
        assert (result[:, 1] >= 0).all()
        if index == 4:
            break
