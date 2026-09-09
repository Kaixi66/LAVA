"""HDF5 -> __getitem__ -> collate regressions for optional action weighting."""
import cv2
import h5py
import numpy as np
import pytest
import torch
from dataloader.dataset import RobotWinTaskDataset, collate_fn
from utils.lava_health import LAVAHealthMonitor

@pytest.fixture
def episode_root(tmp_path):
    folder = tmp_path / 'adjust_bottle' / 'demo_clean' / 'data'
    folder.mkdir(parents=True)
    rng = np.random.default_rng(42)
    with h5py.File(folder / 'episode0.hdf5', 'w') as f:
        f['joint_action/vector'] = rng.normal(size=(128, 14)).astype('float32')
        for side in ('left', 'right'):
            f[f'endpose/{side}_endpose'] = rng.normal(size=(128, 6)).astype('float32')
            f[f'endpose/{side}_gripper'] = rng.normal(size=128).astype('float32')
        images = f.create_dataset('observation/head_camera/rgb', (128,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
        for i in range(128):
            images[i] = cv2.imencode('.png', rng.integers(0, 256, (16, 16, 3), dtype=np.uint8))[1]
    return str(tmp_path)

def make_dataset(root, weighting=False, mode='mixed_batch'):
    return RobotWinTaskDataset(root,
        indices_config=dict(state_indices=[0], action_indices=list(range(32)), camera_indices=[0]),
        camera_names=['head_camera'], image_size=(16, 16), task_set='all',
        use_lava=True, lava_sample_ratio=1., lava_negative_mode=mode,
        lava_action_similarity_weighting=weighting)

@pytest.mark.parametrize('weighting', [False, True])
@pytest.mark.parametrize('mode', ['episode_local', 'mixed', 'mixed_batch'])
@pytest.mark.parametrize('scale', [1, 2, 4, 8, 16])
def test_hdf5_optional_negative_actions(episode_root, weighting, mode, scale):
    dataset = make_dataset(episode_root, weighting, mode)
    sample = dataset[(0, scale)]
    assert sample['evolution_pixel_values'].shape == (scale + 1, 3, 16, 16)
    prefixes = ['temporal_negative'] + (['far_negative'] if mode != 'episode_local' else [])
    for prefix in prefixes:
        assert sample[prefix + '_pixel_values'].shape[0] == scale + 1
        assert (prefix + '_actions' in sample) == weighting
        if weighting:
            assert sample[prefix + '_actions'].shape == (scale + 1, 14)
    batch = collate_fn([sample, dataset[(1, scale)]])
    assert batch['evolution_scales'].tolist() == [scale, scale]
    for prefix in prefixes:
        assert (batch[prefix + '_actions'] is not None) == weighting

def test_loading_error_is_not_recursively_resampled(episode_root, monkeypatch):
    dataset = make_dataset(episode_root)
    calls = []
    def broken(*args):
        calls.append(1)
        raise KeyError('missing_required_data')
    monkeypatch.setattr(dataset, '_load_hdf5_data', broken)
    with pytest.raises(RuntimeError, match='missing_required_data') as error:
        dataset[(0, 4)]
    assert len(calls) == 1
    assert isinstance(error.value.__cause__, KeyError)

def test_workers_preserve_unweighted_lava_paths(episode_root):
    dataset = make_dataset(episode_root)
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=[[(0, 4), (1, 4)]],
                                         num_workers=2, collate_fn=collate_fn)
    batch = next(iter(loader))
    assert len(batch['evolution_pixel_values']) == 2
    assert len(batch['temporal_negative_pixel_values']) == 2
    assert len(batch['far_negative_pixel_values']) == 2
    assert batch['temporal_negative_actions'] is None
    assert batch['far_negative_actions'] is None

def test_health_monitor_rejects_silent_disablement():
    monitor = LAVAHealthMonitor(True, max_empty_batches=3)
    for _ in range(2): monitor.check({}, {'lava_sample_count': 0})
    with pytest.raises(RuntimeError, match='no supervision'):
        monitor.check({}, {'lava_sample_count': 0})
    batch = {'evolution_pixel_values': [torch.ones(2)]}
    with pytest.raises(RuntimeError, match='dropped supervision'):
        LAVAHealthMonitor(True).check(batch, {'lava_sample_count': 0})
    with pytest.raises(RuntimeError, match='autograd'):
        LAVAHealthMonitor(True).check(batch, {'lava_sample_count': 1, 'loss_lava': 1., '_loss_lava_tensor': torch.tensor(1.)})
    monitor.check(batch, {'lava_sample_count': 1, 'loss_lava': 1., '_loss_lava_tensor': torch.tensor(1., requires_grad=True)})
    assert monitor.empty_batches == 0


def test_requested_frames_cannot_silently_disappear(episode_root, monkeypatch):
    dataset = make_dataset(episode_root)
    original = dataset._load_hdf5_data
    def missing(*args):
        result = original(*args)
        del result['evolution_frames']
        return result
    monkeypatch.setattr(dataset, '_load_hdf5_data', missing)
    with pytest.raises(RuntimeError, match='Requested LAVA frames missing'):
        dataset[(0, 4)]
