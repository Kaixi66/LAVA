"""Fail fast when an enabled auxiliary branch stops receiving supervision."""
import math


class LAVAHealthMonitor:
    def __init__(self, enabled, max_empty_batches=100):
        self.enabled = enabled
        self.max_empty_batches = int(max_empty_batches)
        if self.max_empty_batches < 1:
            raise ValueError('max_empty_batches must be positive')
        self.empty_batches = 0
        self.no_candidate_batches = 0

    def check(self, batch, info):
        if not self.enabled:
            return
        paths = batch.get('evolution_pixel_values')
        expected = len(paths) if paths is not None else 0
        actual = int(info.get('lava_sample_count', 0))
        if actual != expected:
            raise RuntimeError(f'LAVA branch dropped supervision: dataset={expected}, loss={actual}')
        if 'paired_hard_count' in info:
            if (actual % 2 or info['paired_hard_count'] != 1
                    or info['paired_world_path_count'] != actual
                    or info['paired_cross_count'] != min(4, actual - 2)
                    or info['paired_distance_over_l'] < 2):
                raise RuntimeError('Invalid paired positive/candidate accounting')
            for key in ('paired_hard_margin', 'paired_hard_probability',
                        'paired_cross_probability', 'paired_positive_probability'):
                if not math.isfinite(float(info[key])):
                    raise RuntimeError(f'Non-finite paired diagnostic: {key}')
        if info.get('episode_balanced_mode'):
            sampled = int(info.get('lava_sampled_anchor_count', -1))
            encoded = int(info.get('lava_encoded_anchor_count', -1))
            scored = int(info.get('lava_scored_anchor_count', -1))
            unavailable = int(info.get('lava_no_negative_anchor_count', -1))
            if sampled != expected or encoded != expected or not 0 <= scored <= encoded or scored + unavailable != encoded:
                raise RuntimeError('Invalid sampled/encoded/scored episode_balanced accounting')
            self.no_candidate_batches = self.no_candidate_batches + 1 if expected and not scored else 0
            if not scored and float(info['loss_lava']) != 0.0:
                raise RuntimeError('No-candidate LAVA batch must return a connected zero loss')
        if actual:
            self.empty_batches = 0
            if not math.isfinite(float(info['loss_lava'])):
                raise RuntimeError('LAVA loss is non-finite')
            loss = info.get('_loss_lava_tensor')
            if loss is None or not loss.requires_grad:
                raise RuntimeError('LAVA loss is disconnected from autograd')
        else:
            self.empty_batches += 1
            if self.empty_batches >= self.max_empty_batches:
                raise RuntimeError(
                    f'LAVA enabled but no supervision for {self.empty_batches} consecutive batches; '
                    'inspect dataset sampling and collate instead of continuing base-only training')
