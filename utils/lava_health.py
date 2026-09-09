"""Fail fast when an enabled auxiliary branch stops receiving supervision."""
import math


class LAVAHealthMonitor:
    def __init__(self, enabled, max_empty_batches=100):
        self.enabled = enabled
        self.max_empty_batches = int(max_empty_batches)
        if self.max_empty_batches < 1:
            raise ValueError('max_empty_batches must be positive')
        self.empty_batches = 0

    def check(self, batch, info):
        if not self.enabled:
            return
        paths = batch.get('evolution_pixel_values')
        expected = len(paths) if paths is not None else 0
        actual = int(info.get('lava_sample_count', 0))
        if actual != expected:
            raise RuntimeError(f'LAVA branch dropped supervision: dataset={expected}, loss={actual}')
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
