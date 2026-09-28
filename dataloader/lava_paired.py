"""V7.1 positive-only paired sampling; no extra negative RGB paths."""
import os
import logging
import numpy as np
import torch

from .dataset import LAVABatchScaleBatchSampler


class LAVAPairedBatchSampler(LAVABatchScaleBatchSampler):
    def __init__(self, dataset, batch_size, scales, seed=0, drop_last=True):
        super().__init__(len(dataset), batch_size, scales, seed, drop_last)
        self.dataset = dataset
        count = batch_size * dataset.lava_sample_ratio
        if count != int(count) or int(count) % 2 or not 2 <= count <= batch_size:
            raise ValueError("paired_batch requires an even integral LAVA sample count")
        self.pair_count = int(count) // 2
        self.pools = {}
        for scale in self.scales:
            tasks = {}
            for meta in dataset.episode_metadata:
                # Every legal interval start, including near episode boundaries.
                low = np.searchsorted(dataset.valid_indices, meta['global_start'])
                high = np.searchsorted(dataset.valid_indices,
                    meta['global_start'] + meta['length'] - scale)
                if high <= low:
                    continue
                local = dataset.valid_indices[low:high] - meta['global_start']
                if local[-1] - local[0] < 2 * scale:
                    continue
                uid = os.path.realpath(meta['hdf5_path'])
                tasks.setdefault(meta['task_name'], []).append((uid, low, local))
            if len({ep[0] for eps in tasks.values() for ep in eps}) < self.pair_count:
                raise ValueError(f"Not enough distinct eligible episodes for paired scale {scale}")
            self.pools[scale] = tasks
            logging.getLogger(__name__).info(
                'V7.1 scale=%d eligible_tasks=%d eligible_episodes=%d pairs_per_batch=%d',
                scale, len(tasks), sum(map(len, tasks.values())), self.pair_count)

    def _draw(self, count):
        return int(torch.randint(count, (), generator=self.generator))

    def __iter__(self):
        for ordinary in super().__iter__():
            scale = ordinary[0][1]
            if len(ordinary) < 2 * self.pair_count:
                raise ValueError("paired_batch requires full batches")
            available = {task: list(eps) for task, eps in self.pools[scale].items()}
            paired = []
            for _ in range(self.pair_count):
                task = list(available)[self._draw(len(available))]
                episodes = available[task]
                uid, low, local = episodes[self._draw(len(episodes))]
                for key in list(available):
                    available[key] = [ep for ep in available[key] if ep[0] != uid]
                    if not available[key]:
                        del available[key]
                first = np.flatnonzero((local - local[0] >= 2 * scale)
                                       | (local[-1] - local >= 2 * scale))
                a = int(first[self._draw(len(first))])
                legal = np.flatnonzero(np.abs(local - local[a]) >= 2 * scale)
                if not len(legal):
                    raise RuntimeError("No legal temporal partner in prevalidated episode")
                b = int(legal[self._draw(len(legal))])
                rel_a = self._draw(min(int(local[a]), self.dataset.chunk_size - scale - 1) + 1)
                rel_b = self._draw(min(int(local[b]), self.dataset.chunk_size - scale - 1) + 1)
                paired.extend([(low + a - rel_a, scale, rel_a),
                               (low + b - rel_b, scale, rel_b)])
            # None disables stochastic LAVA selection on the ordinary samples.
            batch = paired + [(idx, s, None) for idx, s in ordinary[len(paired):]]
            order = torch.randperm(len(batch), generator=self.generator).tolist()
            yield [batch[i] for i in order]


def paired_candidate_indices(episode_uids, starts, scales, device):
    """Return self, unique temporal partner, and up to four cross-episode paths."""
    n = len(episode_uids)
    if len(starts) != n or len(scales) != n:
        raise ValueError("Incomplete paired metadata")
    rows = []
    for i, uid in enumerate(episode_uids):
        partners = [j for j in range(n) if j != i and episode_uids[j] == uid]
        if len(partners) != 1:
            raise ValueError("Each paired anchor must have exactly one same-episode partner")
        j = partners[0]
        if int(scales[i]) != int(scales[j]) or abs(int(starts[i]) - int(starts[j])) < 2 * int(scales[i]):
            raise ValueError("Invalid paired temporal separation or scale")
        cross = [k for k in range(n) if episode_uids[k] != uid and int(scales[k]) == int(scales[i])]
        selected = torch.randperm(len(cross))[:4].tolist()
        rows.append([i, j] + [cross[k] for k in selected] + [-1] * (4 - len(selected)))
    return torch.tensor(rows, dtype=torch.long, device=device)
