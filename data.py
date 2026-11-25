"""Leakage-checked feature cache and deterministic controlled interventions."""
from pathlib import Path

import numpy as np
import torch


SPLITS = ('train', 'val', 'test')
BENCHMARKS = {
    'cmu_mosi': ('text', 'audio', 'video'),
    'ch_sims': ('text', 'audio', 'video'),
    'meld_small': ('text', 'audio'),
    'iemocap': ('text', 'audio', 'video'),
    'upmc_food101': ('image', 'text'),
    'n24news': ('image', 'text'),
    'meld': ('text', 'audio', 'video'),
    'cmu_mosei': ('text', 'audio', 'video'),
    'bhm': ('image', 'text'),
    'pad_ufes20': ('image', 'clinical'),
    'uci_har': ('accelerometer', 'gyroscope'),
}


def load_data(root, name):
    directory = Path(root) / name
    seen_groups, seen_ids, data = set(), set(), {}
    for split in SPLITS:
        path = directory / f'{split}.npz'
        if not path.is_file():
            raise FileNotFoundError(f'Missing {path}. Run prepare.py first.')
        with np.load(path, allow_pickle=False) as src:
            ids, groups = src['ids'].astype(str), src['groups'].astype(str)
            views = [src[f'x{i}'].astype(np.float32) for i in range(len(BENCHMARKS[name]))]
            mask, y = src['mask'].astype(bool), src['y'].astype(np.int64)
        if y.ndim != 1 or not len(y) or mask.shape != (len(y), len(views)) or not mask.any(1).all():
            raise ValueError(f'Invalid labels or availability mask in {path}')
        if (len(ids) != len(y) or len(groups) != len(y) or len(set(ids)) != len(y)
                or seen_groups.intersection(groups) or seen_ids.intersection(ids)):
            raise ValueError(f'Duplicate sample IDs or group shared across splits in {path}')
        seen_groups.update(groups)
        seen_ids.update(ids)
        if any(x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all() for x in views):
            raise ValueError(f'Invalid feature arrays in {path}')
        data[split] = (views, mask, y)
    train, train_mask, train_y = data['train']
    classes = int(train_y.max()) + 1
    if classes < 2 or set(train_y) != set(range(classes)):
        raise ValueError('Training labels must be consecutive from 0 with at least two classes.')
    for i, x in enumerate(train):
        present = x[train_mask[:, i]]
        if not len(present):
            raise ValueError(f'No observed training samples for modality {BENCHMARKS[name][i]}')
        mean, scale = present.mean(0), present.std(0)
        scale[scale < 1e-5] = 1
        for views, mask, y in data.values():
            if y.min() < 0 or y.max() >= classes:
                raise ValueError('A holdout label is absent from train.')
            views[i] = np.where(mask[:, i, None], (views[i] - mean) / scale, 0)
    return {k: ([torch.from_numpy(x) for x in xs], torch.from_numpy(m), torch.from_numpy(y))
            for k, (xs, m, y) in data.items()}, classes


def random_index(mask, rng):
    return torch.rand(mask.shape, generator=rng).masked_fill(~mask, -1).argmax(1)


def inject(clean, mask, strength, rng, kind='gaussian'):
    polluted = []
    for i, x in enumerate(clean):
        amount = (strength[:, i] * mask[:, i])[:, None]
        if kind == 'gaussian':
            value = x + amount * torch.randn(x.shape, generator=rng)
        else:
            dropped = torch.rand(x.shape, generator=rng) < amount
            value = x.masked_fill(dropped, 0)
        polluted.append(value)
    return polluted


def training_batch(clean, original_mask, rng):
    keep = original_mask & (torch.rand(original_mask.shape, generator=rng) >= 0.15)
    empty = ~keep.any(1)
    keep[empty, random_index(original_mask, rng)[empty]] = True
    levels = torch.tensor([0, 0.2, 0.5, 0.9])
    strength = levels[torch.randint(0, len(levels), keep.shape, generator=rng)] * keep
    return inject(clean, keep, strength, rng), keep, strength


def condition(data, kind, argument=None):
    clean, original, labels = data
    rng = torch.Generator().manual_seed(20261122)
    mask = original.clone()
    strength = torch.zeros_like(mask, dtype=torch.float32)
    swaps = torch.zeros_like(mask)
    is_cross_label = torch.zeros(len(labels), dtype=torch.bool)
    inputs = clean
    rows = torch.arange(len(mask))
    if kind in {'gaussian', 'unseen_dropout', 'only_noise'}:
        picked = random_index(mask, rng)
        if kind == 'only_noise':
            picked.fill_(argument)
        amount = 0.5 if kind == 'only_noise' else argument
        strength[rows, picked] = amount * mask[rows, picked]
        noise = 'dropout' if kind == 'unseen_dropout' else 'gaussian'
        inputs = inject(clean, mask, strength, rng, noise)
    elif kind == 'random_missing':
        mask &= torch.rand(mask.shape, generator=rng) >= argument
        lost = ~mask.any(1)
        picked = random_index(original, rng)
        mask[rows[lost], picked[lost]] = True
    elif kind == 'missing':
        mask[:, argument] = False
        lost = ~mask.any(1)
        mask[lost] = original[lost]
    elif kind == 'only':
        mask.zero_()
        mask[:, argument] = original[:, argument]
    elif kind == 'conflict':
        inputs = [x.clone() for x in clean]
        target_mod = random_index(mask, rng)
        is_cross_label = torch.rand(len(labels), generator=rng) < 0.5
        for i in range(mask.shape[1]):
            present = mask[:, i].nonzero().flatten()
            groups = {int(c): present[labels[present] == c] for c in labels.unique()}
            candidates = {c: present[labels[present] != c] for c in groups}
            for row in ((target_mod == i) & (mask.sum(1) > 1)).nonzero().flatten().tolist():
                c = int(labels[row])
                pool = candidates[c] if is_cross_label[row] else groups[c][groups[c] != row]
                if len(pool):
                    donor = pool[torch.randint(len(pool), (), generator=rng)]
                    inputs[i][row] = clean[i][donor]
                    swaps[row, i] = True
    return inputs, mask, strength, swaps, is_cross_label

