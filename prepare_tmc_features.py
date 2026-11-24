"""Convert the published MMSA and DialogueGCN features used by validate_tmc."""
from argparse import ArgumentParser
from pathlib import Path
import pickle

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from prepare import atomic_cache, pooled


SPLITS = ('train', 'val', 'test')


def mmsa(source, name):
    blocks = []
    for key in ('train', 'valid', 'test'):
        block = source[key]
        labels = np.sign(np.asarray(block['regression_labels']).reshape(-1)).astype(np.int64) + 1
        views = [pooled(block[k], block.get(k + '_lengths'))[0] for k in ('text', 'audio', 'vision')]
        ids = np.asarray(block['id']).astype(str)
        groups = np.array([item.split('$_$')[0] for item in ids])
        blocks.append((views, labels, ids, groups))
    if name != 'ch_sims':
        return blocks
    views = [np.concatenate(items) for items in zip(*(b[0] for b in blocks))]
    labels, ids, groups = (np.concatenate(items) for items in zip(*(b[1:] for b in blocks)))
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate CH-SIMS IDs')
    splitter = StratifiedGroupKFold(5, shuffle=True, random_state=20261122)
    folds = [hold for _, hold in splitter.split(np.zeros(len(labels)), labels, groups)]
    indices = (np.concatenate(folds[:3]), folds[3], folds[4])
    return [([x[index] for x in views], labels[index], ids[index], groups[index]) for index in indices]


def iemocap(source):
    _, _, labels, text, audio, video, _, train, test = source
    train, test = sorted(train), sorted(test)

    def group(key):
        return str(key).split('_')[0]

    a, b = next(GroupShuffleSplit(n_splits=1, test_size=.1, random_state=20261122)
                .split(train, groups=[group(k) for k in train]))
    blocks = []
    for keys in ([train[i] for i in a], [train[i] for i in b], test):
        views = [np.concatenate([np.asarray(m[k])[:len(labels[k])] for k in keys])
                 for m in (text, audio, video)]
        y = np.concatenate([labels[k] for k in keys])
        ids = np.asarray([f'{k}:{i}' for k in keys for i in range(len(labels[k]))])
        groups = np.asarray([group(k) for k in keys for _ in labels[k]])
        blocks.append((views, y, ids, groups))
    return blocks


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=('cmu_mosi', 'ch_sims', 'iemocap'))
    parser.add_argument('source', type=Path, help='Trusted dataset publisher pickle')
    parser.add_argument('--output', type=Path, default=Path('data/features'))
    args = parser.parse_args()
    with args.source.open('rb') as file:
        source = pickle.load(file, encoding='latin1')
    blocks = iemocap(source) if args.dataset == 'iemocap' else mmsa(source, args.dataset)
    output = args.output / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    for split, (views, labels, ids, groups) in zip(SPLITS, blocks):
        views = [np.asarray(x, dtype=np.float32) for x in views]
        mask = np.stack([np.isfinite(x).all(1) & (x != 0).any(1) for x in views], axis=1)
        keep = mask.any(1)
        atomic_cache(output / f'{split}.npz', [np.nan_to_num(x[keep]) for x in views],
                     mask[keep], np.asarray(labels, dtype=np.int64)[keep], ids[keep], groups[keep])
        print(args.dataset, split, int(keep.sum()), mask.sum(0).tolist(), flush=True)


if __name__ == '__main__':
    main()
