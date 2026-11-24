"""Relabel local sentiment caches and derive session-disjoint IEMOCAP folds."""
import argparse
from pathlib import Path
import pickle
import re

import numpy as np

from data import BENCHMARKS, SPLITS, load_data
from prepare import atomic_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('data/features'))
    parser.add_argument('--output', type=Path, default=Path('data/extended'))
    parser.add_argument('--mosi', type=Path, required=True)
    parser.add_argument('--sims', type=Path, required=True)
    args = parser.parse_args()
    for name, source in [('cmu_mosi', args.mosi), ('ch_sims', args.sims), ('meld', None), ('iemocap', None)]:
        scores = {}
        if source:
            with source.open('rb') as file:
                original = pickle.load(file, encoding='latin1')
            for block in original.values():
                scores.update(zip(np.asarray(block['id']).astype(str), np.asarray(block['regression_labels']).flatten()))
        blocks = []
        for split in SPLITS:
            with np.load(args.cache / name / f'{split}.npz', allow_pickle=False) as src:
                ids, groups, labels, mask = (src[k] for k in ('ids', 'groups', 'y', 'mask'))
                features = [src[f'x{i}'] for i in range(3)]
            if source:
                values = np.array([scores[i] for i in ids.astype(str)])
                keep = values != 0 if name == 'cmu_mosi' else np.ones(len(ids), dtype=bool)
                labels = (values > 0).astype(int) if name == 'cmu_mosi' else np.digitize(values, [-.6, -.2, .2, .6])
                ids, groups, labels, mask = (x[keep] for x in (ids, groups, labels, mask))
                features = [x[keep] for x in features]
            blocks.append((*features, mask, labels, ids, groups))
        def save(destination, split, block):
            destination.mkdir(parents=True, exist_ok=True)
            atomic_cache(destination / f'{split}.npz', list(block[:3]), *block[3:])
        for split, block in zip(SPLITS, blocks):
            save(args.output / name, split, block)
        load_data(args.output, name)
        if name == 'iemocap':
            merged = [np.concatenate(parts) for parts in zip(*blocks)]
            session = np.array([int(re.match(r'Ses0([1-5])', group)[1]) for group in merged[-1].astype(str)])
            for test in range(1, 6):
                val = test % 5 + 1
                destination = args.output / f'loso{test}'
                for split, keep in zip(SPLITS, ((session != test) & (session != val), session == val, session == test)):
                    save(destination / name, split, [x[keep] for x in merged])
                load_data(destination, name)


if __name__ == '__main__':
    main()
