"""Rebuild IEMOCAP LOSO text features from transcripts without emotion labels."""
from argparse import ArgumentParser
from pathlib import Path
import pickle

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from data import SPLITS, load_data
from prepare import atomic_cache


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('transcripts', type=Path, help='Trusted DialogueGCN IEMOCAP_features.pkl')
    parser.add_argument('--cache', type=Path, default=Path('data/extended'))
    parser.add_argument('--output', type=Path, default=Path('data/crib_iemocap'))
    args = parser.parse_args()
    with args.transcripts.open('rb') as handle:
        _, _, _, _, _, _, utterances, _, _ = pickle.load(handle, encoding='latin1')
    vectorizer = HashingVectorizer(n_features=512, ngram_range=(1, 2), token_pattern=r'(?u)\b\w+\b',
                                  dtype=np.float32, alternate_sign=True)
    for fold in range(1, 6):
        source = args.cache / f'loso{fold}' / 'iemocap'
        out = args.output / f'loso{fold}'
        dest = out / 'iemocap'
        dest.mkdir(parents=True, exist_ok=True)
        for split in SPLITS:
            with np.load(source / f'{split}.npz', allow_pickle=False) as src:
                mask, y, ids, groups = (src[k] for k in ('mask', 'y', 'ids', 'groups'))
                transcripts = [utterances[dialogue][int(index)]
                               for dialogue, index in (sid.rsplit(':', 1) for sid in ids.astype(str))]
                text = vectorizer.transform(transcripts).toarray()
                mask[:, 0] = (text != 0).any(1)
                keep = mask.any(1)
                views = [x[keep] for x in (text, src['x1'], src['x2'])]
                atomic_cache(dest / f'{split}.npz', views, mask[keep], y[keep], ids[keep], groups[keep])
        data, _ = load_data(out, 'iemocap')
        print(f'LOSO {fold}: ' + ', '.join(f'{split}={len(data[split][2])}' for split in SPLITS), flush=True)


if __name__ == '__main__':
    main()
