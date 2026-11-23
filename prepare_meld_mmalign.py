"""Convert the official MM-Align MELD train/dev/test pickles to feature caches."""
from argparse import ArgumentParser
from hashlib import file_digest
import json
from pathlib import Path
import pickle

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from prepare import atomic_cache


def read_split(path, split):
    with path.open('rb') as file:
        source = pickle.load(file)
    ids, groups, text, audio, video, labels = [], [], [], [], [], []
    for key, item in sorted(source.items()):
        dialogue, _ = key.split('_', 1)
        ids.append(f'{split}:{key}')
        groups.append(f'{split}:{dialogue}')
        text.append(' '.join(str(int(token)) for token in item['token_ids']))
        for target, field in ((audio, 'audio_features'), (video, 'video_features')):
            values = np.asarray(item[field], dtype=np.float32)
            usable = np.isfinite(values).all(1) & (values != 0).any(1)
            target.append(values[usable].mean(0) if usable.any() else np.zeros(values.shape[1], np.float32))
        labels.append(item['label'])
    return ids, groups, text, np.stack(audio), np.stack(video), np.asarray(labels, dtype=np.int64)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path, help='Directory containing trusted author-released pickles')
    parser.add_argument('--output', type=Path, default=Path('data/features'))
    args = parser.parse_args()
    destination = args.output / 'meld'
    destination.mkdir(parents=True, exist_ok=True)
    tfidf = TfidfVectorizer(token_pattern=r'\b\d+\b', dtype=np.float32, min_df=2)
    reducer = TruncatedSVD(128, random_state=20261122)
    info = {}
    # The author release uses split-local dialogue and utterance indices.
    for split, stem in [('train', 'train'), ('val', 'dev'), ('test', 'test')]:
        path = args.source / f'{stem}.pkl'
        ids, groups, text, audio, video, labels = read_split(path, split)
        if split == 'train':
            texts = reducer.fit_transform(tfidf.fit_transform(text))
        else:
            texts = reducer.transform(tfidf.transform(text))
        views = [texts, audio, video]
        mask = np.stack([(x != 0).any(1) for x in views], axis=1)
        keep = mask.any(1)
        atomic_cache(destination / f'{split}.npz', [x[keep] for x in views], mask[keep],
                     labels[keep], np.asarray(ids)[keep], np.asarray(groups)[keep])
        with path.open('rb') as file:
            fingerprint = file_digest(file, 'sha256').hexdigest()
        info[split] = dict(rows=int(keep.sum()), available=mask.sum(0).tolist(), source_sha256=fingerprint)
        print(split, info[split], flush=True)
    (destination / 'provenance.json').write_text(json.dumps(info, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
