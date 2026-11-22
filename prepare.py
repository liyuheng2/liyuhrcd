"""Build fixed-dimensional feature caches from official splits or documented metadata."""
from argparse import ArgumentParser
from collections import Counter, defaultdict
from csv import DictReader
from hashlib import file_digest
import json
from pathlib import Path
import pickle
import shutil
import subprocess
from urllib.parse import urlsplit
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageOps
from scipy.signal import stft
from scipy.fft import dct
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
import torch

from data import BENCHMARKS, SPLITS


MOSEI_SHA256 = 'ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f'


def media_index(root, extensions):
    indexed = defaultdict(list)
    for path in root.rglob('*'):
        if path.suffix.lower() in extensions and path.is_file():
            indexed[path.stem].append(path)
    return indexed


def food_rows(root, splits):
    records = {}
    common = []
    for split, original in zip(SPLITS, ('train', 'dev', 'test')):
        with (splits / f'{original}.jsonl').open(encoding='utf-8') as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        records[split] = [(str(r['id']), str(r['id']), r['label'], r['text'], Path(r['img'])) for r in rows]
        common.append(records[split][0][-1])
    roots = [root, *sorted({p.parent for p in root.rglob('images') if p.is_dir()})]
    image_root = next((base for base in roots if any((base / p).is_file() for p in common)), root)
    return {k: [(i, g, y, t, image_root / p) for i, g, y, t, p in rows]
            for k, rows in records.items()}


def news_rows(root, metadata):
    rows = json.loads(metadata.read_text(encoding='utf-8'))
    images = media_index(root, {'.jpg', '.jpeg', '.png', '.webp'})
    parents = {}

    def representative(key):
        while key in parents:
            key = parents[key]
        return key

    pairs = []
    for row in rows:
        article = urlsplit(row['article_url']).path
        image = urlsplit(row['image_url']).path
        a, b = representative(('article', article)), representative(('image', image))
        if a != b:
            parents[b] = a
        pairs.append(('article', article))
    records = []
    for row, key in zip(rows, pairs):
        identity = str(row['image_id'])
        paths = images.get(identity, [])
        if len(paths) > 1:
            raise ValueError(f'Multiple N24News images have ID {identity}. Use one image directory.')
        text = ' '.join(row[k] for k in ('headline', 'abstract') if row.get(k))
        records.append((identity, repr(representative(key)), row['section'], text, paths[0] if paths else None))
    labels = [r[2] for r in records]
    groups = [r[1] for r in records]
    folds = [hold for _, hold in StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=20261122)
             .split(np.zeros(len(rows)), labels, groups)]
    return {'train': [records[i] for i in sorted(np.concatenate(folds[2:]))],
            'val': [records[i] for i in sorted(folds[0])],
            'test': [records[i] for i in sorted(folds[1])]}


def meld_rows(root, annotations):
    videos = media_index(root, {'.mp4'})
    records = {}
    for split, original in zip(SPLITS, ('train', 'dev', 'test')):
        rows = []
        with (annotations / f'{original}_sent_emo.csv').open(encoding='utf-8-sig', errors='replace', newline='') as handle:
            for r in DictReader(handle):
                dialogue, utterance = r['Dialogue_ID'], r['Utterance_ID']
                clip = f'dia{dialogue}_utt{utterance}'
                matches = videos[clip]
                accepted = {'train': {'train', 'train_splits'}, 'dev': {'dev', 'dev_splits', 'dev_splits_complete'},
                            'test': {'test', 'test_splits', 'output_repeated_splits_test'}}[original]
                candidates = [p for p in matches if accepted.intersection(p.relative_to(root).parts[:-1])]
                if len(candidates) > 1:
                    raise ValueError(f'Multiple {original} MELD videos for {clip}')
                identity = f'{original}:{clip}'
                rows.append((identity, f'{original}:{dialogue}', r['Emotion'], r['Utterance'],
                             candidates[0] if candidates else None))
        records[split] = rows
    return records


class VisionEncoder:
    def __init__(self, device):
        from torchvision.models import resnet18, ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.transform = weights.transforms()
        self.net = resnet18(weights=weights)
        self.net.fc = torch.nn.Identity()
        self.net.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def encode(self, media):
        frames, owners = [], []
        for row, images in enumerate(media):
            for im in images:
                frames.append(self.transform(im))
                owners.append(row)
        output = np.zeros((len(media), 512), dtype=np.float32)
        counts = np.bincount(owners, minlength=len(media))
        if frames:
            x = self.net(torch.stack(frames).to(self.device)).cpu().numpy()
            np.add.at(output, owners, x)
        return output / np.maximum(counts, 1).astype(np.float32)[:, None], counts > 0


def open_image(path):
    if path is None:
        return []
    try:
        with Image.open(path) as im:
            return [ImageOps.exif_transpose(im).convert('RGB')]
    except (OSError, ValueError):
        return []


def decode(command):
    try:
        return subprocess.run(command, capture_output=True, check=True, timeout=60).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return b''


def open_video(path):
    if path is None:
        return [], None
    command = ['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1', '-filter_threads', '1', '-i', str(path)]
    video = decode(command + ['-an', '-vf', 'fps=1:start_time=0,scale=224:224',
                              '-frames:v', '6', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'])
    if not video:
        video = decode(command + ['-an', '-vf', 'scale=224:224', '-frames:v', '1',
                                  '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-'])
    audio = decode(command + ['-vn', '-t', '20', '-ac', '1', '-ar', '16000', '-f', 'f32le', '-'])
    pixels = np.frombuffer(video, np.uint8)
    frames = [] if not len(pixels) else [
        Image.fromarray(frame) for frame in pixels.reshape(-1, 224, 224, 3)[::2]
    ]
    wave = np.frombuffer(audio, dtype='<f4')
    if len(wave) < 512:
        return frames, None
    # Fixed log-spectral cepstral summary, untrained, one vector per utterance.
    _, _, spectrum = stft(wave, fs=16000, nperseg=512, noverlap=352)
    bands = np.add.reduceat(np.abs(spectrum) ** 2, np.arange(0, 256, 8), axis=0)
    cepstrum = dct(np.log(bands.clip(1e-10)), type=2, axis=0, norm='ortho')[:20]
    return frames, np.r_[cepstrum.mean(1), cepstrum.std(1)]


def atomic_cache(path, views, mask, labels, ids, groups):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as handle:
        np.savez(handle, mask=mask, y=labels, ids=np.asarray(ids, dtype=str), groups=np.asarray(groups, dtype=str),
                 **{f'x{i}': np.asarray(x, dtype=np.float32) for i, x in enumerate(views)})
    temp.replace(path)


def make_media_cache(records, out, name, device):
    out.mkdir(parents=True, exist_ok=True)
    labels = {label: i for i, label in enumerate(sorted({r[2] for r in records['train']}))}
    text_model = TfidfVectorizer(max_features=40000, ngram_range=(1, 2), sublinear_tf=True)
    matrix = text_model.fit_transform(r[3] or '' for r in records['train'])
    dimensions = min(256, min(matrix.shape) - 1)
    if dimensions < 1:
        raise ValueError('Insufficient training text vocabulary.')
    reduce_text = TruncatedSVD(dimensions, random_state=11).fit(matrix)
    vision = VisionEncoder(device)
    is_video = name == 'meld'
    if is_video and shutil.which('ffmpeg') is None:
        raise RuntimeError('MELD requires ffmpeg on PATH. See README.')
    with ThreadPoolExecutor(max_workers=4) as workers:
        for split, rows in records.items():
            ids, groups, y, strings, paths = zip(*rows)
            text_mask = np.array([bool((s or '').strip()) for s in strings])
            text = reduce_text.transform(text_model.transform(s or '' for s in strings))
            output = np.zeros((len(rows), 512), dtype=np.float32)
            video_mask = np.zeros(len(rows), dtype=bool)
            audio_mask = np.zeros(len(rows), dtype=bool)
            audio = np.zeros((len(rows), 40), dtype=np.float32)
            for start in range(0, len(rows), 32):
                end = min(start + 32, len(rows))
                if is_video:
                    batch = list(workers.map(open_video, paths[start:end]))
                    media, acoustic = zip(*batch)
                    for i, feature in enumerate(acoustic, start):
                        if feature is not None:
                            audio[i] = feature
                            audio_mask[i] = True
                else:
                    media = list(workers.map(open_image, paths[start:end]))
                output[start:end], video_mask[start:end] = vision.encode(media)
                if start % 1024 == 0:
                    print(f'{name} {split}: {end}/{len(rows)}', flush=True)
            if is_video:
                views, mask = (text, audio, output), np.stack((text_mask, audio_mask, video_mask), axis=1)
            else:
                views, mask = (output, text), np.stack((video_mask, text_mask), axis=1)
            if not mask.any(0).all():
                raise ValueError(f'{name} {split} has an entirely missing modality; check extracted media paths.')
            valid = mask.any(1)
            print(f'{name} {split}: rows={len(rows)}, available={mask.sum(0).tolist()}, discarded={sum(~valid)}')
            atomic_cache(out / f'{split}.npz', [x[valid] for x in views], mask[valid],
                         np.array([labels[label] for label in y])[valid], np.asarray(ids)[valid], np.asarray(groups)[valid])
    (out / 'labels.json').write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding='utf-8')


def pooled(values, lengths=None):
    values = np.asarray(values)
    if values.ndim != 3:
        raise ValueError(f'Expected [N,T,D] MOSEI features, got {values.shape}')
    pool = np.empty((len(values), values.shape[-1]), dtype=np.float32)
    mask = np.zeros(len(values), dtype=bool)
    lengths = np.asarray(lengths).reshape(-1) if lengths is not None else None
    for start in range(0, len(values), 256):
        batch = slice(start, start + 256)
        frames = np.asarray(values[batch], dtype=np.float32)
        valid = np.isfinite(frames).all(2)
        if lengths is not None:
            valid &= np.arange(values.shape[1])[None, :] < lengths[batch, None]
        else:
            valid &= (frames != 0).any(2)
        clean = np.where(valid[..., None], frames, 0)
        pool[batch] = clean.sum(1) / valid.sum(1, keepdims=True).clip(min=1)
        mask[batch] = valid.any(1)
    return pool, mask


def make_mosei_cache(path, out):
    with path.open('rb') as handle:
        if file_digest(handle, 'sha256').hexdigest() != MOSEI_SHA256:
            raise ValueError('Expected MOSEI/Processed/unaligned_50.pkl from official MMSA with matching SHA-256.')
        handle.seek(0)
        source = pickle.load(handle)
    out.mkdir(parents=True, exist_ok=True)
    for split, original in zip(SPLITS, ('train', 'valid', 'test')):
        block = source.pop(original)
        labels = np.sign(np.asarray(block['regression_labels']).reshape(-1)).astype('int64') + 1
        views, masks = [], []
        for key in ('text', 'audio', 'vision'):
            lengths = block.get(f'{key}_lengths')
            x = block[key]
            feature, mask = pooled(x, lengths)
            views.append(feature)
            masks.append(mask)
        mask = np.stack(masks, axis=1)
        ids = np.asarray(block['id']).astype(str)
        groups = np.array([sample.split('$_$')[0] for sample in ids])
        valid = mask.any(1)
        print(f'cmu_mosei {split}: rows={len(ids)}, available={mask.sum(0).tolist()}, discarded={sum(~valid)}')
        atomic_cache(out / f'{split}.npz', [x[valid] for x in views], mask[valid], labels[valid], ids[valid], groups[valid])
    (out / 'labels.json').write_text('{"negative":0,"neutral":1,"positive":2}', encoding='utf-8')


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=BENCHMARKS)
    parser.add_argument('--source', required=True, type=Path, help='Raw media root, or official MOSEI .pkl')
    parser.add_argument('--metadata', type=Path, help='Food split folder, N24News JSON, or MELD annotation folder')
    parser.add_argument('--output', type=Path, default=Path('data/features'))
    parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda'))
    parser.add_argument('--limit-per-class', type=int, help='Preprocessing pilot only, sample cap per class per split')
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    if args.limit_per_class is not None and args.limit_per_class < 2:
        parser.error('--limit-per-class must be at least 2')
    target = args.output / args.dataset
    if args.limit_per_class and args.dataset == 'cmu_mosei':
        parser.error('--limit-per-class is supported for raw media datasets only')
    if args.limit_per_class and args.output == Path('data/features'):
        parser.error('Use a separate --output for pilot data.')
    if (target / 'labels.json').exists():
        parser.error(f'Cache already exists at {target}. Use a separate --output to rebuild.')
    if args.dataset == 'cmu_mosei':
        make_mosei_cache(args.source, target)
        return
    if args.metadata is None:
        parser.error('--metadata is required for media datasets.')
    adapter = {'upmc_food101': food_rows, 'n24news': news_rows, 'meld': meld_rows}[args.dataset]
    records = adapter(args.source, args.metadata)
    if args.limit_per_class:
        rng = np.random.default_rng(20261122)
        for split, rows in records.items():
            counter, picked = Counter(), []
            for i in rng.permutation(len(rows)):
                row = rows[i]
                if counter[row[2]] < args.limit_per_class:
                    picked.append(row)
                    counter[row[2]] += 1
            records[split] = picked
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    make_media_cache(records, target, 'cpu' if device == 'auto' else device)


if __name__ == '__main__':
    main()
