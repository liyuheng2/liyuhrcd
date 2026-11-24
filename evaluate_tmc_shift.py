"""Score paired clean and feature-shifted samples using completed checkpoints."""
from argparse import ArgumentParser
from csv import DictWriter
import json
from pathlib import Path

import numpy as np
import torch

from data import condition, load_data
from main import digest_cache
from metrics import detection
from tmc_hcrd import TMCHCRD
from validate_tmc import DATASETS, PROTOCOL


SHIFTS = [('gaussian', .9), ('unseen_dropout', .5)]


@torch.inference_mode()
def scores(model, inputs, mask):
    values = {}
    for rows in torch.arange(len(mask)).split(256):
        keep = mask[rows]
        out = model([x[rows] for x in inputs], keep)
        batch = {'vacuity': 1 - out['score'], 'one_minus_maxprob': 1 - out['p'].amax(-1)}
        if model.uses_risk:
            batch['max_pollution_risk'] = out['o'].masked_fill(~keep, 0).amax(-1)
        for name, value in batch.items():
            values.setdefault(name, []).append(value.numpy())
    return {name: np.concatenate(parts) for name, parts in values.items()}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--feature-dir', type=Path, default=Path('data/features'))
    parser.add_argument('--run-dir', type=Path, default=Path('runs/tmc_hcrd/protocol'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    fields = ['dataset', 'method', 'seed', 'condition', 'score', 'auroc', 'auprc']
    path = args.run_dir / 'shift_detection.csv'
    partial = path.with_suffix('.tmp')
    with partial.open('w', newline='', encoding='utf-8-sig') as file:
        writer = DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for name in DATASETS:
            reports = sorted((args.run_dir / name).glob('*/*/result.json'))
            if not reports:
                continue
            data, classes = load_data(args.feature_dir, name)
            test = data['test']
            fingerprint = digest_cache(args.feature_dir, name)
            shifted = [(kind, arg, condition(test, kind, arg)[:2]) for kind, arg in SHIFTS]
            for report in reports:
                run = json.loads(report.read_text(encoding='utf-8'))['run']
                if run['fingerprint'] != fingerprint or run['protocol'] != PROTOCOL or run['smoke']:
                    raise ValueError(f'Unexpected dataset fingerprint or protocol in {report}')
                settings = run['settings']
                model = TMCHCRD([x.shape[1] for x in test[0]], classes, run['method'],
                                width=settings['width'], latent=settings['latent']).eval()
                model.load_state_dict(torch.load(report.with_name('model.pt'), map_location='cpu',
                                                 weights_only=True))
                clean = scores(model, *test[:2])
                labels = np.repeat([0, 1], len(test[2]))
                for kind, arg, (inputs, mask) in shifted:
                    moved = scores(model, inputs, mask)
                    for score, baseline in clean.items():
                        result = detection(labels, np.concatenate((baseline, moved[score])))
                        writer.writerow(dict(dataset=name, method=run['method'], seed=run['seed'],
                                             condition=f'{kind}_{arg}', score=score, **result))
            print(f'{name}: {len(reports)} evaluated checkpoints', flush=True)
    partial.replace(path)


if __name__ == '__main__':
    main()
