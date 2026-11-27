"""Export the exact data/seed/method matrix and paired differences for EC-RIB."""
from argparse import ArgumentParser
from csv import DictWriter
from itertools import product
import json
from pathlib import Path
from statistics import mean, stdev

from main import SEEDS, VALIDATION, export_summary


DATASETS = ('ch_sims', 'mintrec', 'upmc_food101')
KEY_CASES = ('clean', 'gaussian_0.9', 'unseen_dropout_0.5', 'conflict')
METRICS = ('accuracy', 'macro_f1', 'aurc', 'error_auroc', 'ece15', 'nll')


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('runs/recommended'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reports, missing = {}, []
    for dataset, method, seed in product(DATASETS, VALIDATION, SEEDS):
        key = dataset, method, seed
        file = args.run_dir / 'protocol' / dataset / method / str(seed) / 'result.json'
        if not file.is_file():
            missing.append(key)
            continue
        report = json.loads(file.read_text(encoding='utf-8'))
        cfg = report['run']
        if (cfg['dataset'], cfg['method'], cfg['seed']) != key or cfg['quick']:
            raise ValueError(f'Unexpected run metadata in {file}')
        reports[key] = report
    if not reports:
        parser.error('No full protocol results found.')
    compatibility = {(r['run']['protocol'], json.dumps(r['run']['config'], sort_keys=True)) for r in reports.values()}
    if len(compatibility) != 1:
        raise ValueError('Reports describe different fixed experimental settings.')
    digests = {}
    for (dataset, _, _), report in reports.items():
        fingerprint = report['run']['fingerprint']
        if digests.setdefault(dataset, fingerprint) != fingerprint:
            raise ValueError(f'Reports for {dataset} used different data.')
    args.output.mkdir(parents=True, exist_ok=True)
    export_summary(reports.values(), args.output / 'ec_rib_summary.csv')
    differences = []
    for dataset, method, case, metric in product(DATASETS, VALIDATION[1:], KEY_CASES, METRICS):
        seeds, deltas = [], []
        for seed in SEEDS:
            values = [reports.get((dataset, key, seed), {}).get('conditions', {}).get(case, {}).get(metric)
                      for key in ('full', method)]
            if all(value is not None for value in values):
                seeds.append(seed)
                deltas.append(values[0] - values[1])
        if deltas:
            differences.append(dict(dataset=dataset, comparator=method, condition=case, metric=metric,
                                    ec_rib_minus_comparator=mean(deltas),
                                    std=stdev(deltas) if len(deltas) > 1 else '', seeds=' '.join(map(str, seeds))))
    write_csv(args.output / 'ec_rib_paired_differences.csv', differences,
              ('dataset', 'comparator', 'condition', 'metric', 'ec_rib_minus_comparator', 'std', 'seeds'))
    status = dict(expected=len(DATASETS) * len(VALIDATION) * len(SEEDS), complete=len(reports), missing=missing,
                  dataset_sha256=digests)
    (args.output / 'ec_rib_status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    print(f"{status['complete']}/{status['expected']} complete")


if __name__ == '__main__':
    main()
