"""Run fixed-parameter CIBF comparisons on the existing feature caches."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path
from time import perf_counter

from filelock import FileLock
import torch

from baselines import Baseline
from data import BENCHMARKS, condition, load_data
from main import Settings, atomic_json, batches, digest_cache
from metrics import detection, regression, summarize
from tmc_cibf import TMC_CIBF, VARIANTS, evidential_objective
from validate_tmc import SEEDS
from validate_tmc_crib import Interventions


DATASETS = ('cmu_mosi', 'ch_sims', 'meld', 'iemocap')
METHODS = ('tmc', 'tmc_aug', *VARIANTS)
BALANCE = .5
PROTOCOL = 'cibf-fixed-1'


def fit(model, method, train, seed, settings):
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    intervention = Interventions(train)
    shuffle = torch.Generator().manual_seed(seed + 100)
    corruption = torch.Generator().manual_seed(seed + 200)
    epochs = settings.clean_epochs + settings.joint_epochs
    logs = []
    for epoch in range(1, epochs + 1):
        warmup = epoch <= settings.clean_epochs
        anneal = min(1., 10 * epoch / epochs)
        totals = torch.zeros(4)
        for *views, mask, y, rows in batches(train, settings.batch, shuffle):
            view_labels = y[:, None].expand_as(mask)
            if not warmup and method != 'tmc':
                views, mask, _, _, _, donors = intervention.apply(views, mask, y, rows, corruption,
                                                                    return_donors=True)
                view_labels = train[2][donors]
            if isinstance(model, TMC_CIBF):
                out = model(views, mask, warmup)
                loss, terms = model.loss(out, y, view_labels, mask, anneal, settings.beta, BALANCE, warmup)
            else:
                out = model(views, mask)
                loss = evidential_objective(out['joint_dirichlet'], out['dirichlet'], y, view_labels, mask, anneal)
                terms = (loss, loss.new_zeros(()), loss.new_zeros(()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals += torch.stack((loss, *terms)).detach() * len(y)
        if not totals.isfinite().all():
            raise FloatingPointError(f'Non-finite training loss for {method} at epoch {epoch}')
        logs.append(dict(zip(('loss', 'evidential', 'ib', 'contribution'), (totals / len(train[2])).tolist())))
    return logs


def contribution_metrics(out, target, mask, swapped, cross):
    valid = out['valid'].bool()
    current = out['current'].bool()
    predicted, actual = out['edge'][valid], target[valid]
    metrics = regression(actual.numpy(), predicted.numpy())
    metrics['zero_mae'] = float(actual.abs().mean())
    metrics['bias'] = float((predicted - actual).mean())
    metrics['harmful_detection'] = detection((target[current] < 0).numpy(), (-out['edge'][current]).numpy())
    metrics['cross_swap_detection'] = detection((swapped & cross[:, None])[mask].numpy(),
                                                 (-out['context'][mask]).numpy())
    return metrics


@torch.inference_mode()
def evaluate(models, data, dataset, batch):
    y = data[2]
    names = BENCHMARKS[dataset]
    cases = [('clean', None), *(('gaussian', x) for x in (.2, .5, .9)), ('unseen_dropout', .5),
             ('conflict', None), *(('missing', i) for i in range(len(names))),
             *(('only', i) for i in range(len(names)))]
    results = {}
    for kind, arg in cases:
        views, mask, _, swaps, cross = condition(data, kind, arg)
        rows = mask.any(1).nonzero().flatten()
        if not len(rows):
            continue
        case = kind if arg is None else f'{kind}_{names[arg] if kind in {"missing", "only"} else arg}'
        scores = {}
        for method, model in models.items():
            chunks = []
            for idx in rows.split(batch):
                out = model([x[idx] for x in views], mask[idx])
                if isinstance(model, TMC_CIBF):
                    c = out['contribution']
                    chunks.append(dict(p=out['p'], u=out['u'], conflict=out['conflict'],
                                       discount=out['discount'], rate=out['rate'], compression=out['compression'],
                                       **{k: c[k] for k in ('edge', 'valid', 'current', 'context')},
                                       target=model.targets(c, y[idx])))
                else:
                    chunks.append(dict(p=out['p'], u=1 - out['score']))
            out = {key: torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}
            p, u, label = out['p'], out['u'], y[rows]
            report, _ = summarize(label.numpy(), p.numpy(), (1 - u).numpy())
            report['mean_uncertainty'] = float(u.mean())
            if 'conflict' in out:
                present = mask[rows]
                report.update(mean_conflict=float(out['conflict'].mean()),
                              mean_retention=float(out['discount'][present].mean()),
                              mean_kl_per_dim=float(out['rate'][present].mean()),
                              mean_compression=float(out['compression'][present].mean()))
                if method != 'vib_ds':
                    report['smce'] = contribution_metrics(out, out['target'], present, swaps[rows], cross[rows])
            scores[method] = report
        results[case] = dict(samples=len(rows), methods=scores)
    return results


def run_cell(dataset, fold, seed, settings, args):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    root = args.iemocap_features / f'loso{fold}' if fold else args.features
    directory = args.output / (f'iemocap_loso{fold}' if fold else dataset) / str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    result = directory / 'result.json'
    metadata = dict(protocol=PROTOCOL, dataset=dataset, fold=fold, seed=seed, settings=asdict(settings),
                    contribution_weight=BALANCE, torch=str(torch.__version__), methods=list(METHODS),
                    sha256=digest_cache(root, dataset))
    with FileLock(str(directory / '.lock')):
        if result.is_file():
            if json.loads(result.read_text(encoding='utf-8'))['run'] != metadata:
                raise ValueError(f'Incompatible existing results at {result}')
            if (directory / 'models.pt').is_file():
                return str(result)
        print(f'Training {dataset} fold={fold} seed={seed}', flush=True)
        splits, classes = load_data(root, dataset)
        train = splits['train']
        dims = [x.shape[-1] for x in train[0]]
        models, logs, seconds = {}, {}, {}
        for method in METHODS:
            torch.manual_seed(seed)
            model = (TMC_CIBF(dims, classes, settings.width, settings.latent, method) if method in VARIANTS
                     else Baseline(dims, classes, 'tmc', len(train[2]), settings.width))
            start = perf_counter()
            logs[method] = fit(model, method, train, seed, settings)
            seconds[method] = perf_counter() - start
            models[method] = model.eval()
        metrics = evaluate(models, splits['test'], dataset, settings.batch)
        parameters = {name: sum(p.numel() for p in model.parameters()) for name, model in models.items()}
        checkpoint = directory / 'models.pt'
        temporary = checkpoint.with_suffix('.tmp')
        torch.save({name: model.state_dict() for name, model in models.items()}, temporary)
        temporary.replace(checkpoint)
        atomic_json(result, dict(run=metadata, splits={k: len(v[2]) for k, v in splits.items()}, metrics=metrics,
                                 train_seconds=seconds, parameters=parameters, logs=logs))
    return str(result)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument('--folds', nargs='+', type=int, choices=range(1, 6), default=range(1, 6))
    parser.add_argument('--features', type=Path, default=Path('data/extended'))
    parser.add_argument('--iemocap-features', type=Path, default=Path('data/crib_iemocap'))
    parser.add_argument('--output', type=Path, default=Path('runs/tmc_cibf'))
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    settings = replace(Settings(), joint_epochs=15)
    if args.smoke:
        settings = replace(settings, clean_epochs=1, joint_epochs=1)
        args.output /= 'smoke'
    datasets = sorted(args.datasets, key=lambda name: name != 'meld') if args.workers > 1 else args.datasets
    jobs = [(name, fold, seed, settings, args) for name in datasets
            for fold, seed in product(args.folds if name == 'iemocap' else (None,), args.seeds)]
    if args.workers == 1:
        for job in jobs:
            print(run_cell(*job), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(run_cell, *job) for job in jobs]):
                print(future.result(), flush=True)


if __name__ == '__main__':
    main()
