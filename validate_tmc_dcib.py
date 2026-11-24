"""Reproduce the fixed TMC-HCRD-DCIB v3 experiment on local feature caches."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, replace
from functools import lru_cache
import json
from math import log, sqrt
import os
from pathlib import Path
from time import perf_counter

from filelock import FileLock
import numpy as np
from sklearn.metrics import f1_score
import torch
from torch.nn import functional as F

from data import BENCHMARKS, condition, load_data
from main import Settings, atomic_json, batches, digest_cache, synchronize
from metrics import detection, regression
from tmc_crib import classification, risk_losses, teacher_targets
from tmc_dcib import DCIB, HCRDLite, VARIANTS, adjust, context, geometry, objective, proposal
from tmc_hcrd import mean_valid
from validate_tmc import SEEDS
from validate_tmc_crib import (Interventions, classification_curve, make_model, snapshot,
                               to_device, LAMBDA as V2_LAMBDA)


DATASETS = ('cmu_mosi', 'ch_sims', 'meld', 'iemocap')
LAMBDA = .5
MC = 8
DELTA = .05
THRESHOLDS = (0., .0001, .0005, .001, .0025, .005, .01, .025, .05)
BASELINES = ('tmc', 'tmc_aug', 'rib', 'crib_v2')
PROBES = ('v2_stopgrad', 'v2_utility_only', 'v2_utility_ib')


def rng(seed):
    return torch.Generator().manual_seed(seed)


def loader(train, batch, seed, augmented):
    offset = 1200 if augmented else 300
    shuffle, perturb = rng(seed + offset), rng(seed + offset + 1)
    source = Interventions(train) if augmented else None
    while True:
        yield ((clean, y, source.apply(clean, mask, y, rows, perturb) if augmented else
                (clean, mask, None, None, None))
               for *clean, mask, y, rows in batches(train, batch, shuffle))


def optimizer(parameters, cfg):
    return torch.optim.AdamW((p for p in parameters if p.requires_grad), lr=cfg.lr,
                             weight_decay=cfg.weight_decay)


def update(opt, loss):
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


def fit_baseline(train, dims, classes, seed, cfg, device, method, teacher=None):
    torch.manual_seed(seed)
    is_v2 = method == 'crib_v2' or method in PROBES
    variant = {'v2_utility_only': 'no_ib'}.get(method, 'full' if is_v2 else method)
    model = make_model(dims, classes, variant, cfg).to(device)
    augmented = teacher is not None
    if augmented:
        model.encoders.load_state_dict(teacher.encoders.state_dict())
        model.heads.load_state_dict(teacher.heads.state_dict())
    if is_v2:
        model.encoders.requires_grad_(False)
        model.heads.requires_grad_(False)
    opt = optimizer(model.parameters(), cfg)
    epochs = cfg.joint_epochs if augmented else cfg.clean_epochs
    stream = loader(train, cfg.batch, seed, augmented)
    logs = []
    for epoch in range(epochs):
        total = torch.zeros((), device=device)
        for clean, y, (x, m, strength, swapped, cross) in next(stream):
            x, m, y = to_device(x, device), m.to(device), y.to(device)
            out = model(x, m)
            loss = classification(out, y, m, epoch + (cfg.clean_epochs if augmented else 0),
                                  swapped.to(device) if augmented else None)
            if is_v2:
                targets = teacher_targets(teacher, to_device(clean, device), x, m, y)
                if method in {'v2_utility_only', 'v2_utility_ib'}:
                    valid = m & (m.sum(1, keepdim=True) > 1)
                    target = (1 + targets[1]) / 2
                    error = F.binary_cross_entropy_with_logits(out['d_logit'], target, reduction='none')
                    loss = mean_valid(error, valid)
                    if method == 'v2_utility_ib':
                        loss = loss + cfg.beta * mean_valid(out['kl_d'], valid)
                else:
                    risk, kl = risk_losses(model, out, m, strength.to(device) > 0, cross.to(device), targets)
                    loss = (loss if method == 'crib_v2' else 0.) + V2_LAMBDA * risk + cfg.beta * kl
            elif method == 'rib':
                refs = model.references(teacher, to_device(clean, device), x)
                loss = loss + cfg.gamma * model.risk_loss(out, m, strength.to(device), refs, cfg.beta)
            update(opt, loss)
            total += loss.detach() * len(y)
        logs.append(float(total / len(train[2])))
        if not np.isfinite(logs[-1]):
            raise FloatingPointError(f'{method} epoch {epoch + 1}: non-finite loss')
    return model.eval().requires_grad_(False), logs


@torch.no_grad()
def frozen_inputs(backbone, x, mask):
    h, e = backbone.evidence(x)
    return (h, *backbone.distribution(e)), geometry(e, mask)


def fit_v3(train, teacher, seed, cfg, device):
    torch.manual_seed(seed + 600)
    hcrd = HCRDLite(teacher, cfg.width).to(device)
    opt = optimizer(hcrd.parameters(), cfg)
    stream = loader(train, cfg.batch, seed, True)
    logs = {'hcrd': []}
    start = perf_counter()
    for _ in range(cfg.risk_epochs):
        total = torch.zeros((), device=device)
        for _, y, (x, m, strength, _, cross) in next(stream):
            m = m.to(device)
            with torch.no_grad():
                h, e = teacher.evidence(to_device(x, device))
                p, a, u = teacher.distribution(e)
            loss = hcrd.loss(hcrd(h, p, a, u, m), m, strength.to(device), cross.to(device))
            update(opt, loss)
            total += loss.detach() * len(y)
        logs['hcrd'].append(float(total / len(train[2])))
    hcrd.eval().requires_grad_(False)
    risk_seconds = perf_counter() - start

    models, optimizers, elapsed = {}, {}, dict.fromkeys(VARIANTS, 0.)
    stream = loader(train, cfg.batch, seed, True)
    latent_rng = rng(seed + 730)
    for _ in range(cfg.joint_epochs):
        totals = torch.zeros(len(VARIANTS), device=device)
        for _, y, (x, m, _, _, _) in next(stream):
            m, y = m.to(device), y.to(device)
            stats, g = frozen_inputs(teacher, to_device(x, device), m)
            h, p, _, u = stats
            risks = hcrd(*stats, m)
            noise = torch.randn((1, len(y), cfg.latent), generator=latent_rng).to(device)
            for idx, variant in enumerate(VARIANTS):
                start = perf_counter()
                v, s = context(h, p, u, g, risks, variant)
                if variant not in models:
                    if not models:
                        initial = DCIB(v.shape[1], s.shape[1], teacher.classes, cfg.width, cfg.latent, variant)
                    model = deepcopy(initial).to(device)
                    model.variant = variant
                    models[variant], optimizers[variant] = model, optimizer(model.parameters(), cfg)
                model = models[variant]
                q, kl = model(v, s, g['p0'], noise)
                loss = objective(model, q, kl, g, y, LAMBDA, cfg.beta)
                update(optimizers[variant], loss)
                totals[idx] += loss.detach() * len(y)
                elapsed[variant] += perf_counter() - start
        values = (totals / len(train[2])).tolist()
        if not np.isfinite(values).all() or not np.isfinite(logs['hcrd']).all():
            raise FloatingPointError('Non-finite HCRD/DCIB loss')
        for variant, value in zip(VARIANTS, values):
            logs.setdefault(variant, []).append(value)
    return hcrd, {k: m.eval().requires_grad_(False) for k, m in models.items()}, logs, risk_seconds, elapsed


def half_brier(p, y):
    return .5 * (p - F.one_hot(y, p.shape[-1])).square().sum(-1)


def group_means(values, groups):
    keys, indices = np.unique(groups, return_inverse=True)
    return np.bincount(indices, weights=values) / np.bincount(indices, minlength=len(keys))


@torch.inference_mode()
def calibrate(data, groups, teacher, hcrd, model, cfg, device):
    errors, temperatures = [], []
    sample = rng(20261125)
    grid = torch.logspace(-1, 1, 81, device=device)
    for *x, m, y, _ in batches(data, cfg.batch):
        m, y = m.to(device), y.to(device)
        stats, g = frozen_inputs(teacher, to_device(x, device), m)
        v, s = context(stats[0], stats[1], stats[3], g, hcrd(*stats, m), 'dcib')
        noise = inference_noise(len(y), cfg.latent, sample, device)
        q, _ = model(v, s, g['p0'], noise)
        action = proposal(q, g)
        p0_loss = half_brier(g['p0'], y)
        errors.append(torch.stack([half_brier(adjust(g, action, t)['p'], y) - p0_loss for t in THRESHOLDS]).cpu())
        logits = g['p0'].log()[None] / grid[:, None, None]
        temperatures.append(-logits.log_softmax(-1)[:, torch.arange(len(y), device=device), y].sum(1))
    means = [group_means(row.numpy(), groups) for row in torch.cat(errors, dim=1)]
    n = len(means[0])
    radius = sqrt(2 * log(len(THRESHOLDS) / DELTA) / n)
    candidates = [dict(threshold=t, group_mean=float(x.mean()), upper=float(x.mean() + radius))
                  for t, x in zip(THRESHOLDS, means)]
    selected = min((row for row in candidates if row['upper'] <= 0),
                   key=lambda row: row['group_mean'], default=None)
    temperature = float(grid[torch.stack(temperatures).sum(0).argmin()])
    return dict(n_groups=n, n_samples=len(data[2]), radius=radius, thresholds=candidates,
                threshold=None if selected is None else selected['threshold'], temperature=temperature)


def inference_noise(n, latent, generator, device):
    # Draw in row order so each sample gets its own fixed set of posterior draws.
    return torch.randn((n, MC, latent), generator=generator).transpose(0, 1).to(device)


def append(outputs, key, **values):
    series = outputs.setdefault(key, {})
    for label, value in values.items():
        series.setdefault(label, []).append(value.cpu())


def collect(series):
    return {name: torch.cat(values) for name, values in series.items()}


def diagnostic_metrics(out, mask, strength, cross, i, j):
    pair = mask[:, i] & mask[:, j]
    labels = cross[:, i].long() + 2 * cross[:, j].long()
    result = dict(observation=detection((strength > 0)[mask], out['o'][mask]),
                  observation_strength=regression(strength[mask].numpy(), out['o'][mask].numpy()),
                  conflict=detection(labels[pair] > 0, out['conflict'][pair]))
    if pair.any():
        pred = out['logits'].argmax(-1)
        positive = pair & (labels > 0)
        result['responsibility'] = dict(
            accuracy=float((pred[pair] == labels[pair]).float().mean()),
            macro_f1=float(f1_score(labels[pair], pred[pair], labels=[0, 1, 2], average='macro', zero_division=0)),
            endpoint_accuracy=float((out['logits'][..., 1:].argmax(-1)[positive] + 1 ==
                                     labels[positive]).float().mean()) if positive.any() else None)
    return result


def scores(out, y, p0, oracle):
    stats = classification_curve(y, out['p'], out['score'])
    loss = half_brier(out['p'], y)
    stats['half_brier_gain'] = float((half_brier(p0, y) - loss).mean())
    stats['oracle_gap'] = float((loss - half_brier(oracle, y)).mean())
    if 'adjusted' in out:
        stats['adjust_coverage'] = float(out['adjusted'].float().mean())
    if 'first' in out:
        diff = out['first'][out['valid']] - out['target'][out['valid']]
        stats['direction_mae'] = float(diff.abs().mean()) if len(diff) else None
        stats['direction_bias'] = float(diff.mean()) if len(diff) else None
        stats['conditional_kl'] = float(out['kl'].mean())
    if 'utility' in out:
        diff = out['utility'][out['valid']] - out['utility_target'][out['valid']]
        stats['utility_mae'] = float(diff.abs().mean()) if len(diff) else None
        stats['utility_bias'] = float(diff.mean()) if len(diff) else None
        stats['utility_spearman'] = regression(out['utility_target'][out['valid']].numpy(),
                                               out['utility'][out['valid']].numpy())['spearman']
    return stats


@torch.inference_mode()
def evaluate(data, name, teacher, baseline, hcrd, models, cal, cfg, device):
    clean, _, y = data
    modalities = BENCHMARKS[name]
    cases = [('clean', None), *(('gaussian', level) for level in (.2, .5, .9)), ('unseen_dropout', .5),
             ('conflict', None), *(('missing', i) for i in range(len(clean))),
             *(('only', i) for i in range(len(clean)))]
    summary = {}
    for kind, arg in cases:
        sample = rng(20261125)
        current, mask, strength, swapped, cross = condition(data, kind, arg)
        rows = mask.any(1).nonzero().flatten()
        if not len(rows):
            continue
        outputs = {}
        for index in rows.split(cfg.batch):
            x = to_device([v[index] for v in current], device)
            m, label = mask[index].to(device), y[index].to(device)
            stats, g = frozen_inputs(teacher, x, m)
            h, p, _, u = stats
            risks = hcrd(*stats, m)
            p0 = g['p0']
            onehot = F.one_hot(label, p0.shape[-1]).to(p0.dtype)
            oracle = adjust(g, proposal(onehot, g))['p']
            base_score = 1 - p0.shape[-1] / g['alpha'].sum(-1)
            target = proposal(onehot, g)[-1]
            utility = target - .5 * g['norm']
            append(outputs, 'tmc', p=p0, score=base_score)
            append(outputs, 'oracle', p=oracle, score=oracle.max(1).values)
            append(outputs, 'temperature_tmc', p=(p0.log() / cal['temperature']).softmax(-1), score=base_score)
            append(outputs, 'zero_predictor', p=p0, score=base_score, utility=torch.zeros_like(utility),
                   utility_target=utility, valid=g['valid'])
            append(outputs, 'hcrd', o=risks['o'], conflict=risks['conflict'], logits=risks['conflict_logits'])
            for method, model in baseline.items():
                out = model(x, m)
                more = (dict(utility=out['delta'], utility_target=utility, valid=g['valid'])
                        if method == 'crib_v2' or method in PROBES else {})
                append(outputs, method, p=out['p'], score=out['score'], **more)
                if method in {'crib_v2', 'rib'}:
                    conflict = out.get('conflict', out.get('excess'))
                    cv = out.get('responsibility', out['a'])
                    logits = out.get('conflict_logits', torch.stack((1 - conflict, conflict * cv[:, model.i],
                                                                    conflict * cv[:, model.j]), -1))
                    append(outputs, method + '_risk', o=out['o'], conflict=conflict, logits=logits)
            noise = inference_noise(len(index), cfg.latent, sample, device)
            for variant, model in models.items():
                v, s = context(h, p, u, g, risks, variant)
                controls = (None, 'mean', 'zero', 'shuffle') if variant == 'dcib' and kind in {'clean', 'conflict'} else (None,)
                for control in controls:
                    q, kl = model(v, s, p0, None if control == 'mean' else noise, control)
                    action = proposal(q, g)
                    result = adjust(g, action)
                    key = variant if control is None else f'dcib_latent_{control}'
                    append(outputs, key, p=result['p'], score=1 - result['vacuity'], adjusted=result['adjusted'],
                           first=action[-1], target=target, valid=g['valid'], kl=kl)
                    if variant == 'dcib' and control is None:
                        append(outputs, 'direct_q', p=q, score=q.max(1).values)
                        for key, fusion in (
                            ('discrete', adjust(g, proposal(q, g, discrete=True))),
                            ('dcib_calibrated', adjust(g, action, float('inf') if cal['threshold'] is None else cal['threshold'])),
                        ):
                            append(outputs, key, p=fusion['p'], score=1 - fusion['vacuity'], adjusted=fusion['adjusted'])
        out = {k: collect(v) for k, v in outputs.items()}
        label = y[rows]
        p0, oracle = out['tmc']['p'], out['oracle']['p']
        diagnostics = {}
        for key in ('hcrd', 'crib_v2_risk', 'rib_risk'):
            if key in out:
                diagnostics[key] = diagnostic_metrics(out.pop(key), mask[rows], strength[rows],
                                                      (swapped & cross[:, None])[rows], hcrd.i.cpu(), hcrd.j.cpu())
        cname = kind if arg is None else f'{kind}_{modalities[arg] if kind in {"missing", "only"} else arg}'
        summary[cname] = dict(methods={k: scores(v, label, p0, oracle) for k, v in out.items()},
                              diagnostics=diagnostics, eligible_fraction=len(rows) / len(y))
    return summary


@torch.inference_mode()
def latency(data, teacher, baseline, hcrd, model, cfg, device):
    x = to_device([v[:1] for v in data[0]], device)
    mask = data[1][:1].to(device)
    generator = rng(20261125)

    def predict_v3():
        stats, g = frozen_inputs(teacher, x, mask)
        v, s = context(stats[0], stats[1], stats[3], g, hcrd(*stats, mask), 'dcib')
        q, _ = model(v, s, g['p0'], inference_noise(1, cfg.latent, generator, device))
        return adjust(g, proposal(q, g))

    forwards = {name: (lambda module=module: module(x, mask)) for name, module in dict(tmc=teacher, **baseline).items()}
    forwards['dcib'] = predict_v3
    results = {}
    for name, forward in forwards.items():
        for _ in range(5):
            forward()
        synchronize(device)
        start = perf_counter()
        for _ in range(30):
            forward()
        synchronize(device)
        results[name] = (perf_counter() - start) * 1000 / 30
    return results


def run_cell(data, groups, cfg, device, output, seed, name, fold, fingerprint, threads):
    path = output / (f'iemocap_loso{fold}' if fold else name) / str(seed)
    path.mkdir(parents=True, exist_ok=True)
    report = path / 'result.json'
    run = dict(protocol='dcib-v3-fixed-1', dataset=name, fold=fold, seed=seed, settings=asdict(cfg),
               sha256=fingerprint, device=str(device), torch=str(torch.__version__), threads=threads,
               lambda_=LAMBDA, mc_samples=MC, delta=DELTA, thresholds=list(THRESHOLDS))
    with FileLock(str(path / '.lock')):
        if report.exists():
            if json.loads(report.read_text(encoding='utf-8'))['run'] != run:
                raise ValueError(f'Incompatible run metadata in {path}')
            return
        print(f'{name} fold={fold} seed={seed}', flush=True)
        splits, classes = data
        train = splits['train']
        dims = [x.shape[1] for x in train[0]]
        logs, times, baseline = {}, {}, {}
        start = perf_counter()
        teacher, logs['tmc'] = fit_baseline(train, dims, classes, seed, cfg, device, 'tmc')
        times['tmc'] = perf_counter() - start
        methods = (*BASELINES[1:], *PROBES) if name in {'cmu_mosi', 'meld'} else BASELINES[1:]
        for method in methods:
            start = perf_counter()
            baseline[method], logs[method] = fit_baseline(train, dims, classes, seed, cfg, device, method, teacher)
            times[method] = perf_counter() - start
        start = perf_counter()
        hcrd, models, v3_logs, risk_seconds, head_seconds = fit_v3(train, teacher, seed, cfg, device)
        times.update(hcrd=risk_seconds, dcib_all_ablation_wall=perf_counter() - start, dcib_heads=head_seconds)
        logs.update(v3_logs)
        cal = calibrate(splits['val'], groups, teacher, hcrd, models['dcib'], cfg, device)
        metrics = evaluate(splits['test'], name, teacher, baseline, hcrd, models, cal, cfg, device)
        weights = dict(tmc=snapshot(teacher), hcrd=snapshot(hcrd),
                       **{k: snapshot(m) for k, m in dict(**baseline, **models).items()})
        torch.save(weights, path / 'models.pt')
        parameters = {k: sum(p.numel() for p in m.parameters())
                      for k, m in dict(tmc=teacher, hcrd=hcrd, **baseline, **models).items()}
        parameters['dcib_end_to_end'] = parameters['tmc'] + parameters['hcrd'] + parameters['dcib']
        atomic_json(report, dict(run=run, splits={k: len(v[2]) for k, v in splits.items()},
                                 calibration=cal, metrics=metrics, logs=logs, parameters=parameters,
                                 train_seconds=times, latency_ms=latency(splits['test'], teacher, baseline, hcrd,
                                                                       models['dcib'], cfg, device)))
        print(f'  complete: {path}', flush=True)


@lru_cache(maxsize=1)
def load_experiment(root, name, fold):
    dataset = load_data(root, name)
    with np.load(root / name / 'val.npz', allow_pickle=False) as cache:
        groups = cache['groups'].astype(str)
    if fold:
        groups = np.asarray([group[:5] for group in groups])
    return dataset, groups, digest_cache(root, name)


def run_experiment(root, name, fold, seed, cfg, args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    dataset, groups, fingerprint = load_experiment(root, name, fold)
    run_cell(dataset, groups, cfg, torch.device(args.device), args.output, seed, name, fold,
             fingerprint, args.threads)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument('--folds', nargs='+', type=int, choices=range(1, 6), default=range(1, 6))
    parser.add_argument('--features', type=Path, default=Path('data/extended'))
    parser.add_argument('--iemocap-features', type=Path, default=Path('data/crib_iemocap'))
    parser.add_argument('--output', type=Path, default=Path('runs/tmc_dcib'))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--workers', type=int, default=1, help='Independent CPU experiment processes')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1 or (args.device == 'cuda' and args.workers != 1):
        parser.error('Use positive worker and thread counts, with one worker on CUDA')
    cfg = replace(Settings(), joint_epochs=15)
    if args.smoke:
        cfg = replace(cfg, clean_epochs=1, risk_epochs=1, joint_epochs=1)
        args.output /= 'smoke'
    if args.device == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    names = args.datasets if args.workers == 1 else sorted(args.datasets, key=lambda name: name != 'meld')
    jobs = [(args.iemocap_features / f'loso{fold}' if fold else args.features, name, fold, seed, cfg, args)
            for name in names for fold in (args.folds if name == 'iemocap' else (None,)) for seed in args.seeds]
    if args.workers == 1:
        for job in jobs:
            run_experiment(*job)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = [pool.submit(run_experiment, *job) for job in jobs]
            for job in as_completed(pending):
                job.result()


if __name__ == '__main__':
    main()
