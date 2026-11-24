"""Train, calibrate, and compare TMC-HCRD-ACRIB using existing feature caches."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import f1_score
import torch
from filelock import FileLock

from data import BENCHMARKS, condition, load_data
from main import Settings, atomic_json, batches, digest_cache
from metrics import detection, regression, summarize
from tmc_acrib import (TASKS, TMCACRIB, VARIANTS, action, classification_loss, counterfactual,
                       gains, probabilities, risk_loss, select_alpha)
from tmc_crib import TMCcrib, classification, discount_evidence, risk_losses, teacher_targets
from tmc_hcrd import TMCHCRD, fuse
from validate_tmc import DATASETS, SEEDS
from validate_tmc_crib import Interventions, snapshot


LAMBDA = 1.
JOINT_LR = .0001
MC = 8
THRESHOLDS = (float('inf'), .1, .05, .02, .01, .005, 0.)
MODEL_NAMES = ('tmc', 'tmc_aug', 'crib_v2', *VARIANTS, 'acrib_joint')
PROTOCOL = 'acrib-v1'


def rng(seed):
    return torch.Generator().manual_seed(seed)


def optimizer(model, cfg, lr):
    return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                             lr=lr, weight_decay=cfg.weight_decay)


def backbone(model, trainable):
    model.encoders.requires_grad_(trainable)
    model.heads.requires_grad_(trainable)


def initialize(model, teacher):
    model.encoders.load_state_dict(teacher.encoders.state_dict())
    model.heads.load_state_dict(teacher.heads.state_dict())


def train(model, data, seed, cfg, epochs, phase, teacher=None):
    is_acrib = isinstance(model, TMCACRIB)
    frozen = phase == 'frozen'
    backbone(model, not frozen)
    opt = optimizer(model, cfg, JOINT_LR if phase == 'joint' else cfg.lr)
    source = Interventions(data) if phase != 'teacher' else None
    offset = {'teacher': 300, 'frozen': 1200, 'joint': 2400}[phase]
    order, perturb, latent = (rng(seed + offset + i) for i in range(3))
    logs = []
    for epoch in range(epochs):
        totals = torch.zeros(3)
        for *clean, mask, y, rows in batches(data, cfg.batch, order):
            x = clean
            view_labels = y[:, None].expand_as(mask)
            if source is not None:
                x, mask, strength, swapped, cross, donors = source.apply(clean, mask, y, rows, perturb,
                                                                        return_donors=True)
                view_labels = data[2][donors]
            anneal_epoch = epoch if teacher is None else cfg.clean_epochs + epoch
            risk = kl = torch.tensor(0.)
            if isinstance(model, TMCcrib):
                result = model(x, mask)
                risk, kl = risk_losses(model, result, mask, strength > 0, cross,
                                       teacher_targets(teacher, clean, x, mask, y))
                task = classification(result, y, mask, anneal_epoch, swapped)
            else:
                h, e = model.evidence(x)
                alpha, candidates = counterfactual(e, mask)
                task = classification_loss(alpha, e, mask, y, view_labels, anneal_epoch)
                if is_acrib:
                    with torch.no_grad():
                        ambiguity = teacher.distribution(teacher.evidence(clean)[1])[1]
                        delta = gains(alpha.detach(), candidates.detach(), y)
                    output = model.risk(h, e, mask, alpha, candidates, latent)
                    risk, kl = risk_loss(model, output, strength, cross, ambiguity, delta)
            loss = (0 if frozen else task) + LAMBDA * risk + cfg.beta * kl
            if not loss.isfinite():
                raise FloatingPointError(f'Non-finite {phase} loss in epoch {epoch + 1}')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            totals += torch.stack((task, risk, kl)).detach() * len(y)
        logs.append(dict(zip(('classification', 'risk', 'kl'), (totals / len(data[2])).tolist())))
    return logs


@torch.inference_mode()
def match_capacity(model, data, cfg, samples):
    """Fit fixed and randomized gate centers on training interventions only."""
    source = Interventions(data)
    perturb, latent = rng(7311), rng(7312)
    values = [[], [], []]
    for *clean, mask, y, rows in batches(data, cfg.batch):
        x, mask, *_ = source.apply(clean, mask, y, rows, perturb)
        out = model.infer(x, mask, latent, samples)
        for t in range(3):
            values[t].append(out['cap'][t][out['valid'][t]])
    target = torch.stack([torch.cat(v).mean() if sum(map(len, v)) else torch.tensor(0.) for v in values])
    index = model.channel.index
    offsets = (torch.zeros(1), model.channel.random_offsets)
    centers = []
    for offset in offsets:
        low, high = torch.zeros(3), torch.full((3,), len(index), dtype=torch.float32)
        for _ in range(24):
            center = (low + high) / 2
            capacity = (center[:, None, None] + offset[:, None] - index).sigmoid().square().sum(-1).mean(-1)
            low, high = torch.where(capacity < target, center, low), torch.where(capacity < target, high, center)
        centers.append((low + high) / 2)
    return target, centers


@torch.inference_mode()
def calibrate(model, data, cfg, samples):
    # A fixed perturbation mix on the original validation split. No test data enters this procedure.
    source = Interventions(data)
    perturb, latent = rng(8411), rng(8412)
    totals = torch.zeros(len(THRESHOLDS), dtype=torch.float64)
    for *clean, mask, y, rows in batches(data, cfg.batch):
        x, mask, *_ = source.apply(clean, mask, y, rows, perturb)
        out = model.infer(x, mask, latent, samples)
        potential = gains(out['alpha'], out['candidates'], y)
        for index, threshold in enumerate(THRESHOLDS):
            mod, choice, execute = action(out['delta'], mask, threshold)
            totals[index] += (potential[torch.arange(len(y)), mod, choice] * execute).double().sum()
    best = int(totals.argmax())
    return THRESHOLDS[best], {str(t): float(v) / len(data[2]) for t, v in zip(THRESHOLDS, totals)}


def scenarios(dataset):
    yield 'clean', 'clean', None
    for level in (.2, .5, .9):
        yield f'gaussian_{level}', 'gaussian', level
    for level in (.2, .5):
        yield f'unseen_dropout_{level}', 'unseen_dropout', level
    yield 'swap', 'conflict', None
    for i, name in enumerate(BENCHMARKS[dataset]):
        yield f'missing_{name}', 'missing', i
        yield f'only_{name}', 'only', i


def append(bucket, **values):
    for key, value in values.items():
        bucket.setdefault(key, []).append(value.cpu())


def moments(x):
    if not len(x):
        return {'mean': None, 'var': None}
    return {'mean': x.mean().item(), 'var': x.var(unbiased=False).item()}


def risk_metrics(bucket):
    data = {k: torch.cat(v).numpy() for k, v in bucket.items()}
    pair_target, pair_prob = data['pair_target'], data['pair_prob']
    endpoint = pair_target > 0
    prediction = pair_prob.argmax(1)
    delta, delta_true = data['delta'], data['delta_true']
    return dict(observation=detection(data['polluted'], data['o']),
                ambiguity=regression(data['ambiguity'], data['a']),
                conflict=detection(endpoint, 1 - pair_prob[:, 0]),
                responsibility_accuracy=float((prediction == pair_target).mean()) if len(pair_target) else None,
                responsibility_f1=float(f1_score(pair_target, prediction, labels=[0, 1, 2],
                                                  average='macro', zero_division=0)) if len(pair_target) else None,
                endpoint_accuracy=float((pair_prob[endpoint, 1:].argmax(1) + 1 ==
                                         pair_target[endpoint]).mean()) if endpoint.any() else None,
                action=dict(regression(delta_true, delta), bias=float((delta-delta_true).mean()) if len(delta) else None))


def score_predictions(bucket):
    row = {k: torch.cat(v) for k, v in bucket.items()}
    summary, _ = summarize(row['y'].numpy(), probabilities(row['alpha']).numpy(),
                           (1 - row['alpha'].shape[-1] / row['alpha'].sum(-1)).numpy())
    summary.update(applied_rate=row['applied'].float().mean().item(),
                   realized_half_brier_gain=row['gain'].mean().item(),
                   harmful_among_applied=(row['gain'][row['applied']] < -1e-7).float().mean().item()
                   if row['applied'].any() else None,
                   incorrect_discount_rate=(row['applied'] & (row['gain'] <= 1e-7)).float().mean().item())
    if 'prediction' in row:
        selected = row['applied']
        summary['selected_bias'] = (row['prediction'][selected] - row['gain'][selected]).mean().item() \
            if selected.any() else None
    return summary


@torch.inference_mode()
def evaluate(model, teacher, data, dataset, cfg, threshold, samples, probe=None):
    results = {}
    is_acrib = isinstance(model, TMCACRIB)
    for label, kind, amount in scenarios(dataset):
        if probe is not None and label not in {'clean', 'gaussian_0.5', 'unseen_dropout_0.5', 'swap'}:
            continue
        x, mask, strength, swaps, cross = condition(data, kind, amount)
        valid = mask.any(1)
        x, mask, strength, cross = [a[valid] for a in x], mask[valid], strength[valid], swaps[valid] & cross[valid, None]
        clean = [a[valid] for a in data[0]]
        y = data[2][valid]
        latent = rng(9511)
        predictions, risk = {}, {}
        capacities, divergences = [[], [], []], [[], [], []]
        for start in range(0, len(y), cfg.batch):
            sl = slice(start, start + cfg.batch)
            current, m, target = [a[sl] for a in x], mask[sl], y[sl]
            evidence = model.evidence(current)[1] if not is_acrib else None
            selected = {}
            if is_acrib:
                out = model.infer(current, m, latent, samples, probe)
                alpha, candidates, evidence = out['alpha'], out['candidates'], out['evidence']
                potential = gains(alpha, candidates, target)
                raw = probabilities(alpha)
                ambiguity = teacher.distribution(teacher.evidence([v[sl] for v in clean])[1])[1]
                pair = m[:, model.i] & m[:, model.j]
                pair_target = cross[sl, model.i].long() + 2*cross[sl, model.j].long()
                eligible = out['valid'][2]
                append(risk, o=out['o'][m], polluted=(strength[sl] > 0)[m], ambiguity=ambiguity[m], a=out['a'][m],
                       pair_target=pair_target[pair], pair_prob=out['pi'][pair],
                       delta_true=potential[eligible].flatten(), delta=out['delta'][eligible].flatten())
                for t in range(3):
                    capacities[t].append(out['cap'][t][out['valid'][t]])
                    divergences[t].append(out['kl'][t][out['valid'][t]])
                adjusted, weight = select_alpha(alpha, candidates, out['delta'], m, threshold)
                mod, choice, _ = action(out['delta'], m, threshold)
                selected['acrf'] = adjusted, (weight != 1).any(1), out['delta'][torch.arange(len(m)), mod, choice]
                heuristic = (1-out['o']) * (1-out['a']) / (1+out['c'])
                heuristic = torch.where(m.sum(1, keepdim=True) > 1, heuristic, 1.)
                selected['product'] = fuse(discount_evidence(evidence, heuristic, m)), ((heuristic < 1) & m).any(1), None
            else:
                alpha, candidates = counterfactual(evidence, m)
                raw = probabilities(alpha)
                if isinstance(model, TMCcrib):
                    out = model(current, m)
                    selected['removal'] = out['alpha'], ((out['discount'] < 1) & m).any(1), None
            selected['ds'] = alpha, torch.zeros_like(target, dtype=torch.bool), None
            raw_error = (raw - torch.nn.functional.one_hot(target, raw.shape[-1])).square().sum(-1)
            for policy, (adjusted, applied, predicted) in selected.items():
                target_error = (probabilities(adjusted) - torch.nn.functional.one_hot(target, raw.shape[-1])).square().sum(-1)
                append(predictions.setdefault(policy, {}), alpha=adjusted, y=target, applied=applied,
                       gain=(raw_error - target_error)/2, **({'prediction': predicted} if predicted is not None else {}))
        if not predictions:
            continue
        results[label] = dict(policies={k: score_predictions(v) for k, v in predictions.items()})
        if is_acrib:
            results[label].update(risks=risk_metrics(risk),
                                  capacity={t: moments(torch.cat(v)) for t, v in zip(TASKS, capacities)},
                                  kl={t: moments(torch.cat(v)) for t, v in zip(TASKS, divergences)})
    return results


def experiment(features, output, dataset, seed, smoke, fold):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg = replace(Settings(), beta=.01, clean_epochs=1 if smoke else 5,
                  risk_epochs=1 if smoke else 10, joint_epochs=1 if smoke else 5)
    samples = 2 if smoke else MC
    data, classes = load_data(features, dataset)
    dims = [x.shape[1] for x in data['train'][0]]
    dest = output / (f'iemocap_loso{fold}' if fold else dataset) / str(seed)
    dest.mkdir(parents=True, exist_ok=True)
    metadata = dict(protocol=PROTOCOL, dataset=dataset, seed=seed, fold=fold, settings=asdict(cfg),
                    samples=samples, joint_lr=JOINT_LR, risk_weight=LAMBDA, torch=str(torch.__version__),
                    sha256=digest_cache(features, dataset), splits={k: len(v[2]) for k, v in data.items()})
    with FileLock(str(dest / '.lock')):
        teacher = TMCHCRD(dims, classes, 'tmc', cfg.width, cfg.latent)
        match = frozen_state = None
        for name in MODEL_NAMES:
            path, weights = dest / f'{name}.json', dest / f'{name}.pt'
            torch.manual_seed(seed)
            if name == 'tmc':
                teacher = model = TMCHCRD(dims, classes, 'tmc', cfg.width, cfg.latent)
            elif name == 'tmc_aug':
                model = TMCHCRD(dims, classes, 'tmc', cfg.width, cfg.latent)
            elif name == 'crib_v2':
                model = TMCcrib(dims, classes, 'full', cfg.width, cfg.latent)
            else:
                model = TMCACRIB(dims, classes, 'acrib' if name == 'acrib_joint' else name, cfg.width, cfg.latent)
            if path.exists() and weights.exists():
                existing = json.loads(path.read_text(encoding='utf-8'))
                if existing['run'] != metadata:
                    raise ValueError(f'Incompatible run metadata in {path}')
                model.load_state_dict(torch.load(weights, weights_only=True))
            else:
                print(f'{dataset} fold={fold} seed={seed} {name}', flush=True)
                start = perf_counter()
                if name != 'tmc':
                    initialize(model, teacher)
                if name in {'fixed_acrib', 'random_acrib'}:
                    model.channel.matched_center.copy_(match[1][int(name == 'random_acrib')])
                if name == 'acrib_joint':
                    model.load_state_dict(frozen_state)
                phase = 'teacher' if name == 'tmc' else 'joint' if name in {'tmc_aug', 'acrib_joint'} else 'frozen'
                epochs = dict(teacher=cfg.clean_epochs, joint=cfg.joint_epochs, frozen=cfg.risk_epochs)[phase]
                logs = train(model, data['train'], seed, cfg, epochs, phase, None if name == 'tmc' else teacher)
                seconds = perf_counter() - start
                model.eval()
                threshold, grid = (calibrate(model, data['val'], cfg, samples) if isinstance(model, TMCACRIB)
                                   else (float('inf'), {}))
                metrics = evaluate(model, teacher, data['test'], dataset, cfg, threshold, samples)
                probes = {}
                if name in {'acrib', 'acrib_joint'}:
                    for probe in ('zero', 'shuffle'):
                        probes[probe] = evaluate(model, teacher, data['test'], dataset, cfg, threshold, samples, probe)
                torch.save(snapshot(model), weights)
                atomic_json(path, dict(run=metadata, method=name, phase=phase, logs=logs, seconds=seconds,
                                       threshold=threshold if np.isfinite(threshold) else None, calibration_gain=grid,
                                       capacity_target=match[0].tolist() if name in {'fixed_acrib', 'random_acrib'} else None,
                                       metrics=metrics, probes=probes))
            model.eval()
            if name == 'tmc':
                teacher.requires_grad_(False)
            if name == 'acrib':
                frozen_state = snapshot(model)
                match = match_capacity(model, data['train'], cfg, samples)
        print(f'complete {dest}', flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--seeds', nargs='+', choices=SEEDS, type=int, default=SEEDS)
    parser.add_argument('--features', type=Path, default=Path('data/extended'))
    parser.add_argument('--output', type=Path, default=Path('runs/tmc_acrib'))
    parser.add_argument('--folds', nargs='+', choices=range(1, 6), type=int)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    output = args.output / 'smoke' if args.smoke else args.output
    datasets = ([('iemocap', fold, args.features / f'loso{fold}') for fold in args.folds] if args.folds else
                [(name, None, args.features) for name in args.datasets])
    jobs = [(features, output, dataset, seed, args.smoke, fold)
            for dataset, fold, features in datasets for seed in args.seeds]
    if args.workers == 1 or len(jobs) == 1:
        for job in jobs:
            experiment(*job)
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            futures = [pool.submit(experiment, *job) for job in jobs]
            for future in futures:
                future.result()


if __name__ == '__main__':
    main()
