"""Fixed-protocol comparison of TMC-HCRD-CRIB on existing feature caches."""
from argparse import ArgumentParser
from dataclasses import asdict, replace
from itertools import product
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from filelock import FileLock
from sklearn.metrics import f1_score, precision_score, recall_score

from data import BENCHMARKS, condition, inject, load_data, random_index
from main import Settings, atomic_json, batches, digest_cache
from metrics import detection, regression, summarize
from tmc_crib import VARIANTS, TMCcrib, classification, risk_losses, teacher_targets
from tmc_hcrd import TMCHCRD, js
from validate_tmc import DATASETS, SEEDS


METHODS = ('tmc', 'tmc_aug', 'rib', *VARIANTS)
LAMBDA = .1


class Interventions:
    """Draw a donor only from available rows in this split, excluding the recipient."""
    def __init__(self, data):
        self.views, mask, y = data
        self.pools = {}
        for i in range(mask.shape[1]):
            for label in y.unique().tolist():
                for cross in (False, True):
                    selected = (y != label) if cross else (y == label)
                    self.pools[i, label, cross] = (mask[:, i] & selected).nonzero().flatten()

    def apply(self, clean, available, y, rows, rng, *, return_donors=False):
        mode = torch.randint(0, 4, (len(y),), generator=rng)  # clean, noise, same-class, cross-class
        keep = available & (torch.rand(available.shape, generator=rng) >= .15)
        fallback = torch.nn.functional.one_hot(random_index(available, rng), available.shape[1]).bool()
        keep |= (~keep.any(1, keepdim=True)) & fallback
        levels = torch.tensor([0., .2, .5, .9])
        strength = levels[torch.randint(4, keep.shape, generator=rng)] * keep * (mode == 1)[:, None]
        current = inject(clean, keep, strength, rng)
        selected = random_index(keep, rng)
        swapped = torch.zeros_like(keep)
        donors = rows[:, None].expand_as(keep).clone() if return_donors else None
        for (i, label, cross), pool in self.pools.items():
            target = (selected == i) & (y == label) & (mode == 2 + cross) & (keep.sum(1) > 1)
            indices = target.nonzero().flatten()
            if not len(indices) or len(pool) <= int(not cross):
                continue
            size = len(pool) - int(not cross)
            donor = torch.randint(size, (len(indices),), generator=rng)
            if not cross:
                donor += donor >= torch.searchsorted(pool, rows[indices])
            current[i][indices] = self.views[i][pool[donor]]
            swapped[indices, i] = True
            if donors is not None:
                donors[indices, i] = pool[donor]
        result = current, keep, strength, swapped, swapped & (mode == 3)[:, None]
        return (*result, donors) if return_donors else result


def to_device(views, device):
    return [x.to(device) for x in views]


def snapshot(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def fit(model, data, seed, cfg, device, epochs, augment=False, teacher=None):
    weights = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(weights, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sources = {split: Interventions(data[split]) for split in ('train', 'val')} if augment else {}
    has_crib, is_rib = isinstance(model, TMCcrib), model.method == 'full'
    offset = 1200 if augment else 300
    shuffle = torch.Generator().manual_seed(seed + offset)
    train_rng = torch.Generator().manual_seed(seed + offset + 1)
    best, state, logs = float('inf'), None, []
    for epoch in range(epochs):
        row = {'epoch': epoch + 1}
        for split in ('train', 'val'):
            training = split == 'train'
            model.train(training)
            rng = train_rng if training else torch.Generator().manual_seed(20261124)
            totals = torch.zeros(5, device=device)
            for *clean, mask, y, indices in batches(data[split], cfg.batch, shuffle if training else None):
                with torch.set_grad_enabled(training):
                    if augment:
                        current, mask, strength, swapped, cross = sources[split].apply(clean, mask, y, indices, rng)
                        strength, swapped, cross = strength.to(device), swapped.to(device), cross.to(device)
                    else:
                        current, swapped = clean, None
                    current = to_device(current, device)
                    clean = to_device(clean, device) if augment else current
                    mask, y = mask.to(device), y.to(device)
                    result = model(current, mask)
                    task = classification(result, y, mask, epoch + (cfg.clean_epochs if augment else 0), swapped)
                    risk = kl = task.new_zeros(())
                    if has_crib:
                        refs = teacher_targets(teacher, clean, current, mask, y)
                        risk, kl = risk_losses(model, result, mask, strength > 0, cross, refs)
                    elif is_rib:
                        refs = model.references(teacher, clean, current)
                        risk = model.risk_loss(result, mask, strength, refs, cfg.beta)
                    loss = task + (LAMBDA if has_crib else cfg.gamma) * risk + cfg.beta * kl
                    grad = task.new_zeros(())
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        if has_crib:
                            grad = torch.linalg.vector_norm(model.crib.encoder[0].weight.grad)
                        optimizer.step()
                    p = result['p'][torch.arange(len(y), device=device), y]
                    totals += torch.stack((loss, risk, kl, -p.clamp_min(1e-8).log().mean(), grad)).detach() * len(y)
            stats = (totals / len(data[split][2])).tolist()
            if not np.isfinite(stats).all():
                raise FloatingPointError(f'Non-finite {split} loss at epoch {epoch + 1}')
            row[split] = dict(zip(('loss', 'risk', 'kl', 'nll', 'encoder_grad_norm'), stats))
        # Fixed, repeatable validation perturbations. Select by NLL plus the trained risk objective.
        score = row['val']['nll']
        if has_crib:
            score += LAMBDA * row['val']['risk'] + cfg.beta * row['val']['kl']
        elif is_rib:
            score += cfg.gamma * row['val']['risk']
        row['selection_score'] = score
        logs.append(row)
        if score < best:
            best, state = score, snapshot(model)
    model.load_state_dict(state)
    return min(logs, key=lambda x: x['selection_score'])['epoch'], logs


def errors(target, prediction):
    target, prediction = target.numpy(), prediction.numpy()
    return dict(regression(target, prediction), rmse=float(np.sqrt(np.mean((target - prediction) ** 2)))
                if len(target) else None)


def classification_curve(y, p, score):
    stats, curve = summarize(y.numpy(), p.numpy(), score.numpy())
    curve = curve[np.ceil(np.linspace(.05, 1, 20) * len(curve)).astype(int) - 1]
    return dict(stats, risk_coverage=curve.tolist())


@torch.inference_mode()
def evaluate(model, teacher, data, name, device, batch):
    model.eval()
    clean, _, y = data
    has_crib = isinstance(model, TMCcrib)
    tests = [('clean', None), *(('gaussian', level) for level in (.2, .5, .9)), ('unseen_dropout', .5),
             ('conflict', None), *(('missing', i) for i in range(len(clean))),
             *(('only', i) for i in range(len(clean)))]
    if has_crib and model.variant == 'full':
        tests.extend((('zero', None), ('shuffle', None), ('zero', 'conflict'), ('shuffle', 'conflict')))
    summary = {}
    for test, arg in tests:
        latent = test if test in {'zero', 'shuffle'} else None
        current, mask, strength, swaps, cross = condition(data, (arg or 'clean') if latent else test,
                                                          None if latent else arg)
        valid_rows = mask.any(1).nonzero().flatten()
        if not len(valid_rows):
            continue
        outputs, ambiguity, delta = {}, [], []
        for index in valid_rows.split(batch):
            xx = to_device([x[index] for x in current], device)
            mm, yy = mask[index].to(device), y[index].to(device)
            result = model(xx, mm, intervention=latent) if has_crib else model(xx, mm)
            singles = result['single_alpha']
            p, h, u = model.distribution(singles - 1)
            result.setdefault('o', u)
            result.setdefault('a', h)
            result.setdefault('raw', js(p[:, model.i], p[:, model.j]))
            result.setdefault('conflict', result.get('excess', result['raw']))
            cv = result.get('responsibility', h)
            result.setdefault('pair_mask', mm[:, model.i] & mm[:, model.j])
            heuristic = torch.stack((1 - result['conflict'], result['conflict'] * cv[:, model.i],
                                     result['conflict'] * cv[:, model.j]), -1)
            result.setdefault('conflict_logits', heuristic)
            for k in ('p', 'score', 'o', 'a', 'pair_mask', 'conflict', 'conflict_logits', 'delta', 'discount',
                      'kl_a', 'kl_c', 'kl_d'):
                if k in result:
                    outputs.setdefault(k, []).append(result[k].cpu())
            a, d = teacher_targets(teacher, to_device([x[index] for x in clean], device), xx, mm, yy)
            ambiguity.append(a.cpu())
            delta.append(d.cpu())
        out = {k: torch.cat(v) for k, v in outputs.items()}
        m = mask[valid_rows]
        i, j = model.i.cpu(), model.j.cpu()
        switched = swaps[valid_rows] & cross[valid_rows, None]
        pair = out['pair_mask']
        responsibility = switched[:, i].long() + 2 * switched[:, j].long()
        pred = out['conflict_logits'].argmax(-1)
        metrics = classification_curve(y[valid_rows], out['p'], out['score'])
        metrics.update(observation=detection((strength[valid_rows] > 0)[m], out['o'][m]),
                       ambiguity=errors(torch.cat(ambiguity)[m], out['a'][m]),
                       conflict=detection(responsibility[pair] > 0, out['conflict'][pair]),
                       mean_discount=float(out['discount'][m].mean()),
                       fraction_discounted=float((out['discount'][m] < .9999).float().mean()))
        if (test == 'conflict' or arg == 'conflict') and pair.any():
            positive = pair & (responsibility > 0)
            endpoint = out['conflict_logits'][..., 1:].argmax(-1)[positive] + 1
            metrics['responsibility'] = dict(accuracy=float((pred[pair] == responsibility[pair]).float().mean()),
                macro_f1=float(f1_score(responsibility[pair], pred[pair], labels=[0, 1, 2], average='macro', zero_division=0)),
                endpoint_accuracy=float((endpoint == responsibility[positive]).float().mean()) if positive.any() else None,
                no_swap_accuracy=float((responsibility[pair] == 0).float().mean()), endpoint_random_accuracy=.5)
        if has_crib:
            multi = m & (m.sum(1, keepdim=True) > 1)
            target, estimated = torch.cat(delta)[multi], out['delta'][multi]
            metrics['delta'] = errors(target, estimated)
            metrics['mean_absolute_target_delta'] = float(target.abs().mean()) if len(target) else None
            harmful = target > 0
            metrics['harm'] = detection(harmful, estimated)
            if len(target):
                metrics['harm'].update(precision=float(precision_score(harmful, estimated > 0, zero_division=0)),
                                       recall=float(recall_score(harmful, estimated > 0, zero_division=0)))
            metrics['kl'] = {k: np.quantile(out[k][valid].numpy(), [.1, .5, .9]).tolist() if valid.any() else None
                             for k, valid in (('kl_a', m), ('kl_c', pair), ('kl_d', multi))}
        label = test if arg is None else f'{test}_{BENCHMARKS[name][arg] if test in {"missing", "only"} else arg}'
        summary[label] = metrics
    return summary


def make_model(dims, classes, method, cfg):
    if method in VARIANTS:
        return TMCcrib(dims, classes, method, cfg.width, cfg.latent)
    return TMCHCRD(dims, classes, 'full' if method == 'rib' else 'tmc', cfg.width, cfg.latent)


def run_cell(data, dims, classes, name, method, seed, cfg, device, root, fingerprint, threads):
    dest = root / name / str(seed) / method
    dest.mkdir(parents=True, exist_ok=True)
    run = dict(protocol='tmc-crib-v2-fixed-1', dataset=name, method=method, seed=seed, settings=asdict(cfg),
               risk_weight=LAMBDA, sha256=fingerprint, torch=str(torch.__version__), device=str(device), threads=threads)
    report = dest / 'result.json'
    with FileLock(str(dest / '.lock')):
        if report.exists():
            if json.loads(report.read_text(encoding='utf-8'))['run'] != run:
                raise ValueError(f'Incompatible run metadata in {dest}')
            return
        print(f'{name} seed={seed} {method}', flush=True)
        torch.manual_seed(seed)
        model = make_model(dims, classes, method, cfg).to(device)
        teacher = model
        if method != 'tmc':
            teacher = make_model(dims, classes, 'tmc', cfg).to(device)
            teacher.load_state_dict(torch.load(dest.parent / 'tmc' / 'model.pt', map_location=device, weights_only=True))
            teacher.eval().requires_grad_(False)
            model.encoders.load_state_dict(teacher.encoders.state_dict())
            model.heads.load_state_dict(teacher.heads.state_dict())
            if method in VARIANTS:
                model.encoders.requires_grad_(False)
                model.heads.requires_grad_(False)
        start = perf_counter()
        epoch, logs = fit(model, data, seed, cfg, device, cfg.clean_epochs if method == 'tmc' else cfg.joint_epochs,
                          method != 'tmc', teacher)
        seconds = perf_counter() - start
        model.eval()
        with torch.inference_mode():
            x = to_device([x[:1] for x in data['val'][0]], device)
            m = data['val'][1][:1].to(device)
            for _ in range(5):
                model(x, m)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            start = perf_counter()
            for _ in range(30):
                model(x, m)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            latency = (perf_counter() - start) * 1000 / 30
        torch.save(snapshot(model), dest / 'model.pt')
        metrics = evaluate(model, teacher, data['test'], name, device, cfg.batch)
        atomic_json(report, dict(run=run, epoch=epoch, logs=logs, metrics=metrics, parameters=sum(p.numel() for p in model.parameters()),
                                 trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                                 train_seconds=seconds, latency_ms=latency, split_sizes={k: len(v[2]) for k, v in data.items()}))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument('--features', type=Path, default=Path('data/extended'))
    parser.add_argument('--output', type=Path, default=Path('runs/tmc_crib'))
    parser.add_argument('--fold', type=int, choices=range(1, 6))
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    cfg = replace(Settings(), joint_epochs=15)
    if args.smoke:
        cfg = replace(cfg, clean_epochs=1, joint_epochs=1)
        args.output /= 'smoke'
    if args.fold:
        args.features /= f'loso{args.fold}'
        args.output /= f'loso{args.fold}'
        args.datasets = ['iemocap']
    if args.device == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    for name in args.datasets:
        data, classes = load_data(args.features, name)
        dims = [x.shape[1] for x in data['train'][0]]
        fingerprint = digest_cache(args.features, name)
        for seed, method in product(args.seeds, dict.fromkeys(('tmc', *args.methods))):
            run_cell(data, dims, classes, name, method, seed, cfg, device, args.output, fingerprint, args.threads)


if __name__ == '__main__':
    main()
