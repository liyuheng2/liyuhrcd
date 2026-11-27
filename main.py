"""Run the fixed HRD-CF-M-HCRD-RIB v2 experimental matrix."""
from argparse import ArgumentParser
from csv import DictWriter
from dataclasses import asdict, dataclass
from hashlib import file_digest
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from baselines import BASELINES, Baseline
from data import BENCHMARKS, DEFAULT_BENCHMARKS, SPLITS, condition, load_data, training_batch
from metrics import detection, regression, summarize
from model import ABLATIONS, HCRDRIB, classification_loss
from paper_baselines import CDPRFeatures, SeDUDFeatures


PROTOCOL = 'v2-fixed-1'
CORE = ('late', 'tmc', 'etmc', 'qmf', 'mmdynamics', 'no_ib', 'fixed_ib', 'dynamic_kl', 'dynamic_risk', 'full')
VALIDATION = ('full', 'qmf', 'sed_ud_features', 'cdpr_features', 'fixed_ib',
              'no_ib', 'dynamic_kl', 'dynamic_risk', 'no_c', 'no_csf')
ALL = CORE + tuple(sorted((BASELINES | ABLATIONS) - set(CORE)))
SEEDS = (11, 22, 33)


@dataclass(frozen=True)
class Settings:
    beta: float = 0.001
    gamma: float = 1.0
    lr: float = 0.001
    batch: int = 128
    width: int = 128
    latent: int = 16
    clean_epochs: int = 5
    risk_epochs: int = 5
    joint_epochs: int = 30
    weight_decay: float = 0.0001


def batches(data, batch_size, rng=None):
    views, mask, labels = data
    indices = torch.arange(len(labels))
    return DataLoader(TensorDataset(*views, mask, labels, indices), batch_size=batch_size,
                      shuffle=rng is not None, generator=rng, num_workers=0)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def fit(model, train, method, seed, settings, device, cuda_graph=False):
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay,
                                  fused=True if cuda_graph else None, capturable=cuda_graph)
    loader = batches(train, settings.batch, torch.Generator().manual_seed(seed + 351))
    perturb = torch.Generator().manual_seed(seed + 712)
    shuffle_evidence = torch.Generator().manual_seed(seed + 831)
    phases = [('clean', settings.clean_epochs), ('joint', settings.joint_epochs)]
    if method not in BASELINES:
        phases.insert(1, ('risk', settings.risk_epochs))
    logs = []
    classification_epoch = 0
    for phase, epochs in phases:
        graph, static, recorded = None, None, None
        warmup = 0
        # Reset the joint data RNG so HCRD's risk-only phase cannot alter baseline perturbations.
        if phase == 'joint':
            loader = batches(train, settings.batch, torch.Generator().manual_seed(seed + 1351))
            perturb.manual_seed(seed + 1712)
            if isinstance(model, SeDUDFeatures):
                model.enable_selection()
        for epoch in range(epochs):
            model.train()
            totals = np.zeros(4)
            device_totals = torch.zeros(4, device=device) if cuda_graph else None
            n = 0
            for *clean, mask, y, rows in loader:
                if phase == 'clean':
                    x = clean
                    strength = torch.zeros_like(mask, dtype=torch.float32)
                else:
                    x, mask, strength = training_batch(clean, mask, perturb)
                arity = len(x)
                packed = (*x, *clean, mask, y, rows, strength)

                def step(items, capture=False):
                    noisy, original = items[:arity], items[arity:2 * arity]
                    present, labels, row_ids, severity = items[-4:]
                    zero = labels.new_zeros((), dtype=torch.float32)
                    kl, risk = zero, zero
                    if method in BASELINES:
                        result = model(noisy, present)
                        task = model.loss(result, labels, present, row_ids, classification_epoch)
                    else:
                        result = model(noisy, present, uniform=phase == 'clean')
                        task = zero if phase == 'risk' else classification_loss(
                            result, labels, present, use_unimodal=method != 'no_unimodal')
                        if phase != 'clean':
                            targets, evidence = model.references(original, result, severity)
                            kl, risk = model.risk_losses(result, targets, evidence, shuffle_evidence, capture=capture)
                    total = task + settings.beta * kl + settings.gamma * risk
                    if not cuda_graph and not torch.isfinite(total):
                        raise FloatingPointError(f'{method} {phase} {epoch + 1}: non-finite loss')
                    optimizer.zero_grad(set_to_none=True)
                    if total.requires_grad:
                        total.backward()
                        if phase == 'clean' and isinstance(model, SeDUDFeatures):
                            model.record_importance()
                        optimizer.step()
                    return torch.stack((total.detach(), task.detach(), kl.detach(), risk.detach()))

                if cuda_graph and len(y) == settings.batch and warmup >= 3:
                    if graph is None:
                        torch.cuda.synchronize(device)
                        static = tuple(item.to(device) for item in packed)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            recorded = step(static, capture=True)
                    else:
                        for target, current in zip(static, packed):
                            target.copy_(current)
                    graph.replay()
                    values = recorded
                else:
                    values = step(tuple(item.to(device) for item in packed))
                    warmup += 1
                n += len(y)
                if cuda_graph:
                    device_totals.add_(values, alpha=len(y))
                else:
                    totals += values.cpu().numpy().astype(np.float64) * len(y)
            if cuda_graph:
                totals = device_totals.cpu().numpy()
                if not np.isfinite(totals).all():
                    raise FloatingPointError(f'{method} {phase} {epoch + 1}: non-finite loss')
            if phase != 'risk':
                classification_epoch += 1
            log = dict(zip(('total', 'task', 'kl', 'risk'), (totals / n).tolist()))
            log.update(phase=phase, epoch=epoch + 1)
            logs.append(log)
            if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
                print(f'  {phase} {epoch + 1}/{epochs} loss={log["total"]:.5f}', flush=True)
    return logs


def evaluation_plan(modalities, quick=False):
    cases = [('clean', None), ('gaussian', 0.2), ('gaussian', 0.5), ('gaussian', 0.9),
             ('unseen_dropout', 0.5), ('conflict', None)]
    for i, name in enumerate(modalities):
        cases.extend((('only_noise', i), ('missing', i), ('only', i)))
    if quick:
        cases = [('clean', None), ('gaussian', 0.5), ('conflict', None), ('only', 0)]
    return [(kind if v is None else f'{kind}_{modalities[v] if isinstance(v, int) else v}', kind, v)
            for kind, v in cases]


@torch.inference_mode()
def evaluate(model, data, kind, argument, device, settings, threshold=None):
    model.eval()
    clean, _, labels = data
    inputs, mask, strength, swaps, is_cross_label = condition(data, kind, argument)
    valid = mask.any(1)
    indices = valid.nonzero().flatten()
    if not len(indices):
        return None
    p, score, alpha, risk_predictions, risk_targets, risk_masks, extra_conflict = [], [], [], [], [], [], []
    forward_time = 0
    for rows in indices.split(settings.batch):
        x = [v[rows].to(device) for v in inputs]
        m = mask[rows].to(device)
        synchronize(device)
        started = perf_counter()
        result = model(x, m)
        synchronize(device)
        forward_time += perf_counter() - started
        p.append(result['p'].cpu())
        score.append(result['score'].cpu())
        alpha.append(result['alpha'].cpu())
        if isinstance(model, HCRDRIB):
            refs, _ = model.references([v[rows].to(device) for v in clean], result, strength[rows].to(device))
            risk_predictions.append([t.cpu() for t in result['pred_risk']])
            risk_targets.append([t.cpu() for t in refs])
            risk_masks.append([t.cpu() for t in result['masks']])
            extra_conflict.append(result['r_c'].cpu())
    p, score, alpha = (torch.cat(t).numpy() for t in (p, score, alpha))
    stats, curve = summarize(labels[indices].numpy(), p, score, threshold)
    stats.update(sample_fraction=float(valid.float().mean()), batch=settings.batch,
                 milliseconds_per_sample=1000 * forward_time / len(indices))
    valid_mask = mask[indices].numpy()
    for i in range(alpha.shape[1]):
        observed = valid_mask[:, i]
        stats[f'alpha_mod{i}'] = float(alpha[observed, i].mean()) if observed.any() else None
        target = swaps[indices, i] if kind == 'conflict' else strength[indices, i] > 0
        if target.any():
            stats[f'target_alpha_mod{i}'] = float(alpha[target.numpy(), i].mean())
    if risk_predictions and kind != 'conflict':
        for i, task in enumerate(('O', 'A', 'C')):
            if model.mode == f'no_{task.lower()}' or (task == 'C' and model.mode == 'raw_jsd'):
                continue
            select = torch.cat([m[i] for m in risk_masks]).numpy()
            target = torch.cat([v[i] for v in risk_targets]).numpy()[select]
            prediction = torch.cat([v[i] for v in risk_predictions]).numpy()[select]
            stats.update({f'{task}_{k}': v for k, v in regression(target, prediction).items()})
    if kind == 'conflict':
        replaced = swaps[indices].any(1).numpy()
        shifted = is_cross_label[indices].numpy()
        stats['swap_cross_label_n'] = int((replaced & shifted).sum())
        stats['swap_same_label_n'] = int((replaced & ~shifted).sum())
        correct = p.argmax(1) == labels[indices].numpy()
        for key, group in (('cross_label', replaced & shifted), ('same_label', replaced & ~shifted)):
            stats[f'swap_{key}_accuracy'] = float(correct[group].mean()) if group.any() else None
        if extra_conflict:
            conflict = torch.cat(extra_conflict).numpy()
            involved = np.stack([(swaps[indices, i] | swaps[indices, j]).numpy() & valid_mask[:, i] & valid_mask[:, j]
                                for i, j in combinations(range(alpha.shape[1]), 2)], axis=1)
            detected = detection(np.broadcast_to(shifted[:, None], involved.shape)[involved], conflict[involved])
            stats.update({f'conflict_{k}': v for k, v in detected.items()})
    return stats, curve, score


@torch.inference_mode()
def linear_macs(model, features, mask, device):
    total = 0

    def count(layer, args, result):
        nonlocal total
        total += result.numel() * layer.in_features

    hooks = [layer.register_forward_hook(count) for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
    try:
        model.eval()
        model([x[:1].to(device) for x in features], mask[:1].to(device))
    finally:
        for hook in hooks:
            hook.remove()
    return total


def atomic_json(path, content):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def run_one(name, data, classes, method, seed, directory, settings, device, fingerprint, quick, cuda_graph=False):
    directory.mkdir(parents=True, exist_ok=True)
    config = {'protocol': PROTOCOL, 'dataset': name, 'method': method, 'seed': seed,
              'config': asdict(settings), 'device': str(device), 'torch': str(torch.__version__),
              'fingerprint': fingerprint, 'quick': quick}
    if cuda_graph:
        config['cuda_graph'] = True
    report = directory / 'result.json'
    if report.exists():
        previous = json.loads(report.read_text(encoding='utf-8'))
        if previous['run'] != config:
            raise ValueError(f'{directory} has results under a different configuration. Use another --run-dir.')
        print(f'{name} {method} seed={seed} already complete', flush=True)
        return previous
    torch.manual_seed(seed)
    widths = [x.shape[-1] for x in data['train'][0]]
    if method == 'sed_ud_features':
        model = SeDUDFeatures(widths, classes, settings.width, BENCHMARKS[name].index('text'))
    elif method == 'cdpr_features':
        model = CDPRFeatures(widths, classes, settings.width)
    elif method in BASELINES:
        model = Baseline(widths, classes, method, len(data['train'][2]), settings.width)
    else:
        model = HCRDRIB(widths, classes, method, settings.width, settings.latent)
    model.to(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    print(f'{name} {method} seed={seed} device={device}', flush=True)
    started = perf_counter()
    loss = fit(model, data['train'], method, seed, settings, device, cuda_graph)
    elapsed = perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == 'cuda' else None
    _, _, val_score = evaluate(model, data['val'], 'clean', None, device, settings)
    threshold = float(np.quantile(val_score, 0.2))
    conditions = {}
    for key, kind, argument in evaluation_plan(BENCHMARKS[name], quick):
        result = evaluate(model, data['test'], kind, argument, device, settings, threshold)
        if result is None:
            continue
        stats, curve, _ = result
        conditions[key] = stats
        np.savetxt(directory / f'coverage_{key}.csv', curve, delimiter=',', comments='',
                   header='coverage,risk,accuracy,maxprob_coverage,maxprob_risk')
    for case, metrics in conditions.items():
        if case.startswith(('only_noise_', 'gaussian_', 'unseen_dropout_')):
            for i in range(len(widths)):
                target = metrics.get(f'target_alpha_mod{i}')
                # Compare against the identical subset under clean input, not the clean population mean.
                if target is None:
                    continue
                plan = next((k, v) for n, k, v in evaluation_plan(BENCHMARKS[name], quick) if n == case)
                _, mask, strength, _, _ = condition(data['test'], *plan)
                selected = (strength[:, i] > 0) & mask[:, i]
                clean_batch = data['test'][0]
                with torch.inference_mode():
                    weights = [model([x[rows].to(device) for x in clean_batch], mask[rows].to(device))['alpha'][:, i]
                               for rows in selected.nonzero().flatten().split(settings.batch)]
                    clean_alpha = float(torch.cat(weights).mean().cpu())
                metrics[f'clean_target_alpha_mod{i}'] = clean_alpha
                metrics[f'target_delta_alpha_mod{i}'] = target - clean_alpha
    ckpt = directory / 'model.tmp'
    torch.save(model.state_dict(), ckpt)
    ckpt.replace(directory / 'model.pt')
    output = {'run': config, 'trainable_parameters': sum(p.numel() for p in model.parameters()),
              'train_seconds': elapsed, 'peak_cuda_mib': peak, 'val_80pct_threshold': threshold,
              'linear_macs_per_sample': linear_macs(model, *data['val'][:2], device),
              'modalities': BENCHMARKS[name], 'conditions': conditions, 'training': loss}
    atomic_json(report, output)
    return output


def digest_cache(root, name):
    result = {}
    for split in SPLITS:
        with (root / name / f'{split}.npz').open('rb') as handle:
            result[split] = file_digest(handle, 'sha256').hexdigest()
    return result


def export_summary(reports, destination):
    grouped = {}
    for report in reports:
        run = report['run']
        for case, values in report['conditions'].items():
            group = grouped.setdefault((run['dataset'], run['method'], case), {})
            resource = {k: report[k] for k in ('trainable_parameters', 'train_seconds', 'peak_cuda_mib', 'linear_macs_per_sample')}
            for metric, value in (values | resource).items():
                if value is not None:
                    group.setdefault(metric, []).append(value)
    columns = ('dataset', 'method', 'condition', 'metric', 'mean', 'std', 'n_seeds')
    partial = destination.with_suffix('.tmp')
    with partial.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for (dataset, method, case), group in sorted(grouped.items()):
            for metric, values in sorted(group.items()):
                writer.writerow(dict(zip(columns, [dataset, method, case, metric, np.mean(values),
                                                   np.std(values, ddof=1) if len(values) > 1 else '', len(values)])))
    partial.replace(destination)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=BENCHMARKS)
    parser.add_argument('--suite', choices=('core', 'full', 'validation'), default='core')
    parser.add_argument('--methods', nargs='+', choices=ALL)
    parser.add_argument('--feature-dir', type=Path, default=Path('data/features'))
    parser.add_argument('--run-dir', type=Path, default=Path('runs/fixed_v2'))
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--skip-summary', action='store_true', help='Defer summary export when running disjoint workers.')
    parser.add_argument('--cuda-graph', action='store_true', help='Replay fixed-size training batches with CUDA graphs.')
    parser.add_argument('--smoke', action='store_true', help='One epoch per stage, one seed, four test scenarios.')
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Check your PyTorch interpreter or use --device cpu.')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    settings = Settings(clean_epochs=1, risk_epochs=1, joint_epochs=1) if args.smoke else Settings()
    seeds = SEEDS[:1] if args.smoke else SEEDS
    methods = args.methods or {'core': CORE, 'full': ALL, 'validation': VALIDATION}[args.suite]
    if args.cuda_graph and (device.type != 'cuda' or not set(methods).issubset(VALIDATION)):
        parser.error('--cuda-graph requires CUDA and methods from the validation suite.')
    out = args.run_dir / ('smoke' if args.smoke else 'protocol')
    defaults = ('ch_sims', 'mintrec', 'upmc_food101') if args.suite == 'validation' else DEFAULT_BENCHMARKS
    datasets = dict.fromkeys(args.datasets or defaults)
    missing = [name for name in datasets if any(not (args.feature_dir / name / f'{s}.npz').is_file() for s in SPLITS)]
    if missing:
        parser.error(f'Missing prepared caches in {args.feature_dir}: {", ".join(missing)}. See prepare.py and README.')
    reports = []
    for name in datasets:
        data, classes = load_data(args.feature_dir, name)
        print(f'{name}: K={classes}; ' + ', '.join(f'{k}={len(v[2])} masks={v[1].sum(0).tolist()}'
                                                   for k, v in data.items()), flush=True)
        if args.inspect:
            continue
        fingerprint = digest_cache(args.feature_dir, name)
        capture = args.cuda_graph or (args.suite == 'validation' and name == 'upmc_food101'
                                       and device.type == 'cuda' and set(methods).issubset(VALIDATION))
        for method in dict.fromkeys(methods):
            for seed in seeds:
                dest = out / name / method / str(seed)
                report = run_one(name, data, classes, method, seed, dest, settings, device,
                                 fingerprint, args.smoke, capture)
                reports.append(report)
                if not args.skip_summary:
                    export_summary(reports, out / 'summary.csv')
    if reports and not args.skip_summary:
        print(f'Summary: {out / "summary.csv"}')


if __name__ == '__main__':
    main()
