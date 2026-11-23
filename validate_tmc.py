"""Five-seed feature-space evaluation of TMC-HCRD-RIB."""
from argparse import ArgumentParser
from copy import deepcopy
from csv import DictWriter
from dataclasses import asdict
from itertools import product
from pathlib import Path
from time import perf_counter
import json

import numpy as np
import torch
from filelock import FileLock

from data import BENCHMARKS, condition, load_data, training_batch
from main import Settings, atomic_json, batches, digest_cache, evaluation_plan, export_summary, linear_macs
from metrics import detection, regression, summarize
from tmc_hcrd import METHODS, TMCHCRD, task_loss


DATASETS = ('cmu_mosi', 'ch_sims', 'meld', 'iemocap')
SEEDS = (11, 22, 33, 44, 55)
PROTOCOL = 'tmc-hcrd-feature-v1'


def fit(model, train, seed, settings, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    teacher = None
    classification_epoch = 0
    for phase, epochs, offset in [('clean', settings.clean_epochs, 300),
                                  ('risk', settings.risk_epochs, 700),
                                  ('joint', settings.joint_epochs, 1300)]:
        if phase == 'risk':
            if not model.uses_risk:
                continue
            teacher = deepcopy(model).eval().requires_grad_(False)
        loader = batches(train, settings.batch, torch.Generator().manual_seed(seed + offset))
        rng = torch.Generator().manual_seed(seed + offset + 1)
        for epoch in range(epochs):
            model.train()
            total = torch.zeros((), device=device)
            for *clean, mask, y, _ in loader:
                if phase == 'clean':
                    current = clean
                else:
                    current, mask, strength = training_batch(clean, mask, rng)
                current = [x.to(device) for x in current]
                mask, y = mask.to(device), y.to(device)
                output = model(current, mask, warmup=phase == 'clean', risk_only=phase == 'risk')
                loss = task_loss(output, y, mask, classification_epoch) if phase != 'risk' else 0
                if teacher is not None:
                    refs = model.references(teacher, [x.to(device) for x in clean], current)
                    loss = loss + settings.gamma * model.risk_loss(output, mask, strength.to(device),
                                                                   refs, settings.beta)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total += loss.detach() * len(y)
            if not total.isfinite():
                raise FloatingPointError(f'{phase} epoch {epoch + 1}: non-finite loss')
            if phase != 'risk':
                classification_epoch += 1
            if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == epochs:
                print(f'  {phase} {epoch + 1}/{epochs} loss={float(total) / len(train[2]):.5f}', flush=True)
    return teacher


@torch.inference_mode()
def evaluate(model, teacher, data, kind, argument, device, batch, threshold=None):
    model.eval()
    clean, _, y = data
    inputs, mask, strength, swaps, cross_label = condition(data, kind, argument)
    rows = mask.any(1).nonzero().flatten()
    results = []
    reference = []
    for index in rows.split(batch):
        current = [x[index].to(device) for x in inputs]
        outputs = model(current, mask[index].to(device))
        results.append({k: v.cpu() for k, v in outputs.items()})
        if teacher is not None and kind != 'conflict':
            reference.append(tuple(v.cpu() for v in model.references(
                teacher, [x[index].to(device) for x in clean], current)))
    out = {k: torch.cat([r[k] for r in results]).numpy() for k in results[0]}
    score = out['score']
    stats, curve = summarize(y[rows].numpy(), out['p'], score, threshold)
    stats['sample_fraction'] = len(rows) / len(y)
    valid = mask[rows].numpy()
    stats['discount_mean'] = float(out['discount'][valid].mean())
    stats['discount_vacuity_spearman'] = regression(out['u'][valid], out['discount'][valid])['spearman']
    polluted = (strength[rows].numpy() > 0) & valid
    if polluted.any():
        stats['polluted_discount_mean'] = float(out['discount'][polluted].mean())
    if reference:
        delta, normal, clean_h = [torch.cat(group).numpy() for group in zip(*reference)]
        pair = out['pair_mask']
        stats['pollution_auroc'] = detection(polluted[valid], out['o'][valid])['auroc']
        for name, target, estimate, selected in [('O', strength[rows].numpy(), out['o'], valid),
                ('A_delta', delta, out['delta'], valid), ('A_residual', clean_h, out['a'], valid),
                ('C', normal, out['normal'], pair)]:
            stats.update({f'{name}_{key}': value for key, value in
                          regression(target[selected], estimate[selected]).items()})
    if kind == 'conflict':
        replaced = swaps[rows].any(1).numpy()
        cross = cross_label[rows].numpy()
        correct = out['p'].argmax(1) == y[rows].numpy()
        for name, selected in [('same_class', replaced & ~cross), ('other_class', replaced & cross)]:
            stats[f'{name}_swap_n'] = int(selected.sum())
            stats[f'{name}_swap_accuracy'] = float(correct[selected].mean()) if selected.any() else None
        if teacher is not None:
            i, j = model.i.cpu(), model.j.cpu()
            affected = (swaps[rows][:, i] | swaps[rows][:, j]).numpy() & out['pair_mask']
            labels = np.broadcast_to(cross[:, None], affected.shape)[affected]
            stats['conflict_auroc'] = detection(labels, out['excess'][affected])['auroc']
            stats['raw_js_auroc'] = detection(labels, out['raw'][affected])['auroc']
    return stats, curve, score


def run_cell(name, data, classes, method, seed, root, cfg, device, fingerprint, smoke):
    folder = root / name / method / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'result.json'
    config = dict(protocol=PROTOCOL, dataset=name, method=method, seed=seed, settings=asdict(cfg),
                  fingerprint=fingerprint, device=str(device), torch=str(torch.__version__), smoke=smoke)
    with FileLock(str(folder / '.lock')):
        if path.exists():
            existing = json.loads(path.read_text(encoding='utf-8'))
            if existing['run'] != config:
                raise ValueError(f'Experiment configuration changed for {folder}. Choose a fresh run directory.')
            return existing
        torch.manual_seed(seed)
        model = TMCHCRD([x.shape[1] for x in data['train'][0]], classes, method, cfg.width, cfg.latent).to(device)
        print(f'{name} {method} seed={seed} [{device}]', flush=True)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        started = perf_counter()
        teacher = fit(model, data['train'], seed, cfg, device)
        duration = perf_counter() - started
        peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else None
        _, _, score = evaluate(model, None, data['val'], 'clean', None, device, cfg.batch)
        threshold = float(np.quantile(score, .2))
        conditions = {}
        plan = evaluation_plan(BENCHMARKS[name], smoke)
        if not smoke:
            plan = plan + [('gaussian_0.7', 'gaussian', .7)]
        for key, kind, arg in plan:
            stats, curve, _ = evaluate(model, teacher, data['test'], kind, arg, device, cfg.batch, threshold)
            conditions[key] = stats
            np.savetxt(folder / f'coverage_{key}.csv', curve, delimiter=',', comments='',
                       header='coverage,risk,accuracy,maxprob_coverage,maxprob_risk')
        torch.save(model.state_dict(), folder / 'model.tmp')
        (folder / 'model.tmp').replace(folder / 'model.pt')
        report = dict(run=config, conditions=conditions, train_seconds=duration, peak_cuda_mib=peak,
                      trainable_parameters=sum(p.numel() for p in model.parameters()),
                      linear_macs_per_sample=linear_macs(model, *data['val'][:2], device))
        atomic_json(path, report)
        return report


def paired_comparisons(reports, path):
    lookup = {(r['run']['dataset'], r['run']['method'], r['run']['seed']): r for r in reports}
    comparisons = [('simple', 'tmc'), ('hcrd', 'simple'), ('fixed_ib', 'hcrd'),
                   ('dynamic_ib', 'fixed_ib'), ('full', 'dynamic_ib'), ('full', 'tmc')]
    comparisons += [('full', method) for method in ('no_o', 'no_a', 'no_c', 'only_o', 'only_a')]
    fields = ['dataset', 'condition', 'metric', 'method', 'reference', 'n',
              'delta_mean', 'delta_std', 'sign_flip_p']
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for name in sorted({key[0] for key in lookup}):
            for method, reference in comparisons:
                pairs = [(lookup[name, method, s], lookup[name, reference, s]) for s in SEEDS
                         if (name, method, s) in lookup and (name, reference, s) in lookup]
                if not pairs:
                    continue
                for case in pairs[0][0]['conditions']:
                    for metric in ('accuracy', 'macro_f1', 'ece15', 'nll', 'brier', 'aurc',
                                   'pollution_auroc', 'A_delta_mae', 'C_mae', 'conflict_auroc'):
                        values = [(a['conditions'].get(case, {}).get(metric),
                                   b['conditions'].get(case, {}).get(metric)) for a, b in pairs]
                        delta = np.array([a - b for a, b in values if a is not None and b is not None])
                        if not len(delta):
                            continue
                        n = len(delta)
                        permuted = np.array(list(product((-1, 1), repeat=n))) @ delta / n
                        p = np.mean(np.abs(permuted) >= abs(delta.mean()) - 1e-12)
                        writer.writerow(dict(zip(fields, [name, case, metric, method, reference, n,
                            delta.mean(), delta.std(ddof=1) if n > 1 else '', p if n == len(SEEDS) else ''])))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=(*DATASETS, 'meld_small'), default=DATASETS)
    parser.add_argument('--methods', nargs='+', choices=METHODS, default=METHODS)
    parser.add_argument('--seeds', nargs='+', type=int, choices=SEEDS)
    parser.add_argument('--feature-dir', type=Path, default=Path('data/features'))
    parser.add_argument('--run-dir', type=Path, default=Path('runs/tmc_hcrd'))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable in this interpreter. Select a CUDA-enabled PyTorch environment.')
    device = torch.device(args.device)
    cfg = Settings(clean_epochs=1, risk_epochs=1, joint_epochs=1) if args.smoke else Settings()
    seeds = args.seeds or (SEEDS[:1] if args.smoke else SEEDS)
    root = args.run_dir / ('smoke' if args.smoke else 'protocol')
    for name in args.datasets:
        data, classes = load_data(args.feature_dir, name)
        fingerprint = digest_cache(args.feature_dir, name)
        for seed, method in product(seeds, args.methods):
            run_cell(name, data, classes, method, seed, root, cfg, device, fingerprint, args.smoke)
        with FileLock(str(root / '.summary.lock')):
            reports = [json.loads(path.read_text(encoding='utf-8')) for path in root.glob('*/*/*/result.json')]
            export_summary(reports, root / 'summary.csv')
            paired_comparisons(reports, root / 'paired.csv')


if __name__ == '__main__':
    main()
