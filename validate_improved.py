"""Run the fixed-parameter task-risk validation against cached matched baselines."""
from argparse import ArgumentParser
from copy import deepcopy
from csv import DictWriter
from dataclasses import asdict
from itertools import product
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
import json

# main initializes CUDA DLLs before any module imports PyTorch on Windows.
from main import (SEEDS, Settings, atomic_json, batches, digest_cache,
                  evaluation_plan, export_summary, linear_macs, torch)
from filelock import FileLock
import numpy as np

from data import BENCHMARKS, condition, load_data, training_batch
from improved import MODES, TaskRisk
from metrics import detection, summarize
from model import classification_loss


ROOT = Path(__file__).resolve().parent
DATASETS = ('cmu_mosi', 'ch_sims', 'meld_small', 'iemocap')
BASELINES = ('late', 'concat', 'gate', 'tmc', 'etmc', 'qmf', 'mmdynamics',
             'full', 'fixed_ib', 'no_ib', 'no_csf')
PROTOCOL = 'task-risk-fixed-1'


def fit(model, train, settings, seed, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    swap_inputs, _, _, swapped, _ = condition(train, 'conflict')
    logs = []
    for phase, epochs, offset in (('clean', settings.clean_epochs, 351),
                                  ('risk', settings.risk_epochs, 1351),
                                  ('joint', settings.joint_epochs, 1351)):
        if phase == 'risk':
            teacher = deepcopy(model).eval().requires_grad_(False)
        loader = batches(train, settings.batch, torch.Generator().manual_seed(seed + offset))
        perturb = torch.Generator().manual_seed(seed + (1712 if phase == 'joint' else 712))
        swap_rng = torch.Generator().manual_seed(seed + (3321 if phase == 'joint' else 3320))
        shuffle_rng = torch.Generator().manual_seed(seed + 833)
        for epoch in range(epochs):
            model.train()
            totals = torch.zeros(4, device=device)
            for *clean, original_mask, labels, rows in loader:
                y = labels.to(device)
                clean_input = [x.to(device) for x in clean]
                zero = y.new_zeros((), dtype=torch.float32)
                kl, risk = zero, zero
                if phase == 'clean':
                    mask = original_mask.to(device)
                    result = model(clean_input, mask, uniform=True)
                    task = classification_loss(result, y, mask)
                else:
                    noisy, keep, _ = training_batch(clean, original_mask, perturb)
                    mask = keep.to(device)
                    noisy = [x.to(device) for x in noisy]
                    eligible = swapped[rows].any(1)
                    select = (torch.rand(len(rows), generator=swap_rng) < 0.25) & eligible
                    source_rows = rows[select]
                    selected = select.to(device)
                    # Auxiliary swapped copies add risk supervision. The classification
                    # minibatch remains the same as in the original protocol.
                    observed = [torch.cat((x, sx[source_rows].to(device))) for x, sx in zip(noisy, swap_inputs)]
                    observed_mask = torch.cat((mask, original_mask[select].to(device)))
                    all_y = torch.cat((y, y[selected]))
                    result = model(observed, observed_mask)
                    with torch.no_grad():
                        _, _, clean_p = teacher.encode(clean_input)
                        _, _, noisy_p = teacher.encode(observed)
                        targets = teacher.targets(torch.cat((clean_p, clean_p[selected])), noisy_p, all_y)
                    sw = torch.cat((torch.zeros_like(keep), swapped[source_rows])).to(device)
                    swap_pairs = sw[:, model.left] | sw[:, model.right]
                    kl, risk = model.supervised_risk(result, targets, swap_pairs, shuffle_rng)
                    result = {k: result[k][:len(y)] for k in ('p', 'single')}
                    task = zero if phase == 'risk' else classification_loss(result, y, mask)
                total = task + settings.beta * kl + settings.gamma * risk
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                optimizer.step()
                totals += torch.stack((total, task, kl, risk)).detach() * len(y)
            values = (totals / len(train[2])).tolist()
            if not np.isfinite(values).all():
                raise FloatingPointError(f'Non-finite loss in {phase} epoch {epoch + 1}')
            logs.append(dict(zip(('total', 'task', 'kl', 'risk'), values), phase=phase, epoch=epoch + 1))
            if epoch == 0 or epoch + 1 == epochs:
                print(f'  {phase} {epoch + 1}/{epochs} loss={values[0]:.5f}', flush=True)
    return logs


@torch.inference_mode()
def evaluate(model, split, kind, argument, settings, device, threshold=None):
    inputs, mask, strength, swapped, cross = condition(split, kind, argument)
    valid = mask.any(1)
    if not valid.any():
        return None, None, None
    inputs, mask, labels = [x[valid] for x in inputs], mask[valid], split[2][valid]
    outputs = [model([x[rows].to(device) for x in inputs], mask[rows].to(device))
               for rows in torch.arange(len(labels)).split(settings.batch)]
    collect = lambda key: torch.cat([out[key].cpu() for out in outputs])
    p, score, alpha = collect('p'), collect('score'), collect('alpha')
    stats, curve = summarize(labels.numpy(), p.numpy(), score.numpy(), threshold)
    stats['sample_fraction'] = valid.float().mean().item()
    for i in range(mask.shape[1]):
        if mask[:, i].any():
            stats[f'alpha_mod{i}'] = alpha[mask[:, i], i].mean().item()
        targeted = (swapped if kind == 'conflict' else strength)[valid, i] > 0
        if targeted.any():
            stats[f'target_alpha_mod{i}'] = alpha[targeted, i].mean().item()
    if kind == 'conflict':
        swapped, cross = swapped[valid], cross[valid]
        paired = swapped.any(1)
        correct = p.argmax(1) == labels
        for name, group in (('cross_label', paired & cross), ('same_label', paired & ~cross)):
            stats[f'swap_{name}_n'] = int(group.sum())
            stats[f'swap_{name}_accuracy'] = correct[group].float().mean().item() if group.any() else None
        left, right = model.left.cpu(), model.right.cpu()
        touched = (swapped[:, left] | swapped[:, right]) & mask[:, left] & mask[:, right]
        harm = torch.cat([out['risks'][2].cpu() for out in outputs])
        anomaly = collect('swap_logit')
        for metric, value in detection(cross[:, None].expand_as(touched)[touched], harm[touched]).items():
            stats[f'conflict_{metric}'] = value
        if touched.any():
            clean = [x[valid] for x in split[0]]
            clean_logits = torch.cat([model([x[rows].to(device) for x in clean],
                                mask[rows].to(device))['swap_logit'].cpu()
                                for rows in torch.arange(len(labels)).split(settings.batch)])
            for name, group in (('all', touched), ('same', touched & ~cross[:, None]),
                                ('cross', touched & cross[:, None])):
                n = int(group.sum())
                if n:
                    target = np.repeat([0, 1], n)
                    scores = torch.cat((clean_logits[group], anomaly[group]))
                    for metric, value in detection(target, scores).items():
                        stats[f'paired_swap_{name}_{metric}'] = value
    return stats, curve, score.numpy()


def run_one(dataset, method, seed, data, classes, fingerprint, settings, device, folder, smoke):
    run = dict(protocol=PROTOCOL, dataset=dataset, method=method, seed=seed, config=asdict(settings),
               fingerprint=fingerprint, device=str(device), torch=str(torch.__version__), quick=smoke)
    path = folder / 'result.json'
    if path.exists():
        report = json.loads(path.read_text(encoding='utf-8'))
        if report['run'] != run:
            raise ValueError(f'Saved settings differ in {folder}; use another run directory.')
        return report
    torch.manual_seed(seed)
    model = TaskRisk([x.shape[1] for x in data['train'][0]], classes, method,
                     settings.width, settings.latent).to(device)
    print(f'{dataset} {method} seed={seed}', flush=True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    logs = fit(model, data['train'], settings, seed, device)
    elapsed = perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == 'cuda' else None
    model.eval()
    _, _, val_score = evaluate(model, data['val'], 'clean', None, settings, device)
    threshold = float(np.quantile(val_score, 0.2))
    conditions = {}
    for key, kind, argument in evaluation_plan(BENCHMARKS[dataset], smoke):
        stats, curve, _ = evaluate(model, data['test'], kind, argument, settings, device, threshold)
        if stats is not None:
            conditions[key] = stats
            np.savetxt(folder / f'coverage_{key}.csv', curve, delimiter=',', comments='',
                       header='coverage,risk,accuracy,maxprob_coverage,maxprob_risk')
    ckpt = folder / 'model.tmp'
    torch.save(model.state_dict(), ckpt)
    ckpt.replace(folder / 'model.pt')
    report = dict(run=run, conditions=conditions, training=logs, train_seconds=elapsed,
                  peak_cuda_mib=peak, val_80pct_threshold=threshold,
                  trainable_parameters=sum(p.numel() for p in model.parameters()),
                  linear_macs_per_sample=linear_macs(model, *data['val'][:2], device))
    atomic_json(path, report)
    return report


def paired_table(reports, destination, datasets=DATASETS, baselines=BASELINES):
    index = {(r['run']['dataset'], r['run']['method'], r['run']['seed']): r['conditions'] for r in reports}
    names = ('clean', 'gaussian_0.9', 'unseen_dropout_0.5', 'random_missing_0.8', 'conflict')
    comparisons = [('task_fixed', b) for b in (*baselines, 'task_uniform')]
    comparisons += [('task_dynamic', 'task_fixed'), ('task_dynamic', 'task_shuffled')]
    fields = ('dataset', 'method', 'baseline', 'condition', 'metric', 'n_seeds', 'difference',
              'ci95_low', 'ci95_high', 'p_unadjusted', 'seed_differences')
    with destination.open('w', encoding='utf-8-sig', newline='') as file:
        writer = DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for dataset, (method, baseline), name, metric in product(
                datasets, comparisons, names, ('accuracy', 'macro_f1', 'aurc', 'maxprob_aurc')):
            pairs = [tuple(index.get((dataset, m, seed), {}).get(name, {}).get(metric)
                           for m in (method, baseline)) for seed in SEEDS]
            differences = [a - b for a, b in pairs if a is not None and b is not None]
            if len(differences) != len(SEEDS):
                continue
            delta = mean(differences)
            margin = 2.776445 * stdev(differences) / len(SEEDS) ** 0.5
            flips = list(product((-1, 1), repeat=len(SEEDS)))
            p = mean(abs(mean(s * d for s, d in zip(sign, differences))) >= abs(delta) - 1e-12
                     for sign in flips)
            writer.writerow(dict(zip(fields, (dataset, method, baseline, name, metric, len(SEEDS), delta,
                                              delta - margin, delta + margin, p, differences))))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--features', type=Path, default=ROOT / 'data/features')
    parser.add_argument('--reference', type=Path, default=ROOT / 'runs/small_fixed/protocol')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/task_risk_fixed')
    parser.add_argument('--worker-results', type=Path, help='Reuse per-dataset runs under this directory.')
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable in this interpreter. Select a CUDA-enabled PyTorch environment.')
    device = torch.device(args.device)
    settings = Settings(clean_epochs=1, risk_epochs=1, joint_epochs=1) if args.smoke else Settings()
    seeds = SEEDS[:1] if args.smoke else SEEDS
    out = args.out / ('smoke' if args.smoke else 'protocol')
    reports = []
    datasets = tuple(dict.fromkeys(args.datasets))
    fingerprints = {name: digest_cache(args.features, name) for name in datasets}
    if not args.smoke:
        for name, method, seed in product(datasets, BASELINES, seeds):
            path = args.reference / name / method / str(seed) / 'result.json'
            report = json.loads(path.read_text(encoding='utf-8'))
            run = report['run']
            if (run['protocol'] != 'v2-fixed-1' or run['fingerprint'] != fingerprints[name]
                    or run['config'] != asdict(settings) or run['quick']
                    or (run['dataset'], run['method'], run['seed']) != (name, method, seed)):
                raise ValueError(f'Baseline does not match split, budget or seed: {path}')
            reports.append(report)
    for name in datasets:
        data, classes = load_data(args.features, name)
        dataset_out = out / name
        if args.worker_results:
            dataset_out = args.worker_results / name / out.name / name
        for method, seed in product(MODES, seeds):
            folder = dataset_out / method / str(seed)
            folder.mkdir(parents=True, exist_ok=True)
            with FileLock(str(folder / '.run.lock')):
                reports.append(run_one(name, method, seed, data, classes, fingerprints[name],
                                       settings, device, folder, args.smoke))
    out.mkdir(parents=True, exist_ok=True)
    export_summary(reports, out / 'summary.csv')
    if not args.smoke:
        paired_table(reports, out / 'paired.csv')
    atomic_json(out / 'COMPLETE.json', dict(protocol=PROTOCOL, datasets=datasets, seeds=seeds,
        new_runs=len(datasets) * len(seeds) * len(MODES), matched_runs=len(reports), smoke=args.smoke))
    print(f'Completed {len(reports)} runs including matched baselines: {out}', flush=True)


if __name__ == '__main__':
    main()
