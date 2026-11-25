"""Fixed-parameter validation of v3-S using cached features and matched controls."""
from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import asdict
from itertools import combinations, product
from pathlib import Path
from time import perf_counter
import json

# Import main first to initialize Windows CUDA DLLs.
from main import (SEEDS, Settings, atomic_json, digest_cache, evaluation_plan, export_summary,
                  linear_macs, synchronize, torch)
from filelock import FileLock
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from data import BENCHMARKS, condition, load_data, training_batch
from metrics import regression, summarize
from model import EPS, classification_loss
from validate_improved import BASELINES, DATASETS
from v3s import (Classifier, MODES, V3S, fusion_weights, proxy_targets, task_masks, task_targets)


ROOT = Path(__file__).resolve().parent
PROTOCOL = 'v3s-fixed-1'
FOLDS = 3
TRAIN_VIEWS = 4
EVAL_BATCH = 1024
FUSIONS = ('uniform', 'original_csf', 'softmax', 'conservative', 'oracle')
PAIR_CONDITIONS = ('clean', 'gaussian_0.9', 'unseen_dropout_0.5', 'random_missing_0.8', 'conflict')


def batches(n, batch, device, rng=None):
    order = (torch.arange(n) if rng is None else torch.randperm(n, generator=rng)).to(device)
    return order.split(batch)


def subset(views, rows):
    return [x[rows] for x in views]


def train_teacher(dims, classes, split, settings, seed, device):
    torch.manual_seed(seed)
    model = Classifier(dims, classes, settings.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay,
                                  fused=device.type == 'cuda')
    views, mask, labels = split
    rng = torch.Generator().manual_seed(seed + 351)
    for _ in range(settings.clean_epochs):
        for rows in batches(len(labels), settings.batch, device, rng):
            optimizer.zero_grad(set_to_none=True)
            loss = classification_loss(model(subset(views, rows), mask[rows]), labels[rows], mask[rows])
            loss.backward()
            optimizer.step()
    return model.eval()


@torch.no_grad()
def teacher_predictions(model, views, batch):
    return torch.cat([model.encode(subset(views, rows))[2]
                      for rows in batches(len(views[0]), batch, views[0].device)])


def fold_transform(views, mask):
    stats = []
    for i, x in enumerate(views):
        present = x[mask[:, i]]
        if len(present) == 0:
            raise ValueError(f'No training-fold observations for modality {i}')
        scale = present.std(0, unbiased=False)
        stats.append((present.mean(0), torch.where(scale < 1e-5, 1, scale)))
    return stats


def transform(views, stats):
    return [(x - mean) / scale for x, (mean, scale) in zip(views, stats)]


def same_fold_swaps(labels, mask, rng):
    source = torch.arange(len(labels))[:, None].expand_as(mask).clone()
    eligible = mask.sum(1) > 1
    selected = eligible & (torch.rand(len(labels), generator=rng) < 0.25)
    cross = torch.rand(len(labels), generator=rng) < 0.5
    picked = torch.rand(mask.shape, generator=rng).masked_fill(~mask, -1).argmax(1)
    for i in range(mask.shape[1]):
        available = mask[:, i].nonzero().flatten()
        for row in (selected & (picked == i)).nonzero().flatten().tolist():
            different = labels[available] != labels[row]
            pool = available[(different == cross[row]) & (available != row)]
            if len(pool):
                source[row, i] = pool[torch.randint(len(pool), (), generator=rng)]
    return source


def training_views(train, groups, dims, classes, settings, seed, device, pairs, smoke):
    views, mask, labels = train
    left, right = pairs
    folds = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    rng = torch.Generator().manual_seed(seed + 1712)
    fold_splits = list(folds.split(groups, labels, groups))
    banks = []
    for train_rows, held_rows in fold_splits:
        fit_views, fit_mask = subset(views, train_rows), mask[train_rows]
        fit_views, fit_mask = [x.to(device) for x in fit_views], fit_mask.to(device)
        scales = fold_transform(fit_views, fit_mask)
        teacher = train_teacher(dims, classes, (transform(fit_views, scales), fit_mask,
                                labels[train_rows].to(device)), settings, seed, device)
        clean, original, y = subset(views, held_rows), mask[held_rows], labels[held_rows]
        clean_p = teacher_predictions(teacher, transform([x.to(device) for x in clean], scales), settings.batch)
        local = torch.arange(len(y))[:, None].expand_as(original)
        for version in range(1 if smoke else TRAIN_VIEWS):
            if version == len(banks):
                banks.append([])
            noisy, keep, strength = training_batch(clean, original, rng)
            donor = same_fold_swaps(y, keep, rng)
            swapped = donor != local
            attached = swapped.any(1)
            source = torch.cat((local, donor[attached]))
            # The restored pairing receives the identical observation-noise realization.
            actual = [torch.cat((x, c[donor[attached, i]] + (x - c)[attached]))
                      for i, (x, c) in enumerate(zip(noisy, clean))]
            reference = [torch.cat((x, x[attached])) for x in noisy]
            targets_mask = torch.cat((keep, keep[attached])).to(device)
            strength = torch.cat((strength, strength[attached])).to(device)
            y_actual = torch.cat((y, y[attached])).to(device)
            changed = torch.cat((torch.zeros_like(swapped), swapped[attached])).to(device)
            actual, reference = [x.to(device) for x in actual], [x.to(device) for x in reference]
            source = source.to(device)
            source_p = clean_p[source, torch.arange(mask.shape[1], device=device)]
            source_y = y.to(device)[source]
            current_p = teacher_predictions(teacher, transform(actual, scales), settings.batch)
            ref_p = teacher_predictions(teacher, transform(reference, scales), settings.batch)
            targets = task_targets(source_p, current_p, ref_p, source_y, y_actual, targets_mask,
                                   changed, left, right)
            proxies = proxy_targets(source_p, current_p, strength, targets, left, right)
            is_primary = torch.arange(len(y_actual), device=device) < len(y)
            banks[version].append((actual, targets_mask, y_actual, is_primary, targets, proxies))
    packed = []
    for bank in banks:
        views = [torch.cat([part[0][i] for part in bank]) for i in range(len(dims))]
        mask, y, primary = [torch.cat([part[i] for part in bank]) for i in (1, 2, 3)]
        targets, proxies = [tuple(torch.cat(values) for values in zip(*(part[i] for part in bank)))
                            for i in (4, 5)]
        valid = task_masks(mask, left, right)
        packed.append((views, mask, y, primary, targets, proxies, valid))
    return packed


def risk_plan(banks, method, seed):
    ib = MODES[method]
    shuffle = torch.Generator().manual_seed(seed + 833)
    targets = [b[5] if method == 'old_risk' else b[4] for b in banks]
    raw = []
    for target, bank in zip(targets, banks):
        delta = (*target[:3], (2 * target[3] - 1).abs(), target[4])
        weight = [(1 + d).reciprocal() if ib in {'dynamic', 'shuffled'} else
                  1 + d if ib == 'reverse' else torch.ones_like(d) for d in delta]
        if ib == 'shuffled':
            for w, valid in zip(weight, bank[6]):
                values = w[valid]
                w[valid] = values[torch.randperm(len(values), generator=shuffle).to(w.device)]
        raw.append(weight)
    total = [sum(bank[6][t].sum() for bank in banks).clamp_min(1) for t in range(5)]
    normalizers = [sum((w[t] * b[6][t]).sum() for w, b in zip(raw, banks)) / total[t]
                   for t in range(5)]
    return [(target, [w / norm.clamp_min(EPS) for w, norm in zip(weight, normalizers)])
            for target, weight in zip(targets, raw)]



def joined_batches(banks, batch):
    """Keep fixed shapes for replay and mark padding separately from the modality mask."""
    start = 0
    schedule = []
    slots = 0
    for bank in banks:
        primary = bank[3].cpu()
        main_rows = primary.nonzero().flatten() + start
        aux_rows = (~primary).nonzero().flatten() + start
        n_steps = (len(main_rows) + batch - 1) // batch
        slots = max(slots, (len(aux_rows) + n_steps - 1) // n_steps)
        schedule.append((main_rows, aux_rows, n_steps))
        start += len(primary)
    return schedule, batch + slots


def epoch_batches(schedule, batch, size, rng, device):
    main, aux, steps = schedule
    main = main[torch.randperm(len(main), generator=rng)].split(batch)
    aux = aux[torch.randperm(len(aux), generator=rng)].tensor_split(steps)
    indices = torch.zeros((steps, size), dtype=torch.long)
    present = torch.zeros((steps, size), dtype=torch.bool)
    for i, (primary, extra) in enumerate(zip(main, aux)):
        indices[i, :len(primary)] = primary
        indices[i, batch:batch + len(extra)] = extra
        present[i, :len(primary)] = True
        present[i, batch:batch + len(extra)] = True
    return indices.to(device), present.to(device)


def fit(model, banks, settings, seed, device, eager):
    use_graph = device.type == 'cuda' and not eager
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay,
                                  fused=device.type == 'cuda', capturable=use_graph)
    plans = risk_plan(banks, model.method, seed)
    schedule, size = joined_batches(banks, settings.batch)
    views = [torch.cat(parts) for parts in zip(*(b[0] for b in banks))]
    mask, y = [torch.cat([b[i] for b in banks]) for i in (1, 2)]
    targets = tuple(torch.cat(parts) for parts in zip(*(plan[0] for plan in plans)))
    weights = tuple(torch.cat(parts) for parts in zip(*(plan[1] for plan in plans)))
    valid = tuple(torch.cat(parts) for parts in zip(*(b[6] for b in banks)))
    tasks = [i for i, flag in enumerate(model.active) if flag and valid[i].any()]
    counts = torch.stack([torch.stack([b[6][i].sum() for i in tasks]) for b in banks])
    rows = torch.zeros(size, dtype=torch.long, device=device)
    occupied = torch.zeros(size, dtype=torch.bool, device=device)
    scale = torch.zeros(len(tasks), device=device)
    totals = torch.zeros(5, device=device)
    logs = []
    for phase, epochs, offset in (('risk', settings.risk_epochs, 1351), ('joint', settings.joint_epochs, 2351)):
        generator = torch.Generator().manual_seed(seed + offset)
        joint = phase == 'joint'
        model.backbone.requires_grad_(joint)
        model.train()

        def step():
            keep, labels = mask[rows], y[rows]
            out = model(subset(views, rows), keep)
            task = totals.new_zeros(())
            if joint:
                main = slice(settings.batch)
                main_y, main_mask = labels[main], keep[main]
                fused = out['p'][main].gather(-1, main_y[:, None]).squeeze(-1).clamp_min(EPS)
                single = out['single'][main].gather(-1, main_y[:, None, None].expand(-1, keep.shape[1], 1))
                single_nll = -(single.squeeze(-1).clamp_min(EPS).log() * main_mask).sum(1) / main_mask.sum(1)
                task = ((-fused.log() + single_nll) * occupied[main]).sum() / occupied[main].sum().clamp_min(1)
            errors, rates, raw_rates = [], [], []
            for j, i in enumerate(tasks):
                eligible = valid[i][rows] & occupied[:, None]
                raw_kl = out['kl'][i] * eligible
                error = torch.nn.functional.binary_cross_entropy_with_logits(
                    out['risk_logits'][i], targets[i][rows], reduction='none')
                errors.append((error * eligible).sum() * scale[j])
                raw_rates.append(raw_kl.sum() * scale[j])
                rates.append((raw_kl * weights[i][rows]).sum() * scale[j])
            risk, kl, raw = (torch.stack(v).mean() for v in (errors, rates, raw_rates))
            loss = task + settings.beta * kl + settings.gamma * risk
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals.add_(torch.stack((loss, task, kl, risk, raw)).detach())

        graph = None
        warmed = 0
        for epoch in range(epochs):
            version = epoch % len(banks)
            plan, present = epoch_batches(schedule[version], settings.batch, size, generator, device)
            # Epoch-wide task counts preserve equal task contributions with sparse masks.
            scale.copy_(len(plan) / counts[version].clamp_min(1))
            totals.zero_()
            for indices, available in zip(plan.unbind(0), present.unbind(0)):
                rows.copy_(indices)
                occupied.copy_(available)
                if use_graph and warmed >= 3:
                    if graph is None:
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            step()
                    graph.replay()
                else:
                    step()
                    warmed += 1
            values = (totals / len(plan)).tolist()
            if not np.isfinite(values).all():
                raise FloatingPointError(f'Non-finite {phase} epoch {epoch + 1}')
            log = dict(zip(('total', 'task', 'kl', 'risk', 'raw_kl'), values),
                       phase=phase, epoch=epoch + 1)
            if epoch + 1 == epochs:
                gradients = [p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
                log['combined_gradient_norm_last_batch'] = torch.stack(gradients).norm().item()
                log['kl_weight_means'] = [(w[m].mean().item() if m.any() else None) for w, m in zip(weights, valid)]
            logs.append(log)
            if epoch == 0 or epoch + 1 == epochs:
                print(f'  {phase} {epoch + 1}/{epochs} loss={values[0]:.5f}', flush=True)
        del graph
        if phase == 'risk':
            yield deepcopy(model).eval()
    yield logs


@torch.inference_mode()
def evaluation_data(split, teacher, device, modalities, smoke, pairs):
    clean_p_all = teacher_predictions(teacher, [x.to(device) for x in split[0]], EVAL_BATCH)
    for key, kind, argument in evaluation_plan(modalities, smoke):
        views, mask, _, swapped, cross = condition(split, kind, argument)
        valid = mask.any(1)
        if not valid.any():
            continue
        current = [x[valid].to(device) for x in views]
        keep, y, swapped = mask[valid].to(device), split[2][valid].to(device), swapped[valid].to(device)
        clean_p = clean_p_all[valid.to(device)]
        p = clean_p if views is split[0] else teacher_predictions(teacher, current, EVAL_BATCH)
        source_p = torch.where(swapped[..., None], p, clean_p)
        source_y = y[:, None].expand_as(keep)
        # A on replaced modalities needs the donor label. The legacy evaluator does not
        # expose donor indices, so A is scored only on unchanged modalities here.
        target = task_targets(source_p, p, clean_p, source_y, y, keep, swapped, *pairs)
        yield key, current, keep, y, swapped, cross[valid].to(device), target, float(valid.float().mean())


@torch.inference_mode()
def predictions(model, views, mask, batch):
    outputs = [model(subset(views, rows), mask[rows])
               for rows in batches(len(mask), batch, mask.device)]
    single = torch.cat([out['single'] for out in outputs])
    utility = torch.cat([out['utility'] for out in outputs])
    risks = tuple(torch.cat(parts) for parts in zip(*(out['risks'] for out in outputs)))
    conflict = torch.cat([out['mod_conflict'] for out in outputs])
    return single, utility, risks, conflict


def fused_probabilities(single, utility, risks, conflict, mask, rule, oracle=None):
    if rule == 'original_csf':
        weight = (1 - risks[0]) * (1 - risks[1]) * (1 - conflict) * mask
        mass = weight.sum(1, keepdim=True)
        alpha = torch.where(mass > EPS, weight / mass.clamp_min(EPS), mask / mask.sum(1, keepdim=True))
    elif rule == 'oracle':
        alpha = fusion_weights(oracle, mask)
    else:
        alpha = fusion_weights(utility, mask, rule)
    return (alpha[..., None] * single).sum(1)


def risk_metrics(pred, targets, mask, swapped, pairs, active):
    result = {}
    valid = list(task_masks(mask, *pairs))
    valid[0] = valid[0] & ~swapped
    valid[1] = valid[1] & ~swapped
    # Only evaluate conflict damage where a valid pair was actually changed.
    valid[2] = valid[2] & (swapped[:, pairs[0]] | swapped[:, pairs[1]])
    for name, prediction, target, eligible, enabled in zip('oacu', pred, targets, valid, active):
        if not enabled:
            continue
        prediction, target = prediction[eligible], target[eligible]
        if name == 'u':
            prediction, target = 2 * prediction - 1, 2 * target - 1
        for metric, value in regression(target.cpu().numpy(), prediction.cpu().numpy()).items():
            result[f'{name}_{metric}'] = value
    if active[3]:
        left, right = pairs
        actual = targets[3][:, left] - targets[3][:, right]
        estimated = pred[3][:, left] - pred[3][:, right]
        unequal = valid[3][:, left] & valid[3][:, right] & (actual.abs() > 1e-6)
        if unequal.any():
            result['u_ranking_accuracy'] = ((estimated[unequal] * actual[unequal] > 0).float()
                                           + 0.5 * (estimated[unequal] == 0)).mean().item()
    return result


@torch.inference_mode()
def evaluate(model, cases, val, settings, rules, diagnostics):
    model.eval()
    val_views, val_mask, _ = val
    single, utility, risks, conflict = predictions(model, val_views, val_mask, EVAL_BATCH)
    thresholds = {rule: float(np.quantile(fused_probabilities(single, utility, risks, conflict, val_mask,
                                                             rule).amax(1).cpu(), 0.2))
                  for rule in rules if rule != 'oracle'}
    summaries = {rule: {} for rule in rules}
    curves = {}
    pairs = model.left, model.right
    clean_ms = None
    for key, views, mask, y, swapped, cross, targets, fraction in cases:
        single, utility, risks, conflict = predictions(model, views, mask, EVAL_BATCH)
        if key == 'clean':
            synchronize(mask.device)
            start = perf_counter()
            predictions(model, views, mask, settings.batch)
            synchronize(mask.device)
            clean_ms = (perf_counter() - start) / len(y) * 1000
        for rule in rules:
            oracle = 2 * targets[3] - 1 if rule == 'oracle' else None
            p = fused_probabilities(single, utility, risks, conflict, mask, rule, oracle)
            stats, curve = summarize(y.cpu().numpy(), p.cpu().numpy(), p.amax(1).cpu().numpy(),
                                     thresholds.get(rule))
            incorrect = round((1 - stats['accuracy']) * len(y))
            accepted = np.arange(1, len(y) + 1)
            optimal = (np.maximum(0, accepted - (len(y) - incorrect)) / accepted).mean()
            stats.update(sample_fraction=fraction, e_aurc=stats['aurc'] - optimal)
            if diagnostics and rule == 'conservative':
                stats.update(risk_metrics(risks, targets, mask, swapped, pairs, model.active))
            if key == 'conflict':
                group_scores = []
                for name, selected in (('same_label', swapped.any(1) & ~cross),
                                       ('cross_label', swapped.any(1) & cross)):
                    if selected.any():
                        acc = (p[selected].argmax(1) == y[selected]).float().mean().item()
                        stats[f'swap_{name}_accuracy'] = acc
                        stats[f'swap_{name}_n'] = int(selected.sum())
                        group_scores.append(acc)
                if group_scores:
                    stats['swap_worst_group_accuracy'] = min(group_scores)
            summaries[rule][key] = stats
            if rule == 'conservative':
                curves[key] = curve
    for stats in summaries.values():
        worst = min(case['accuracy'] for key, case in stats.items() if key != 'clean')
        stats['clean']['worst_perturbation_accuracy'] = worst
    return summaries, curves, thresholds, clean_ms


def on_device(split, device):
    views, mask, labels = split
    return [x.to(device) for x in views], mask.to(device), labels.to(device)


def history(directory, protocol, methods, dataset, settings, fingerprint, seeds):
    for method, seed in product(methods, seeds):
        path = directory / dataset / method / str(seed) / 'result.json'
        report = json.loads(path.read_text(encoding='utf-8'))
        run = report['run']
        checks = dict(protocol=protocol, dataset=dataset, method=method, seed=seed,
                      config=asdict(settings), fingerprint=fingerprint, quick=False)
        if any(run.get(k) != value for k, value in checks.items()):
            raise ValueError(f'Historical settings, seed or feature fingerprint differ: {path}')
        yield report


def paired_table(reports, path, seeds):
    from csv import DictWriter
    index = {(r['run']['dataset'], r['run']['method'], r['run']['seed']): r['conditions'] for r in reports}
    datasets = sorted({key[0] for key in index})
    methods = sorted({key[1] for key in index})
    comparisons = [('v3s_fixed', m) for m in methods if m != 'v3s_fixed']
    comparisons += [('v3s_full', m) for m in MODES if m not in {'v3s_full', 'old_risk'}]
    comparisons += [('v3s_fixed', 'old_risk')]
    comparisons += [(f'{stage}_conservative', f'{stage}_{rule}')
                    for stage, rule in product(('frozen', 'joint'), FUSIONS) if rule != 'conservative']
    comparisons = list(dict.fromkeys(comparisons))
    fields = ('dataset', 'method', 'baseline', 'condition', 'metric', 'n_seeds', 'difference',
              'ci95_low', 'ci95_high', 'p_unadjusted', 'seed_differences')
    signs = np.array(list(product((-1, 1), repeat=len(seeds))))
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset, (method, baseline), condition, metric in product(
                datasets, comparisons, PAIR_CONDITIONS, ('accuracy', 'macro_f1', 'nll', 'brier', 'ece15',
                                                        'maxprob_aurc', 'maxprob_error_auroc',
                                                        'o_mae', 'a_mae', 'c_mae', 'u_mae')):
            values = [tuple(index.get((dataset, name, seed), {}).get(condition, {}).get(metric)
                            for name in (method, baseline)) for seed in seeds]
            if len(seeds) != len(SEEDS) or any(a is None or b is None for a, b in values):
                continue
            d = np.array([a - b for a, b in values])
            margin = 2.776445 * d.std(ddof=1) / len(d) ** 0.5
            p = (np.abs(signs @ d / len(d)) >= abs(d.mean()) - 1e-12).mean()
            writer.writerow(dict(zip(fields, (dataset, method, baseline, condition, metric, len(seeds),
                                  d.mean(), d.mean() - margin, d.mean() + margin, p, d.tolist()))))


def run_method(model, banks, make_cases, val, settings, seed, device, directory, run, teacher_seconds, eager):
    result = directory / 'result.json'
    if result.exists():
        report = json.loads(result.read_text(encoding='utf-8'))
        if report['run'] != run:
            raise ValueError(f'Saved experiment configuration differs: {result}')
        return report
    print(f"{run['dataset']} {model.method} seed={seed}", flush=True)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    started = perf_counter()
    iterations = fit(model, banks, settings, seed, device, eager)
    frozen = next(iterations)
    logs = next(iterations)
    synchronize(device)
    seconds = perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / 1024 ** 2 if device.type == 'cuda' else None
    measurements = {}
    if model.method == 'v3s_fixed':
        measurements['frozen'], *_ = evaluate(frozen, make_cases(), val, settings, FUSIONS, False)
        measurements['joint'], curves, thresholds, inference_ms = evaluate(
            model, make_cases(), val, settings, FUSIONS, True)
        metrics = measurements['joint']['conservative']
    else:
        final, curves, thresholds, inference_ms = evaluate(
            model, make_cases(), val, settings, ('conservative',), True)
        metrics = final['conservative']
    metrics['clean'].update(inference_ms_per_sample=inference_ms, shared_preparation_seconds=teacher_seconds)
    for key, curve in curves.items():
        np.savetxt(directory / f'coverage_{key}.csv', curve, delimiter=',', comments='',
                   header='coverage,risk,accuracy,maxprob_coverage,maxprob_risk')
    checkpoint = directory / 'model.tmp'
    torch.save(model.state_dict(), checkpoint)
    checkpoint.replace(directory / 'model.pt')
    report = dict(run=run, conditions=metrics, fusion=measurements, training=logs,
                  train_seconds=seconds, shared_preparation_seconds=teacher_seconds,
                  inference_ms_per_sample=inference_ms, inference_timing='clean_batch_128',
                  peak_cuda_mib=peak, val_80pct_threshold=thresholds['conservative'],
                  trainable_parameters=sum(p.numel() for p in model.parameters()),
                  linear_macs_per_sample=linear_macs(model, *val[:2], device))
    atomic_json(result, report)
    return report


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--methods', nargs='+', choices=MODES, default=tuple(MODES))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--features', type=Path, default=ROOT / 'data/features')
    parser.add_argument('--reference', type=Path, default=ROOT / 'runs/small_fixed/protocol')
    parser.add_argument('--previous', type=Path, default=ROOT / 'runs/task_risk_fixed/protocol')
    parser.add_argument('--previous-workers', type=Path, default=ROOT / 'runs/task_risk_workers')
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/v3s')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--eager', action='store_true', help='Disable CUDA Graph replay.')
    args = parser.parse_args(argv)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable in this interpreter. Select a CUDA-enabled PyCharm interpreter.')
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device(args.device)
    settings = Settings(clean_epochs=1, risk_epochs=1, joint_epochs=1) if args.smoke else Settings()
    seeds = SEEDS[:1] if args.smoke else SEEDS
    datasets, methods = tuple(dict.fromkeys(args.datasets)), tuple(dict.fromkeys(args.methods))
    out = args.out / ('smoke' if args.smoke else 'protocol')
    out.mkdir(parents=True, exist_ok=True)
    reports = []
    for dataset in datasets:
        modalities = BENCHMARKS[dataset]
        train_data, classes = load_data(args.features, dataset)
        fingerprint = digest_cache(args.features, dataset)
        if not args.smoke:
            reports.extend(history(args.reference, 'v2-fixed-1', BASELINES, dataset, settings, fingerprint, seeds))
            previous = args.previous
            if not (previous / dataset).is_dir():
                previous = args.previous_workers / dataset / 'protocol'
            if (previous / dataset).is_dir():
                reports.extend(history(previous, 'task-risk-fixed-1',
                                       ('task_uniform', 'task_fixed', 'task_dynamic', 'task_shuffled'),
                                       dataset, settings, fingerprint, seeds))
        with np.load(args.features / dataset / 'train.npz', allow_pickle=False) as cache:
            groups = cache['groups'].astype(str)
        train = train_data['train']
        val = on_device(train_data['val'], device)
        dims = [x.shape[1] for x in train[0]]
        pairs = tuple(torch.tensor(v, device=device) for v in zip(*combinations(range(len(dims)), 2)))
        for seed in seeds:
            shared = None
            test_cases = None
            for method in methods:
                directory = out / dataset / method / str(seed)
                directory.mkdir(parents=True, exist_ok=True)
                run = dict(protocol=PROTOCOL, dataset=dataset, method=method, seed=seed,
                           config=asdict(settings), fingerprint=fingerprint, device=str(device),
                           torch=str(torch.__version__), quick=args.smoke, cuda_graph=device.type == 'cuda' and not args.eager)
                with FileLock(str(directory / '.run.lock')):
                    result = directory / 'result.json'
                    if result.exists():
                        report = json.loads(result.read_text(encoding='utf-8'))
                        if report['run'] != run:
                            raise ValueError(f'Saved experiment configuration differs: {result}')
                    else:
                        if shared is None:
                            start = perf_counter()
                            backbone = train_teacher(dims, classes, on_device(train, device), settings, seed, device)
                            banks = training_views(train, groups, dims, classes, settings, seed, device, pairs, args.smoke)
                            synchronize(device)
                            teacher_seconds = perf_counter() - start
                            shared = backbone, banks, teacher_seconds
                            print(f'{dataset} seed={seed} prepared cross-fit targets in {teacher_seconds:.1f}s', flush=True)
                        backbone, banks, teacher_seconds = shared
                        torch.manual_seed(seed + 210)
                        model = V3S(deepcopy(backbone), len(dims), classes, method,
                                    settings.width, settings.latent).to(device)

                        def make_cases():
                            nonlocal test_cases
                            if test_cases is None:
                                test_cases = list(evaluation_data(train_data['test'], backbone, device,
                                                                  modalities, args.smoke, pairs))
                            return test_cases

                        report = run_method(model, banks, make_cases, val, settings, seed, device,
                                            directory, run, teacher_seconds, args.eager)
                reports.append(report)
    extras = []
    for report in reports:
        for stage, rules in report.get('fusion', {}).items():
            for rule, conditions in rules.items():
                extras.append(dict(report, run=dict(report['run'], method=f'{stage}_{rule}'),
                                   conditions=conditions))
    export_summary(reports + extras, out / 'summary.csv')
    paired_table(reports + extras, out / 'paired.csv', seeds)
    atomic_json(out / 'COMPLETE.json', dict(protocol=PROTOCOL, datasets=datasets, seeds=seeds, methods=methods,
                new_runs=len(datasets) * len(seeds) * len(methods), historical_runs=len(reports)
                - len(datasets) * len(seeds) * len(methods), fusion_readouts=len(extras), smoke=args.smoke))
    print(f'Completed {len(reports)} runs plus {len(extras)} fusion readouts at {out}', flush=True)


if __name__ == '__main__':
    main()


