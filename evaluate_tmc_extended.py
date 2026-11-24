"""Evaluate fixed-setting TMC checkpoints and write paper-ready tables and figures."""
from argparse import ArgumentParser
from collections import defaultdict
from csv import DictWriter
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch

from data import BENCHMARKS, condition, inject, load_data
from main import Settings, digest_cache
from metrics import detection, risk_coverage, summarize
from tmc_hcrd import entropy, fuse
from train_tmc_extended import METHODS, ROOT, make_model
from validate_tmc import DATASETS, SEEDS


def intervention(data, selected, level, kind):
    views, mask, _ = data
    strength = torch.zeros_like(mask, dtype=torch.float32)
    strength[:, list(selected)] = level * mask[:, list(selected)]
    return inject(views, mask, strength, torch.Generator().manual_seed(752), kind), mask, strength


def table(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report_metrics(y, p, score, threshold=None):
    stats, curve = summarize(y.numpy(), p.numpy(), score.numpy(), threshold)
    correct = (p.argmax(1) == y).numpy()
    stats['entropy_aurc'] = float(risk_coverage(correct, -entropy(p).numpy())[1].mean())
    return stats, curve


def mse(est, target, mask):
    diff = (est - target)[mask]
    return dict(mae=float(diff.abs().mean()), mse=float(diff.square().mean())) if len(diff) else {}


@torch.inference_mode()
def evaluate(model, teacher, tmc, data, key, groups, output, level, kind, views, mask, strength,
             tables, val_info, split='test'):
    clean, _, y = data[split]
    out = model(views, mask)
    threshold = val_info.get('confidence')
    stats, curve = report_metrics(y, out['p'], out['score'], threshold)
    row = dict(key, **stats)
    if level == 0:
        tables['main_comparison'].append(row)
    else:
        tables['robustness'].append(row)
    tables['calibration'].append(row)
    sample_output = dict(p=out['p'].numpy(), score=out['score'].numpy(), y=y.numpy(), mask=mask.numpy(),
                         groups=groups)
    if level == 0:
        classes = np.arange(out['p'].shape[1])
        pred = out['p'].argmax(1)
        pr, rc, f1, n = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
        conf = confusion_matrix(y, pred, labels=classes)
        for i, c in enumerate(classes):
            tables['per_class'].append(dict(key, label=c, precision=pr[i], recall=rc[i], f1=f1[i], n=n[i]))
            for j, predicted in enumerate(classes):
                tables['confusion'].append(dict(key, label=c, prediction=predicted, n=conf[i, j]))
        tables['curve'].extend(dict(key, coverage=cover, risk=risk, maxprob_risk=prob_risk)
                               for cover, risk, prob_risk in curve[np.linspace(0, len(curve) - 1, 41).astype(int)][:, [0, 1, 4]])
    if 'o' in out:
        sample_output.update({k: out[k].numpy() for k in ('o', 'discount', 'a', 'delta', 'normal', 'excess',
                                                        'responsibility', 'kl_a', 'kl_c') if k in out})
        base = model(clean, mask)
        baseline = tmc(views, mask)
        for i, modality in enumerate(('text', 'audio', 'video')):
            polluted = (strength[:, i] > 0) & mask[:, i]
            if polluted.any():
                if key['method'] == 'full':
                    tables['discount_response'].append(dict(key, modality=modality,
                        clean_discount=float(base['discount'][polluted, i].mean()),
                        corrupted_discount=float(out['discount'][polluted, i].mean()),
                        mean_confidence_change=float((out['p'].amax(-1) - base['p'].amax(-1))[polluted].mean())))
                for name, low, high in [('risk', base['o'][:, i], out['o'][:, i]),
                                        ('tmc_u', tmc(clean, mask)['u'][:, i], baseline['u'][:, i])]:
                    low, high = low[polluted], high[polluted]
                    targets = np.repeat([0, 1], len(low))
                    tables['risk_observation'].append(dict(key, modality=modality, proxy=name,
                        difference=float((high-low).mean()), **detection(targets, torch.cat((low, high)))))
        if 'delta' in out:
            delta, normal, h = model.references(teacher, clean, views)
            _, current_entropy, _ = model.distribution(out['single_alpha'] - 1)
            _, teacher_entropy, _ = teacher.distribution(teacher.evidence(views)[1])
            for label, est in [('raw_entropy', current_entropy),
                               ('simple_subtract', (current_entropy - out['o']).clamp(0, 1)),
                               ('hcrd', out['a']),
                               ('teacher_scale', (teacher_entropy - out['delta']).clamp(0, 1)),
                               ('corrected_student', (current_entropy - val_info['offset'] - out['delta']).clamp(0, 1))]:
                tables['risk_ambiguity'].append(dict(key, proxy=label, **mse(est, h, mask)))
            pair = out['pair_mask']
            tables['risk_ambiguity'].append(dict(key, proxy='delta', **mse(out['delta'], delta, mask)))
            tables['risk_ambiguity'].append(dict(key, proxy='zero_delta', **mse(torch.zeros_like(delta), delta, mask)))
            for label, est in [('hcrd', out['normal']), ('mean_jsd', val_info['normal_mean'].expand_as(normal))]:
                tables['risk_conflict'].append(dict(key, proxy=label, **mse(est, normal, pair)))
            wa, wc = model.dynamic_weights(out, strength, normal, h)
            for label, omega, weights, valid in [('A', wa, out['kl_a'], mask), ('C', wc, out['kl_c'], pair)]:
                values = omega[valid]
                kl = weights[valid]
                if len(values):
                    tables['ib_diagnostics'].append(dict(key, task=label, omega_mean=float(values.mean()),
                        omega_sd=float(values.std(unbiased=False)), kl_mean=float(kl.mean()),
                        weighted_kl=float((values * kl).mean())))
            sample_output.update(omega_a=wa.numpy(), omega_c=wc.numpy())
            if key['method'] == 'full':
                compare_fusion(model, out, mask, y, key, val_info, tables)
                if kind == 'dropout' and level == .5:
                    latent_check(model, views, mask, y, key, out, delta, normal, tables)
    if level == 0 or (len(key['condition'].split('_')) == 3 and level == .9):
        np.savez_compressed(output / (key['condition'] + '.npz'), **sample_output)


def compare_fusion(model, out, mask, y, key, val_info, tables):
    trust = mask * (1 - out['o']) * (1 - out['a'])
    matrix = model.pair_i + model.pair_j
    equal = (out['excess'] @ matrix) / (out['pair_mask'].float() @ matrix).clamp_min(1)
    for method, discount in [('F0_ds', mask), ('F1_constant', mask * val_info['scale']),
                             ('F2_vacuity', mask * (1-out['u'])), ('F3_oa', trust),
                             ('F4_equal', trust / (1 + equal)), ('F5_csf', out['discount'])]:
        alpha = fuse((out['single_alpha'] - 1) * discount[..., None])
        p = alpha / alpha.sum(-1, keepdim=True)
        for count in (1, 2, 3):
            keep = mask.sum(-1) == count
            if keep.any():
                eval_stats, _ = report_metrics(y[keep], p[keep], 1-model.classes/alpha[keep].sum(-1))
                tables['fusion_ablation'].append(dict(key, fusion=method, modalities=count, **eval_stats))


def latent_check(model, x, mask, y, key, out, delta, normal, tables):
    width = model.rib.head.in_features - model.rib.task.embedding_dim
    for method in ('original', 'zero_latent', 'shuffle_latent'):
        def alter(mod, args):
            value = args[0].clone()
            if method == 'zero_latent':
                value[..., :width] = 0
            else:
                value[..., :width] = value[..., :width].roll(1, dims=0)
            return (value,)
        if method == 'original':
            pred = out
        else:
            hook = model.rib.head.register_forward_pre_hook(alter)
            try:
                pred = model(x, mask)
            finally:
                hook.remove()
        tables['latent_ablation'].append(dict(key, intervention=method,
            accuracy=float((pred['p'].argmax(-1) == y).float().mean()),
            delta_mae=mse(pred['delta'], delta, mask).get('mae'),
            normal_mae=mse(pred['normal'], normal, pred['pair_mask']).get('mae')))


@torch.inference_mode()
def conflict_check(model, data, key, tables, high_cut, thresholds, baseline, adjacent=False):
    clean, mask, _ = data['test']
    clean_out = model(clean, mask)
    if 'excess' not in clean_out:
        return
    x, m, _, swaps, cross = condition(data['test'], 'conflict_adjacent' if adjacent else 'conflict')
    out = model(x, m)
    base = baseline(clean, mask)
    pair = clean_out['pair_mask']
    affected = (swaps[:, model.i] | swaps[:, model.j]) & pair
    high = pair & (base['raw'] >= high_cut)
    def scores(result):
        evidence = result['single_alpha'] - 1
        belief = evidence / result['single_alpha'].sum(-1, keepdim=True)
        left, right = belief[:, model.i], belief[:, model.j]
        ds = left.sum(-1) * right.sum(-1) - (left * right).sum(-1)
        return {'raw_jsd': result['raw'], 'ds': ds, 'residual': result['excess']}
    clean_scores, swap_scores = scores(clean_out), scores(out)
    subsets = [('adjacent', affected)] if adjacent else [('same', affected & ~cross[:, None]),
                                                        ('different', affected & cross[:, None])]
    for label, subset in subsets:
        n = int(subset.sum())
        if n:
            for proxy, measured in swap_scores.items():
                reference = clean_scores[proxy]
                values = torch.cat((reference[subset], measured[subset]))
                tables['risk_conflict'].append(dict(key, swap=label, proxy=proxy,
                    natural_high_fpr=float((reference[high] > thresholds[proxy]).float().mean()) if high.any() else None,
                    **detection(np.repeat([0, 1], n), values)))
    for count in (2, 3):
        keep = (m.sum(-1) == count) & swaps.any(-1)
        for label, score in [('responsibility', out['responsibility']),
                             ('incident_jsd', out['raw'] @ (model.pair_i + model.pair_j))]:
            score = score.masked_fill(~m, -torch.inf)
            top = (score == score.amax(-1, keepdim=True)) & m
            hit = (top & swaps).sum(-1) / top.sum(-1)
            if keep.any():
                tables['conflict_localization'].append(dict(key, modalities=count, proxy=label, adjacent=adjacent,
                                                           n=int(keep.sum()), top1=float(hit[keep].mean()),
                                                           random=1/count))


@torch.inference_mode()
def run_dataset(cache, root, name, directory, tables, only_clean=False):
    data, classes = load_data(cache, name)
    fingerprint = digest_cache(cache, name)
    with np.load(cache / name / 'test.npz', allow_pickle=False) as info:
        groups = info['groups'].astype(str)
    vx, vm, _ = data['val']
    clean, mask, _ = data['test']
    cfg = Settings()
    for report in sorted((root / name).glob('*/*/result.json')):
        result = json.loads(report.read_text(encoding='utf-8'))
        run = result['run']
        method, seed = run['method'], run['seed']
        if run['fingerprint'] != fingerprint or run['protocol'] != 'extended-v1':
            raise ValueError(f'Checkpoint protocol mismatch in {report}')
        if only_clean and method not in {'tmc', 'full'}:
            continue
        def load(variant, path):
            model = make_model(data, classes, variant, cfg).eval()
            model.load_state_dict(torch.load(path, weights_only=True))
            return model
        model = load(method, report.with_name('model.pt'))
        key = dict(dataset=name, method=method, seed=seed, fold=root.name if only_clean else '')
        val = model(vx, vm)
        val_info = {'confidence': float(val['score'].quantile(.2))}
        tmc = teacher = reference = None
        if 'o' in val:
            tmc = load('tmc', root / name / 'tmc' / str(seed) / 'model.pt')
            teacher = load(method, report.with_name('teacher.pt'))
        if 'delta' in val:
            _, teacher_h, _ = teacher.distribution(teacher.evidence(vx)[1])
            _, student_h, _ = model.distribution(val['single_alpha'] - 1)
            val_info.update(offset=((student_h-teacher_h)*vm).sum(0)/vm.sum(0).clamp_min(1),
                            scale=float(val['discount'][vm].mean()))
            reference = load('full', root / name / 'full' / str(seed) / 'model.pt')
            tx, tm, _ = data['train']
            _, normal, _ = model.references(teacher, tx, tx)
            train_pair = tm[:, model.i] & tm[:, model.j]
            val_info['normal_mean'] = (normal * train_pair).sum(0) / train_pair.sum(0).clamp_min(1)
        out_dir = directory / 'predictions' / (root.name if only_clean else 'main') / name / method / str(seed)
        out_dir.mkdir(exist_ok=True, parents=True)
        evaluate(model, teacher, tmc, data, dict(key, condition='clean'), groups, out_dir, 0,
                 'clean', clean, mask, torch.zeros_like(mask, dtype=torch.float32), tables, val_info)
        if only_clean:
            continue
        for batch in (1, 32):
            sample = [x[:batch] for x in clean]
            availability = mask[:batch]
            for _ in range(5):
                model(sample, availability)
            start = perf_counter()
            for _ in range(30):
                model(sample, availability)
            elapsed = (perf_counter() - start)/30
            tables['complexity'].append(dict(key, batch=batch, cpu_ms=1000*elapsed, samples_per_sec=batch/elapsed,
                **{k: result[k] for k in ('parameters', 'linear_macs', 'train_seconds', 'selected_epoch', 'joint_epochs')}))
        for chosen in (list(combinations(range(3), 1)) + list(combinations(range(3), 2)) + [(0, 1, 2)]):
            keep = mask[:, list(chosen)].all(1)
            selected_mask = torch.zeros_like(mask[keep])
            selected_mask[:, list(chosen)] = True
            y = data['test'][2][keep]
            observed = model([x[keep] for x in clean], selected_mask)
            key_subset = dict(key, condition=''.join(map(str, chosen)), modality_count=len(chosen))
            tables['missing_modalities'].append(dict(key_subset, **report_metrics(y, observed['p'], observed['score'])[0]))
            if method == 'full':
                compare_fusion(model, observed, selected_mask, y, key_subset, val_info, tables)
        for kind in ('gaussian', 'dropout'):
            for chosen in (list(combinations(range(3), 1)) + list(combinations(range(3), 2)) + [(0, 1, 2)]):
                levels = (.2, .5, .9) if len(chosen) == 1 else (.5,)
                risk_series = []
                for level in levels:
                    inp, m, s = intervention(data['test'], chosen, level, kind)
                    label = f'{kind}_{"".join(map(str, chosen))}_{level}'
                    evaluate(model, teacher, tmc, data, dict(key, condition=label), groups, out_dir,
                             level, kind, inp, m, s, tables, val_info)
                    if 'o' in val and len(chosen) == 1:
                        i = chosen[0]
                        risk_series.append(model(inp, m)['o'][m[:, i], i].numpy())
                if risk_series:
                    i = chosen[0]
                    risk_series.insert(0, model(clean, mask)['o'][mask[:, i], i].numpy())
                    corr = spearmanr(np.repeat([0, *levels], len(risk_series[0])), np.concatenate(risk_series)).statistic
                    tables['risk_observation'].append(dict(key, condition=kind, modality=BENCHMARKS[name][i],
                                                           proxy='risk_dose', spearman=corr))
        if reference is not None:
            ref_val = reference(vx, vm)
            high_cut = ref_val['raw'][ref_val['pair_mask']].quantile(.9)
            ev = val['single_alpha'] - 1
            belief = ev / val['single_alpha'].sum(-1, keepdim=True)
            left, right = belief[:, model.i], belief[:, model.j]
            ds = left.sum(-1)*right.sum(-1)-(left*right).sum(-1)
            thresholds = {k: v[val['pair_mask']].quantile(.95) for k, v in
                          [('raw_jsd', val['raw']), ('ds', ds), ('residual', val['excess'])]}
            conflict_check(model, data, dict(key, condition='swap'), tables, high_cut, thresholds, reference)
            if name == 'ch_sims':
                conflict_check(model, data, dict(key, condition='swap_adjacent'), tables,
                               high_cut, thresholds, reference, adjacent=True)
        if name == 'ch_sims':
            correct = model(clean, mask)['p'].argmax(-1).numpy() == data['test'][2].numpy()
            for group in np.unique(groups):
                indices = groups == group
                tables['source_video'].append(dict(key, group=group, n=int(indices.sum()),
                                                   accuracy=float(correct[indices].mean())))
        print(name, method, seed, root.name, flush=True)


def paired_bootstrap(directory, rng):
    rows = []
    predictions = directory / 'predictions/main'
    for dataset in DATASETS:
        for method, reference in [('full', 'tmc'), ('full', 'simple'), ('full', 'capacity'),
                                  ('full', 'dynamic_equal'), ('dynamic_equal', 'fixed_equal')]:
            differences = []
            for seed in SEEDS:
                paths = [predictions / dataset / variant / str(seed) / 'clean.npz' for variant in (method, reference)]
                if not all(path.is_file() for path in paths):
                    continue
                a, b = map(np.load, paths)
                assert np.array_equal(a['groups'], b['groups']) and np.array_equal(a['y'], b['y'])
                diff = (a['p'].argmax(1) == a['y']).astype(float) - (b['p'].argmax(1) == b['y'])
                differences.append(diff)
                _, grouped = np.unique(a['groups'], return_inverse=True)
                a.close()
                b.close()
            if len(differences) != len(SEEDS):
                continue
            differences = np.asarray(differences)
            count = np.bincount(grouped)
            delta = np.stack([np.bincount(grouped, weights=row) for row in differences])
            sampled_groups = rng.integers(0, len(count), size=(2000, len(count)))
            sampled_seeds = rng.integers(0, len(SEEDS), size=(2000, len(SEEDS)))
            boot = delta[sampled_seeds[:, :, None], sampled_groups[:, None, :]].sum((1, 2))
            boot /= len(SEEDS)*count[sampled_groups].sum(1)
            rows.append(dict(dataset=dataset, method=method, reference=reference, metric='accuracy',
                             n_seeds=len(SEEDS), n_groups=len(count), mean=differences.mean(),
                             ci_low=np.quantile(boot, .025), ci_high=np.quantile(boot, .975)))
    return rows


def aggregate(rows, measures):
    values = defaultdict(list)
    for row in rows:
        group = tuple(row[k] for k in ('dataset', 'fold', 'method', 'condition'))
        for metric in measures:
            if row.get(metric) is not None:
                values[(*group, metric)].append(row[metric])
    return [dict(zip(('dataset', 'fold', 'method', 'condition', 'metric', 'mean', 'sd', 'n'),
                     (*key, float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else None, len(x))))
            for key, x in sorted(values.items())]


def plots(out, tables):
    names = ('tmc', 'simple', 'capacity', 'fixed_equal', 'dynamic_equal', 'full')
    fig, axs = plt.subplots(4, 3, figsize=(13, 11), sharex=True)
    for i, dataset in enumerate(DATASETS):
        for modality, ax in enumerate(axs[i]):
            for method in names:
                series = [[row['accuracy'] for row in tables['main_comparison' if level == 0 else 'robustness']
                           if row['dataset'] == dataset and row['method'] == method and not row['fold']
                           and row['condition'] == ('clean' if level == 0 else f'gaussian_{modality}_{level}')]
                          for level in (0, .2, .5, .9)]
                if all(series):
                    y = np.mean(series, axis=1)
                    sd = [np.std(values, ddof=1) if len(values) > 1 else 0 for values in series]
                    ax.plot([0, .2, .5, .9], y, marker='o', label=method)
                    ax.fill_between([0, .2, .5, .9], y-sd, y+sd, alpha=.11)
            ax.set_title(f'{dataset} / {BENCHMARKS[dataset][modality]}')
            ax.set_ylabel('Accuracy')
    axs[-1, 1].set_xlabel('Gaussian feature noise')
    if axs[0, 0].lines:
        axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'robustness.png', dpi=180)
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(9, 7))
    for dataset, ax in zip(DATASETS, axs.flat):
        for method in names:
            runs = [[row for row in tables['curve'] if row['dataset'] == dataset and row['method'] == method
                     and not row['fold'] and row['seed'] == seed] for seed in SEEDS]
            if all(runs):
                ax.plot([row['coverage'] for row in runs[0]],
                        np.mean([[row['risk'] for row in run] for run in runs], axis=0), label=method)
        ax.set(title=dataset, xlabel='Coverage', ylabel='Error risk')
    if axs[0, 0].lines:
        axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'risk_coverage.png', dpi=180)
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    for dataset, ax in zip(DATASETS, axs.flat):
        omega, kl = [], []
        for path in (out / 'predictions/main' / dataset / 'full').glob('*/gaussian_1_0.9.npz'):
            with np.load(path, allow_pickle=False) as data:
                omega.extend(data['omega_a'][data['mask']])
                kl.extend(data['kl_a'][data['mask']])
        if omega:
            points = ax.hexbin(omega, kl, gridsize=30, mincnt=1, bins='log', cmap='viridis')
            fig.colorbar(points, ax=ax, label='Count')
        ax.set(title=dataset, xlabel='Dynamic coefficient', ylabel='KL per latent dimension')
    fig.tight_layout()
    fig.savefig(out / 'dynamic_kl.png', dpi=180)
    plt.close(fig)
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    for dataset, ax in zip(DATASETS, axs.flat):
        for method in ('fixed_equal', 'dynamic_equal', 'mean_equal', 'shuffle_equal', 'reverse_equal', 'full'):
            means = []
            for level in (.2, .5, .9):
                data = [row['kl_mean'] for row in tables['ib_diagnostics'] if row['dataset'] == dataset
                        and row['method'] == method and row['task'] == 'A'
                        and row['condition'] == f'dropout_1_{level}']
                means.append(np.mean(data) if data else np.nan)
            ax.plot([.2, .5, .9], means, marker='o', label=method)
        ax.set(title=dataset, xlabel='Unseen audio feature dropout', ylabel='Mean KL per dimension')
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'kl_by_severity.png', dpi=180)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--feature-dir', type=Path, default=Path('data/extended'))
    parser.add_argument('--run-dir', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=Path('results/tmc_extended'))
    parser.add_argument('--allow-partial', action='store_true', help='Permit incomplete matrices for smoke tests')
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    tables = defaultdict(list)
    for name in DATASETS:
        run_dataset(args.feature_dir, args.run_dir, name, args.output, tables)
    for i in range(1, 6):
        fold = f'loso{i}'
        run_dataset(args.feature_dir / fold, args.run_dir / fold, 'iemocap', args.output, tables, only_clean=True)
    if not args.allow_partial:
        expected = {(name, '', method, seed) for name in DATASETS for method in METHODS for seed in SEEDS}
        expected.update(('iemocap', f'loso{i}', method, seed) for i in range(1, 6)
                        for method in ('tmc', 'full') for seed in SEEDS)
        observed = {(r['dataset'], r['fold'], r['method'], r['seed']) for r in tables['main_comparison']}
        if observed != expected:
            raise ValueError(f'Missing {len(expected-observed)} cells, unexpected {len(observed-expected)}')
    tables['significance'] = paired_bootstrap(args.output, np.random.default_rng(1124))
    tables['main_summary'] = aggregate(tables['main_comparison'],
        ('accuracy','macro_f1','weighted_f1','nll','brier','ece15','aurc','maxprob_aurc','entropy_aurc'))
    for name, rows in tables.items():
        if name != 'curve':
            table(args.output / f'{name}.csv', rows)
    plots(args.output, tables)


if __name__ == '__main__':
    main()
