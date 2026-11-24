"""Check the complete v3 matrix and aggregate test metrics over paired seeds."""
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t
import torch
from torch import nn

from main import Settings, atomic_json
from tmc_dcib import DCIB, HCRDLite, predict
from validate_tmc_dcib import BASELINES, DATASETS, PROBES, SEEDS, VARIANTS
from validate_tmc_crib import make_model


NAMES = dict(cmu_mosi='CMU-MOSI', ch_sims='CH-SIMS', meld='MELD', iemocap='IEMOCAP LOSO')
CASES = ('clean', 'gaussian_0.9', 'unseen_dropout_0.5', 'conflict')
COMPARATORS = ('tmc', 'tmc_aug', 'rib', 'crib_v2', 'temperature_tmc', 'direct_q',
               'discrete', 'dcib', 'dcib_calibrated', 'oracle')


def flatten(mapping, prefix=''):
    for key, value in mapping.items():
        key = prefix + key
        if isinstance(value, dict):
            yield from flatten(value, key + '.')
        elif isinstance(value, (float, int)):
            yield key, value


def paired_ci(values):
    average = mean(values)
    spread = stdev(values)
    radius = float(t.ppf(.975, len(values) - 1)) * spread / len(values) ** .5
    return dict(mean=average, sd=spread, lower=average - radius, upper=average + radius, n=len(values))


def read_reports(root):
    expected = [(dataset, fold, seed) for dataset in DATASETS
                for fold, seed in product(range(1, 6) if dataset == 'iemocap' else (None,), SEEDS)]
    reports, missing = [], []
    for dataset, fold, seed in expected:
        cell = root / (f'iemocap_loso{fold}' if fold else dataset) / str(seed)
        if not (cell / 'result.json').is_file() or not (cell / 'models.pt').is_file():
            missing.append(str(cell))
            continue
        report = json.loads((cell / 'result.json').read_text(encoding='utf-8'))
        run = report['run']
        if ((run['dataset'], run['fold'], run['seed']) != (dataset, fold, seed)
                or run['protocol'] != 'dcib-v3-fixed-1'
                or tuple(run['settings'][k] for k in ('clean_epochs', 'risk_epochs', 'joint_epochs')) != (5, 5, 15)):
            raise ValueError(f'Invalid protocol or cell metadata in {cell}')
        required = set((*BASELINES, *VARIANTS, *COMPARATORS))
        if dataset in ('cmu_mosi', 'meld'):
            required.update(PROBES)
        if len(report['metrics']) != 12:
            raise ValueError(f'Incomplete test conditions in {cell}')
        for case in report['metrics'].values():
            if not required <= case['methods'].keys():
                raise ValueError(f'Missing comparison method in {cell}')
        reports.append(report)
    if missing:
        raise ValueError(f'{len(missing)} of {len(expected)} experiment cells missing: ' + ', '.join(missing))
    return reports


def aggregate(reports):
    metrics_by_seed = defaultdict(lambda: defaultdict(list))
    raw = defaultdict(list)
    calibration = defaultdict(list)
    for report in reports:
        run = report['run']
        dataset, seed = run['dataset'], run['seed']
        for metric, value in flatten(dict(test=report['metrics'], latency=report['latency_ms'],
                                           parameters=report['parameters'])):
            metrics_by_seed[dataset, metric][seed].append(value)
        for case, tests in report['metrics'].items():
            methods = tests['methods']
            for baseline in ('tmc', 'tmc_aug', 'rib', 'crib_v2', 'direct_q', 'discrete', 'temperature_tmc', *VARIANTS[1:]):
                for metric in ('brier', 'accuracy', 'macro_f1'):
                    delta = methods['dcib'][metric] - methods[baseline][metric]
                    metrics_by_seed[dataset, f'paired.{case}.{baseline}.{metric}'][seed].append(delta)
        calibration[dataset].append(dict(seed=seed, fold=run['fold'], **report['calibration']))
        raw[dataset].append(dict(seed=seed, fold=run['fold'], data_sha256=run['sha256'], splits=report['splits'],
                                 clean=report['metrics']['clean']['methods']['dcib'], train_seconds=report['train_seconds']))
    result = {name: {} for name in DATASETS}
    for (dataset, metric), seeds in metrics_by_seed.items():
        values = [mean(folds) for folds in seeds.values()]
        result[dataset][metric] = paired_ci(values) if len(values) > 1 else dict(mean=values[0], n=1)
    return dict(metrics=result, calibration=dict(calibration), cells=dict(raw))


@torch.inference_mode()
def linear_macs(root):
    """Count executed linear-layer multiply-accumulates using a full-mask sample."""
    torch.set_num_threads(1)
    results = {}
    for name in DATASETS:
        cell = root / ('iemocap_loso1' if name == 'iemocap' else name) / str(SEEDS[0])
        weights = torch.load(cell / 'models.pt', map_location='cpu', weights_only=True)
        report = json.loads((cell / 'result.json').read_text(encoding='utf-8'))
        cfg = replace(Settings(), **report['run']['settings'])
        dims = [p.shape[1] for k, p in weights['tmc'].items() if k.startswith('encoders.') and k.endswith('.weight')]
        classes = len(weights['tmc']['heads.0.bias'])
        tmc = make_model(dims, classes, 'tmc', cfg).eval()
        crib = make_model(dims, classes, 'full', cfg).eval()
        hcrd = HCRDLite(tmc, cfg.width).eval()
        condition_dim = weights['dcib']['encoder.0.weight'].shape[1] - len(dims) * cfg.width - 1
        dcib = DCIB(len(dims) * cfg.width, condition_dim, classes, cfg.width, cfg.latent, 'dcib').eval()
        modules = dict(tmc=tmc, crib_v2=crib, hcrd=hcrd, dcib=dcib)
        for method, module in modules.items():
            module.load_state_dict(weights[method])
        x, mask = [torch.zeros(1, dim) for dim in dims], torch.ones(1, len(dims), dtype=torch.bool)
        macs = []

        def count(layer, inputs, output):
            macs.append(output.numel() * layer.in_features)

        hooks = [layer.register_forward_hook(count) for model in modules.values()
                 for layer in model.modules() if isinstance(layer, nn.Linear)]
        results[name] = {}
        for method, forward in (('tmc', lambda: tmc(x, mask)), ('crib_v2', lambda: crib(x, mask)),
                                ('dcib', lambda: predict(tmc, hcrd, dcib, x, mask))):
            macs.clear()
            forward()
            results[name][method] = sum(macs)
        for hook in hooks:
            hook.remove()
    return results


def md_table(headers, rows):
    return '\n'.join('|' + ' | '.join(row) + '|' for row in (headers, ['---'] * len(headers), *rows))


def report_text(summary):
    metrics, calibration = summary['metrics'], summary['calibration']

    def test(dataset, case, method, metric):
        return metrics[dataset].get(f'test.{case}.methods.{method}.{metric}', {}).get('mean')

    def diagnostic(dataset, case, method, metric):
        return metrics[dataset].get(f'test.{case}.diagnostics.{method}.{metric}', {}).get('mean')

    def fmt(value, digits=4, factor=1):
        return '—' if value is None else f'{value * factor:.{digits}f}'

    lines = ['# TMC-HCRD-DCIB v3 固定参数验证',
             '4 个数据集，40 个正式实验单元。IEMOCAP 每个种子先等权平均 5 个测试会话，其余数据集使用固定测试集。']
    lines += ['## 干净测试集', md_table(['数据集', '方法', 'Accuracy %', 'Macro-F1 %', 'Brier', 'NLL', 'ECE', 'AURC'], [
        [NAMES[d], method, *(fmt(test(d, 'clean', method, k), 2, 100) for k in ('accuracy', 'macro_f1')),
         *(fmt(test(d, 'clean', method, k)) for k in ('brier', 'nll', 'ece15', 'aurc'))]
        for d, method in product(DATASETS, COMPARATORS)
    ])]
    lines += ['## 扰动条件', '每格为 `Accuracy % / Brier`。换配条件混合同类与异类换配。',
              md_table(['数据集', '条件', *COMPARATORS], [
                  [NAMES[d], case, *(f'{fmt(test(d, case, m, "accuracy"), 2, 100)} / {fmt(test(d, case, m, "brier"))}'
                                     for m in COMPARATORS)]
                  for d, case in product(DATASETS, CASES[1:])
              ])]
    lines += ['## 决策收益与独立校准',
              'Brier 差定义为 DCIB 减 TMC，负值表示改善。区间是同种子配对差的 95% t 区间，描述固定测试集上的训练种子波动。校准的 95% Hoeffding 上界另外按验证组计算。',
              md_table(['数据集', '条件', 'Brier 差', '95% 区间', '调整率 %', '校准调整率 %', 'Oracle 差距（½ Brier）'], [
                  [NAMES[d], case, fmt(interval['mean']), f'[{interval["lower"]:.4f}, {interval["upper"]:.4f}]',
                   fmt(test(d, case, 'dcib', 'adjust_coverage'), 2, 100),
                   fmt(test(d, case, 'dcib_calibrated', 'adjust_coverage'), 2, 100),
                   fmt(test(d, case, 'dcib', 'oracle_gap'))]
                  for d, case in product(DATASETS, CASES)
                  for interval in [metrics[d][f'paired.{case}.tmc.brier']]
              ]), md_table(['数据集', '独立校准组', '通过阈值检验', '最小风险上界范围'], [
                  [NAMES[d], ', '.join(map(str, sorted({c['n_groups'] for c in cells}))),
                   f'{sum(c["threshold"] is not None for c in cells)}/{len(cells)}',
                   f'{min(upper):.4f}–{max(upper):.4f}']
                  for d, cells in calibration.items()
                  for upper in [[min(row['upper'] for row in c['thresholds']) for c in cells]]
              ])]
    lines += ['## 信息瓶颈与三风险消融',
              '每格为 `Brier / 一阶方向收益 MAE`。后验均值、置零和批内打乱复用完整模型的权重。',
              md_table(['变体', *[NAMES[d] + ' 干净' for d in DATASETS], *[NAMES[d] + ' 换配' for d in DATASETS]], [
                  [m, *(f'{fmt(test(d, case, m, "brier"))} / {fmt(test(d, case, m, "direction_mae"))}'
                         for case, d in product(('clean', 'conflict'), DATASETS))]
                  for m in (*VARIANTS, 'dcib_latent_mean', 'dcib_latent_zero', 'dcib_latent_shuffle')
              ]), '完整模型的干净测试平均条件 KL（nats）为 ' + '，'.join(
                  f'{NAMES[d]} {fmt(test(d, "clean", "dcib", "conditional_kl"))}' for d in DATASETS) + '。']
    lines += ['## v2 收益回归与梯度隔离',
              '目标为完全移除模态后的实际半 Brier 降幅。偏差为预测减目标。',
              md_table(['数据集', '条件', '方法', 'MAE', '偏差', 'Spearman'], [
                  [NAMES[d], case, method, *(fmt(test(d, case, method, m))
                                            for m in ('utility_mae', 'utility_bias', 'utility_spearman'))]
                  for d, case, method in product(('cmu_mosi', 'meld'), ('clean', 'conflict'),
                                                 ('crib_v2', *PROBES, 'zero_predictor'))
              ])]
    lines += ['## HCRD 风险诊断',
              '观测污染指标来自未见坐标丢弃，冲突指标来自同类与异类换配。端点准确率只统计确实发生异常换配的有效模态对。',
              md_table(['数据集', '方法', '观测 AUROC', '观测 AUPRC', '强度 Spearman', '冲突 AUROC', '冲突 AUPRC',
                        '责任 Macro-F1', '端点准确率 %'], [
                  [NAMES[d], m, *(fmt(diagnostic(d, CASES[2], m, k))
                                  for k in ('observation.auroc', 'observation.auprc', 'observation_strength.spearman')),
                   *(fmt(diagnostic(d, 'conflict', m, k))
                     for k in ('conflict.auroc', 'conflict.auprc', 'responsibility.macro_f1')),
                   fmt(diagnostic(d, 'conflict', m, 'responsibility.endpoint_accuracy'), 2, 100)]
                  for d, m in product(DATASETS, ('hcrd', 'crib_v2_risk', 'rib_risk'))
              ])]
    lines += ['## 效率', '延迟为已有特征的单样本前向均值（30 次），实际测量受到本机并行负载影响。',
              md_table(['数据集', 'TMC 参数', 'CRIB v2 参数', 'DCIB v3 总参数', 'TMC ms', 'CRIB v2 ms', 'DCIB v3 ms'], [
                  [NAMES[d], *(fmt(metrics[d][f'parameters.{k}']['mean'], 0) for k in ('tmc', 'crib_v2', 'dcib_end_to_end')),
                   *(fmt(metrics[d][f'latency.{k}']['mean'], 3) for k in ('tmc', 'crib_v2', 'dcib'))]
                  for d in DATASETS
              ]), '以下是完整三模态输入的线性层 MACs。一个 MAC 对应一次乘法和一次加法，未计入激活、归一化和证据融合等算子。',
              md_table(['数据集', 'TMC MACs', 'CRIB v2 MACs', 'DCIB v3 MACs'], [
                  [NAMES[d], *(f'{summary["linear_macs"][d][m]:,}' for m in ('tmc', 'crib_v2', 'dcib'))]
                  for d in DATASETS
              ]), '完整条件、缺失与单模态指标、配对区间、逐折数据指纹和校准阈值见本地生成的 `summary.json`。']
    return '\n\n'.join(lines) + '\n'


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('runs/tmc_dcib'))
    parser.add_argument('--output', type=Path, default=Path('results/tmc_dcib'))
    args = parser.parse_args()
    summary = aggregate(read_reports(args.runs))
    summary['linear_macs'] = linear_macs(args.runs)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / 'summary.json', summary)
    (args.output / 'REPORT.md').write_text(report_text(summary), encoding='utf-8')
    print(f'40 complete cells; report at {args.output / "REPORT.md"}')


if __name__ == '__main__':
    main()
