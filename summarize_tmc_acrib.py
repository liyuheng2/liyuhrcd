"""Aggregate the fixed ACRIB experiment and generate a Chinese Markdown report."""
from argparse import ArgumentParser
from csv import DictWriter
import json
from pathlib import Path

import numpy as np

from validate_tmc_acrib import MODEL_NAMES, PROTOCOL
from validate_tmc import DATASETS, SEEDS


PRIMARY = dict(tmc='ds', tmc_aug='ds', crib_v2='removal')
DISPLAY = dict(cmu_mosi='CMU-MOSI', ch_sims='CH-SIMS', meld='MELD', iemocap='IEMOCAP')


def flatten(row, prefix=''):
    for key, value in row.items():
        field = f'{prefix}.{key}' if prefix else key
        if isinstance(value, dict):
            yield from flatten(value, field)
        else:
            yield field, value


def average(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def format_value(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def paired_ci(values):
    values = np.asarray(values)
    samples = np.random.default_rng(611).choice(values, size=(10000, len(values)), replace=True).mean(1)
    low, high = np.quantile(samples, [.025, .975])
    return f'{values.mean():+.2f} [{low:+.2f}, {high:+.2f}]'


def read_results(root):
    records = {}
    for dataset in DATASETS:
        for seed in SEEDS:
            cell = root / dataset / str(seed)
            for method in MODEL_NAMES:
                file = cell / f'{method}.json'
                if not file.exists():
                    raise FileNotFoundError(f'Incomplete experiment: {file}')
                run = json.loads(file.read_text(encoding='utf-8'))
                if (run['run']['protocol'], run['run']['dataset'], run['run']['seed'], run['method']) != (
                    PROTOCOL, dataset, seed, method
                ) or run['run']['samples'] != 8:
                    raise ValueError(f'Unexpected protocol or smoke result: {file}')
                records[dataset, seed, method] = run
    reference = records[DATASETS[0], SEEDS[0], 'tmc']['run']
    for (dataset, seed, method), record in records.items():
        metadata = record['run']
        if any(metadata[k] != reference[k] for k in ('settings', 'joint_lr', 'risk_weight', 'torch')):
            raise ValueError(f'Inconsistent experiment settings: {dataset} {seed} {method}')
        if metadata['sha256'] != records[dataset, SEEDS[0], 'tmc']['run']['sha256']:
            raise ValueError(f'Inconsistent dataset fingerprint: {dataset} {seed} {method}')
    return records


def write_metrics(records, output):
    rows = []
    for (dataset, seed, method), run in records.items():
        for probe, scenarios in {'standard': run['metrics'], **run['probes']}.items():
            for scenario, values in scenarios.items():
                for key, value in flatten(values):
                    rows.append(dict(dataset=dataset, seed=seed, method=method, probe=probe, condition=scenario,
                                     metric=key, value=value))
    with (output / 'metrics.csv').open('w', encoding='utf-8', newline='') as file:
        writer = DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_loso(root):
    folders = [root / f'iemocap_loso{fold}' / '11' for fold in range(1, 6)]
    if not any(folder.exists() for folder in folders):
        return {}
    records = {}
    for fold, folder in enumerate(folders, 1):
        for method in MODEL_NAMES:
            file = folder / f'{method}.json'
            if not file.exists():
                raise FileNotFoundError(f'Incomplete IEMOCAP LOSO experiment: {file}')
            result = json.loads(file.read_text(encoding='utf-8'))
            meta = result['run']
            if (meta['protocol'], meta['fold'], meta['seed'], meta['samples'], result['method']) != (
                PROTOCOL, fold, 11, 8, method
            ):
                raise ValueError(f'Unexpected LOSO protocol: {file}')
            records[f'iemocap_loso{fold}', 11, method] = result
    return records


def report(records, loso):
    def metric(dataset, method, field, scenario='clean', policy=None, probe=None):
        key = policy or PRIMARY.get(method, 'acrf')
        return [(records[dataset, seed, method]['metrics'] if probe is None else
                 records[dataset, seed, method]['probes'][probe])[scenario]['policies'][key][field] for seed in SEEDS]

    def diagnosis(dataset, method, scenario, probe=None):
        return [(records[dataset, seed, method]['metrics'] if probe is None else
                 records[dataset, seed, method]['probes'][probe])[scenario]['risks'] for seed in SEEDS]

    def diagnostic_field(dataset, method, field, scenario, probe=None):
        return average([dict(flatten(row)).get(field) for row in diagnosis(dataset, method, scenario, probe)])

    n = len(records)
    clean_delta = [100 * (average(metric(d, 'acrib', 'accuracy')) - average(metric(d, 'fixed_acrib', 'accuracy')))
                   for d in DATASETS]
    retain = sum(row['threshold'] is None for (_, _, method), row in records.items() if method == 'acrib')
    decisions = len(DATASETS) * len(SEEDS)
    lines = [
        '# TMC-HCRD-ACRIB 固定参数验证',
        '',
        f'完成 {n} 个训练单元。每个数据集使用五个随机种子，所有方法复用相同的特征缓存和原始划分。',
        '表中分类指标为五种子均值，Accuracy 和 Macro-F1 以百分比表示。详细逐种子、条件、融合方式和诊断指标见本地生成的 `metrics.csv`。',
        '',
        '## 主要结果',
        '',
        '冻结 ACRIB 相对容量匹配的 Fixed-ACRIB，干净准确率差值为 ' +
        '、'.join(f'{DISPLAY[d]} {v:+.2f} 个百分点' for d, v in zip(DATASETS, clean_delta)) + '。',
        f'验证集在 {decisions} 个数据集与种子组合中有 {retain} 个选择保留全部证据。',
        '固定门控按训练集的平均容量拟合。实测 KL 随方法变化，因此配对差值也包含信息压缩强度差异。',
        '',
        '## 原始测试集',
        '',
        '| 数据集 | 模型 | Accuracy % | Macro-F1 % | NLL | Brier | ECE |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for dataset in DATASETS:
        for method in MODEL_NAMES:
            fields = [format_value(100 * average(metric(dataset, method, k)), 2) for k in ('accuracy', 'macro_f1')]
            fields += [format_value(average(metric(dataset, method, k))) for k in ('nll', 'brier', 'ece15')]
            lines.append(f'| {DISPLAY[dataset]} | `{method}` | ' + ' | '.join(fields) + ' |')
    lines += [
        '',
        '## 与容量匹配的固定 ACRIB 对比',
        '',
        '差值为冻结 ACRIB 减去 Fixed-ACRIB。Accuracy 和 Macro-F1 差值以百分点表示，均值后为种子配对重采样的 95% 区间。五种子的区间只反映这组训练种子的波动。',
        '',
        '| 数据集 | 测试条件 | Accuracy Δ | Macro-F1 Δ | ACRIB Brier | Fixed Brier |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for dataset in DATASETS:
        for scenario in ('clean', 'gaussian_0.5', 'unseen_dropout_0.5', 'swap', 'missing_text'):
            diffs = [100 * (np.asarray(metric(dataset, 'acrib', k, scenario)) -
                            np.asarray(metric(dataset, 'fixed_acrib', k, scenario))) for k in ('accuracy', 'macro_f1')]
            brier = [format_value(average(metric(dataset, m, 'brier', scenario))) for m in ('acrib', 'fixed_acrib')]
            lines.append(f'| {DISPLAY[dataset]} | {scenario} | ' + ' | '.join([*map(paired_ci, diffs), *brier]) + ' |')
    lines += [
        '',
        '## 融合策略与动作风险',
        '',
        '按原始输入、0.5 高斯噪声、0.5 坐标丢弃、混合同类及异类换配四种条件等权汇总。动作收益使用半 Brier，正值代表改善。负收益比例只在实际执行的动作中计算，按各条件实际动作次数加权。',
        '',
        '| 数据集 | 模型 / 融合 | Accuracy % | Brier | 执行率 % | 负收益比例 % |',
        '|---|---|---:|---:|---:|---:|',
    ]
    policies = [('tmc', 'ds'), ('tmc_aug', 'ds'), ('crib_v2', 'removal'), ('no_ib', 'acrf'),
                ('fixed_acrib', 'acrf'), ('acrib', 'ds'), ('acrib', 'product'), ('acrib', 'acrf'),
                ('acrib_joint', 'ds'), ('acrib_joint', 'acrf')]
    for dataset in DATASETS:
        for method, policy in policies:
            rows = [records[dataset, seed, method]['metrics'][scenario]['policies'][policy]
                    for seed in SEEDS for scenario in ('clean', 'gaussian_0.5', 'unseen_dropout_0.5', 'swap')]
            counts = [v['n'] * v['applied_rate'] for v in rows]
            harm = 100 * sum(n * (v['harmful_among_applied'] or 0) for n, v in zip(counts, rows)) / sum(counts) \
                if sum(counts) else None
            fields = [100*average([v['accuracy'] for v in rows]), average([v['brier'] for v in rows]),
                      100*average([v['applied_rate'] for v in rows]), harm]
            lines.append(f'| {DISPLAY[dataset]} | {method} / {policy} | ' + ' | '.join(map(format_value, fields)) + ' |')
    lines += [
        '',
        '## 瓶颈容量与风险诊断',
        '',
        '容量与 KL 取 0.5 坐标丢弃条件。各任务按有效实例统计，C 项的 KL 是单方向均值，训练损失使用双方向之和。固定与随机门控在训练扰动上匹配平均容量，表中的实际 KL 可用于核对可比性。',
        '',
        '| 数据集 | 模型 | 容量均值 A/C/D | 同任务容量方差 A/C/D | KL 均值 A/C/D |',
        '|---|---|---|---|---|',
    ]
    for dataset in DATASETS:
        for method in ('crib', 'acrib', 'fixed_acrib', 'random_acrib', 'unbounded_acrib'):
            rows = [records[dataset, seed, method]['metrics']['unseen_dropout_0.5'] for seed in SEEDS]
            fields = [' / '.join(format_value(average([v[kind][t][stat] for v in rows])) for t in ('A', 'C', 'D'))
                      for kind, stat in (('capacity', 'mean'), ('capacity', 'var'), ('kl', 'mean'))]
            lines.append(f'| {DISPLAY[dataset]} | {method} | ' + ' | '.join(fields) + ' |')
    if loso:
        lines += [
            '',
            '## IEMOCAP 留一会话验证',
            '',
            f'PyCharm 运行了 {len(loso)} 个训练单元。固定种子 11，五个会话轮流作测试集，文本使用无标签转录哈希。下表对五折等权平均。',
            '',
            '| 模型 | 干净 Accuracy % | 干净 Macro-F1 % | 未知坐标丢弃 Accuracy % | 混合换配 Accuracy % |',
            '|---|---:|---:|---:|---:|',
        ]
        for method in MODEL_NAMES:
            policy = PRIMARY.get(method, 'acrf')
            values = [100 * average([row['metrics'][scenario]['policies'][policy][key]
                                     for (_, _, name), row in loso.items() if name == method])
                      for scenario, key in [('clean', 'accuracy'), ('clean', 'macro_f1'),
                                            ('unseen_dropout_0.5', 'accuracy'), ('swap', 'accuracy')]]
            lines.append(f'| {method} | ' + ' | '.join(format_value(v, 2) for v in values) + ' |')
    lines += [
        '',
        '## 潜变量干预',
        '',
        '以下诊断使用冻结 ACRIB 的同一检查点。污染 AUROC 来自未知坐标丢弃，冲突和收益指标来自换配条件。',
        '',
        '| 数据集 | 潜变量干预 | 污染 AUROC | 冲突 AUROC | 端点定位准确率 | 动作 MAE | 动作 Bias | 动作 Spearman |',
        '|---|---|---:|---:|---:|---:|---:|---:|',
    ]
    for dataset in DATASETS:
        for probe in (None, 'zero', 'shuffle'):
            d = [diagnostic_field(dataset, 'acrib', 'observation.auroc', 'unseen_dropout_0.5', probe)]
            d += [diagnostic_field(dataset, 'acrib', k, 'swap', probe) for k in
                  ('conflict.auroc', 'endpoint_accuracy', 'action.mae', 'action.bias', 'action.spearman')]
            lines.append(f'| {DISPLAY[dataset]} | {probe or "标准"} | ' + ' | '.join(map(format_value, d)) + ' |')
    lines += [
        '',
        '## 适用范围',
        '',
        '- 测试覆盖原始划分、预提取特征扰动、模态缺失、单模态输入和受控换配。自然语义冲突及原始媒体污染需要单独实测。',
        '- IEMOCAP 主表沿用已有普通划分及有监督的 DialogueGCN 文本特征。该表仅支持当前缓存协议下的配对比较。',
        '- KL 为变分上界相关诊断量，容量为门控代理。两者的实测值用于比较通道行为，互信息定理来自解析推导。',
        '- 训练协议、测试条件和复现命令见 [TMC_ACRIB.md](../../TMC_ACRIB.md)。',
    ]
    return '\n'.join(lines) + '\n'


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('runs/tmc_acrib'))
    parser.add_argument('--output', type=Path, default=Path('results/tmc_acrib'))
    parser.add_argument('--include-loso', action='store_true')
    args = parser.parse_args()
    records = read_results(args.runs)
    loso = read_loso(args.runs) if args.include_loso else {}
    if args.include_loso and not loso:
        raise FileNotFoundError(f'IEMOCAP LOSO results missing in {args.runs}')
    args.output.mkdir(parents=True, exist_ok=True)
    write_metrics(records | loso, args.output)
    (args.output / 'REPORT.md').write_text(report(records, loso), encoding='utf-8')
    print(f'Summarized {len(records) + len(loso)} completed runs into {args.output}')


if __name__ == '__main__':
    main()
