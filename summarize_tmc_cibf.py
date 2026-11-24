"""Check the full experiment matrix and summarize paired seed comparisons."""
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t

from data import BENCHMARKS
from main import Settings, atomic_json
from validate_tmc_cibf import BALANCE, DATASETS, METHODS, PROTOCOL, SEEDS


NAMES = dict(cmu_mosi='CMU-MOSI', ch_sims='CH-SIMS', meld='MELD', iemocap='IEMOCAP LOSO')
KEY_CASES = ('clean', 'gaussian_0.9', 'unseen_dropout_0.5', 'conflict')


def read_cells(root):
    settings = asdict(replace(Settings(), joint_epochs=15))
    cells, digests, missing = [], {}, []
    for dataset in DATASETS:
        for fold, seed in product(range(1, 6) if dataset == 'iemocap' else (None,), SEEDS):
            name = f'iemocap_loso{fold}' if fold else dataset
            path = root / name / str(seed)
            report = path / 'result.json'
            if not (report.is_file() and (path / 'models.pt').is_file()):
                missing.append(f'{name}/{seed}')
                continue
            row = json.loads(report.read_text(encoding='utf-8'))
            run = row['run']
            expected = {*KEY_CASES, 'gaussian_0.2', 'gaussian_0.5',
                        *(f'{kind}_{view}' for kind in ('missing', 'only') for view in BENCHMARKS[dataset])}
            fingerprint = (run['sha256'], row['splits'], run['torch'])
            if ((run['dataset'], run['fold'], run['seed']) != (dataset, fold, seed)
                    or run['protocol'] != PROTOCOL or run['settings'] != settings
                    or run['contribution_weight'] != BALANCE or run['methods'] != list(METHODS)
                    or set(row['metrics']) != expected
                    or any(set(case['methods']) != set(METHODS) for case in row['metrics'].values())
                    or digests.setdefault(name, fingerprint) != fingerprint):
                raise ValueError(f'Unexpected protocol, data or incomplete evaluation in {report}')
            cells.append(row)
    if missing:
        raise ValueError(f'Missing {len(missing)} of 40 fixed-protocol cells: {", ".join(missing)}')
    return cells


def flatten(tree, prefix=''):
    for key, value in tree.items():
        if isinstance(value, dict):
            yield from flatten(value, f'{prefix}{key}.')
        elif isinstance(value, (int, float)):
            yield f'{prefix}{key}', value


def interval(values):
    average = mean(values)
    result = dict(mean=average, n=len(values))
    if len(values) > 1:
        spread = stdev(values)
        radius = float(t.ppf(.975, len(values) - 1)) * spread / len(values) ** .5
        result.update(sd=spread, lower=average - radius, upper=average + radius)
    return result


def aggregate(cells):
    values = defaultdict(lambda: defaultdict(list))
    for cell in cells:
        run = cell['run']
        dataset, seed = run['dataset'], run['seed']
        for condition, case in cell['metrics'].items():
            methods = case['methods']
            for key, value in flatten(methods):
                values[dataset, f'{condition}.{key}'][seed].append(value)
            for method in METHODS:
                if method == 'cibf':
                    continue
                for metric in ('accuracy', 'macro_f1', 'brier', 'nll', 'aurc', 'ece15'):
                    delta = methods['cibf'][metric] - methods[method][metric]
                    values[dataset, f'paired.{condition}.{method}.{metric}'][seed].append(delta)
        for key, value in flatten(dict(train_seconds=cell['train_seconds'], parameters=cell['parameters'])):
            values[dataset, key][seed].append(value)
    summary = {name: {} for name in DATASETS}
    for (dataset, key), seeds in values.items():
        folds = 5 if dataset == 'iemocap' else 1
        averages = [mean(scores) for scores in seeds.values() if len(scores) == folds]
        if averages:
            summary[dataset][key] = interval(averages)
    return summary


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)])


def report(summary):
    def value(dataset, key, fmt='.4f', scale=1):
        item = summary[dataset].get(key)
        return format(item['mean'] * scale, fmt) if item else '—'

    def difference(dataset, key, fmt='+.4f', scale=1):
        row = summary[dataset][key]
        return ' / '.join(format(row[k] * scale, fmt) for k in ('mean', 'lower', 'upper'))

    head = []
    for d in DATASETS:
        accuracy = value(d, 'paired.clean.tmc_aug.accuracy', '+.2f', 100)
        brier = value(d, 'paired.clean.tmc_aug.brier', '+.4f')
        head.append((NAMES[d], value(d, 'clean.cibf.accuracy', '.2f', 100), accuracy, brier,
                     value(d, 'paired.conflict.tmc_aug.brier', '+.4f'),
                     value(d, 'clean.cibf.aurc'), value(d, 'clean.tmc_aug.aurc')))
    overview = table(('数据集', 'CIBF 干净 Accuracy %', '干净 Accuracy 差（百分点）', '干净 Brier 差',
                      '换配 Brier 差', 'CIBF 干净 AURC', 'TMC-Aug 干净 AURC'), head)
    rows = [(NAMES[d], method, *(value(d, f'clean.{method}.{m}', '.2f', 100) for m in ('accuracy', 'macro_f1')),
             *(value(d, f'clean.{method}.{m}') for m in ('brier', 'nll', 'ece15', 'aurc')))
            for d in DATASETS for method in METHODS]
    clean = table(('数据集', '方法', 'Accuracy %', 'Macro-F1 %', 'Brier', 'NLL', 'ECE', 'AURC'), rows)
    rows = [(NAMES[d], case, *(f"{value(d, f'{case}.{method}.accuracy', '.2f', 100)} / "
                               f"{value(d, f'{case}.{method}.brier')}" for method in METHODS))
            for d in DATASETS for case in KEY_CASES[1:]]
    robustness = table(('数据集', '条件', *METHODS), rows)
    rows = [(NAMES[d], case, method, difference(d, f'paired.{case}.{method}.accuracy', '+.2f', 100),
             difference(d, f'paired.{case}.{method}.brier'))
            for d in DATASETS for case in KEY_CASES for method in ('tmc', 'tmc_aug', 'vib_ds')]
    paired = table(('数据集', '条件', '对照', 'Accuracy 差（百分点）', 'Brier 差'), rows)
    diagnostics = ('smce.mae', 'smce.zero_mae', 'smce.spearman', 'smce.harmful_detection.auroc',
                   'smce.cross_swap_detection.auroc', 'mean_retention', 'mean_conflict', 'mean_uncertainty')
    rows = [(NAMES[d], case, *(value(d, f'{case}.cibf.{m}') for m in diagnostics))
            for d in DATASETS for case in ('clean', 'conflict')]
    contribution = table(('数据集', '条件', '贡献 MAE', '零预测 MAE', 'Spearman', '负贡献 AUROC',
                          '异类换配 AUROC', '证据保留率', '冲突质量', '证据不确定性'), rows)
    return f'''# TMC-CIBF 固定参数验证

四个已有数据集，40 个数据集或折 × 种子单元。每个单元训练 9 种方法，共 360 次拟合。IEMOCAP 按种子先等权平均五个测试会话，再对五个种子汇总。协议与复现入口见 [TMC-CIBF 实验协议](../../TMC_CIBF.md)。

## 与同增强 TMC 的比较

表中各差值为 CIBF 减 TMC-Aug。Brier 和 AURC 越小越好。

{overview}

## 干净测试集

Brier 为逐类别平方误差和。ECE 使用 15 个固定宽度区间。AURC 使用证据确定性排序，数值越小越好。

{clean}

## 高斯噪声、未见坐标丢弃与模态换配

每格为 Accuracy % / Brier。换配包含同类和异类样本，测试扰动对所有方法一致。

{robustness}

## 与基线的配对差

每格为 CIBF 减对照的均值 / 95% 区间下界 / 上界。区间是固定划分上的配对种子 t 区间，未做多重比较校正。Brier 差为负表示改善。

{paired}

## 贡献估计与证据诊断

贡献监督目标是当前模型在加入模态前后的半 Brier 差。零预测 MAE 使用同一目标。负贡献 AUROC 只看当前完整上下文。异类换配 AUROC 的正例是受异类换配的模态。没有可定义结果时显示「—」。

{contribution}

本地 `summary.json` 包含完整测试条件、消融、训练耗时、参数量和逐折数据 SHA-256。
'''


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('runs/tmc_cibf'))
    parser.add_argument('--output', type=Path, default=Path('results/tmc_cibf'))
    args = parser.parse_args()
    cells = read_cells(args.runs)
    summary = aggregate(cells)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / 'summary.json', dict(metrics=summary, cells=[dict(run=c['run'], splits=c['splits'])
                                                                            for c in cells]))
    (args.output / 'REPORT.md').write_text(report(summary), encoding='utf-8')


if __name__ == '__main__':
    main()
