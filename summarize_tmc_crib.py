"""Aggregate complete, matched TMC-HCRD-CRIB seed sets."""
from argparse import ArgumentParser
from collections import defaultdict
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from main import atomic_json
from validate_tmc_crib import DATASETS, METHODS, SEEDS


CLASSIFICATION = {'accuracy': 'Acc', 'macro_f1': 'Macro-F1', 'ece15': 'ECE', 'nll': 'NLL',
                  'brier': 'Brier', 'aurc': 'AURC'}


def scalar_metrics(source, prefix=''):
    for key, value in source.items():
        field = f'{prefix}{key}'
        if isinstance(value, dict):
            yield from scalar_metrics(value, f'{field}.')
        elif isinstance(value, (int, float)) and np.isfinite(value):
            yield field, value


def stats(values):
    return {'n': len(values), 'mean': float(np.mean(values)), 'sd': float(np.std(values, ddof=1))
            if len(values) > 1 else None}


def load_cells(root, datasets, methods):
    reports = {}
    for name, seed, method in product(datasets, SEEDS, methods):
        path = root / name / str(seed) / method / 'result.json'
        if not path.is_file():
            raise FileNotFoundError(f'Incomplete experiment matrix, missing {path}')
        report = json.loads(path.read_text(encoding='utf-8'))
        run = report['run']
        if (run['dataset'], run['seed'], run['method']) != (name, seed, method):
            raise ValueError(f'Inconsistent run identity in {path}')
        reports[name, seed, method] = report
    for name in datasets:
        signatures = [{k: v for k, v in reports[name, seed, method]['run'].items() if k not in {'seed', 'method'}}
                      for seed, method in product(SEEDS, methods)]
        if any(spec != signatures[0] for spec in signatures):
            raise ValueError(f'Inconsistent data or settings in {root / name}')
    return reports


def aggregate(reports, name, method):
    values = defaultdict(list)
    for seed in SEEDS:
        row = reports[name, seed, method]
        selected = row['logs'][row['epoch'] - 1]
        items = dict(runtime={key: row[key] for key in ('parameters', 'trainable', 'train_seconds', 'latency_ms')},
                     selected_epoch=row['epoch'], gradient=selected['train']['encoder_grad_norm'],
                     validation_kl=selected['val']['kl'], **row['metrics'])
        for key, value in scalar_metrics(items):
            values[key].append(value)
    return {k: stats(v) for k, v in values.items()}


def paired(reports, name, baseline, condition, metric):
    values = [reports[name, seed, 'full']['metrics'][condition][metric]
              - reports[name, seed, baseline]['metrics'][condition][metric] for seed in SEEDS]
    result = stats(values)
    width = t.ppf(.975, len(values) - 1) * result['sd'] / np.sqrt(len(values))
    result['lo'], result['hi'] = result['mean'] - width, result['mean'] + width
    return result


def show(row, key, deviation=False):
    if key not in row:
        return '—'
    s = row[key]
    return f"{s['mean']:.4f}" + (f" ± {s['sd']:.4f}" if deviation and s['sd'] is not None else '')


def table(headers, rows):
    lines = [headers, ['---'] * len(headers), *rows]
    return ['| ' + ' | '.join(map(str, line)) + ' |' for line in lines]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('runs/tmc_crib'))
    parser.add_argument('--loso', type=Path, default=Path('runs/tmc_crib_loso'))
    parser.add_argument('--output', type=Path, default=Path('results/tmc_crib'))
    args = parser.parse_args()
    reports = load_cells(args.runs, DATASETS, METHODS)
    summary = {name: {method: aggregate(reports, name, method) for method in METHODS} for name in DATASETS}
    lines = ['# TMC-HCRD-CRIB v2 固定参数验证结果', '',
             '四个数据集、五个匹配种子、十个方法，共 200 个实验单元。表格为种子均值，± 后为样本标准差。', '',
             '## 干净测试集', '']
    lines += table(['数据集', '方法', *CLASSIFICATION.values()],
                   [[name, method, *(show(summary[name][method], f'clean.{key}', key in ('accuracy', 'macro_f1'))
                                     for key in CLASSIFICATION)] for name, method in product(DATASETS, METHODS)])
    lines += ['', '## 分类准确率的配对差值', '',
              'Full 减基线。区间为按五个种子计算的 95% 配对 t 区间，使用同一个测试集，未做多重比较校正。', '']
    contrasts = {}
    rows = []
    for name, cond, ref in product(DATASETS, ('clean', 'gaussian_0.9', 'unseen_dropout_0.5', 'conflict'),
                                   ('tmc', 'tmc_aug', 'rib')):
        s = paired(reports, name, ref, cond, 'accuracy')
        contrasts.setdefault(name, {}).setdefault(cond, {})[ref] = s
        rows.append([name, cond, ref, f"{s['mean'] * 100:+.2f}", f"[{s['lo'] * 100:+.2f}, {s['hi'] * 100:+.2f}]"])
    lines += table(['数据集', '测试条件', '基线', '差值（百分点）', '95% 区间'], rows)
    lines += ['', '## 风险任务与消融', '',
              '观测 AUROC 取未见坐标丢弃。模糊性 MAE 取高斯强度 0.9。其余风险指标取换配测试。',
              '端点定位仅统计已知异类换配的有效模态对，随机命中率为 0.5。', '']
    fields = {'unseen_dropout_0.5.observation.auroc': '观测 AUROC', 'gaussian_0.9.ambiguity.mae': '模糊性 MAE',
              'conflict.conflict.auroc': '冲突 AUROC', 'conflict.responsibility.endpoint_accuracy': '端点定位 Acc',
              'conflict.delta.spearman': '移除收益 ρ', 'conflict.harm.auroc': '有害证据 AUROC',
              'conflict.fraction_discounted': '折扣比例', 'validation_kl': '验证 KL',
              'gradient': '编码器梯度范数'}
    lines += table(['数据集', '方法', *fields.values()],
                   [[name, method, *(show(summary[name][method], key) for key in fields)]
                    for name, method in product(DATASETS, METHODS)])
    lines += ['', '## 潜变量干预', '', '对 Full 的同一检查点置零或循环打乱批内潜变量。', '']
    lines += table(['数据集', '潜变量', '干净 Acc', '冲突 AUROC', '端点定位 Acc', '移除收益 ρ'],
                   [[name, op, show(summary[name]['full'], f'{test}.accuracy'),
                     *(show(summary[name]['full'], f'{conflict}.{k}') for k in
                       ('conflict.auroc', 'responsibility.endpoint_accuracy', 'delta.spearman'))]
                    for name in DATASETS for op, test, conflict in
                    [('原始', 'clean', 'conflict'), ('置零', 'zero', 'zero_conflict'), ('打乱', 'shuffle', 'shuffle_conflict')]])
    lines += ['', '## IEMOCAP 跨会话复核', '',
              '用转录文本的 512 维哈希特征替换监督文本特征。每折保留一个会话测试、一个验证、三个训练。',
              '每个会话均报告五个种子，跨折总均值对会话等权。沿用现有音频与视觉特征。', '']
    fold_methods = ('tmc', 'tmc_aug', 'full')
    loso = {}
    rows = []
    for fold in range(1, 6):
        folder = args.loso / f'loso{fold}'
        runs = load_cells(folder, ('iemocap',), fold_methods)
        loso[str(fold)] = {method: aggregate(runs, 'iemocap', method) for method in fold_methods}
        for method, row in loso[str(fold)].items():
            rows.append([fold, method, *(show(row, f'clean.{key}', key in ('accuracy', 'macro_f1'))
                                         for key in CLASSIFICATION)])
    for method in fold_methods:
        all_folds = {key: stats([fold[method][key]['mean'] for fold in loso.values()])
                     for key in (f'clean.{metric}' for metric in CLASSIFICATION)}
        rows.append(['五折均值', method, *(show(all_folds, f'clean.{key}') for key in CLASSIFICATION)])
    lines += table(['会话', '方法', *CLASSIFICATION.values()], rows)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    atomic_json(args.output / 'summary.json', dict(main=summary, paired_accuracy=contrasts, loso=loso))
    print('Verified 200 main runs and 75 LOSO runs.')


if __name__ == '__main__':
    main()
