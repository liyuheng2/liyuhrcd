"""Summarize the fixed-setting experiment, retaining seed-level pairings."""
from argparse import ArgumentParser
from collections import defaultdict
from csv import DictReader, DictWriter
from itertools import product
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DATASETS = {'cmu_mosi': 'CMU-MOSI', 'ch_sims': 'CH-SIMS', 'meld': 'MELD', 'iemocap': 'IEMOCAP'}


def read(root, name):
    with (root / (name + '.csv')).open(encoding='utf-8-sig', newline='') as file:
        return list(DictReader(file))


def write(root, name, rows):
    with (root / (name + '.csv')).open('w', encoding='utf-8-sig', newline='') as file:
        writer = DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def group_mean(rows, metric):
    return mean(float(row[metric]) for row in rows if row.get(metric))


def missing_plot(root):
    data = read(root, 'missing_modalities')
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=False)
    conditions = ('0', '1', '2', '01', '02', '12', '012')
    labels = ('T', 'A', 'V', 'T+A', 'T+V', 'A+V', 'T+A+V')
    for (ds, label), ax in zip(DATASETS.items(), axes.flat):
        for method in ('tmc', 'simple', 'full', 'full_no_dropout'):
            selected = [r for r in data if r['dataset'] == ds and r['method'] == method]
            vals = [[float(r['accuracy']) for r in selected if r['condition'] == c] for c in conditions]
            ax.errorbar(labels, list(map(mean, vals)), yerr=list(map(stdev, vals)),
                        marker='o', capsize=2, label=method)
        ax.set(title=label, ylabel='Accuracy')
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root / 'missing_modalities.png', dpi=180)
    plt.close(fig)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/tmc_extended'))
    root = parser.parse_args().output
    data = read(root, 'main_comparison')
    clean = {(r['dataset'], r['method'], r['seed']): r for r in data if not r['fold']}
    robustness = read(root, 'robustness')
    for row in robustness:
        baseline = float(clean[row['dataset'], row['method'], row['seed']]['accuracy'])
        current = float(row['accuracy'])
        row.update(accuracy_drop=baseline-current, retention=current/baseline if baseline else '')
    write(root, 'robustness', robustness)
    paired = read(root, 'significance')
    fusion = read(root, 'fusion_ablation')
    conflict = read(root, 'conflict_localization')
    observation = read(root, 'risk_observation')
    efficiency = read(root, 'complexity')
    risks = read(root, 'risk_ambiguity')
    summary = read(root, 'main_summary')
    methods = ['tmc', 'capacity', 'simple', 'only_o', 'hcrd_equal', 'fixed_equal', 'dynamic_equal', 'full']
    clean_report = []
    for dataset, method in product(DATASETS, methods):
        rows = {r['metric']: r for r in summary if r['dataset'] == dataset and r['method'] == method and not r['fold']}
        clean_report.append(dict(dataset=dataset, method=method,
            **{key: float(rows[key]['mean']) for key in ('accuracy', 'macro_f1', 'nll', 'ece15', 'brier', 'aurc')},
            accuracy_sd=float(rows['accuracy']['sd']), n_seeds=int(rows['accuracy']['n'])))
    write(root, 'paper_main_table', clean_report)
    report = [
        '# TMC-HCRD-RIB 补充验证结果', '',
        f'完成 {len(clean)} 个主矩阵训练单元和 {len(data)-len(clean)} 个 IEMOCAP 跨会话训练单元。',
        '所有训练使用现有冻结特征和固定超参数。MOSI 为排除零分的二分类，CH-SIMS 为五分类，MELD 为七分类，IEMOCAP 为六分类。',
        '使用本机 CPU 计时。实验协议和复现命令见 [TMC_EXTENDED.md](../../TMC_EXTENDED.md)。', '',
        '## 干净测试集', '',
        '| 数据集 | TMC Acc | 完整 Acc | TMC → 完整 Macro-F1 | TMC → 完整 NLL | 准确率差及 95% CI |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    by_key = {(r['dataset'], r['method']): r for r in clean_report}
    for ds, label in DATASETS.items():
        a, b = by_key[ds, 'tmc'], by_key[ds, 'full']
        ci = next(r for r in paired if r['dataset'] == ds and r['method'] == 'full' and r['reference'] == 'tmc')
        delta, low, high = [100 * float(ci[k]) for k in ('mean', 'ci_low', 'ci_high')]
        report.append(f"| {label} | {100*a['accuracy']:.2f}% | {100*b['accuracy']:.2f}% | "
                      f"{a['macro_f1']:.3f} → {b['macro_f1']:.3f} | {a['nll']:.3f} → {b['nll']:.3f} | "
                      f"{delta:+.2f} [{low:+.2f}, {high:+.2f}] pp |")
    report += ['', '固定划分的 IEMOCAP 测试集只有两个重采样组，该区间仅作描述。会话间表现见下方 LOSO 结果。']
    report += ['', '## 机制对照', '',
               '坐标丢弃强度 0.5，熵变化 MAE 在三个单模态污染条件上等权平均。冲突定位使用已知替换模态，平局按并列数折算。', '',
               '| 数据集 | 固定 KL | 动态 KL | 等均值 KL | 打乱 KL | 反向 KL | 三模态责任定位 |',
               '|---|---:|---:|---:|---:|---:|---:|']
    for ds, label in DATASETS.items():
        maes = []
        for method in ('fixed_equal', 'dynamic_equal', 'mean_equal', 'shuffle_equal', 'reverse_equal'):
            rows = [r for r in risks if r['dataset'] == ds and r['method'] == method and r['proxy'] == 'delta'
                    and r['condition'] in {f'dropout_{i}_0.5' for i in range(3)}]
            maes.append(f"{group_mean(rows, 'mae'):.4f}")
        located = [r for r in conflict if r['dataset'] == ds and r['method'] == 'full'
                   and r['modalities'] == '3' and r['proxy'] == 'responsibility' and r['condition'] == 'swap']
        report.append('| ' + ' | '.join([label, *maes, f"{group_mean(located, 'top1'):.3f}"]) + ' |')
    report += ['', '## 跨会话与融合对照', '',
               'IEMOCAP 五折属于冻结缓存上的探索性结果。DialogueGCN 的文本提取器使用情绪标签训练，缓存上游的训练划分未针对本实验各折重建。TMC 在会话 1–4 的准确率约为 89%–92%，会话 5 约为 52%。五折均值不宜作为严格的跨会话泛化证据。 [DialogueGCN 论文第 3.2 节](https://aclanthology.org/D19-1015/)', '']
    for method in ('tmc', 'full'):
        folds = defaultdict(list)
        for row in data:
            if row['method'] == method and row['fold']:
                folds[row['fold']].append(float(row['accuracy']))
        acc = [mean(scores) for scores in folds.values()]
        report.append(f'- IEMOCAP LOSO {method} 的折均准确率为 {100*mean(acc):.2f}%，折间标准差为 {100*stdev(acc):.2f} 个百分点。')
    for ds, label in DATASETS.items():
        selected = [r for r in fusion if r['dataset'] == ds and r['condition'] == 'clean'
                    and r['modalities'] == '3' and not r['fold']]
        equal, csf = [group_mean([r for r in selected if r['fusion'] == f], 'nll') for f in ('F4_equal', 'F5_csf')]
        report.append(f'- {label} 三模态干净测试中，冲突均分与责任分配的 NLL 分别为 {equal:.4f} 与 {csf:.4f}。')
    report += ['', '## 未见污染与计算成本', '',
               '单模态坐标丢弃均值对三种目标模态和三个强度等权计算。延迟为 CPU 批量 32 的每批均值。', '',
               '| 数据集 | TMC → 完整污染 Acc | 观测风险 AUROC | TMC vacuity AUROC | TMC → 完整延迟（ms） |',
               '|---|---:|---:|---:|---:|']
    for ds, label in DATASETS.items():
        rates = [group_mean([r for r in robustness if r['dataset'] == ds and r['method'] == method
                             and r['condition'].startswith('dropout_') and len(r['condition'].split('_')[1]) == 1],
                            'accuracy') for method in ('tmc', 'full')]
        auroc = [group_mean([r for r in observation if r['dataset'] == ds and r['method'] == 'full'
                             and r['proxy'] == proxy and r['condition'].startswith('dropout_')
                             and len(r['condition'].split('_')[1]) == 1], 'auroc')
                 for proxy in ('risk', 'tmc_u')]
        times = [group_mean([r for r in efficiency if r['dataset'] == ds and r['method'] == method
                             and r['batch'] == '32'], 'cpu_ms') for method in ('tmc', 'full')]
        report.append(f'| {label} | {100*rates[0]:.2f}% → {100*rates[1]:.2f}% | {auroc[0]:.3f} | '
                      f'{auroc[1]:.3f} | {times[0]:.3f} → {times[1]:.3f} |')
    report += ['', '## 图表', '', '- [主结果表](paper_main_table.csv) · [配对置信区间](significance.csv)',
               '- [抗污染曲线](robustness.png) · [风险–覆盖率](risk_coverage.png)',
               '- [动态系数与 KL 分布](dynamic_kl.png) · [污染等级与 KL](kl_by_severity.png)',
               '- [七种模态组合](missing_modalities.png)', '']
    (root / 'README.md').write_text('\n'.join(report), encoding='utf-8')
    missing_plot(root)


if __name__ == '__main__':
    main()
