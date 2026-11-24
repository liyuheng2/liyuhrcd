# TMC-HCRD-ACRIB 固定参数协议

## PyCharm 复现

打开项目目录并选择已安装 `requirements.txt` 的 Python 3.12 解释器。项目工作目录为仓库根目录。共享 PyCharm 运行配置 **TMC ACRIB Full** 执行四数据集五种子主实验。

```powershell
python -m unittest -v test_tmc_acrib
python validate_tmc_acrib.py --smoke --datasets cmu_mosi --seeds 11 --workers 1
python validate_tmc_acrib.py --workers 8
python summarize_tmc_acrib.py
```

使用现有 `data/extended/{cmu_mosi,ch_sims,meld,iemocap}/{train,val,test}.npz`。每个 NPZ 需要 `x0,x1,x2,mask,y,ids,groups`。三模态顺序为文本、音频、视觉。数据加载器校验跨划分的组和样本 ID，并仅根据训练划分拟合标准化统计。CMU-MOSI 为去除零分后的二分类，CH-SIMS 为五分类，MELD 为七分类，IEMOCAP 为六分类。使用原有三模态缓存时，各数据集沿用此前 TMC-CRIB 实验划分。IEMOCAP 另有 **TMC ACRIB IEMOCAP LOSO** 运行配置。该配置用现有 `data/crib_iemocap` 缓存在种子 11 下逐一留出五个会话，文本特征为无标签的转录哈希特征。

主实验每个数据集、种子和方法保存独立 JSON、训练日志、阈值校准结果、特征 SHA-256 和模型权重。重复运行时复用元数据匹配的完整单元。汇总程序要求四数据集的 280 个单元齐全，生成 `results/tmc_acrib/REPORT.md` 和逐种子逐条件的 `metrics.csv`。数据、模型权重和逐单元运行目录保存在本地。

```powershell
python validate_tmc_acrib.py --features data/crib_iemocap --folds 1 2 3 4 5 --seeds 11 --workers 5
python summarize_tmc_acrib.py --include-loso
```

如需从原发布的特征包重建缓存，可用以下入口。MOSI 和 CH-SIMS 使用包含回归标签及样本 ID 的 MMSA 特征文件，MELD 使用带官方划分的 MMAlign 特征目录，IEMOCAP 使用 DialogueGCN 特征文件。

```powershell
python prepare_tmc_features.py cmu_mosi PATH_TO_MOSI_PKL
python prepare_tmc_features.py ch_sims PATH_TO_CH_SIMS_PKL
python prepare_tmc_features.py iemocap PATH_TO_IEMOCAP_PKL
python prepare_meld_mmalign.py PATH_TO_MELD_FEATURES
python prepare_tmc_extended.py --mosi PATH_TO_MOSI_PKL --sims PATH_TO_CH_SIMS_PKL
python prepare_tmc_crib_iemocap.py PATH_TO_IEMOCAP_PKL
```

## 参数与训练

| 参数 | 固定值 |
|---|---|
| 主干 / 风险隐藏层 / 最大潜维度 | 128 / 64 / 16 |
| 标准化均值偏移与对数方差比的绝对值上界 | 2 / 1 |
| 批量 / 优化器 / 衰减 | 128 / AdamW / 0.0001 |
| 预训练 / 冻结风险学习 / 联合训练轮次 | 5 / 10 / 5 |
| 预训练与风险学习 / 联合训练学习率 | 0.001 / 0.0001 |
| 风险 λ / KL β | 1 / 0.01 |
| 随机种子 | 11、22、33、44、55 |
| 后验采样数 | 训练 1，正式评估 8 |
| 证据保留动作 | 0、0.5、1 |

各模型从相同的种子配对 TMC 分类主干初始化。正式实验采用固定轮次，因此验证集用于动作阈值校准。校准在独立验证划分采用固定随机种子的训练式扰动混合，以平均半 Brier 收益选择阈值。阈值候选为 ∞、0.1、0.05、0.02、0.01、0.005、0。并列优先较保守的策略，候选动作并列优先保留 0.5。每个样本最多调整一个有效模态。联合版的分类损失使用原始 DS 融合和来源模态标签。A 的监督取冻结教师的原始熵，D 使用当前模型的停止梯度反事实证据。污染和责任诊断使用人工干预参考标签。

分类 Brier 为各类平方误差和。D 使用其一半。报告中的损害率按实际执行动作的真实负收益计算。错误折扣率指全部样本中执行了无正收益动作的占比。无变化数量或单类标签导致部分指标无定义时记录为空。

## 对照

| 代码名称 | 实验定义 |
|---|---|
| `tmc` / `tmc_aug` | 基础 TMC / 与联合版等轮次、等学习率并使用相同干预的增强 TMC |
| `crib_v2` | 现有 CRIB v2 的移除收益监督和连续可靠性折扣。按本轮轮次和 λ、β 重新训练 |
| `no_ib` / `context_only` | 确定性编码 / 只提供条件背景 |
| `fixed_vib` / `rib` | 标准高斯固定先验 / 同先验下用停止梯度的风险信号缩放 KL |
| `crib` / `idab_crib` | 条件先验 / 受 IDAB 启发的条件残余硬截断，以直通估计训练 |
| `fixed_acrib` / `random_acrib` | 仅用训练数据匹配任务平均容量的固定门控 / 样本独立的随机门控。另报告实际 KL |
| `unbounded_acrib` | 移除 tanh 残余约束的门控版本。沿用数值稳定的对数方差截取 |
| `acrib` / `acrib_joint` | 冻结版 / 从冻结版继续联合训练的完整模型 |

每个 ACRIB 系列检查点支持原始 DS、风险乘积折扣及 ACRF 融合，比较时复用同一组证据和风险输出。测试包含干净输入、三个高斯强度、两档未知坐标丢弃、同类与异类混合换配、逐模态缺失和单模态输入。完整模型额外对 A、C、D 的潜变量做置零和按有效批内行位置循环打乱。该测试只干预推理链路，沿用标准模型的校准阈值。
