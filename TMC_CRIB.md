# TMC-HCRD-CRIB v2 实验协议

实验基于现有的 CMU-MOSI（二分类，去掉零分）、CH-SIMS（五分类）、MELD（七分类）和 IEMOCAP（六分类）特征缓存。`data.load_data` 检查跨划分样本和组是否重叠，标准化参数由该折训练集拟合。四个数据集和五个种子均使用同一套超参数。

## PyCharm 复现

在 PyCharm 打开项目，选择安装 `requirements.txt` 依赖的解释器。主实验读取 `data/extended`。准备方法见 [TMC_VALIDATION.md](TMC_VALIDATION.md) 和 [TMC_EXTENDED.md](TMC_EXTENDED.md)。

```powershell
python validate_tmc_crib.py --smoke --datasets cmu_mosi --seeds 11
python validate_tmc_crib.py

python prepare_tmc_crib_iemocap.py PATH_TO_IEMOCAP_features.pkl
foreach ($fold in 1..5) {
    python validate_tmc_crib.py --features data/crib_iemocap --output runs/tmc_crib_loso --fold $fold --methods tmc tmc_aug full
}
python summarize_tmc_crib.py
```

也可运行共享的 `TMC CRIB v2 Full` 配置。主实验支持按数据集或种子并行和自动续跑。每单元保存模型、数据 SHA-256、训练日志、选中轮次和逐测试条件指标。汇总脚本检查 200 个主实验和 75 个 LOSO 单元齐全，输出 `results/tmc_crib/REPORT.md` 和 `results/tmc_crib/summary.json`。

## 固定设置与对照

- PyCharm 项目的 Python 3.12.3、PyTorch 2.6.0 CPU。每进程一个 Torch 线程，种子为 11、22、33、44、55。
- AdamW，学习率 0.001，权重衰减 0.0001，批量 128，主干维度 128，CRIB 隐藏维度 64，潜变量 16，任务嵌入 4，风险权重 λ=0.1，KL 权重 β=0.001。
- TMC 在原始配对上预训练 5 轮，以干净验证 NLL 选择教师。之后训练 15 轮。v2 的学生主干与证据头冻结，并用验证 NLL 加风险及 KL 加权项选检查点。验证扰动固定。`tmc_aug` 解冻主干并继续在同样扰动上训练，以验证 NLL 选择检查点。`rib` 以相同教师初始化，解冻主干，保留上一版 RIB 的风险目标与线性折扣，选点附加其风险损失。
- 训练扰动等概率选择原始、观测污染、同类换配、异类换配。另对可用模态独立做 15% 缺失模拟，保留至少一个可用模态。污染强度为 0、0.2、0.5、0.9。换配只在同一划分选取有对应模态的不同样本，单模态分类项屏蔽换配模态。
- `full` 为条件先验与主观逻辑折扣。`context_only` 将额外风险特征及可用标记置零，使用确定性潜变量。`no_ib` 保留特征但去掉采样与 KL。`fixed_ib` 使用标准高斯先验。`no_responsibility` 移除冲突三分类监督。`no_harm` 把折扣替换成旧的三风险乘积形式，保留 D 预测供诊断。`linear` 将主观逻辑折扣替换为线性折扣。完整条件先验在 `full` 中实现，换用其他先验或折扣的差异由对应消融测量。

## 指标与边界

- 测试包括原始样本、高斯噪声、未见坐标丢弃、同类和异类换配、逐模态缺失及单模态条件。风险指标针对人工干预。模糊性以冻结教师对原始模态的熵为参照。有害证据以教师移除模态后的 Brier 降幅大于零定义。汇报的分类 Brier 为逐类别平方误差和，反事实监督使用该量的一半。
- 冲突 AUROC/AUPRC 使用所有有效模态对。责任三分类同时包含无异类换配、左端异类换配和右端异类换配。端点定位只在已知受干预的模态对上计算。TMC 使用原始 JSD 和单模态熵作为冲突与责任启发式分数。
- Full 检查点额外评估批内潜变量置零和循环打乱。逐样本诊断保存每任务 KL 的 10%、50%、90% 分位，逐轮日志记录 CRIB 第一层权重的训练梯度范数。推理延迟为单样本 30 次前向的平均值，受本机并发负载影响。
- IEMOCAP 的普通主表使用现有 DialogueGCN 文本特征，其文本提取器用情绪标签训练。跨会话表使用原始转录文本的 512 维哈希一元组和二元组替换该文本特征。音频与视觉仍来自原发布缓存。每折保持训练、验证、测试会话互斥。 [DialogueGCN 第 3.2 节](https://aclanthology.org/D19-1015.pdf)
- 当前验证覆盖预提取特征的人工扰动与缺失。自然语义冲突、原始模态损坏及真实缺失仍需额外标注或原始模态测试。

## 本轮结论

Full 在四个原始测试集上均取得高于旧 RIB 的平均准确率。相比原始 TMC，CMU-MOSI、CH-SIMS 和 IEMOCAP 的干净准确率差值不超过 0.15 个百分点。MELD 低 1.47 个百分点。Full 的干净 NLL、Brier 和 ECE 在四个数据集上都高于 TMC，校准目标未达成。

异类换配检测 AUROC 为 0.489–0.505，受干预端点的定位准确率为 49.19%–51.76%，与 50% 随机基准接近。未见坐标丢弃的观测 AUROC 为 0.664–0.801，四个数据集均低于旧 RIB。Full 对 Context-Only 和 Fixed-IB 的增益随数据集与指标变化。完整消融和潜变量干预结果未显示条件先验带来稳定的独立收益。

替换 IEMOCAP 监督文本特征后，五折平均准确率为 56.06%，TMC 为 56.15%，增强训练 TMC 为 54.87%。对应 Macro-F1 为 53.87%、53.73%、53.64%。 [完整结果](results/tmc_crib/REPORT.md)
