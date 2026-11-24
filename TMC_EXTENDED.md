# TMC-HCRD-RIB 固定参数补充验证

在项目目录使用 PyCharm 的 Python 3.12 解释器并安装 `requirements.txt` 中的依赖。补充实验复用本地冻结特征。MOSI 与 CH-SIMS 需要本地 MMSA 评分文件，将评分按样本 ID 回填到现有缓存。

```powershell
python prepare_tmc_extended.py --mosi PATH_TO_ALIGNED_50.pkl --sims PATH_TO_UNALIGNED_39.pkl
python train_tmc_extended.py
foreach ($fold in 1..5) {
    python train_tmc_extended.py --fold $fold --methods tmc full
}
python evaluate_tmc_extended.py
python summarize_tmc_extended.py
```

- MOSI 使用正负二分类并排除零分。CH-SIMS 按 −0.6、−0.2、0.2、0.6 切分，边界归高一档。CH-SIMS 保留现有源视频不重叠的三套划分。MELD 七类、IEMOCAP 六类保持原缓存。
- 主矩阵每数据集 20 个方法，每方法 5 个种子，共 400 个单元。包含 8 个主模型、4 个外部融合基线、4 个额外 KL 对照、2 个干净训练对照和 2 个无模态 Dropout 对照。内部消融统一证据头。TMC-Capacity 使用按总参数量近似匹配的残差 MLP。外部动态证据对照 ETMC 采用仓库的联合伪视图实现。
- 训练沿用上一轮固定优化参数和最多 5 轮分类预热、5 轮风险预热、30 轮联合训练。联合阶段按干净验证 NLL 选择检查点，耐心为 10 轮。固定潜变量维度 16、β=0.001、风险损失权重为 1。IB-0 的 β=0 是消融。平均权重对照使用同种子动态模型在训练样本上的任务均值。各单元以锁保护，可按数据集和种子并行或续跑。
- IEMOCAP 的 LOSO 包含 5 个外层会话。每折用剩余会话中的一个作验证，其余三个用于优化。TMC 和完整模型各跑 5 个种子，共 50 个单元。
- IEMOCAP LOSO 使用上游情绪标签监督得到的 DialogueGCN 文本特征。分类器与标准化在每折重新拟合，特征提取器沿用缓存，因此五折结果用于探索会话变化。可直接查看原测试会话的独立折结果。 [DialogueGCN 论文第 3.2 节](https://aclanthology.org/D19-1015/)
- 风险和融合测试在训练后使用固定检查点。F1 的常数折扣取干净验证集上完整模型的平均折扣。换配测试以对应干净配对作阴性对照。自然高分歧子集由同种子完整参考模型的验证集 JSD 90% 分位定义，检出阈值取各分数在验证集的 95% 分位。缺失实验中每个组合只使用原始具有所选模态的样本。
- 测试扰动应用在预提取特征。高斯为训练类型，坐标丢弃为未见类型。单模态污染用 0.2、0.5 和 0.9，双模态及全模态污染用 0.5。换配样本用于冲突诊断。冻结的教师检查点用于风险参考。
- IB 诊断表中的动态系数是在各检查点上按统一公式计算的候选系数。固定和打乱对照的训练权重分别由其配置决定。折扣替换与潜变量扰动属于冻结检查点的诊断实验。
- `results/tmc_extended` 保存逐种子表、汇总表、按种子和视频或对话分组重采样的准确率配对置信区间、PNG 图和压缩预测文件。模型、教师和训练日志保存于 `runs/tmc_extended`。复杂度表使用线性层 MACs 和本机 CPU 实测，训练时长包括教师与风险预热并受到并发训练影响。
