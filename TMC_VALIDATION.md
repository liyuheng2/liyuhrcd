# TMC-HCRD-RIB 固定参数验证

## 数据与复现

所有模型共用预提取特征、划分、训练集标准化和扰动参数。CMU-MOSI 使用 MMSA aligned_50 的原始划分，CH-SIMS 使用 MMSA unaligned_39 并按源视频重新分组，避免发布划分中的源视频重叠。MOSI 和 SIMS 按情感值的负、零、正映射为三类。IEMOCAP 使用 DialogueGCN 的六类情绪特征，保留发布测试集并按源视频从训练集留出验证组。MELD 使用 MM-Align 发布的三模态七分类数据，沿用 train/dev/test，对音频和视频逐话语池化，对 token ID 的 TF-IDF 和 SVD 仅在训练集拟合。

- [MMSA 数据下载](https://github.com/thuiar/MMSA#2-datasets)
- [DialogueGCN IEMOCAP 特征](https://github.com/declare-lab/conv-emotion/tree/master/DialogueGCN/IEMOCAP_features)
- [MM-Align 数据说明](https://github.com/declare-lab/MM-Align#meld-dataset) · [MELD 三模态归档](https://drive.google.com/file/d/1RjrYSMpXxg_6r_nUQaysaPyMsldLpMcb/view)

在 PyCharm 使用 Python 3.12 项目解释器，从仓库根目录安装 `requirements.txt`。将各数据发布文件放在下方对应位置。MELD 解包后目录包含 `train.pkl`、`dev.pkl` 和 `test.pkl`。仅加载可信发布者提供的 pickle 文件。

```powershell
python prepare_tmc_features.py cmu_mosi data/raw/cmu_mosi/Processed/aligned_50.pkl
python prepare_tmc_features.py ch_sims data/raw/ch_sims/Processed/unaligned_39.pkl
python prepare_tmc_features.py iemocap data/raw/iemocap/IEMOCAP_features.pkl
python prepare_meld_mmalign.py data/raw/meld_mmalign
python validate_tmc.py --datasets cmu_mosi --methods tmc full --device cpu --smoke
python validate_tmc.py --device cpu
python evaluate_tmc_shift.py
```

PyCharm 提供 **TMC HCRD Smoke** 和 **TMC HCRD Full**。已有旧版 MELD 缓存时，可给四个准备命令传入 `--output data/tmc_features`，并在运行时加上 `--feature-dir data/tmc_features`。

## 模型与训练

TMC 在同一预提取特征协议下使用 Dirichlet 证据、退火 EDL 损失和 DS 融合。共享条件网络预测有符号的熵变化及正常 JSD 基线。DS 实现先在恢复证据的公式中消去冲突归一化分母，避免接近完全冲突时做小分母除法。缺失模态提供零证据，单模态的冲突责任为零。

| 变体 | 证据折扣 | 风险潜变量 |
|---|---|---|
| `tmc` | 无 | 无 |
| `simple` | 证据确定度 $1-u$ | 无 |
| `hcrd` | 观测风险和残余模糊性 | 确定性 |
| `fixed_ib` | 观测风险和残余模糊性 | 变分，单位 KL 权重 |
| `dynamic_ib` | 观测风险和残余模糊性 | 变分，动态 KL |
| `full` | 观测风险、残余模糊性和责任分配 | 变分，动态 KL |
| `no_o` / `no_a` / `no_c` | 分别移除一种风险的折扣作用 | 同完整模型 |
| `only_o` / `only_a` | 仅一种风险参与折扣 | 同完整模型 |

折扣消融保留辅助风险监督，用于检验风险通过折扣进入融合的贡献。按此定义，`no_c` 是 `dynamic_ib` 的实验重复。动态系数使用冻结预热模型产生的干净熵、污染强度标签及当前 JSD 相对干净基线的正残差作为代理，仅调整 KL 项和停止梯度。推理时采用潜变量均值，预测只接收当前特征和存在掩码。干净参考用于训练和离线诊断。

AdamW 学习率 0.001，权重衰减 0.0001，批大小 128，隐藏宽度 128，潜变量宽度 16，$\lambda=1$，$\beta=0.001$。种子为 11、22、33、44、55。分类预热 5 轮，风险模块预热 5 轮时冻结分类主干，联合训练 30 轮并评估最终轮。TMC 和简单折扣有相同的 35 轮分类更新。训练扰动为标准化特征上的高斯噪声和随机模态缺失。

## 结果

本轮五种子结果见 [实验报告](results/tmc_hcrd/README.md)。

`runs/tmc_hcrd/protocol` 包含每个训练单元的权重、`result.json` 及各条件的风险–覆盖率曲线。已完成单元按数据 SHA-256、设备、协议与配置校验后复用。汇总表 `summary.csv` 报告均值、种子间样本标准差和有效种子数。配对表 `paired.csv` 报告同种子差值以及双侧精确符号翻转检验，五对种子的最小双侧 p 值为 0.0625，p 值仅用于描述本轮配对结果。

测试项包括干净特征、多级高斯噪声、未见坐标丢弃、逐模态污染与缺失、仅单模态和受控换配。主要指标为 Accuracy、Macro-F1、ECE、NLL、Brier 和 AURC。诊断指标包含污染检测、正常冲突基线、熵变化与残余熵的参考误差，以及原始 JSD 和额外冲突检出同类与异类换配的能力。`shift_detection.csv` 记录将同一批干净样本和受扰动样本混合后，用证据不确定性、最大类别概率和观测风险区分分布偏移的 AUROC 与 AUPRC。实验结论适用于预提取特征以及这些受控干预。
