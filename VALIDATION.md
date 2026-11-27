# EC-RIB 三数据集固定参数实验

本协议使用 **CH-SIMS、MIntRec 初版、UPMC Food-101**。三个数据集均比较 QMF、SeD-UD 和 CDPR 的统一特征适配版，以及完整 EC-RIB 和六项消融。所有方法共享各数据集的输入特征、训练集统计量、随机种子、污染样本和测试条件。

| 数据集 | 数据及划分 |
| --- | --- |
| CH-SIMS | 使用 MMSA 的音频、视频和文本预提取序列。对有效帧池化，按情感分数的负、零、正构造三类。对源视频做固定互斥划分，原始发布版的训练、验证、测试划分不适用于本表。 |
| MIntRec | 使用作者的 20 类 train/dev/test，且每个样本 ID 只出现一次。官方划分在三个集合间共享电视剧集，因而只支持同剧集条件下的结论。正文采用作者发布的音视频序列和训练集拟合的文本向量。 |
| UPMC Food-101 | 用 QMF 发布的图文 JSONL 划分。图像使用冻结的 ImageNet ResNet-18，文本使用训练集拟合的 TF-IDF + SVD。 |

- CH-SIMS 下载 [MMSA 入口](https://github.com/thuiar/MMSA#2-datasets) 的 `unaligned_39.pkl`。下载镜像与 SHA-256 记录见 `prepare.py`。
- MIntRec 从 [作者项目](https://github.com/thuiar/MIntRec#4-usage) 下载。将 `train.tsv`、`dev.tsv`、`test.tsv` 和 `audio_data/audio_feats.pkl`、`video_data/video_feats.pkl` 放在 MIntRec 数据根目录。支持从作者发布的原始 MP4 提取替代特征时，应用独立的缓存目录并单独标注。
- UPMC 图文归档用 [Kaggle 作者基线推荐的数据](https://www.kaggle.com/datasets/gianmarco96/upmcfood101)。解压后让媒体根目录含 `images`，评测清单取 [QMF 仓库](https://github.com/QingyangZhang/QMF/tree/main/text-image-classification/datasets/food101)。

```powershell
python prepare.py ch_sims --source data/raw/ch_sims/unaligned_39.pkl
python prepare.py mintrec --source data/raw/mintrec --metadata data/raw/mintrec
python prepare.py upmc_food101 --source data/raw/upmc_food101/extracted --metadata data/raw/upmc_food101/metadata
python main.py --suite validation --inspect
python main.py --suite validation --device auto --run-dir runs/recommended
```

PyCharm 可运行 **EC-RIB Three Dataset Validation**。每个实验单元保存参数、特征缓存 SHA-256、结果和风险覆盖曲线。同一命令可继续未完成的单元。可用 `--datasets ch_sims` 限定单个数据集。

## 固定设置

AdamW，学习率 0.001，权重衰减 0.0001，批大小 128，隐藏宽度 128，风险潜变量 16，β 为 0.001，γ 为 1。种子使用 11、22、33。分类预热 5 轮、风险预热 5 轮、联合阶段 30 轮。基线保持相同的分类更新轮数。采用最后一轮权重，验证集用于选择 80% 目标覆盖的甄别阈值。

UPMC 的 CUDA 运行使用 CUDA Graph 及 fused AdamW 执行相同的固定批量更新。并行运行时的耗时反映了当时的资源竞争，不应用于方法之间的效率排名。

QMF 采用现有的能量加权和历史排序损失。SeD-UD 保留按输入和排序结果确定压缩量、先去冗余后融合去噪的结构。CDPR 保留共享与独立表示、共识与推理两路、原型和 Jensen–Shannon 分歧门控。SeD-UD 和 CDPR 更换了原始预训练序列编码与部分辅助目标，增加了缺失掩码，并将融合推广到图文两模态。结果应标注 *feature-level adaptation*。 [SeD-UD 原版](https://github.com/9meiye/SeD-UD) · [CDPR 原版](https://github.com/Hebust-NLP/CDPR) · [QMF 原版](https://github.com/QingyangZhang/QMF)

消融为去掉 IB、固定 IB、仅证据调节 KL、仅证据调节风险监督、去掉 C、用均匀权重替代 CSF。测试覆盖原始输入、已见的高斯扰动、未见的特征坐标丢弃、单模态缺失以及模态错配。概率质量用 NLL、ECE，选择性分类用错误检测 AUROC 和 AURC。错配以同类换配作对照，原始标签作为受控压力测试的参考目标。实验的扰动施加在特征层。
