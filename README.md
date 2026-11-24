# HRD-CF-M-HCRD-RIB v2 固定参数实验

以 CR-E、共享风险瓶颈 HCRD-RIB、冲突责任融合 CSF 实现 v2。风险任务使用 O（观测污染强度）、A（污染引起的预测熵变化）、C（模态对正常冲突基线）。分类损失叠加融合与单模态分类监督。EC-RIB 对同一份配对证据使用逆向权重，KL 乘以 `1/(1+δ)`，风险误差乘以 `1+δ`，然后对有效风险任务做 LogMeanExp。评估时只用待分类样本的模态，配对干净输入仅用于离线构造诊断参考目标。

## 在 Windows PyCharm 运行

在 PyCharm 的 **Get from VCS** 中克隆 `https://github.com/liyuheng2/liyuhrcd.git`，项目目录使用 `liyuhrcd`。选择 Python 3.12 的虚拟环境作为项目解释器。以 PyCharm Terminal 中已激活的项目解释器运行：

```powershell
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

CPU 环境可以直接安装 `requirements.txt` 并传入 `--device cpu`。在有可用 CUDA 的机器上传入 `--device cuda` 会在 CUDA 不可用时直接报错。PyCharm 项目名改过时，在 **Edit Configurations** 中选择实际项目的解释器。项目根目录为工作目录。

## 四个数据集

| CLI 名称 | 输入与分类任务 | 划分和来源 |
|---|---|---|
| `upmc_food101` | 食物图像、配对网页文本，101 类 | 采用 QMF 附带的 MMBT train/dev/test；[UPMC 论文](https://cord.isir.upmc.fr/pdfs/publis/CordCooking2015icme.pdf)、[QMF 数据说明](https://github.com/QingyangZhang/QMF#dataset-preparation) |
| `n24news` | 新闻图像、标题和摘要，24 类 | 从公开 JSON 固定构造约 80/10/10 分组划分，按文章 URL 或图像 URL 重合关系分组；[N24News 原项目](https://github.com/billywzh717/N24News) |
| `meld` | 台词、音频、视频，7 类情绪 | 官方 train/dev/test，按片段做分类；[MELD 原项目](https://github.com/declare-lab/MELD)、[作者发布的原始视频归档](https://huggingface.co/datasets/declare-lab/MELD) |
| `cmu_mosei` | 文本、音频、视频，3 类情感 | MMSA 提供的 train/valid/test，按情感分数的负、零、正划分类别；[MMSA 下载与校验](https://github.com/thuiar/MMSA#2-datasets) |

先下载各数据集，解压到自选目录。也可使用 [UPMC 图文归档镜像](https://huggingface.co/datasets/kkim0451/UPMC-Food101) 和 [N24News 图像归档镜像](https://huggingface.co/datasets/kkim0451/N24News)。[MOSEI 特征镜像](https://huggingface.co/datasets/AZYoung/MOSEI_processed) 的 SHA-256 与 MMSA 原项目一致。未对齐的 MOSEI 文件约 13.7 GB，读取原始 pickle 是内存峰值所在，建议在具备至少 24 GB 可用内存的环境中先生成特征缓存，再用该缓存训练。下面以 `data/raw/` 为例。UPMC 图片应能匹配清单中的 `images/train/类别/文件名.jpg` 和 `images/test/类别/文件名.jpg`。N24News 镜像解包后的图像目录叫 `imgs`。MELD 需解包嵌套的视频归档，并保留各 split 的文件夹，因为不同 split 可出现同名影片。MELD 需要 FFmpeg 在 PATH 上，第一次提取图像特征需要下载 torchvision 的 ResNet-18 权重。

```powershell
git clone https://github.com/QingyangZhang/QMF.git data/raw/QMF
git clone https://github.com/billywzh717/N24News.git data/raw/N24News_metadata
git clone https://github.com/declare-lab/MELD.git data/raw/MELD_metadata

python prepare.py upmc_food101 --source data/raw/UPMC-Food-101 --metadata data/raw/QMF/text-image-classification/datasets/food101
python prepare.py n24news --source data/raw/N24News/imgs --metadata data/raw/N24News_metadata/nytimes_dataset.json
python prepare.py meld --source data/raw/MELD --metadata data/raw/MELD_metadata/data/MELD
python prepare.py cmu_mosei --source data/raw/MOSEI/Processed/unaligned_50.pkl
```

UPMC 使用真正配对的图文数据。MELD、UPMC 和 N24News 使用冻结的 ImageNet ResNet-18，MELD 对采样帧取平均，音频使用固定的对数频带倒谱统计。三个媒体数据集的文本使用仅在训练集拟合的 TF-IDF 和 SVD。MOSEI 使用官方 MMSA 预提取特征，按有效时步池化，读取前校验与 MMSA 公布的 SHA-256 一致。各方法共用同一批特征和训练集标准化统计。解码缺失的媒体会进入模态掩码，预处理会打印各 split 的可用行数。训练入口检查分组泄漏与特征形状。

## 首轮训练协议

- 固定 `β=0.001`，`γ=1.0`，AdamW 学习率 `0.001`，权重衰减 `0.0001`，批大小 128，随机种子 11、22、33。
- 分类预热 5 轮，风险预热 5 轮，联合训练 30 轮。对比方法参加 5 轮分类预热和 30 轮含干预的训练，获得相同的分类更新轮数。使用每次训练的最后一轮，验证集用于确定可靠性筛选阈值。
- 在标准化特征层施加干预。训练污染强度为 0、0.2、0.5、0.9，随机模态缺失率 15%，每个样本至少保留一个原本可用模态。测试包括原始输入、固定高斯强度、未见类型的坐标丢弃、指定模态污染、指定模态缺失、仅单模态输入，以及同类对照和异类换配。若去掉某模态后样本完全不可用，缺失实验保留该行原本唯一的可用模态。仅单模态输入实验会报告可评样本比例。同类和异类换配用于受控冲突分析。
- 默认矩阵包含 Late Fusion、TMC、ETMC、QMF、MMDynamics、无 IB、固定 IB、仅动态 KL、仅动态风险监督和完整 EC-RIB。TMC、ETMC、QMF、MMDynamics 是统一输入特征和掩码条件下的算法改写，采用各自的证据融合、能量排序或特征与模态加权机制。相关作者实现见 [TMC / ETMC](https://github.com/Han-Zongbo/TMC)、[QMF](https://github.com/QingyangZhang/QMF)、[MMDynamics](https://github.com/TencentAILabHealthcare/mmdynamics)。
- 扩展矩阵增加拼接、学习门控，以及独立 IB、证据打乱、去掉 O/A/C、直接 JSD、普通 JSD 监督目标、均匀融合、关闭梯度隔离以及去掉单模态监督。两套矩阵可使用同一输出目录，完成的单元会复用。

```powershell
python main.py --inspect
python main.py --datasets upmc_food101 --methods late fixed_ib full --smoke --device auto
python main.py --suite core --device cuda
python main.py --suite full --device cuda
```

PyCharm 已提供相应 Run Configurations。完整数据预处理一次即可复用。终止后用同样的命令重跑会跳过已生成完整结果的单元。需要先处理少量媒体时，按类别抽样用 `prepare.py ... --limit-per-class 12 --output data/pilot`，训练时带 `--feature-dir data/pilot --run-dir runs/pilot --smoke`。

## 结果文件

`runs/fixed_v2/protocol/summary.csv` 是按数据集、方法、测试条件和指标汇总的均值、样本标准差与可用种子数。每个单元另存 `result.json`、模型权重及风险—覆盖率曲线。烟雾运行输出在 `smoke/`。复用已有结果前会检查协议、配置、设备和数据缓存 SHA-256。

指标包括 Accuracy、Macro/Weighted-F1、NLL、Brier、ECE、分类别 ECE、错误检测 AUROC/AUPRC、AURC、选择性覆盖率、O/A/C 参考目标 MAE 和 Spearman、换配检测 AUROC/AUPRC、污染前后的目标模态权重、参数量、线性层 MACs、训练和推理耗时以及 CUDA 峰值显存。错误检测同时记录方法可靠性与最大分类概率。单类或常数输入下无定义的指标留空。耗时与 MACs 的范围是已提取特征后的模型。

## TMC-HCRD-RIB 验证

五种子消融、MELD 三模态特征准备及 PyCharm 运行配置见 [TMC-HCRD-RIB 验证协议](TMC_VALIDATION.md)。

## TMC-HCRD-CRIB v2 验证

固定参数、数据准备、PyCharm 复现与消融定义见 [实验协议](TMC_CRIB.md)。四数据集主实验和 IEMOCAP 跨会话结果见 [验证报告](results/tmc_crib/REPORT.md)。

## TMC-HCRD-DCIB v3 验证

双目标信息瓶颈、解析证据折扣和独立校准见 [实验协议](TMC_DCIB.md)。PyCharm 运行配置为 **TMC DCIB v3 Full**，汇总报告生成在 `results/tmc_dcib/REPORT.md`。
