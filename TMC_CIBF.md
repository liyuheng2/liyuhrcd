# TMC-CIBF 固定参数验证

TMC-CIBF 将联盟贡献感知信息瓶颈（CA-IB）、有符号边际贡献估计（SMCE）和保守冲突证据融合（CCEF）用于同一证据分类模型。四数据集结果、基线与消融分析见[实验报告](results/tmc_cibf/REPORT.md)和[算法评审](results/tmc_cibf/REVIEW.md)。

## 数据与复现

使用已经提取的文本、音频、视频特征。CMU-MOSI 为排除零分后的二分类，CH-SIMS 为五分类，MELD 为七分类。IEMOCAP 六分类采用五折留一会话测试，每折三个训练会话、一个独立保留会话和一个测试会话。IEMOCAP 文本使用转录内容的 512 维哈希 n-gram，音频和视觉复用已有缓存。上游预训练语料未独立审计。

把 `train.npz`、`val.npz`、`test.npz` 放在 `data/extended/{cmu_mosi,ch_sims,meld}` 和 `data/crib_iemocap/loso{1…5}/iemocap`。标准化统计只用每折训练集拟合。加载时检查样本和分组跨划分重复。原始固定参数实验保留验证集，追加的温度校准仅在该验证集上拟合。[缓存准备协议](https://github.com/liyuheng2/liyuhrcd/blob/e66729d2f20af12c384e402d2e5b32d72a975e09/TMC_ACRIB.md)

在 PyCharm 打开项目，选择安装了 `requirements.txt` 的 Python 3.12 解释器，然后运行 **TMC CIBF Fixed**。训练结束后运行 **TMC CIBF Summary** 生成报告。如果 IDE 提示找不到 `liyuhrcd` 模块，在该配置中重新选择当前项目的解释器。命令行入口如下。

```powershell
python -m unittest test_tmc_cibf
python validate_tmc_cibf.py --workers 5
python summarize_tmc_cibf.py
python review_tmc_cibf.py
```

PyCharm 中的 **TMC CIBF Review** 使用正式实验的模型权重，重放干净与混合换配测试，输出校准、证据折扣和逐类诊断。

可以用 `--features` 和 `--iemocap-features` 指向已有缓存目录。`--smoke --datasets cmu_mosi --seeds 11` 输出到独立的 `smoke` 目录。正式运行用相同命令续跑，复用元数据与数据 SHA-256 一致且含检查点的单元。每个单元写入训练损失、测试指标与检查点。汇总要求全部 40 个正式单元。`tmc_cibf.predict` 接受标准化输入和可用性掩码，返回类别、概率、证据不确定性、冲突质量、当前贡献和证据保留系数。全缺失时类别为 `-1`，概率为均匀占位。

## 固定协议

- 种子 11、22、33、44、55。CPU 每进程一个 Torch 线程，启用确定性算子。AdamW 学习率 0.001，权重衰减 0.0001，批大小 128，统一特征宽度 128，潜变量 16。β 固定为 0.001，λ 固定为 0.5。融合与单模态的 EDL 退火为 `min(1, 10 × epoch / 20)`。
- 所有方法均训练 20 轮并使用末轮检查点。前 5 轮用干净数据预热，CIBF 的折扣与 SMCE 监督在预热后启用，预热期使用单位 IB 权重。后 15 轮，增强方法共用等概率原配、高斯噪声、同类换配、异类换配及 15% 模态缺失模拟，始终保留至少一个可用模态。换配模态的单模态损失使用供体标签，融合和 SMCE 使用原标签。
- CIBF 在训练时同时监督采样与后验均值的证据，推理使用后验均值。SMCE 用未折扣意见预测联盟边上的半 Brier 改善。Shapley 加权的平均贡献调节 IB，完整上下文的负贡献折扣证据。贡献权重、折扣和监督目标使用停止梯度。CCEF 保留剩余冲突并计入不确定性。
- 基线 `tmc` 使用干净数据。`tmc_aug` 使用与 CIBF 相同的增强、训练步数和标签规则。`vib_ds` 共用瓶颈和证据头的结构，使用固定 IB 与 DS 融合，代码保留未训练的 SMCE 参数以共享模型结构。消融 `fixed_ib` 将权重固定为 1，`no_ib` 去掉 KL 项，`full_context_ib` 用单一完整上下文贡献调节 IB，`no_discount` 移除证据折扣，`ds_fusion` 保留折扣并改用 DS 归一化。记录的 `mean_compression` 是贡献函数提出的系数，消融仍按各自的规则施加 KL。
- 共同评估干净输入、三级高斯噪声、未见坐标丢弃、混合换配、逐模态缺失及单模态输入。测试标签只用于评估和离线诊断。受控换配和噪声实验的结论适用于现有预提取特征及固定划分。配对区间按种子计算，IEMOCAP 先按种子等权汇总五个测试会话。
