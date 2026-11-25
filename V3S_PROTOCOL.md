# HRD-CF-M-HCRD-TD-RIB v3-S 固定参数验证

## 运行

在 PyCharm 中打开项目根目录，选用可访问 CUDA 的 Python 环境，运行 **V3-S validation (CUDA)**。本机验证使用 Python 3.11、PyTorch 2.5.0、CUDA 12.1 和 RTX 4060 Laptop GPU。已有 `data/features` 缓存可直接使用。重建缓存所需的公开来源和划分见 [小数据集协议](SMALL_EXPERIMENT_PROTOCOL.txt)。

```powershell
python validate_v3s.py --smoke --device cuda
python validate_v3s.py --device cuda
```

也可直接在 PyCharm 的 Python 控制台导入验证入口。

```python
from validate_v3s import main

main(["--device", "cuda"])
```

CUDA Graph 不可用时可传入 `--eager`，无 CUDA 时可传入 `--device cpu`。换执行模式时使用另一个 `--out`，各模式内相同配置可续跑。若缺少已有 v2 基线，先运行以下命令。前一版任务损害基线支持集中目录和按数据集运行的工作目录。

```powershell
python main.py --datasets cmu_mosi ch_sims meld_small iemocap --methods late concat gate tmc etmc qmf mmdynamics full fixed_ib no_ib no_csf --run-dir runs/small_fixed --device cuda
python validate_improved.py --device cuda
```

## 数据和训练协议

- CMU-MOSI、CH-SIMS、MELD-small、IEMOCAP 共用既有特征与划分。CH-SIMS 按源视频分组，MELD-small 使用文本和音频。历史结果载入前逐个校验数据文件 SHA-256、配置、方法和种子。
- 种子固定为 11、22、33、44、55。AdamW 学习率 0.001、权重衰减 0.0001、批大小 128、共享特征维数 128、瓶颈维数 16，β=0.001、γ=1。先预热分类 5 轮，再冻结分类器训练风险 5 轮，最后联合训练 30 轮，取最终轮。融合比例 0.25，效用 softmax 温度 1。
- 每个种子用训练组交叉拟合 3 个教师并为各模态拟合折内训练标准化。每个目标样本与换配供体位于同一留出折。每种子预生成四份独立扰动视图，按轮次循环使用。主批采用历史高斯强度及 15% 缺失训练分布，再附加候选率为 25% 的同类或异类换配样本。换配行进入风险损失，分类更新使用原配对行。C 比较相同噪声下的当前和恢复配对，U 使用逐模态留一效用。
- 风险目标使用等任务权重的软标签 BCE。固定 IB 的 KL 系数恒为 1。动态、任务内打乱和反向的系数均用训练扰动池中的有效任务实例归一化。U 的损害强度为效用绝对值。单模态样本跳过 C/U 监督，分类权重为 1。
- 测试条件沿用历史固定种子的噪声、缺失和换配。测试阶段风险回归使用由完整训练集预热后冻结的共同参考教师。原评估接口未返回供体标签，O/A 回归在换配条件下只评估原来源未替换的模态，C 回归限于发生换配的有效模态对。每一方法的 MCP 阈值为干净验证集的第 20 百分位数。
- 特征扰动和教师诊断按种子缓存。续跑版本使用 1024 行评估批，并单独测量以 128 行为批的干净集推理时间。早期已完成结果的推理耗时为各测试条件中位数，可通过单元结果中的 `inference_timing` 字段识别。

## 对照与解读

- 分类基线包括 Late Fusion、Concat、Gate、TMC、ETMC、QMF、MMDynamics 和既有 EC-RIB v2 消融。方法名 `full` 对应历史 EC-RIB v2，`v3s_full` 对应本轮动态 IB。历史方法是共用特征和掩码的实现。
- IB 对照包括无随机无 KL、仅随机、固定、动态、打乱和反向。风险和层次消融包含旧风险代理、去除 O/A/C/U、Utility-only、去掉层次条件，及在同一框架内增加配对异常任务的 `with_swap_task`。后者是受控重建。
- 同一单模态预测分别用 Uniform、风险乘积、完全效用、Conservative CSF 和 Oracle Utility 融合，在冻结分类器阶段和联合训练后各比较一次。Oracle 使用评估标签和训练侧教师，只用作诊断。
- 输出包括 `runs/v3s/protocol/summary.csv`、`paired.csv`、`COMPLETE.json`，以及单元结果、权重、风险—覆盖率曲线、训练 KL 和各任务归一权重均值。配对差为当前方法减去基线，五种子 t 区间和双侧配对符号翻转 p 值用于探索性比较。指标按原比例保存。
- 该协议复用曾评估的测试划分和预提取特征，结论限定在这四个数据集和特征级扰动。历史墙钟耗时采自其他执行环境或并行任务，参数量与线性层 MACs更适合跨版本比较。交叉拟合、分类预热和目标构造耗时在结果中单列。
