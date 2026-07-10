# FedCLAM Backbone 改进实验 Plan

## 一、实验总目标

FedCLAM 原论文当前 backbone 使用的是 **U-Net + InstanceNorm**，主要创新集中在 **CLAM 客户端自适应动量聚合** 和 **FIM 前景强度匹配损失**。因此，本实验计划固定 FedCLAM 的聚合策略和 FIM loss，只围绕 backbone 进行短期可验证的结构改进。

本计划围绕三个方向展开：

1. **Residual U-Net 方向**：增强编码器特征提取能力，提高训练稳定性。
2. **nnU-Net 思路方向**：引入深监督等稳定训练机制，提高多尺度分割能力。
3. **注意力模块方向**：增强前景区域、边界区域和 skip connection 特征筛选能力。

最终实验不是单独测试某一个模块，而是将前三个方向中表现最优的部分组合起来，形成最终改进模型。

---

## 二、实验设置保持一致

为了保证结果具有可解释性，所有实验尽量保持以下设置不变：

```text
联邦聚合方法：FedCLAM
损失函数：Dice / CE + FIM loss
客户端划分：不变
训练轮数：不变
local epoch：不变
学习率：不变或只做极小范围调整
输入尺寸：优先保持与原论文一致
评价指标：Average Dice、Client Std、Worst-client Dice、参数量、通信量
```

实验重点不是一次性追求最高分，而是判断每个方向是否能够稳定带来提升，尤其要关注弱客户端性能和不同客户端之间的性能波动。

---

## 三、实验 Plan

### E0：原始基线实验

**方法：**

```text
FedCLAM + 原始 U-Net + InstanceNorm
```

**目的：**

作为所有 backbone 改进的对照组，复现 FedCLAM 原始 backbone 设置。

**记录指标：**

```text
Average Dice
每个 Client 的 Dice
Client Std
Worst-client Dice
训练收敛曲线
参数量
每轮通信量
```

**预期作用：**

确认代码、数据划分、训练流程和评价方式可靠。后续所有改进都与 E0 进行比较。

---

### E1：Residual U-Net 方向实验

**方法：**

```text
FedCLAM + Residual Encoder U-Net
```

**改法：**

只将 encoder 中的普通卷积块替换为 residual block，decoder 暂时保持原始结构。

```text
原结构：
Conv → IN → ReLU → Conv → IN → ReLU

替换为：
Conv → IN → ReLU → Conv → IN + shortcut → ReLU
```

**目的：**

验证残差编码器是否能够增强特征提取能力，并改善联邦训练中的收敛稳定性。

**为什么先只改 encoder：**

encoder 负责提取语义特征，是 backbone 表达能力的核心。只改 encoder 可以控制参数量和通信成本，也方便判断残差结构是否真正有效。

**预期结果：**

Average Dice 有小幅提升，训练曲线更平稳，部分弱客户端可能提升更明显。

---

### E2：nnU-Net 思路方向实验

**方法：**

```text
FedCLAM + Residual Encoder U-Net + Deep Supervision
```

**改法：**

在 E1 的基础上加入 nnU-Net 常用的 **deep supervision**，让 decoder 的多个尺度输出都参与训练。

```text
主输出 loss
+ 1/2 尺度辅助输出 loss
+ 1/4 尺度辅助输出 loss
```

短期内不完整复现 nnU-Net，而是优先借鉴其中最容易产生效果、实现成本较低的训练机制。

**目的：**

验证多尺度监督是否能够提升 decoder 的学习能力，改善小目标、边界区域和尺度变化明显区域的分割效果。

**预期结果：**

模型收敛可能更快，Average Dice 进一步提升，尤其是在前景较小或结构尺度变化明显的数据集上更容易看到效果。

---

### E3：注意力方向实验一：SE / scSE

**方法：**

```text
FedCLAM + Residual Encoder U-Net + scSE
```

**改法：**

在 encoder 的中深层加入 scSE 模块。

推荐插入位置：

```text
encoder stage 3
encoder stage 4
bottleneck
```

不建议一开始在所有层加入注意力模块，尤其不要优先加在浅层高分辨率特征上，以免增加计算量并放大低层纹理噪声。

**目的：**

验证空间注意力和通道注意力是否能够增强前景区域表达能力。

**为什么优先 scSE：**

SE 只做通道注意力，而 scSE 同时考虑通道和空间信息，更适合医学图像分割中对前景区域、边界区域的表达需求。

**预期结果：**

对边界、小目标和前景区域有一定提升，参数量和通信成本增加较小。

---

### E4：注意力方向实验二：Attention Gate

**方法：**

```text
FedCLAM + Residual Encoder U-Net + Attention Gate
```

**改法：**

在 U-Net 的 skip connection 上加入 Attention Gate。

```text
encoder skip feature → Attention Gate → decoder fusion
```

**目的：**

筛选 encoder 传递给 decoder 的浅层特征，减少中心特异性纹理、亮度差异、扫描噪声等无关信息对 decoder 的干扰。

**与 FedCLAM 的关系：**

FedCLAM 的 FIM loss 主要从前景强度分布层面缓解不同中心的图像强度差异，而 Attention Gate 可以从特征融合层面减少无关纹理进入 decoder，两者具有互补性。

**预期结果：**

弱边界、小目标以及跨中心差异较大的客户端可能提升更明显，Client Std 有机会下降。

---

## 四、最终组合实验：各方向最优模块融合

### E5：最终组合模型

E5 是最后一个实验，也是主结果模型。它不是单独测试某一个方向，而是组合前三个方向中表现最优的改进。

**基础组合方案：**

```text
FedCLAM
+ Residual Encoder U-Net
+ Deep Supervision
+ Best Attention Module
+ FIM loss
```

其中：

```text
Residual Encoder U-Net 来自 E1
Deep Supervision 来自 E2
Best Attention Module 从 E3 和 E4 中选择
```

### E5 的选择规则

如果 E3 的 scSE 提升更明显，则最终模型为：

```text
FedCLAM + ResEnc U-Net + Deep Supervision + scSE
```

如果 E4 的 Attention Gate 提升更明显，则最终模型为：

```text
FedCLAM + ResEnc U-Net + Deep Supervision + Attention Gate
```

如果 E3 和 E4 都有提升，并且参数量与通信成本增加可接受，则最终模型为：

```text
FedCLAM + ResEnc U-Net + Deep Supervision + scSE + Attention Gate
```

短期内更推荐默认最终模型为：

```text
FedCLAM + ResEnc U-Net + Deep Supervision + Attention Gate
```

原因是 Attention Gate 直接作用于 skip connection，更贴合联邦医学图像中浅层纹理存在中心差异的问题。

---

## 五、完整实验表

| 实验编号 | 方法 | 对应方向 | 目的 |
|---|---|---|---|
| E0 | FedCLAM + U-Net | 原始基线 | 复现原始 backbone |
| E1 | FedCLAM + ResEnc U-Net | Residual U-Net | 验证残差编码器 |
| E2 | FedCLAM + ResEnc U-Net + Deep Supervision | nnU-Net 思路 | 验证多尺度监督 |
| E3 | FedCLAM + ResEnc U-Net + scSE | 注意力模块 | 验证轻量空间-通道注意力 |
| E4 | FedCLAM + ResEnc U-Net + Attention Gate | 注意力模块 | 验证 skip 特征筛选 |
| E5 | FedCLAM + ResEnc U-Net + Deep Supervision + 最优注意力 | 最终组合 | 融合各方向最优改进 |

---

## 六、结果分析方式与最终表述

每个实验至少记录以下指标：

```text
Average Dice：整体分割性能
Client Std：不同客户端之间的公平性
Worst-client Dice：最差客户端性能
Params：模型参数量
Communication per round：每轮通信量
Convergence rounds：达到稳定性能所需轮数
```

分析时不能只看 Average Dice，还要重点关注：

```text
是否降低 Client Std
是否提升弱客户端 Dice
是否明显增加参数量
是否影响 FedCLAM 收敛稳定性
```

如果某个模块 Average Dice 提升不大，但能明显降低 Client Std 或提升 worst-client Dice，也值得保留，因为 FedCLAM 的核心应用场景是非 IID 联邦医学图像分割，稳定性和公平性同样重要。

最终方法可以命名为：

```text
FedCLAM-ResAttUNet
```

方法核心表述如下：

> 在 FedCLAM 原有客户端自适应动量聚合和 FIM loss 的基础上，引入 residual encoder 增强 backbone 的特征表达能力，借鉴 nnU-Net 的 deep supervision 提升多尺度分割稳定性，并通过注意力模块筛选 skip connection 或前景相关特征，从而在通信成本可控的条件下提升联邦医学图像分割性能。

最终实验 E5 应作为主结果模型，E1 至 E4 作为消融实验，用来说明每个方向分别贡献了什么。
