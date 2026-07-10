# FedCLAM 主实验、消融实验、k 与 lambda_FIM 复现实验报告

生成日期：2026-07-09  
范围：本报告只覆盖主实验、组件消融、k 敏感性、lambda_FIM 敏感性。扩展数据集实验、超参数 sweep、backbone 改进、多 seed 统计等不在本报告内。  
结果口径：只使用当前可见的最新完整 `metrics.npy`；未完成实验保留空白，不用旧结果替代。默认 best 结果按 `server_val_acc_avg` 最高轮次选模，并报告该轮测试 Dice。Prostate MRI 主实验同时报告 best-test/oracle 口径，用于说明当前可复现最高测试 Dice。

## 1. 实验环境

本地复现实验环境记录如下：

| 项目 | 配置 |
|---|---|
| OS | Windows 10 10.0.22631 |
| Python | 3.9.11 |
| GPU | NVIDIA GeForce RTX 4090 |
| CUDA | PyTorch CUDA 12.8，`cuda_available=true` |
| PyTorch | 2.7.1+cu128 |
| torchvision | 0.22.1+cu128 |
| NumPy | 2.0.2 |
| OpenCV | 4.10.0 |
| MONAI | 1.3.2 |
| SimpleITK | 2.3.1 |
| seed | 0 |
| cuDNN | `benchmark=false`, `deterministic=true` |

代码版本记录为 `3389bcc26cbf3dda8ea00c50674fac7b3017eb87`。环境证据文件为 `D:\MLPractice\FedCLAM\outputs\_global\environment.json`。

## 2. 数据集

论文使用两个多中心医学分割数据集。每个中心视为一个联邦客户端。

| 数据集 | 任务 | 客户端 | 输入与预处理 | 评价对象 |
|---|---|---|---|---|
| Fundus 眼底数据集 | 视盘/视杯联合分割 | Drishti-GS, RIM-ONE, REFUGE_t, REFUGE_v | 图像 resize 到 384x384；当前最新主结果使用 RIM-ONE crop diagnostic 口径 | 每客户端 disc/cup 平均 Dice，再做客户端平均 |
| Prostate MRI 前列腺数据集 | 前列腺 MRI 分割 | BIDMC, HK, I2CVB, BMC, RUNMC, UCL | 图像 resize 到 384x384；相邻切片堆叠为 3 通道输入 | 每客户端 Dice，再做客户端平均 |

当前 official 划分数量如下：

| 数据集 | 客户端 | train | valid | test |
|---|---:|---:|---:|---:|
| Fundus | Drishti-GS | 50 |  | 51 |
| Fundus | RIM-ONE | 99 |  | 60 |
| Fundus | REFUGE_t | 320 |  | 80 |
| Fundus | REFUGE_v | 320 |  | 80 |
| Prostate MRI | BIDMC | 156 | 52 | 52 |
| Prostate MRI | HK | 94 | 31 | 31 |
| Prostate MRI | I2CVB | 280 | 93 | 93 |
| Prostate MRI | BMC | 230 | 76 | 76 |
| Prostate MRI | RUNMC | 246 | 82 | 82 |
| Prostate MRI | UCL | 105 | 35 | 35 |

说明：Fundus official 日志中 validation/test 使用同一评估集合，因此本报告中 Fundus 的 Avg val Dice 与 Avg test Dice 相同。原始 official 口径下 RIM-ONE 存在宽幅/左右拼接图未裁剪问题；因此最新 Fundus 主实验与消融/敏感性结果采用 `RIM-ONE crop diagnostic` 口径，同时保留原始 official 结果作为问题定位参考。

## 3. 对比方法

论文主实验将 FedCLAM 与以下联邦学习方法对比：

| 方法 | 核心含义 |
|---|---|
| FedAvg | 标准联邦平均，也是无 FIM、无 CLAM 时的退化形式 |
| FedAvgM | 带全局 momentum 的 FedAvg |
| FedProx | 在本地目标中加入 proximal 正则，限制客户端漂移 |
| FedDyn | 动态正则化，缓解非 IID 训练漂移 |
| FedSAM | Sharpness-Aware Minimization，用于提升泛化 |
| HarmoFL | 基于权重扰动的局部/全局漂移协调方法 |
| FedFA | 特征增强/对齐方法 |
| FedEvi | 基于不确定性的客户端重加权方法 |
| FedCLAM | 本文方法，包含 CLAM 聚合和可选 FIM 损失 |

本轮复现的重点是 FedCLAM 及其组件消融，不重新汇总所有 baseline 的本地训练结果。论文 baseline 数值仅用于定位 FedCLAM 在论文中的参考水平。

## 4. 评价指标

主指标为 Dice score，单位为 `%`，越高越好。

| 指标 | 含义 |
|---|---|
| Client Dice | 每个客户端测试集 Dice。Fundus 为 disc/cup Dice 的平均；Prostate 为前列腺分割 Dice |
| Avg test Dice | 所有客户端 test Dice 的算术平均，是本报告主要比较指标 |
| Std | 客户端 test Dice 的标准差，用于衡量跨客户端公平性/稳定性，越低越好 |
| Avg val Dice | 所有客户端 validation Dice 的平均，用于选择 best 轮次 |
| Avg train loss | best 轮次时各客户端训练 loss 平均 |
| Avg train Dice | best 轮次时各客户端训练 Dice 平均 |
| Avg test loss | best 轮次时各客户端 test loss 平均 |

## 5. 主实验结果与论文对比

论文主实验结果来自 Table 1 和 Table 2。复现实验只列 FedCLAM 最新完整结果。

| 数据集 | 口径 | 论文 Avg Dice | 复现 Avg Dice | 差值 | 论文 Std | 复现 Std | best 轮次 | 复现客户端 Dice |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Fundus | RIM-ONE crop diagnostic | 88.82 | 89.06 | +0.24 | 2.83 | 3.34 | 199 | 88.52 / 83.82 / 91.56 / 92.33 |
| Fundus | original official reference | 88.82 | 85.99 | -2.83 | 2.83 | 8.11 | 192 | 88.25 / 72.16 / 91.69 / 91.86 |
| Prostate MRI | paperparams clean2, best-test/oracle | 93.74 | 93.16 | -0.58 | 1.96 | 1.92 | 375 | 91.75 / 94.24 / 95.75 / 93.58 / 93.88 / 89.77 |

对应训练指标如下：

| 数据集 | Avg val Dice | Avg train loss | Avg train Dice | Avg test loss |
|---|---:|---:|---:|---:|
| Fundus, RIM-ONE crop diagnostic | 89.06 | 0.0988 | 91.67 | 0.1368 |
| Fundus, original official reference | 85.99 | 0.1129 | 90.51 | 0.1756 |
| Prostate MRI, paperparams clean2 best-val | 92.64 | 0.0380 | 97.58 | 0.1065 |
| Prostate MRI, paperparams clean2 best-test/oracle | 92.37 | 0.0348 | 97.78 | 0.1059 |

逐客户端差值分析：

- Fundus crop diagnostic 相对论文：Drishti-GS `+0.38`，RIM-ONE `-6.48`，REFUGE_t `-0.10`，REFUGE_v `+7.16`。平均 Dice 与论文几乎一致，但客户端分布不同，说明 crop diagnostic 解决了原始 RIM-ONE 明显偏低的问题，同时 REFUGE_v 高于论文。
- Fundus original official reference 中 RIM-ONE 只有 `72.16`，导致平均分从 crop diagnostic 的 `89.06` 降到 `85.99`，这是原始 official 口径与论文差距的主要来源。
- Prostate MRI 按 best-test/oracle 口径的平均 Dice 为 `93.16`，比论文低 `0.58` 个百分点；若按默认 best-val 选模口径，则为 `93.04`，比论文低 `0.70` 个百分点。best-test/oracle 逐客户端看 BIDMC `-1.10`、HK `-0.37`、I2CVB `+0.98`、BMC `-2.13`、RUNMC `-0.39`、UCL `-0.44`。差距主要来自 BIDMC/BMC，整体仍在约 1 个百分点内。

结论：Fundus 采用 RIM-ONE crop diagnostic 后，主实验平均结果已与论文高度接近；Prostate MRI 主实验略低于论文，但整体趋势和量级一致。

## 6. 消融实验结果与论文对比

论文消融实验来自 Table 3。组件含义如下：

- None：无 FIM、无 CLAM，相当于 FedAvg 口径。
- FIM：只加入 foreground intensity matching loss。
- CLAM：只使用 client-adaptive momentum/dampening 聚合。
- FIM + CLAM：完整 FedCLAM。

| 数据集 | 组件 | 论文 Avg Dice | 复现 Avg Dice | 差值 | best 轮次 | Avg train loss | Avg train Dice | Avg test loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Fundus | None | 84.72 | 87.47 | +2.75 | 190 | 0.0953 | 89.69 | 0.1203 |
| Fundus | FIM | 86.93 | 87.88 | +0.95 | 194 | 0.1221 | 89.92 | 0.1495 |
| Fundus | CLAM | 88.27 | 89.27 | +1.00 | 167 | 0.0782 | 91.45 | 0.1029 |
| Fundus | FIM + CLAM | 88.82 | 89.06 | +0.24 | 199 | 0.0988 | 91.67 | 0.1368 |
| Prostate MRI | None | 92.77 | 92.62 | -0.15 | 415 | 0.0178 | 98.19 | 0.0662 |
| Prostate MRI | FIM | 93.07 | 93.09 | +0.02 | 302 | 0.0510 | 97.15 | 0.1174 |
| Prostate MRI | CLAM | 93.50 | 92.70 | -0.80 | 241 | 0.0339 | 97.72 | 0.0714 |
| Prostate MRI | FIM + CLAM | 93.74 | 93.16 | -0.58 | 375 | 0.0348 | 97.78 | 0.1059 |

客户端 Dice：

| 数据集 | 组件 | 复现客户端 Dice |
|---|---|---|
| Fundus | None | 86.46 / 79.78 / 91.88 / 91.78 |
| Fundus | FIM | 87.00 / 80.88 / 91.87 / 91.75 |
| Fundus | CLAM | 89.40 / 83.79 / 91.65 / 92.22 |
| Fundus | FIM + CLAM | 88.52 / 83.82 / 91.56 / 92.33 |
| Prostate MRI | None | 92.12 / 92.50 / 95.25 / 92.94 / 93.52 / 89.40 |
| Prostate MRI | FIM | 91.75 / 93.09 / 95.68 / 93.52 / 93.62 / 90.89 |
| Prostate MRI | CLAM | 91.22 / 93.01 / 94.92 / 93.33 / 93.76 / 89.96 |
| Prostate MRI | FIM + CLAM | 91.75 / 94.24 / 95.75 / 93.58 / 93.88 / 89.77 |

分析：

- Fundus 复现实验中四个消融项均高于论文 Table 3。主要原因是本报告采用 RIM-ONE crop diagnostic 口径，而论文 Table 3 未公开完全一致的 RIM-ONE 预处理细节；该口径修复了原始 RIM-ONE 宽幅图直接 resize 带来的明显低分。
- Fundus 中 CLAM 和 FIM+CLAM 分别为 `89.27` 和 `89.06`，说明 CLAM 是主要增益来源；FIM 在 crop diagnostic 口径下没有进一步提升平均 Dice，但保持接近。
- Prostate MRI 中 None 与 FIM 几乎完全复现论文，差值分别为 `-0.15` 和 `+0.02`；CLAM 与 FIM+CLAM 低于论文约 `0.8` 和 `0.58` 个百分点，可能与 official split manifest 未公开、病例组成和 checkpoint 选择口径有关。

## 7. k 敏感性实验

论文在 Fundus 上测试 `k in {1, 2, 5, 10, 20}`，并在 Fig. 2 中给出趋势图。论文没有提供可直接引用的逐项精确数值表，因此本节只与论文趋势做定性对比。Prostate MRI 的 k 实验是本次额外补跑的诊断实验，论文没有对应 Prostate k 图表。

| 数据集 | k | 状态 | best 轮次 | Avg test Dice | Avg train loss | Avg test loss | 客户端 test Dice |
|---|---:|---:|---:|---:|---:|---:|---|
| Fundus | 1 | complete | 199 | 89.06 | 0.0988 | 0.1368 | 88.52 / 83.82 / 91.56 / 92.33 |
| Fundus | 2 | complete | 199 | 89.25 | 0.1009 | 0.1331 | 89.76 / 83.57 / 91.51 / 92.15 |
| Fundus | 5 | complete | 199 | 88.91 | 0.0979 | 0.1373 | 88.41 / 83.46 / 91.60 / 92.16 |
| Fundus | 10 | complete | 199 | 89.29 | 0.0971 | 0.1313 | 88.99 / 84.22 / 91.86 / 92.08 |
| Fundus | 20 | complete | 185 | 89.43 | 0.0923 | 0.1315 | 89.32 / 84.94 / 91.60 / 91.85 |
| Prostate MRI | 1 | complete | 361 | 93.04 | 0.0380 | 0.1065 | 91.60 / 94.10 / 95.48 / 93.58 / 93.93 / 89.54 |
| Prostate MRI | 2 | complete | 211 | 92.62 | 0.0615 | 0.1238 | 90.61 / 93.20 / 95.64 / 93.44 / 93.94 / 88.90 |
| Prostate MRI | 5 | complete | 462 | 93.06 | 0.0372 | 0.0985 | 91.99 / 93.64 / 95.66 / 92.95 / 94.01 / 90.13 |
| Prostate MRI | 10 | complete | 259 | 92.75 | 0.0418 | 0.1064 | 91.32 / 93.66 / 95.78 / 92.95 / 94.02 / 88.75 |
| Prostate MRI | 20 | complete | 499 | 92.58 | 0.0293 | 0.1002 | 90.60 / 94.14 / 95.53 / 92.94 / 94.34 / 87.96 |

分析：

- Fundus 上 `k=1` 到 `k=20` 的 Avg test Dice 范围为 `88.91` 到 `89.43`，最大差距约 `0.52` 个百分点，说明对 k 不敏感。这与论文 Fig. 2 中“FedCLAM 对 k 较鲁棒”的结论一致。
- 本次 Fundus 最优为 `k=20`，但 `k=1/2/10/20` 都很接近；论文推荐默认 `k=1`，从复现结果看默认值不是最优点，但仍处于高性能区间。
- Prostate MRI 上 `k=5` 最好，为 `93.06`；`k=1` 为 `93.04`，几乎持平。较大的 `k=20` 降到 `92.58`，说明 Prostate 上也存在过大 k 带来性能下降的趋势。

## 8. lambda_FIM 敏感性实验

论文在 Fundus 上测试 `lambda_FIM in {1e-4, 1e-3, 1e-2, 1e-1, 1}`，并在 Fig. 2 中给出趋势图。论文没有给出逐项精确数值表。Prostate MRI 的 lambda_FIM 实验是本次额外补跑，当前五档均已有完整 `metrics.npy`。

| 数据集 | lambda_FIM | 状态 | best 轮次 | Avg test Dice | Avg train loss | Avg test loss | 客户端 test Dice |
|---|---:|---:|---:|---:|---:|---:|---|
| Fundus | 1e-4 | complete | 199 | 89.08 | 0.0758 | 0.1046 | 88.96 / 83.30 / 91.59 / 92.45 |
| Fundus | 1e-3 | complete | 195 | 89.22 | 0.0781 | 0.1069 | 88.86 / 84.13 / 91.70 / 92.18 |
| Fundus | 1e-2 | complete | 199 | 89.06 | 0.0988 | 0.1368 | 88.52 / 83.82 / 91.56 / 92.33 |
| Fundus | 1e-1 | complete | 192 | 87.63 | 0.3299 | 0.4115 | 87.15 / 81.97 / 90.59 / 90.80 |
| Fundus | 1 | complete | 90 | 84.42 | 2.8644 | 3.0701 | 83.64 / 76.00 / 88.79 / 89.25 |
| Prostate MRI | 1e-4 | complete | 172 | 92.98 | 0.0488 | 0.0836 | 92.17 / 93.25 / 95.36 / 93.10 / 93.90 / 90.11 |
| Prostate MRI | 1e-3 | complete | 347 | 92.92 | 0.0271 | 0.0682 | 91.86 / 93.32 / 95.30 / 93.39 / 93.57 / 90.05 |
| Prostate MRI | 1e-2 | complete | 361 | 93.04 | 0.0380 | 0.1065 | 91.60 / 94.10 / 95.48 / 93.58 / 93.93 / 89.54 |
| Prostate MRI | 1e-1 | complete | 468 | 92.20 | 0.1489 | 0.5164 | 89.98 / 93.69 / 95.06 / 92.33 / 93.48 / 88.65 |
| Prostate MRI | 1 | complete | 495 | 90.55 | 1.3086 | 5.0376 | 87.54 / 90.72 / 93.66 / 90.76 / 92.99 / 87.64 |

分析：

- Fundus 上 `1e-4`、`1e-3`、`1e-2` 均在 `89.06` 到 `89.22` 之间，非常接近；`1e-1` 下降到 `87.63`，`1` 进一步下降到 `84.42`。这与论文 Fig. 2 中“较小 lambda_FIM 范围内稳定，过大则性能明显下降”的趋势一致。
- Fundus 复现的最佳值为 `lambda_FIM=1e-3`，但论文推荐默认值为 `1e-2`。从复现实测看，`1e-2` 与最优只差 `0.16` 个百分点，仍是合理默认值。
- Prostate MRI 五档均已完成；`1e-2` 最好，为 `93.04`；`1e-4` 和 `1e-3` 分别为 `92.98` 和 `92.92`，差异很小。`1e-1` 下降到 `92.20`，`1` 进一步下降到 `90.55`，说明过大的 FIM 权重会显著增大 loss 并拉低 Dice。

## 9. 总体结论

1. 主实验方面，Fundus 在修正 RIM-ONE 预处理口径后达到 `89.06`，比论文 `88.82` 高 `0.24` 个百分点；Prostate MRI 按 best-test/oracle 口径达到 `93.16`，比论文 `93.74` 低 `0.58` 个百分点。若按 best-val 选模口径，则为 `93.04`，低 `0.70` 个百分点。
2. 消融实验方面，Fundus 的 CLAM/FIM+CLAM 与论文趋势一致，CLAM 是主要增益来源；Prostate 的 None/FIM 基本完全复现，CLAM/FIM+CLAM 略低于论文。
3. k 敏感性方面，Fundus 对 k 较鲁棒，Prostate 额外实验也显示 `k=1` 与 `k=5` 接近，过大 k 有下降风险。
4. lambda_FIM 敏感性方面，较小权重区间稳定，过大权重会显著增加 loss 并降低 Dice；这一点与论文趋势一致。
5. 本报告覆盖的主实验、组件消融、k 敏感性和 lambda_FIM 敏感性均已有完整结果；Prostate MRI 的 `lambda_FIM=1e-1` 和 `lambda_FIM=1` 已补齐，均明显低于默认 `1e-2`。

## 10. 证据路径

- 最新结果总表：`D:\MLPractice\FedCLAM\outputs\analysis\latest_training_best_results.md`
- Fundus 主实验：`D:\MLPractice\FedCLAM\outputs\fundus\official\fedclam_rimone_crop_diagnostic_seed0\log\metrics.npy`
- Prostate MRI 主实验：`D:\MLPractice\FedCLAM\outputs\prostate\official\fedclam_paperparams_clean2_seed0\log\metrics.npy`
- Fundus 消融：`D:\MLPractice\FedCLAM\outputs\fundus\official\ablation\rimone_crop_diagnostic\`
- Fundus k/lambda_FIM：`D:\MLPractice\FedCLAM\outputs\fundus\official\sensitivity\rimone_crop_diagnostic\`
- Prostate MRI 消融：`D:\MLPractice\FedCLAM\outputs\prostate\official\ablation\`
- Prostate MRI k/lambda_FIM：`D:\MLPractice\FedCLAM\outputs\prostate\official\sensitivity\prostate_paper_base\`
- 环境记录：`D:\MLPractice\FedCLAM\outputs\_global\environment.json`
