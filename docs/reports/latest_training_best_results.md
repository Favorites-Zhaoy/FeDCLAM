# 最新训练最好结果汇总

生成日期：2026-07-09  
口径：只使用当前可见的最新 `metrics.npy`。没有完整 `metrics.npy` 的实验不填数值，不用旧结果替代。  
best 口径：按 `server_val_acc_avg` 最高轮次选模，并报告该轮 `server_test_acc_avg`。Dice 单位为 `%`。

## 1. 主实验

客户端顺序：

- Fundus：Drishti-GS / RIM-ONE / REFUGE_t / REFUGE_v
- Prostate MRI：BIDMC / HK / I2CVB / BMC / RUNMC / UCL

说明：Fundus 当前最新主结果采用 `RIM-ONE crop diagnostic` 数据口径；原始 official 结果只作为 RIM-ONE 问题定位参考。Fundus official 日志中 validation/test 为同一集合，因此 val Dice 与 test Dice 相同。

| 数据集/实验 | 状态 | best 轮次 | Avg test Dice | Avg val Dice | Avg train loss | Avg train Dice | Avg test loss | 客户端 test Dice |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Fundus main: FIM+CLAM, RIM-ONE crop diagnostic | complete (200轮) | 199 | 89.06 | 89.06 | 0.0988 | 91.67 | 0.1368 | 88.52 / 83.82 / 91.56 / 92.33 |
| Fundus original official reference | complete (200轮) | 192 | 85.99 | 85.99 | 0.1129 | 90.51 | 0.1756 | 88.25 / 72.16 / 91.69 / 91.86 |
| Prostate MRI main: FIM+CLAM, paperparams clean2 | complete (500轮) | 361 | 93.04 | 92.64 | 0.0380 | 97.58 | 0.1065 | 91.60 / 94.10 / 95.48 / 93.58 / 93.93 / 89.54 |

主要变化：

- Fundus 从原始 official 到 RIM-ONE crop diagnostic，平均 test Dice 从 85.99 提高到 89.06，主要来自 RIM-ONE：72.16 -> 83.82。
- Prostate MRI 最新主实验 best-val test Dice 为 93.04，best-test oracle 为 93.16，最后一轮为 93.03。

## 2. 消融实验

| 数据集/组件 | 状态 | best 轮次 | Avg test Dice | Avg val Dice | Avg train loss | Avg train Dice | Avg test loss | 客户端 test Dice |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Fundus None | complete (200轮) | 190 | 87.47 | 87.47 | 0.0953 | 89.69 | 0.1203 | 86.46 / 79.78 / 91.88 / 91.78 |
| Fundus FIM | complete (200轮) | 194 | 87.88 | 87.88 | 0.1221 | 89.92 | 0.1495 | 87.00 / 80.88 / 91.87 / 91.75 |
| Fundus CLAM | complete (200轮) | 167 | 89.27 | 89.27 | 0.0782 | 91.45 | 0.1029 | 89.40 / 83.79 / 91.65 / 92.22 |
| Fundus FIM+CLAM | complete (200轮) | 199 | 89.06 | 89.06 | 0.0988 | 91.67 | 0.1368 | 88.52 / 83.82 / 91.56 / 92.33 |
| Prostate MRI None | complete (500轮) | 415 | 92.62 | 92.03 | 0.0178 | 98.19 | 0.0662 | 92.12 / 92.50 / 95.25 / 92.94 / 93.52 / 89.40 |
| Prostate MRI FIM | complete (500轮) | 302 | 93.09 | 92.47 | 0.0510 | 97.15 | 0.1174 | 91.75 / 93.09 / 95.68 / 93.52 / 93.62 / 90.89 |
| Prostate MRI CLAM | complete (500轮) | 241 | 92.70 | 92.31 | 0.0339 | 97.72 | 0.0714 | 91.22 / 93.01 / 94.92 / 93.33 / 93.76 / 89.96 |
| Prostate MRI FIM+CLAM | complete (500轮) | 361 | 93.04 | 92.64 | 0.0380 | 97.58 | 0.1065 | 91.60 / 94.10 / 95.48 / 93.58 / 93.93 / 89.54 |

消融结论（当前最新结果）：

- Fundus crop diagnostic 口径下，CLAM 最好：89.27；FIM+CLAM 为 89.06，几乎持平。
- Prostate MRI 口径下，FIM 单独最好：93.09；FIM+CLAM 为 93.04；None 为 92.62；CLAM 为 92.70。

## 3. 超参数实验

来源：`D:\MLPractice\remoteMLP\FedCLAM\outputs\hparam_sweeps`。这是当前可见的最新 hparam sweep 产物。  
限制：当前完整 `metrics.npy` 只覆盖 Fundus 的 `lr` 和 `batch_size`；Fundus 的 `alpha`、`local_epochs` 缺完整 metrics，console 显示 CUDA OOM；Prostate hparam sweep 没有完整结果目录。

| 实验 | 状态 | best 轮次 | Avg test Dice | Avg train loss | Avg test loss | 客户端 test Dice |
|---|---:|---:|---:|---:|---:|---|
| Fundus lr=1e-4 | complete (200轮) | 122 | 88.10 | 0.1096 | 0.2032 | 87.89 / 80.01 / 91.96 / 92.54 |
| Fundus lr=3e-4 | complete (200轮) | 120 | 86.81 | 0.0783 | 0.1686 | 88.28 / 74.88 / 91.78 / 92.29 |
| Fundus lr=1e-3 | complete (200轮) | 186 | 85.47 | 0.1135 | 0.1783 | 87.55 / 71.12 / 91.46 / 91.74 |
| Fundus lr=3e-3 | complete (200轮) | 173 | 82.10 | 0.1687 | 0.2161 | 87.32 / 59.78 / 90.73 / 90.58 |
| Fundus lr=1e-2 | complete (200轮) | 197 | 74.06 | 0.2787 | 0.3069 | 78.29 / 46.09 / 86.43 / 85.44 |
| Fundus batch_size=4 | complete (200轮) | 178 | 83.58 | 0.1416 | 0.2011 | 87.46 / 64.50 / 91.23 / 91.14 |
| Fundus batch_size=8 | complete (200轮) | 186 | 85.47 | 0.1135 | 0.1783 | 87.55 / 71.12 / 91.46 / 91.74 |
| Fundus batch_size=16 | complete (200轮) | 175 | 86.87 | 0.1023 | 0.1630 | 87.50 / 75.97 / 91.80 / 92.23 |
| Fundus batch_size=32 | complete (200轮) | 165 | 86.53 | 0.0980 | 0.1691 | 86.85 / 75.21 / 91.67 / 92.38 |
| Fundus alpha=0 | 未完成/缺 metrics |  |  |  |  |  |
| Fundus alpha=0.5 | 未完成/缺 metrics |  |  |  |  |  |
| Fundus alpha=1 | 未完成/缺 metrics |  |  |  |  |  |
| Fundus alpha=2 | 未完成/缺 metrics |  |  |  |  |  |
| Fundus alpha=4 | 未完成/未见结果目录 |  |  |  |  |  |
| Fundus local_epochs=1 | 未完成/缺 metrics |  |  |  |  |  |
| Fundus local_epochs=2 | 未完成/缺 metrics |  |  |  |  |  |
| Fundus local_epochs=3 | 未完成/缺 metrics |  |  |  |  |  |
| Fundus local_epochs=5 | 未完成/缺 metrics |  |  |  |  |  |
| Fundus weight_decay=0 | 未完成/未见结果目录 |  |  |  |  |  |
| Fundus weight_decay=1e-6 | 未完成/未见结果目录 |  |  |  |  |  |
| Fundus weight_decay=1e-5 | 未完成/未见结果目录 |  |  |  |  |  |
| Fundus weight_decay=1e-4 | 未完成/未见结果目录 |  |  |  |  |  |
| Fundus weight_decay=1e-3 | 未完成/未见结果目录 |  |  |  |  |  |
| Prostate MRI hparam sweep | 未完成/未见完整结果目录 |  |  |  |  |  |

超参结论（只基于完成项）：

- 完成的 Fundus lr sweep 中，`lr=1e-4` 最好，Avg test Dice 88.10。
- 完成的 Fundus batch size sweep 中，`batch_size=16` 最好，Avg test Dice 86.87。
- 这些 hparam sweep 使用的是远端原始 Fundus official 数据口径，不是本地最新的 RIM-ONE crop diagnostic 口径，因此不与第 1-2 节直接混排作最终主结论。

## 4. k 实验

| 实验 | 状态 | best 轮次 | Avg test Dice | Avg train loss | Avg test loss | 客户端 test Dice |
|---|---:|---:|---:|---:|---:|---|
| Fundus k=1 | complete (200轮) | 199 | 89.06 | 0.0988 | 0.1368 | 88.52 / 83.82 / 91.56 / 92.33 |
| Fundus k=2 | complete (200轮) | 199 | 89.25 | 0.1009 | 0.1331 | 89.76 / 83.57 / 91.51 / 92.15 |
| Fundus k=5 | complete (200轮) | 199 | 88.91 | 0.0979 | 0.1373 | 88.41 / 83.46 / 91.60 / 92.16 |
| Fundus k=10 | complete (200轮) | 199 | 89.29 | 0.0971 | 0.1313 | 88.99 / 84.22 / 91.86 / 92.08 |
| Fundus k=20 | complete (200轮) | 185 | 89.43 | 0.0923 | 0.1315 | 89.32 / 84.94 / 91.60 / 91.85 |
| Prostate MRI k=1 | complete (500轮) | 361 | 93.04 | 0.0380 | 0.1065 | 91.60 / 94.10 / 95.48 / 93.58 / 93.93 / 89.54 |
| Prostate MRI k=2 | complete (500轮) | 211 | 92.62 | 0.0615 | 0.1238 | 90.61 / 93.20 / 95.64 / 93.44 / 93.94 / 88.90 |
| Prostate MRI k=5 | complete (500轮) | 462 | 93.06 | 0.0372 | 0.0985 | 91.99 / 93.64 / 95.66 / 92.95 / 94.01 / 90.13 |
| Prostate MRI k=10 | complete (500轮) | 259 | 92.75 | 0.0418 | 0.1064 | 91.32 / 93.66 / 95.78 / 92.95 / 94.02 / 88.75 |
| Prostate MRI k=20 | complete (500轮) | 499 | 92.58 | 0.0293 | 0.1002 | 90.60 / 94.14 / 95.53 / 92.94 / 94.34 / 87.96 |

k 结论：

- Fundus 最新 crop diagnostic 口径下，`k=20` 最好，Avg test Dice 89.43。
- Prostate MRI 下，`k=5` 最好，Avg test Dice 93.06；`k=1` 的 93.04 基本持平。

## 5. lambda_FIM 实验

| 实验 | 状态 | best 轮次 | Avg test Dice | Avg train loss | Avg test loss | 客户端 test Dice |
|---|---:|---:|---:|---:|---:|---|
| Fundus lambda_FIM=1e-4 | complete (200轮) | 199 | 89.08 | 0.0758 | 0.1046 | 88.96 / 83.30 / 91.59 / 92.45 |
| Fundus lambda_FIM=1e-3 | complete (200轮) | 195 | 89.22 | 0.0781 | 0.1069 | 88.86 / 84.13 / 91.70 / 92.18 |
| Fundus lambda_FIM=1e-2 | complete (200轮) | 199 | 89.06 | 0.0988 | 0.1368 | 88.52 / 83.82 / 91.56 / 92.33 |
| Fundus lambda_FIM=1e-1 | complete (200轮) | 192 | 87.63 | 0.3299 | 0.4115 | 87.15 / 81.97 / 90.59 / 90.80 |
| Fundus lambda_FIM=1 | complete (200轮) | 90 | 84.42 | 2.8644 | 3.0701 | 83.64 / 76.00 / 88.79 / 89.25 |
| Prostate MRI lambda_FIM=1e-4 | complete (500轮) | 172 | 92.98 | 0.0488 | 0.0836 | 92.17 / 93.25 / 95.36 / 93.10 / 93.90 / 90.11 |
| Prostate MRI lambda_FIM=1e-3 | complete (500轮) | 347 | 92.92 | 0.0271 | 0.0682 | 91.86 / 93.32 / 95.30 / 93.39 / 93.57 / 90.05 |
| Prostate MRI lambda_FIM=1e-2 | complete (500轮) | 361 | 93.04 | 0.0380 | 0.1065 | 91.60 / 94.10 / 95.48 / 93.58 / 93.93 / 89.54 |
| Prostate MRI lambda_FIM=1e-1 | 未完成/缺 metrics |  |  |  |  |  |
| Prostate MRI lambda_FIM=1 | 未完成/缺 metrics |  |  |  |  |  |

lambda_FIM 结论：

- Fundus 最新 crop diagnostic 口径下，`lambda_FIM=1e-3` 最好，Avg test Dice 89.22；`1e-4` 与 `1e-2` 非常接近。
- Prostate MRI 已完成项中，`lambda_FIM=1e-2` 最好，Avg test Dice 93.04。
- Prostate MRI 的 `lambda_FIM=1e-1` 和 `lambda_FIM=1` 不填，因为没有完整 metrics。

## 6. 除上述实验和扩展数据集实验外，当前还没加上的实验

明确没有纳入当前汇总、或没有完整结果的实验：

1. 主实验 baseline 对比：FedAvg、FedAvgM、FedProx、FedSAM、FedDyn、FedFA、FedEVI、FedHarmo、FedBN/Central 等完整主表未整理进来。
2. 多随机种子统计：当前主结果/消融/敏感性主要是 seed=0，未加入 seed=1/2 的均值、标准差和显著性检验。
3. Prostate MRI hparam sweep：脚本支持，但当前未见完整 `outputs/hparam_sweeps/prostate/official/*/metrics.npy`。
4. Fundus hparam 的 `alpha`、`local_epochs`、`weight_decay`：当前未完成或未见完整结果；`alpha/local_epochs` console 显示 CUDA OOM。
5. Prostate MRI 的 `lambda_FIM=1e-1`、`lambda_FIM=1`：当前状态为缺完整 metrics。
6. Fundus 原始 official 口径下完整新版汇总：已有旧口径结果，但本汇总只把它作为参考，不作为最新主结论。
7. per-case / per-slice / per-channel 深度分析：已有部分审计文件，但没有作为正式实验表纳入。
8. Backbone improvement / FedClAM_pro 目录实验：当前未见完整 metrics，因此未加入。
9. 资源与效率实验：训练时长、显存占用、通信量、模型参数量/吞吐未形成完整对比表。
10. 鲁棒性实验：不同数据划分、不同客户端缺失、噪声/标签扰动、跨域留一等不属于扩展数据集主流程的鲁棒性测试未加入。

## 7. 证据路径

- Fundus 最新主实验：`D:\MLPractice\FedCLAM\outputs\fundus\official\fedclam_rimone_crop_diagnostic_seed0\log\metrics.npy`
- Prostate MRI 最新主实验：`D:\MLPractice\FedCLAM\outputs\prostate\official\fedclam_paperparams_clean2_seed0\log\metrics.npy`
- Fundus 最新消融与敏感性：`D:\MLPractice\FedCLAM\outputs\fundus\official\ablation\rimone_crop_diagnostic\`、`D:\MLPractice\FedCLAM\outputs\fundus\official\sensitivity\rimone_crop_diagnostic\`
- Prostate MRI 消融与敏感性：`D:\MLPractice\FedCLAM\outputs\prostate\official\ablation\`、`D:\MLPractice\FedCLAM\outputs\prostate\official\sensitivity\prostate_paper_base\`
- hparam sweep：`D:\MLPractice\remoteMLP\FedCLAM\outputs\hparam_sweeps\`
