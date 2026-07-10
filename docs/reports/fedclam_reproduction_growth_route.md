# FedCLAM 复现问题解决路线与最终结果总汇

生成日期：2026-07-09  
项目目录：`D:\MLPractice\FedCLAM`

本文档记录本轮 FedCLAM 复现过程中，从初始问题定位、代码与数据修正、实验队列设计，到最终主实验、消融实验、`k` 与 `lambda_FIM` 参数敏感性实验的完整脉络。核心目标不是单纯“追分”，而是把每一个分数差异拆解到可解释、可复查、可继续扩展的证据链上。

## 1. 最终结论概览

当前复现已经形成两个稳定结论：

1. **Fundus 主实验在修正 RIM-ONE 预处理后，复现结果略高于论文。**
   - 论文 Avg Dice：`88.82`
   - 当前 crop diagnostic 复现 Avg Dice：`89.06`
   - 差值：`+0.24 pp`

2. **Prostate 主实验使用 paper 参数 clean run 后，best-test/oracle 口径下距离论文约 `0.58 pp`。**
   - 论文 Avg Dice：`93.74`
   - 当前 best-test/oracle 复现 Avg Dice：`93.16`
   - 差值：`-0.58 pp`
   - best-val 选择口径下为 `93.04`，差值 `-0.70 pp`

Prostate 这 `0.58-0.70 pp` 的剩余差异，目前最合理解释不是模型实现大错，而是：

- 论文未公开 Prostate split manifest；
- test case 数很少，BIDMC、BMC、UCL 对病例组成敏感；
- 论文未说明 Dice 聚合口径；
- client 名称与论文 Client 1-6 映射未完全公开；
- 当前训练数据、切片过滤、文件顺序可能与论文作者环境存在细节差异。

## 2. 关键成长路线

### 2.1 从“全量复现”收敛到“FedCLAM 主线”

最开始计划覆盖所有 baseline 与全部数据集，但很快发现这样会让问题空间过大：baseline 数量多、数据预处理不统一、官方代码也存在多处隐含假设。随后把目标收敛为：

- 先复现 FedCLAM 主实验；
- 再复现 Component Ablation；
- 最后复现 `k` 与 `lambda_FIM` 参数变化实验；
- Fundus 已完成后，重点补 Prostate。

这个收敛非常关键。它避免了在多个 baseline 和多个数据问题之间来回跳转，把主线压缩成“论文核心方法是否能在当前代码和数据上稳定复现”。

### 2.2 确定 paper 参数作为主线

论文实现细节给出的关键参数为：

| 项目 | Fundus | Prostate |
|---|---:|---:|
| Backbone | U-Net + InstanceNorm | U-Net + InstanceNorm |
| Optimizer | Adam | Adam |
| Learning rate | `1e-3` | `1e-4` |
| Weight decay | `1e-4` | `1e-4` |
| Rounds | `200` | `500` |
| Local epoch | `1` | `1` |
| Loss | Dice + FIM | Dice + CE + FIM |
| `lambda_FIM` | `1e-2` | `1e-2` |
| `k` | `1.0` | `1.0` |
| `alpha` | `1.0` | `1.0` |
| Augmentation | 原仓库流程 | random flip |

过程中发现 GitHub released prostate 脚本与论文文本不完全一致，例如 released script 中出现过 `lambda_FIM=0.1`、`fim_warmup=100` 的组合。最终处理方式是：

- **主结果以 paper 参数为准**；
- GitHub released 参数只作为补充对照，不作为论文复现主线；
- Prostate 主线固定为 `lambda_FIM=1e-2`、`k=1`、`alpha=1`、`lr=1e-4`、`500 rounds`。

### 2.3 Fundus 问题：RIM-ONE 预处理拖低原始 official 结果

Fundus 初始 official 结果明显低于论文，主要问题集中在 RIM-ONE：

| 口径 | Avg Dice | Client Dice |
|---|---:|---|
| 论文 | `88.82` | 未逐项完全展开 |
| 原始 official 复现 | `85.99` | `88.25 / 72.16 / 91.69 / 91.86` |
| crop diagnostic 修正后 | `89.06` | `88.52 / 83.82 / 91.56 / 92.33` |

可以看到，原始 official 中第二个 client 只有 `72.16`，导致平均值被显著拉低。crop diagnostic 后该 client 提升到 `83.82`，总平均达到 `89.06`，已经略高于论文 `88.82`。

解决方案：

- 把 Fundus original official 与 crop diagnostic 结果分开记录；
- 不把 original official 的低分直接解释为方法失败；
- 最终 Fundus 主报告采用 crop diagnostic 作为有效复现结果；
- 保留 original official 作为数据预处理问题的证据。

### 2.4 Prostate 问题：49% 错误方向被排除

早期 Prostate 曾出现约 `49%` 的异常结果。经过分析，这不是正常实验波动，而是标签、dtype、metric 或数据读取方向上的无效结果。

处理方式：

- 从最终分析主线中删除 `49%` 方向；
- 不再用该结果解释论文差异；
- 重新以 label fix + paper 参数 clean run 作为有效 baseline；
- 后续所有 Prostate 主实验、消融实验和参数实验都以这个修复版为基座。

这一步很重要，因为如果把 49% 当成真实复现结果，会错误地把问题归因到模型方法本身。

### 2.5 Prostate 剩余缺口从 `1 pp` 缩小到 `0.58 pp`

修复后，Prostate 主实验逐步从低约 `0.95-1.08 pp` 收敛到：

| 选择口径 | Repro Avg Dice | Paper Avg Dice | 差值 |
|---|---:|---:|---:|
| best-val selected | `93.04` | `93.74` | `-0.70 pp` |
| best-test/oracle | `93.16` | `93.74` | `-0.58 pp` |

最终主报告中按用户要求使用 best-test/oracle 口径记录最高复现值：

```text
Paper: 93.74
Repro highest: 93.16
Delta: -0.58 pp
Best round: 375
Client Dice: 91.75 / 94.24 / 95.75 / 93.58 / 93.88 / 89.77
```

这个结果已经非常接近论文，并且低分主要不是所有 client 系统性低，而是少数 client 仍有差距。

## 3. 过程中遇到的问题与解决方案

### 3.1 代码启动与参数传递问题

遇到的问题：

- 官方代码默认路径与本地目录不一致；
- Prostate 与 Fundus 的数据读取逻辑不同；
- 一些参数需要从 CLI 显式传入，例如 `alpha`、`beta`、`l_fim`、`fim_warmup`；
- 输出目录需要区分 main、ablation、sensitivity，避免结果覆盖或混淆。

解决方案：

- 固定本地项目目录 `D:\MLPractice\FedCLAM`；
- 通过脚本统一队列执行；
- 输出按任务拆分到 `outputs/` 下不同子目录；
- 生成独立 summary/report 文件，而不是只依赖训练日志。

### 3.2 Prostate split manifest 未公开

这是 Prostate 复现最大的不可控因素。

论文只说明使用 6 个中心的 Prostate MRI，但没有公开：

- 哪些病例进入 train；
- 哪些病例进入 valid；
- 哪些病例进入 test；
- 每个 case 的哪些 slice 参与训练或评估；
- 文件排序与过滤规则；
- Client 1-6 与具体中心的映射。

当前仓库 official 逻辑更像是按切片顺序切分：

```text
前 60% train
中间 20% valid
最后 20% test
```

这会使 test set 对病例顺序、文件排序、过滤规则非常敏感。尤其是当前 Prostate test case 数较少，例如：

| Site | 当前 test case 数 |
|---|---:|
| BIDMC | `3` |
| BMC | `6` |
| UCL | `3` |

少数困难病例就可能造成 `0.5-1 pp` 的平均 Dice 波动。

解决方案：

- 生成 `prostate_official_split_manifest.csv`；
- 记录 site、case、slice、split、前景面积、文件名、hash 等信息；
- 补 per-case / per-site 评估；
- 不再只看全局平均分，而是追踪到底是哪些中心和病例拉低。

### 3.3 Dice 聚合口径未公开

论文没有明确说明 Prostate Dice 到底按哪种方式聚合：

- 每 slice Dice 后平均；
- 每 case Dice 后平均；
- 每 site 先平均再 macro；
- 所有像素 pooled 后计算 Dice；
- foreground-only Dice；
- background + foreground mean Dice。

当前代码主口径更接近：

```text
每个 site 内按 slice 统计 background+foreground mean Dice，
最后对 site/client 取平均。
```

额外评估脚本显示，同一个 checkpoint 在不同聚合口径下可能有明显差异。例如当前 Prostate checkpoint 可以从约 `93.04` 到约 `94.57` 不等。这说明 `93.74` 是否可完全对齐，不只取决于模型本身，也取决于论文未公开的 metric aggregation。

解决方案：

- 补统一评估口径脚本；
- 同时输出 foreground-only、background+foreground、site macro、case macro、pooled 等口径；
- 最终主表明确写明采用何种口径；
- 对论文未公开口径的部分，只给出合理解释，不强行声称完全复现。

### 3.4 Fundus valid/test overlap 的来源

Fundus official 模式中 valid 和 test 存在重合。分析后把它归为：

- 更接近论文作者仓库原码/官方 released code 的行为；
- 不是本轮修复后来新增的改动；
- 因此最终报告中保留该口径，但在说明中标记为 official behavior。

解决方案：

- 不擅自修改 Fundus official valid/test 行为；
- 把 Fundus crop diagnostic 与 original official 分开；
- 用最终可解释分数与论文对比。

### 3.5 随机增强差异

论文明确写 Prostate 使用 random flip。检查后确认当前 Prostate 训练包含随机翻转。

另外发现代码或后续实验中有额外随机 90 度旋转的可能性。最终处理原则：

- 主复现实验以论文描述为准；
- random flip 是必须保留的；
- 额外 90 度旋转不作为 paper main 的必要条件；
- 如果出现含旋转的实验，需单独标注，不能混入 paper clean run。

## 4. 最终主实验对比

### 4.1 Main Experiment

| Dataset | 口径 | Paper Avg Dice | Repro Avg Dice | 差值 | Repro Std | Best Round | Client Dice |
|---|---|---:|---:|---:|---:|---:|---|
| Fundus | crop diagnostic | `88.82` | `89.06` | `+0.24` | `3.34` | `199` | `88.52 / 83.82 / 91.56 / 92.33` |
| Fundus | original official reference | `88.82` | `85.99` | `-2.83` | `8.11` | `192` | `88.25 / 72.16 / 91.69 / 91.86` |
| Prostate | paperparams clean2, best-test/oracle | `93.74` | `93.16` | `-0.58` | `1.92` | `375` | `91.75 / 94.24 / 95.75 / 93.58 / 93.88 / 89.77` |
| Prostate | paperparams clean2, best-val selected | `93.74` | `93.04` | `-0.70` | - | `361` | `91.60 / 94.10 / 95.48 / 93.58 / 93.93 / 89.54` |

## 5. Component Ablation 最终结果

### 5.1 Fundus Component Ablation

| Component | Paper Avg Dice | Repro Avg Dice | 差值 | Best Round | Train Loss | Train Dice | Test Loss | Client Dice |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| None | `84.72` | `87.47` | `+2.75` | `190` | `0.0953` | `89.69` | `0.1203` | `86.46 / 79.78 / 91.88 / 91.78` |
| FIM | `86.93` | `87.88` | `+0.95` | `194` | `0.1221` | `89.92` | `0.1495` | `87.00 / 80.88 / 91.87 / 91.75` |
| CLAM | `88.27` | `89.27` | `+1.00` | `167` | `0.0782` | `91.45` | `0.1029` | `89.40 / 83.79 / 91.65 / 92.22` |
| FIM+CLAM | `88.82` | `89.06` | `+0.24` | `199` | `0.0988` | `91.67` | `0.1368` | `88.52 / 83.82 / 91.56 / 92.33` |

### 5.2 Prostate Component Ablation

| Component | Paper Avg Dice | Repro Avg Dice | 差值 | Best Round | Train Loss | Train Dice | Test Loss | Client Dice |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| None | `92.77` | `92.62` | `-0.15` | `415` | `0.0178` | `98.19` | `0.0662` | `92.12 / 92.50 / 95.25 / 92.94 / 93.52 / 89.40` |
| FIM | `93.07` | `93.09` | `+0.02` | `302` | `0.0510` | `97.15` | `0.1174` | `91.75 / 93.09 / 95.68 / 93.52 / 93.62 / 90.89` |
| CLAM | `93.50` | `92.70` | `-0.80` | `241` | `0.0339` | `97.72` | `0.0714` | `91.22 / 93.01 / 94.92 / 93.33 / 93.76 / 89.96` |
| FIM+CLAM | `93.74` | `93.16` | `-0.58` | `375` | `0.0348` | `97.78` | `0.1059` | `91.75 / 94.24 / 95.75 / 93.58 / 93.88 / 89.77` |

说明：

- Prostate 表中 FIM+CLAM 使用 best-test/oracle 最高值 `93.16` 与论文主表对比。
- 若严格使用 best-val selected，则 FIM+CLAM 为 `93.04`，差值 `-0.70 pp`。

## 6. `k` 参数敏感性实验

### 6.1 Fundus `k` Sensitivity

| k | Best Round | Avg Dice | Train Loss | Test Loss | Client Dice |
|---:|---:|---:|---:|---:|---|
| `1` | `199` | `89.06` | `0.0988` | `0.1368` | `88.52 / 83.82 / 91.56 / 92.33` |
| `2` | `199` | `89.25` | `0.1009` | `0.1331` | `89.76 / 83.57 / 91.51 / 92.15` |
| `5` | `199` | `88.91` | `0.0979` | `0.1373` | `88.41 / 83.46 / 91.60 / 92.16` |
| `10` | `199` | `89.29` | `0.0971` | `0.1313` | `88.99 / 84.22 / 91.86 / 92.08` |
| `20` | `185` | `89.43` | `0.0923` | `0.1315` | `89.32 / 84.94 / 91.60 / 91.85` |

观察：

- Fundus 对 `k` 不极端敏感；
- `k=20` 在当前数据上最高，为 `89.43`；
- 但论文默认 `k=1` 的结果已经达到 `89.06`，高于论文主表。

### 6.2 Prostate `k` Sensitivity

| k | Best Round | Avg Dice | Train Loss | Test Loss | Client Dice |
|---:|---:|---:|---:|---:|---|
| `1` | `361` | `93.04` | `0.0380` | `0.1065` | `91.60 / 94.10 / 95.48 / 93.58 / 93.93 / 89.54` |
| `2` | `211` | `92.62` | `0.0615` | `0.1238` | `90.61 / 93.20 / 95.64 / 93.44 / 93.94 / 88.90` |
| `5` | `462` | `93.06` | `0.0372` | `0.0985` | `91.99 / 93.64 / 95.66 / 92.95 / 94.01 / 90.13` |
| `10` | `259` | `92.75` | `0.0418` | `0.1064` | `91.32 / 93.66 / 95.78 / 92.95 / 94.02 / 88.75` |
| `20` | `499` | `92.58` | `0.0293` | `0.1002` | `90.60 / 94.14 / 95.53 / 92.94 / 94.34 / 87.96` |

观察：

- Prostate 上 `k=1` 与 `k=5` 非常接近；
- `k=5` best-val 口径略高于 `k=1`，但提升只有 `+0.02 pp`；
- `k=10/20` 没有带来稳定提升；
- 因此不建议为了追分偏离论文默认 `k=1`。

## 7. `lambda_FIM` 参数敏感性实验

### 7.1 Fundus `lambda_FIM` Sensitivity

| lambda_FIM | Best Round | Avg Dice | Train Loss | Test Loss | Client Dice |
|---:|---:|---:|---:|---:|---|
| `1e-4` | `199` | `89.08` | `0.0758` | `0.1046` | `88.96 / 83.30 / 91.59 / 92.45` |
| `1e-3` | `195` | `89.22` | `0.0781` | `0.1069` | `88.86 / 84.13 / 91.70 / 92.18` |
| `1e-2` | `199` | `89.06` | `0.0988` | `0.1368` | `88.52 / 83.82 / 91.56 / 92.33` |
| `1e-1` | `192` | `87.63` | `0.3299` | `0.4115` | `87.15 / 81.97 / 90.59 / 90.80` |
| `1` | `90` | `84.42` | `2.8644` | `3.0701` | `83.64 / 76.00 / 88.79 / 89.25` |

观察：

- `1e-3` 在当前 Fundus 上略高；
- `1e-2` 是论文默认值，表现稳定；
- `1e-1` 和 `1` 明显过强，损失升高且 Dice 下降。

### 7.2 Prostate `lambda_FIM` Sensitivity

| lambda_FIM | Best Round | Avg Dice | Train Loss | Test Loss | Client Dice |
|---:|---:|---:|---:|---:|---|
| `1e-4` | `172` | `92.98` | `0.0488` | `0.0836` | `92.17 / 93.25 / 95.36 / 93.10 / 93.90 / 90.11` |
| `1e-3` | `347` | `92.92` | `0.0271` | `0.0682` | `91.86 / 93.32 / 95.30 / 93.39 / 93.57 / 90.05` |
| `1e-2` | `361` | `93.04` | `0.0380` | `0.1065` | `91.60 / 94.10 / 95.48 / 93.58 / 93.93 / 89.54` |
| `1e-1` | `468` | `92.20` | `0.1489` | `0.5164` | `89.98 / 93.69 / 95.06 / 92.33 / 93.48 / 88.65` |
| `1` | `495` | `90.55` | `1.3086` | `5.0376` | `87.54 / 90.72 / 93.66 / 90.76 / 92.99 / 87.64` |

观察：

- Prostate 上论文默认 `1e-2` 是当前 best-val 口径最高；
- `1e-4`、`1e-3` 与 `1e-2` 相差很小，说明轻量 FIM 正则有稳定区间；
- `1e-1` 和 `1` 明显过强，测试损失大幅上升，Dice 下降；
- 这也解释了为什么 GitHub released 的 `0.1` 不适合作为论文主线参数。

## 8. 对论文差异的最终解释

### 8.1 Fundus

Fundus 的论文差异已经基本解决。当前有效结果：

```text
Paper Avg Dice: 88.82
Repro Avg Dice: 89.06
Delta: +0.24 pp
```

主要经验：

- 原始 official 低分不是 FedCLAM 方法问题；
- 低分主要来自 RIM-ONE 预处理/crop 差异；
- 修正后结果与论文高度一致，甚至略高。

### 8.2 Prostate

Prostate 的最终有效结果：

```text
Paper Avg Dice: 93.74
Best-val selected: 93.04, delta -0.70 pp
Best-test/oracle: 93.16, delta -0.58 pp
```

当前差距的最合理解释：

1. **split manifest 未公开**  
   当前无法证明 train/valid/test 病例和论文完全一致。

2. **test case 数少，病例组成敏感**  
   BIDMC、BMC、UCL 等中心 test case 数较少，少数困难病例足以造成 `0.5-1 pp` 波动。

3. **Dice 聚合口径未公开**  
   同一 checkpoint 在不同 Dice 聚合方式下会得到不同平均值。论文没有说明是否按 case、slice、site macro 或 pooled 计算。

4. **client 映射未完全公开**  
   论文表中写 Client 1-6，但没有完全公开与 BIDMC、HK、I2CVB、BMC、RUNMC、UCL 的对应关系。

5. **训练数据细节可能不同**  
   即使使用同一公开数据源，切片过滤、排序、mask 后处理、空切片处理等细节都可能与论文作者环境不同。

因此，当前 Prostate `-0.58 pp` 更像是复现边界中的数据/口径差异，而不是方法实现失败。

## 9. 最终产物与可复查文件

主要报告文件：

- `FedCLAM/outputs/analysis/main_ablation_k_lfim_reproduction_report.md`
- `FedCLAM/outputs/analysis/latest_training_best_results.md`
- `FedCLAM/outputs/prostate/analysis/prostate_ablation_sensitivity_summary.md`
- `FedCLAM/outputs/analysis/fedclam_reproduction_growth_route.md`

关键脚本和工具方向：

- Prostate split manifest 审计；
- checkpoint per-case / per-site 评估；
- metric aggregation 对比；
- Prostate paper 参数 clean run；
- Prostate Component Ablation 队列；
- Prostate `k` sensitivity 队列；
- Prostate `lambda_FIM` sensitivity 队列；
- 汇总报告生成脚本。

## 10. 当前可以写入论文/报告的结论表述

推荐表述如下：

> We reproduced FedCLAM using the paper-reported hyperparameters. On Fundus, after correcting the RIM-ONE preprocessing/cropping inconsistency, our reproduction achieved 89.06 Dice, slightly higher than the reported 88.82. On Prostate MRI, the best-test/oracle reproduction reached 93.16 Dice compared with the reported 93.74, leaving a 0.58 percentage-point gap. Further analysis suggests that the remaining discrepancy is mainly attributable to the unpublished Prostate split manifest, small test-case counts in several clients, unspecified Dice aggregation protocol, and potentially different case/slice filtering rules.

中文报告中可以写为：

> 当前 FedCLAM 主实验已经基本复现。Fundus 在修正 RIM-ONE 预处理差异后达到 89.06 Dice，高于论文 88.82；Prostate 在 paper 参数 clean run 下 best-test/oracle 达到 93.16 Dice，较论文 93.74 低 0.58 个百分点。结合 split manifest 未公开、test 病例数较少、Dice 聚合口径未说明以及 client 映射不完整等证据，剩余差异更可能来自数据划分和评估口径差异，而不是核心方法实现错误。

## 11. 后续若继续推进

若还希望进一步压缩 Prostate 与论文的差距，建议优先级如下：

1. 获取或重建更接近论文的 Prostate split manifest；
2. 对多个 volume-level split seed 做 sensitivity，量化自然波动区间；
3. 在报告中同时列 best-val、best-test/oracle、case macro、site macro、pooled Dice；
4. 不建议通过偏离论文默认参数进行追分，因为 `k` 与 `lambda_FIM` 实验已经表明大幅调参收益有限，且会削弱复现可信度。

