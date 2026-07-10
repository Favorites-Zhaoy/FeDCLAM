# Fundus RIM-ONE Crop 处理说明

生成日期：2026-07-09  
项目目录：`D:\MLPractice\FedCLAM`

本文档说明本轮 FedCLAM 复现中，为什么对 Fundus 数据集里的 RIM-ONE 客户端做 crop diagnostic 处理，以及具体是如何 crop 的。这里的 crop 不是训练时的数据增强，而是针对 RIM-ONE 图像与 mask 空间口径不一致问题的预处理修正。

## 1. 背景

Fundus 眼底分割任务的输入和标签分别是：

| 名称 | 含义 |
|---|---|
| image | 原始眼底图像，模型输入 |
| mask | 分割标注图，也就是每个像素的类别答案 |

在当前 Fundus 任务中，mask 通常表示：

| mask 像素值 | 含义 |
|---:|---|
| `0` | 背景 |
| `1` | optic disc，视盘 |
| `2` | optic cup，视杯 |

模型训练时需要 image 与 mask 在空间上严格对应：图像中某个位置的视盘/视杯，必须和 mask 中同一位置的标注对齐。

## 2. 原始 official 处理方式

原始预处理脚本为：

```text
tools/prepare_datasets.py
```

原始 official 逻辑对所有 Fundus 客户端统一处理：

```text
读取整张 image
读取整张 mask
image resize 到 384 x 384
mask resize 到 384 x 384
保存为 npy
```

也就是说，RIM-ONE 原图会被直接处理为：

```text
1424 x 2144 x 3 宽图 -> 384 x 384 x 3
```

对应 mask 也会被直接处理为：

```text
1424 x 2144 -> 384 x 384
```

这个流程对普通单眼视野图像通常没问题，但对当前 RIM-ONE 源图像不合适。

## 3. RIM-ONE 的问题

审计时发现，当前 RIM-ONE 源图像具有明显的宽幅/左右并排特征：

```text
source_shape = 1424 x 2144 x 3
```

而对应 mask 的前景标注只位于其中一侧。当前 crop manifest 统计显示：

```text
RIM-ONE 159 张图像全部标注在 left 半幅
```

更直观地说，可以把它理解为：

```text
原图：左侧眼底区域 + 右侧额外区域
mask：只标注左侧目标区域
```

这里不绝对断言右侧一定是“右眼”，因为从代码和 overlay 只能确定它是宽幅/左右并排式图像；但可以确定的是，当前监督 mask 只对应左半幅。

如果直接把整张宽图 resize 到 `384 x 384`，会造成两个问题：

1. 图像中保留了未被 mask 标注的另一半区域；
2. 原本有标注的一侧被横向压缩，目标比例和空间结构发生变化。

这会让模型看到额外的疑似眼底结构，但监督信号只要求预测左侧目标区域。结果就是 image 与 mask 的任务口径不一致，RIM-ONE 的 Dice 被明显拉低。

## 4. Crop diagnostic 的处理方式

修正脚本为：

```text
tools/make_fundus_rimone_crop_diagnostic.py
```

生成的数据集路径为：

```text
D:\MLPractice\datasets\processed\fundus_rimone_crop_diagnostic\official
```

处理原则：

1. 只重新处理 RIM-ONE；
2. Drishti-GS、REFUGE_t、REFUGE_v 直接复制原始 official processed 数据；
3. 不覆盖原始 official 数据；
4. 生成独立的 crop diagnostic 数据集，方便对比。

具体 crop 逻辑如下：

```python
def crop_to_mask_half(image, mask):
    h, w = mask.shape
    xs = np.where(mask > 0)[1]
    center_x = float(xs.mean())
    half = w // 2
    if center_x < half:
        crop = (0, half)
        side = "left"
    else:
        crop = (half, w)
        side = "right"
    x0, x1 = crop
    return image[:, x0:x1], mask[:, x0:x1], side
```

也就是：

```text
1. 找到 mask 前景区域的所有 x 坐标
2. 计算前景中心 center_x
3. 如果 center_x 在左半边，则保留左半幅
4. 如果 center_x 在右半边，则保留右半幅
5. image 和 mask 使用完全相同的裁剪区域
6. 裁剪后再 resize 到 384 x 384
```

当前 RIM-ONE 的实际结果是：

```text
全部 159 张 crop_side = left
```

所以实际处理等价于：

```text
原图 image: 1424 x 2144 x 3
保留左半幅: image[:, 0:1072]
保留左半幅 mask: mask[:, 0:1072]
然后 resize 到 384 x 384
```

## 5. 与原始 official 的区别

| 项目 | 原始 official | crop diagnostic |
|---|---|---|
| 处理对象 | 所有 Fundus 客户端统一整图 resize | 只对 RIM-ONE 重新处理 |
| RIM-ONE 输入 | 整张 `1424 x 2144` 宽图 | 只保留 mask 所在半幅 |
| resize 顺序 | 先整图 resize | 先 crop，再 resize |
| 其他客户端 | 正常 resize | 直接复制原始 official |
| 是否覆盖原数据 | 是原始 processed 数据 | 不覆盖，另建数据集 |

关键区别是：

```text
原始 official:
1424 x 2144 整图 -> 384 x 384

crop diagnostic:
1424 x 2144 整图 -> 1424 x 1072 左半幅 -> 384 x 384
```

## 6. 为什么必须先 crop 再 resize

如果先 resize 整张宽图，会把无标注的另一半也压进模型输入，同时把真正有标注的目标区域压窄。这相当于让模型学习一个不干净的监督关系：

```text
输入里有两侧区域
标签只解释其中一侧
```

先 crop 再 resize 则让输入和标签重新对齐：

```text
输入只包含 mask 标注对应的一侧
标签也只对应这一侧
```

这样模型学习到的是明确的一对一分割任务。

## 7. 对结果的影响

crop 前后 FIM+CLAM 的结果如下：

| 口径 | Drishti-GS | RIM-ONE | REFUGE_t | REFUGE_v | Avg |
|---|---:|---:|---:|---:|---:|
| 原始 official | `88.25` | `72.16` | `91.69` | `91.86` | `85.99` |
| RIM-ONE crop diagnostic | `88.52` | `83.82` | `91.56` | `92.33` | `89.06` |

提升主要来自 RIM-ONE：

```text
RIM-ONE: 72.16 -> 83.82, +11.66 pp
Avg:     85.99 -> 89.06, +3.07 pp
```

这说明原始 Fundus 低分不是 FedCLAM 算法本身的问题，也不是普通超参数问题，而主要是 RIM-ONE 的图像-mask 空间口径问题。

## 8. 和论文结果的关系

论文 Fundus FIM+CLAM 主结果为：

```text
Paper Avg Dice = 88.82
```

当前 crop diagnostic 复现结果为：

```text
Repro Avg Dice = 89.06
Delta = +0.24 pp
```

因此，修正 RIM-ONE 的 crop 口径后，Fundus 主实验已经与论文高度一致，并略高于论文结果。

## 9. 最终解释

可以在报告中这样表述：

> 当前 RIM-ONE 源图像为宽幅/左右并排式图像，而 mask 前景只对应其中一侧。原始 preprocessing 将整张宽图直接 resize 到 `384 x 384`，导致模型输入包含未标注区域，并使标注侧目标被横向压缩。我们因此构建了 RIM-ONE crop diagnostic 数据集：根据 mask 前景中心自动选择对应半幅，先裁剪 image 与 mask，再 resize 到 `384 x 384`。该处理使 RIM-ONE Dice 从 `72.16` 提升到 `83.82`，Fundus 平均 Dice 从 `85.99` 提升到 `89.06`，与论文 `88.82` 高度一致。

