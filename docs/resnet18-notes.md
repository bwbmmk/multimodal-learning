# ResNet18 结构笔记

> 来源：`projects/resnet18/model_resnet18.py` 里的代码注释，按概念重新整理。代码本身保持原位，这里只归纳知识点。

## 1. 为什么 CIFAR 版 ResNet 没有 7×7 入口卷积和 MaxPool

ImageNet 版 ResNet 的入口是 `conv7×7 stride2 + maxpool`，把 224×224 一路降到 56×56。CIFAR-10 的图只有 32×32，如果照搬，两次降采样后只剩 8×8，四组残差块就没空间可分了。

所以 CIFAR 版改成：`conv3×3, stride=1, padding=1`，分辨率保持 32×32，直接进 layer1。代价是入口感受野小、第一层计算量偏大，但 32×32 的图小，完全可以接受。

## 2. BasicBlock 的部件与顺序

```text
输入 x
 ├─ 主分支: conv1(3×3, stride) → bn1 → relu → conv2(3×3, stride=1) → bn2
 └─ 捷径:   shortcut(x)
输出: relu(bn2 + shortcut(x))
```

- 下采样只发生在 `conv1` 的 stride 上，`conv2` 永远是 stride=1，这是 ResNet 的标准设计。
- 每个卷积后面都接 BatchNorm，再接 ReLU。
- 残差相加之后才做最后一次 ReLU。

## 3. 两个常被问到的细节：`bias=False` 和 `inplace=True`

**`bias=False`**：卷积后面紧接 BatchNorm，BN 自己带可学习的平移参数 β，卷积的 bias 会被 BN 重新归一化掉，属于冗余参数。关掉它既省参数，也让 BN 的统计更干净。

**`ReLU(inplace=True)`**：直接在输入张量上原地修改，不新分配内存，省显存也省一次内存申请/释放。注意它要求这块内存没有被别的计算引用，所以在残差结构里要**先算完捷径分支**再做原地 ReLU，否则会改坏后面还要用的张量。

## 4. 捷径（shortcut）的两种形态

残差是**逐元素相加**，两边形状必须完全一致，所以捷径分两种：

```python
self.shortcut = nn.Identity()                    # 通道和分辨率都不变：原样传过去
if stride != 1 or inchannels != outchannels:     # 否则用 1×1 卷积 + BN 对齐
    self.shortcut = nn.Sequential(
        nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(outchannels),
    )
```

- 1×1 卷积在这里**不是提特征**，而是做通道的线性组合、顺便用 stride 完成空间下采样。
- 空 `nn.Sequential()` 也能原样返回输入，但 `nn.Identity()` 语义更清楚。
- 属性名用 `shortcut`（torchvision 官方叫法）。2026-11-03 的旧权重里保存的是拼写错误的 `shoutcut`，加载时需要把键名里的 `shoutcut` 替换成 `shortcut`。

## 5. 为什么是"先相加再 ReLU"

如果把 ReLU 放在相加之前，恒等通路的负数会被截断成 0，残差就失去了把信息原样传过去的意义。标准写法是 `relu(bn2 + identity)`：相加得到完整信号，再由 ReLU 统一决定哪些激活。

## 6. 四阶段的通道数与下采样

| 阶段 | 输出通道 | 第一个 block 的 stride | 输出分辨率 |
| --- | ---: | ---: | --- |
| layer1 | 64 | 1 | 32×32 |
| layer2 | 128 | 2 | 16×16 |
| layer3 | 256 | 2 | 8×8 |
| layer4 | 512 | 2 | 4×4 |

`_make_layer` 会把 stride 列表拼成 `[stride] + [1] * (num_blocks - 1)`：只有每组第一个 block 负责下采样，后面的 block 保持 stride=1。每建完一组，就把 `self.inchannels` 更新成 `outchannels * expansion`，供下一组使用。

## 7. `expansion` 是什么

`BasicBlock.expansion = 1`，`Bottleneck.expansion = 4`。它的含义是"该 block 输出通道数相对 `outchannels` 的倍率"：BasicBlock 的输出就是 `outchannels`，Bottleneck 因为最后一层 1×1 升维，输出是 `outchannels × 4`。

代码里 `fc` 写成 `nn.Linear(512 * block.expansion, num_classes)` 是为将来换 Bottleneck 预留的；现在 `×1` 等于没乘。

## 8. 分类头为什么用全局平均池化

```text
layer4 输出 [N,512,4,4]
AdaptiveAvgPool2d((1,1)) → [N,512,1,1]
flatten → [N,512]
Linear(512, 10) → [N,10]
```

- 相比 `Flatten + Linear(512×4×4, 10)`，全局平均池化把参数从 8 万多降到 5120，过拟合风险低得多。
- `AdaptiveAvgPool2d` 不依赖输入分辨率，换输入尺寸也能用。
- `flatten` 从第 1 维开始拉平，保留第 0 维的 batch。

## 9. 参数量 11,173,962 的粗略拆解

| 部分 | 参数量 |
| --- | ---: |
| 入口 conv 3×3×3×64 | 1,728 |
| layer1（2 个 block，各 2 个 3×3 卷积） | 147,456 |
| layer2（含第一个 block 的 1×1 捷径 8,192） | 516,096 |
| layer3（含 1×1 捷径 32,768） | 2,064,384 |
| layer4（含 1×1 捷径 131,072） | 8,257,536 |
| 所有 BatchNorm 的可学习参数 | 约 9,600 |
| 分类层 512×10+10 | 5,130 |
| **合计** | **11,173,962** |

卷积占了绝大部分参数，且越往深（通道越多）增长越快——layer4 一组就占了 74%。

## 10. 形状自检清单

改结构后按这个顺序核对：

1. 每个 `Conv2d` 的 in_channels 等于上一层的 out_channels
2. 每个 `BatchNorm2d` 的参数等于前一个卷积的输出通道数
3. 每个池化/stride=2 之后空间尺寸减半
4. 全局池化后 `flatten` 的特征数等于 `Linear.in_features`
5. 分类层输出数等于类别数（CIFAR-10 为 10）

10 秒冒烟测试：

```python
net = ResNet18()
print(net(torch.randn(2, 3, 32, 32)).shape)   # 期望 torch.Size([2, 10])
```

相关：[训练流程与技巧笔记](training-notes.md) · [CIFAR-10 实验记录](cifar10-experiments.md) · [ResNet 项目](../projects/resnet18/README.md)
