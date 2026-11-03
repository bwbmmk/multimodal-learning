# CIFAR-10 实验记录

CIFAR-10 是从 MNIST 走向彩色图像和真实泛化问题的第二步。这个文件记录数据准备、两轮主要实验、过拟合分析，以及后续实验计划。

## 任务与数据

CIFAR-10 包含 10 类 `32×32` 彩色图片：

```text
airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck
```

| 数据 | 数量 | 用途 |
| --- | ---: | --- |
| 官方训练集 | 50,000 | 再划分为训练集和验证集 |
| 训练集 | 45,000 | 更新参数 |
| 验证集 | 5,000 | 每轮评估、选择最佳模型 |
| 官方测试集 | 10,000 | 最终只评估一次 |

```python
train_data = torchvision.datasets.CIFAR10(
    path, train=True, transform=train_transform, download=True
)
test_data = torchvision.datasets.CIFAR10(
    path, train=False, transform=eval_transform, download=True
)

train_set, val_set = random_split(
    train_data, [45000, 5000], generator=torch.Generator().manual_seed(42)
)
```

## 形状检查

```text
单张图片：[3, 32, 32]
一个 batch：[B, 3, 32, 32]
标签：[B]
```

`3` 是 RGB 通道数。与 MNIST 的 `[1, 28, 28]` 相比，主要变化是通道数和空间尺寸。

## 预处理

训练集允许随机变化，验证集和测试集必须保持确定性：

```python
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
```

### `RandomCrop(32, padding=4)`

- 现在四周各补 4 个像素
- `32×32` 先变成 `40×40`
- 再随机裁回 `32×32`

不是把原图内容拉伸到 `36×36`。

### `ToTensor()` 与 `Normalize()`

```text
原始像素 0~255
→ ToTensor：0~1，形状 HWC → CHW
→ Normalize((x - 0.5) / 0.5)：-1~1
```

三个 `0.5` 分别对应 R、G、B 三个通道。

## 验证集常见错误

错误写法：

```python
train_set, val_set = random_split(train_data, [45000, 5000])
val_set = torchvision.datasets.CIFAR10(
    path, train=True, transform=eval_transform, download=False
)
```

这会把 5,000 个验证索引整体覆盖成完整的 50,000 张训练集，造成训练数据泄漏。

正确写法：

```python
val_set.dataset = torchvision.datasets.CIFAR10(
    path, train=True, transform=eval_transform, download=False
)
```

`random_split` 返回的 `Subset` 保存的是 `indices` 和 `dataset`。替换 `dataset` 不会改变验证索引，也不会重新划分数据。

## 实验一：无正则化的 CNN

### 结构

```text
Conv(3→32) + ReLU + Pool
Conv(32→64) + ReLU + Pool
Conv(64→128) + ReLU + Pool
Conv(128→256) + ReLU + Pool
Flatten(256×2×2)
Linear(1024→256) + ReLU
Linear(256→64) + ReLU
Linear(64→10)
```

### 结果

| 指标 | 数值 |
| --- | ---: |
| 最佳验证准确率 | 75.28% |
| 测试准确率 | 74.85% |
| 第 20 轮训练准确率 | 97.84% |
| 第 20 轮验证准确率 | 73.92% |
| 验证损失变化 | 约 0.77 → 1.48 |

### 判断

这是典型过拟合：

- 训练准确率持续上升
- 验证准确率很早就停止提升
- 验证损失后期明显上升
- 最佳模型出现在训练结束之前

验证损失上升而准确率只是缓慢下降，说明模型对一些错误预测变得越来越自信。交叉熵会惩罚“高置信度错误”，因此损失比准确率更早暴露问题。

## 实验二：数据增强与正则化

### 改动

- 训练集加入随机裁剪与水平翻转
- 每个卷积层后加入 BatchNorm
- 全连接层加入 Dropout(0.3)
- Adam 改为 AdamW，`weight_decay=1e-4`
- 训练 20 轮

### 网络顺序

```text
Conv → BatchNorm → ReLU → MaxPool
```

### 结果

| 指标 | 实验一 | 实验二 |
| --- | ---: | ---: |
| 最佳验证准确率 | 75.28% | **82.54%** |
| 测试准确率 | 74.85% | **81.85%** |
| 训练/验证差距 | 很大 | 约 1.34 个百分点 |

验证错误率从约 24.72% 降到约 17.46%，相对减少接近三成。虽然训练准确率下降，但泛化能力明显提高。

这说明：**训练准确率不是唯一目标。一个 98% 训练准确率但严重过拟合的模型，可能不如一个 83% 训练准确率、82% 验证准确率的模型。**

## 实验三：50 轮与结构扩展

### 当前配置

```text
Conv → BN → ReLU → Conv → BN → ReLU → Pool
Conv → BN → ReLU → Conv → BN → ReLU → Pool
Conv → BN → ReLU → Conv → BN → ReLU → Pool
```

通道变化：

```text
3 → 16 → 32 → 64 → 128 → 256 → 512
空间尺寸：
32 → 16 → 8 → 4
```

全连接输入：

```text
512 × 4 × 4 = 8192
```

训练设置：

```python
optimizer = torch.optim.AdamW(
    net.parameters(), lr=0.001, weight_decay=0.0001
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50
)
```

### 余弦退火公式

```text
lr_t = eta_min + (eta_max - eta_min) / 2 * (1 + cos(t / T_max * pi))
```

参数：

```text
eta_max = 0.001
eta_min = 0
T_max = 50
```

代入：

```text
第 0 轮：lr = 0.001
第 25 轮：lr = 0.0005
第 50 轮：lr = 0
```

余弦角度从 `0` 增加到 `pi`，只使用下降半周期，因此 50 轮内学习率不会重新上升。`weight_decay=0.0001` 是权重衰减，不参与余弦计算。

### 结果

| 指标 | 实验二（20 轮） | 实验三（50 轮） |
| --- | ---: | ---: |
| 最佳验证准确率 | 82.54% | **89.86%**（第 46 轮） |
| 测试准确率 | 81.85% | **89.22%** |
| 测试损失 | — | 0.4163 |
| 最后一轮训练准确率 | 83.20% | 95.93% |
| 最后一轮验证准确率 | 81.86% | 89.78% |
| 训练/验证差距 | 约 1.34 个百分点 | 约 6.15 个百分点 |

训练耗时：50 轮 × 每轮约 14 秒，共约 12 分钟（GPU）。

相对实验二，最佳验证准确率提高 **7.32 个百分点**，测试准确率提高 **7.37 个百分点**，错误率从 18.15% 降到 10.78%。

### 剩余过拟合的判断

第 29 轮之后验证损失基本停在 `0.39 ~ 0.40`，而训练损失继续从 `0.25` 降到 `0.12`：

```text
第 29 轮  Train Loss 0.2493  Val Loss 0.3888
第 40 轮  Train Loss 0.1473  Val Loss 0.3980
第 50 轮  Train Loss 0.1184  Val Loss 0.4003
```

这说明模型仍在持续压缩训练集误差，而这部分改善没有转化到验证集上，属于**轻度到中度过拟合**。

但要区分两种信号：

- **验证损失不再下降**：泛化性能已经到平台期，这是过拟合的证据
- **验证准确率仍在缓慢刷新**（第 46 轮 89.86%）：模型容量尚未被完全浪费

因此当前状态不是"必须立刻正则化"，而是"再加正则化还有空间，但收益会缩小"。

### 与实验一的区别

实验一是**严重过拟合**：验证损失从 0.77 涨到 1.48，最佳模型出现在训练早期。

实验三是**温和过拟合**：验证损失只是停止下降，没有明显回升，最佳模型出现在第 46/50 轮，几乎训练到最后。这说明数据增强和正则化已经把过拟合控制在一个可接受的范围内。

## 实验四：手写 CIFAR ResNet18（2026-11-03）

代码、完整日志和逐轮指标见 [ResNet 项目](../projects/resnet18/README.md)。随机初始化、SGD 0.1 / momentum 0.9 / weight_decay 5e-4、余弦退火 50 轮，输入 32×32，未使用预训练或标签平滑。

| 轮次 | 训练准确率 | 验证准确率 | 验证损失 |
| --- | ---: | ---: | ---: |
| 40 | 98.42% | 93.30% | 0.2413 |
| 48，最佳准确率 | 99.85% | 94.80% | 0.2019 |
| 50 | 99.85% | 94.64% | 0.2015 |

最佳权重测试准确率 **93.91%**，损失 **0.2212**。存在泛化差距，但末期验证损失仍下降，尚无持续恶化证据。相对 CNN 测试提高 4.69 个百分点，伴随模型、优化器、批量大小等变化，不是残差连接的单因素消融。

标签平滑仅为后续可选实验：`CrossEntropyLoss(label_smoothing=0.1)` 在十类下目标为正确类 0.91、其他每类 0.01。验证损失应使用统一普通交叉熵，避免与平滑训练损失混比。没有证据保证其涨点，也没有本项目的实际收益数值。

## 验证函数

核心逻辑：

```python
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / total, correct / total
```

### 为什么乘 `images.size(0)`

`CrossEntropyLoss` 返回当前 batch 的平均损失。最后一个 batch 可能不满，直接累加平均值会让不同大小的 batch 权重不一致。

```text
总损失 = Σ(当前 batch 平均损失 × 当前 batch 样本数)
平均损失 = 总损失 / 总样本数
```

### `torch.max` 与 `argmax`

`torch.max(outputs, dim=1)` 返回两个结果：

```python
values, indices = torch.max(outputs, dim=1)
```

分类只关心索引：

```python
predicted = outputs.argmax(dim=1)
```

不要写成：

```python
predicted = torch.max(outputs.data, 1)
```

这样 `predicted` 是两个返回值的组合，不能直接与标签比较。`torch.no_grad()` 已经关闭梯度，不需要额外使用 `.data`。

## 损失函数与优化器

```python
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    net.parameters(), lr=0.001, weight_decay=0.0001
)
```

- CIFAR-10 仍然是 10 类分类，输出 `[B, 10]`
- 标签是 `0~9` 的整数，形状 `[B]`
- `CrossEntropyLoss` 内部包含 `LogSoftmax`
- 模型最后一层输出 logits，不再加 Softmax
- `AdamW` 把权重衰减与梯度更新解耦

## 网络结构检查清单

每次改动卷积网络后检查：

1. 第一层 `in_channels` 是否等于输入通道数
2. 每个卷积的 `out_channels` 是否等于下一个卷积的 `in_channels`
3. BatchNorm 参数是否等于前一个卷积的输出通道数
4. 每次池化后空间尺寸是否减半
5. Flatten 后特征数是否等于 `Linear.in_features`
6. 分类输出数是否等于类别数

## 下一步

1. 分析 ResNet 错例；可选做一次单变量标签平滑对照，用验证集选择方案。
2. 学习冻结与微调，区分从头训练和预训练迁移。
3. 使用真实图文对做 CLIP 检索；MalImg 保留为选修。

## 三个实验的完整对比

| 项目 | 实验一 | 实验二 | 实验三 |
| --- | ---: | ---: | ---: |
| 卷积层数 | 4（每层后池化） | 4 | 6（两两成块） |
| 池化次数 | 4 | 4 | 3 |
| 最终特征图 | 2×2 | 2×2 | 4×4 |
| 参数量 | 约 0.67M | 约 0.67M | 5.81M |
| 数据增强 | 无 | 有 | 有 |
| BatchNorm | 无 | 有 | 有 |
| Dropout | 无 | 0.3 | 0.3 |
| 训练轮数 | 20 | 20 | 50 |
| 学习率调度 | 无 | 无 | 余弦退火 |
| 最佳验证准确率 | 75.28% | 82.54% | 89.86% |
| 测试准确率 | 74.85% | 81.85% | 89.22% |

这些实验展示了组合改动后的结果，尚不足以确定某一技巧的独立贡献。当前已完成 CNN 与手写 ResNet 两种训练实践，下一步优先完善评测解释与迁移学习，而非继续无边界追分。
