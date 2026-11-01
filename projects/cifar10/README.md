# CIFAR-10 彩色图像分类

从 MNIST 迁移到 CIFAR-10 的第二个完整项目：三通道输入、数据增强、BatchNorm、Dropout、AdamW 与余弦退火。

## 任务

识别 10 类 `32×32` RGB 图片：

```text
airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck
```

## 数据划分

| 数据 | 数量 |
| --- | ---: |
| 训练集 | 45,000 |
| 验证集 | 5,000 |
| 测试集 | 10,000 |

验证集从官方 50,000 张训练集中划分，测试集保持独立。

## 预处理

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

训练集使用随机增强，验证集和测试集只使用确定性变换。

## 实验结果

| 实验 | 关键变化 | 最佳验证准确率 | 测试准确率 |
| --- | --- | ---: | ---: |
| baseline | 4 层卷积，无增强/BN/Dropout | 75.28% | 74.85% |
| regularized | 数据增强 + BN + Dropout + AdamW | 82.54% | 81.85% |
| extended | 6 层卷积、50 轮、余弦退火 | **89.86%** | **89.22%** |

### baseline 的过拟合现象

```text
第 20 轮 Train Acc: 97.84%
第 20 轮 Val Acc:   73.92%
验证损失：约 0.77 → 1.48
```

训练准确率继续上涨，但验证损失明显上升，说明模型记住了训练集细节而没有学到更好的通用特征。

### 正则化后的变化

- 训练准确率降到约 83%
- 最佳验证准确率提高到 82.54%
- 训练/验证差距缩小到约 1.34 个百分点

训练准确率变低，但模型泛化更好，这是期望中的结果。

### extended 实验：50 轮 + 6 层卷积

```text
第 46 轮 Val Acc: 89.86%（最佳，保存为最终权重）
第 50 轮 Train Acc: 95.93%  Val Acc: 89.78%
测试结果：Test Loss 0.4163  Test Acc 89.22%
```

相比 20 轮版本，测试准确率从 81.85% 提升到 89.22%，错误率从 18.15% 降到 10.78%。

这轮仍然有轻度过拟合，但性质和 baseline 完全不同：

```text
第 29 轮 Val Loss 0.3888
第 50 轮 Val Loss 0.4003   ← 只是停止下降，没有明显回升
```

而训练损失从 `0.25` 继续降到 `0.12`。也就是说，模型还在继续压缩训练集误差，但这部分改善没有转化到验证集。

诊断要点：

- **验证损失停止下降** → 泛化到了平台期，是过拟合信号
- **验证准确率仍在缓慢刷新到第 46 轮** → 容量还没被完全浪费
- **最佳模型出现在 46/50 轮** → 说明训练轮数设置基本合理，没有明显训练过头

所以目前属于「温和过拟合 + 容量接近饱和」，下一步优先用标签平滑和更强的数据增强（`RandomErasing`），而不是继续加轮数。

```python
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
```

标签平滑直接针对「训练损失 0.12、验证损失 0.40」这个现象：它不允许模型把正确类的置信度推到接近 1。

## 当前网络

```text
Block 1: Conv(3→16)   + BN + ReLU + Conv(16→32)   + BN + ReLU + Pool
Block 2: Conv(32→64)  + BN + ReLU + Conv(64→128)  + BN + ReLU + Pool
Block 3: Conv(128→256)+ BN + ReLU + Conv(256→512)+ BN + ReLU + Pool
Flatten: 512 × 4 × 4 = 8192
FC: 8192 → 512 → 64 → 10
Dropout: 0.3
```

训练设置：

```python
optimizer = torch.optim.AdamW(
    net.parameters(), lr=0.001, weight_decay=0.0001
)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=50
)
```

## 运行

```bash
python projects/cifar10/train.py
```

脚本中的数据集路径为：

```python
path = 'D:/ai/cifar10_data/'
```

换机器运行前需要修改为自己的目录。第一次运行会自动下载 CIFAR-10，之后直接读取本地数据。

## 注意

- 验证集要用 `val_set.dataset = ...` 替换预处理，不能重新赋值整个 `val_set`
- 训练集增强，验证/测试集不增强
- 最佳模型按验证准确率保存
- 调整卷积结构后，必须重新核对 BatchNorm 通道和 `Linear.in_features`
- 不要使用测试集反复调参

完整实验过程与原理见 [CIFAR-10 实验记录](../../docs/cifar10-experiments.md)。
