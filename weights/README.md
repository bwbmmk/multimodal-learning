# 模型权重

这里保存已经训练好的代表性权重，方便复现和后续推理。Git 不擅长存储大文件，因此只保留有代表性的检查点；训练过程中产生的大量临时权重仍应忽略。

## 文件说明

| 目录 / 文件 | 对应项目 | 参数量 | 说明 |
| --- | --- | ---: | --- |
| `mnist/mnist_best_model_1031_0011.pth` | MNIST | 426,122 | 第 5 轮最佳验证模型，测试准确率 99.26% |
| `cifar10/cifar10_best_model_1101_2111.pth` | CIFAR-10 | 667,914 | 早期 4 层卷积实验权重 |
| `cifar10/cifar10_best_model_1101_2112.pth` | CIFAR-10 | 667,914 | 早期 4 层卷积实验权重 |
| `cifar10/cifar10_best_model_1101_2149.pth` | CIFAR-10 | 669,838 | 加入 BatchNorm 等改动后的 4 层卷积权重 |
| `cifar10/cifar10_best_model_1101_2230.pth` | CIFAR-10 | 5,805,104 | 扩展为 6 层卷积后的实验权重 |

CIFAR-10 的实验版本经过了多次结构调整，文件保留原始时间戳命名，避免错误对应实验编号。具体实验结果见 [CIFAR-10 实验记录](../docs/cifar10-experiments.md)。

## 加载方式

权重文件只包含 `state_dict`，不包含网络结构。加载前必须使用同一版本的模型定义：

```python
model = Net().to(device)
state_dict = torch.load(
    "weights/mnist/mnist_best_model_1031_0011.pth",
    map_location=device,
)
model.load_state_dict(state_dict)
model.eval()
```

如果模型结构已经修改，旧权重可能因参数名或形状不一致而无法加载。

## 为什么没有使用 Git LFS

当前单个权重文件小于 GitHub 的 100 MB 限制，仓库体积也还可以接受，因此直接作为普通文件保存。如果后续加入 ResNet 或更大的权重，建议改用 Git LFS，或在 Release 中提供下载链接。
