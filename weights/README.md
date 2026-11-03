# 模型权重

这里保存已经训练好的代表性权重，方便复现和后续推理。Git 不擅长存储大文件，因此只保留有代表性的检查点；训练过程中产生的大量临时权重仍应忽略。

## 文件说明

| 目录 / 文件 | 对应项目 | 参数量 | 说明 |
| --- | --- | ---: | --- |
| `mnist/mnist_best_model_1031_0011.pth` | MNIST | 426,122 | 第 5 轮最佳验证模型，测试准确率 99.26% |
| `cifar10/cifar10_best_model_1101_2111.pth` | CIFAR-10 | 667,914 | 早期 4 层卷积实验权重 |
| `cifar10/cifar10_best_model_1101_2112.pth` | CIFAR-10 | 667,914 | 早期 4 层卷积实验权重 |
| `cifar10/cifar10_best_model_1101_2149.pth` | CIFAR-10 | 669,838 | 加入 BatchNorm 等改动后的 4 层卷积权重 |
| `cifar10/cifar10_best_model_1101_2230.pth` | CIFAR-10 | 5,803,082 | 6 层卷积 + 50 轮实验的最终最佳权重，第 46 轮验证 89.86%，测试 89.22% |

CIFAR-10 的实验版本经过了多次结构调整，文件保留原始时间戳命名，避免错误对应实验编号。具体实验结果见 [CIFAR-10 实验记录](../docs/cifar10-experiments.md)。

## 最新 ResNet 权重

`cifar10/cifar10_resnet_best_model_1103_2241.pth`：11,173,962 参数，第 48 轮最佳验证 94.80%，测试 93.91%，loss 0.2212。随机初始化训练，无 ImageNet 预训练。SHA256：`b95de7c4e694f62b4ffa88e7fcc812128b7a2d2ee45f70a66ff98fe48f26f739`。

必须匹配 [ResNet 模型定义](../projects/resnet18/model.py) 加载，不能使用历史 CNN 的 `Net`。网页端使用同源权重导出的 `projects/cifar10-demo/model.onnx`，不是直接读取 `.pth`。

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

当前单个权重文件小于 GitHub 的 100 MB 限制，仓库体积也还可以接受，因此直接作为普通文件保存。本次 ResNet 检查点约 42.7 MiB，仍沿用代表性权重归档约定；后续模型增大时应考虑 Git LFS 或 Release 资产。
