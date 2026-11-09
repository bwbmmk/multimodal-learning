# 多模态学习

从零学习 PyTorch：MNIST → CIFAR-10 CNN → 手写 ResNet18 → 迁移学习与图文检索。

这里保留训练代码、实验日志、错误分析和可以在浏览器运行的模型。当前完成的是单模态图像分类；迁移学习、CLIP 和视觉语言模型仍是后续计划。

## 当前进度

| 内容 | 状态 | 结果 |
| --- | --- | --- |
| MNIST 手写数字识别 | 已完成 | 测试准确率 99.26% |
| CIFAR-10 CNN 三轮实验 | 已完成 | 最佳验证 89.86%，测试 89.22% |
| 手写 CIFAR ResNet18，从头训练 | 已完成 | 50 轮基线测试 93.91%；100 轮改进版**验证 96.10%、测试 95.29%** |
| 浏览器与手机演示 | 已更新为 ResNet18 | ONNX + WASM，本地推理，输入 32×32 |
| ImageNet 预训练模型的冻结与微调 | 待完成 | 区分从头训练和迁移学习 |
| CLIP 图文检索 | 待完成 | 真实图文配对、检索评测、错误分析 |
| MalImg | 可选练习 | 不作为进入图文多模态的必经项目 |

## 实验结果

| 数据集 / 实验 | 主要设置 | 最佳验证准确率 | 测试准确率 |
| --- | --- | ---: | ---: |
| MNIST | 3 层 CNN，Adam | 99.20% | 99.26% |
| CIFAR-10 baseline | 4 层 CNN，无增强/BN/Dropout | 75.28% | 74.85% |
| CIFAR-10 regularized | 增强、BN、Dropout、AdamW | 82.54% | 81.85% |
| CIFAR-10 extended | 6 层 CNN，50 轮，余弦退火 | 89.86% | 89.22% |
| CIFAR-10 ResNet18（基线） | 手写残差块，SGD，50 轮，余弦退火 | 94.80% | 93.91% |
| **CIFAR-10 ResNet18（改进）** | **100 轮，标签平滑 0.1，RandomErasing，真实 mean/std，AMP** | **96.10%** | **95.29%** |

两轮 ResNet 实验都保留 32×32 RGB 输入，入口是 3×3、stride=1，没有入口 MaxPool；四组残差块数量为 `[2,2,2,2]`，使用全局平均池化和 10 类分类层。参数量 **11,173,962**，随机初始化、全部参数参与训练，没有加载 ImageNet 权重。

基线（50 轮）配置：

- 数据划分：45,000 训练 / 5,000 验证 / 10,000 测试。
- 训练：随机裁剪、水平翻转；验证和测试不做随机增强。
- 优化：SGD，`lr=0.1`、`momentum=0.9`、`weight_decay=5e-4`；余弦退火 50 轮，batch size 128。
- 第 48 轮验证损失 0.2019，验证准确率 94.80%；加载该轮权重后测试损失 **0.2212**、准确率 **93.91%**。
- 末期训练准确率约 99.85%，存在泛化差距；但第 40→50 轮验证损失仍从 0.2413 降至 0.2015，没有持续恶化的证据。

2026-11-10 追加了一轮 100 epoch 改进实验，在上表基础上启用标签平滑 0.1、RandomErasing、CIFAR-10 真实 mean/std、AMP，batch size 改为 64：

- 第 98 轮最佳验证 **96.10%**（loss 0.6027），加载该轮后测试 **95.29%**（loss 0.6159）。
- 标签平滑会改变 loss 口径（即使全对也有约 0.4 的下限），因此 `0.6027 / 0.6159` 不能与基线的 `0.2019 / 0.2212` 直接比较；能横向比的是准确率。
- 相对基线测试提高 1.38 个百分点，但轮数、损失函数、增强、归一化、精度、batch size 同时改变，**不能归因到单一技巧**。
- 逐轮记录与解读见 [CIFAR-10 实验记录](docs/cifar10-experiments.md) 实验五。

相对旧 CNN，ResNet 基线测试提高 4.69 个百分点；以上都是单次实验，不是多随机种子统计。

原始输出和逐轮指标见 [ResNet 项目](projects/resnet18/README.md)：50 轮基线为 [训练日志](projects/resnet18/training-log.txt) / [指标 CSV](projects/resnet18/metrics.csv)，100 轮改进版为 [训练日志](projects/resnet18/training-log-100ep.txt) / [指标 CSV](projects/resnet18/metrics-100ep.csv)。

## 运行

安装 Python 训练依赖：

```bash
pip install -r requirements.txt
python tools/check_env.py
```

```bash
python projects/mnist/train.py
python projects/cifar10/train.py
python projects/resnet18/CIFAR-10_resnet18_train.py
```

训练脚本仍保留学习时的数据路径，例如 `D:/ai/cifar10_data/`，换机器前需要修改 `path`。新训练产生的权重保存到运行时当前目录。代表性已训练权重位于 [weights](weights/README.md)。

## 浏览器演示

```bash
cd projects/cifar10-demo
py tools/serve.py
```

Windows 也可以双击 `start.bat`，按终端显示的地址访问；默认端口可用时为 `http://127.0.0.1:8000`。手机与电脑连接同一 Wi-Fi 后按页面二维码访问，需要系统防火墙允许局域网连接。

- 当前使用第 98 轮 ResNet18，ONNX **44,686,130 字节，约 42.6 MiB**。
- 模型和 WASM 运行时均随仓库保存，图片不上传，推理发生在当前浏览器。
- 保留原图、32×32 输入预览、裁剪/拉伸选项和 Top-5 结果。
- 十张示例经过规则筛选以便展示，不能当作随机评测；模型只认识 CIFAR-10 的十类。
- 浏览器使用版本化模型 URL，元数据禁止缓存，以免更换模型后加载旧资产。

本次导出用十张 32×32 RGB 示例做了数值对照：ONNX 与 PyTorch CPU 的 logits 最大绝对误差约 **5.96e-7**，十张预测类别均一致；浏览器端一致性脚本（`projects/cifar10-demo/verification/`）已按新模型和新预处理更新，需要在本地重新跑一次。该结果仅覆盖这组样本，不代表全部输入逐位一致。推理耗时由页面现场测量，不沿用旧模型的速度。

详细操作见 [Demo README](projects/cifar10-demo/README.md)。

## 目录与记录

```text
projects/
  mnist/           MNIST 训练
  cifar10/         历史 CNN 训练
  resnet18/        手写残差结构、训练、日志与指标
  cifar10-demo/    当前 ResNet18 网页演示与导出脚本
docs/              实验、学习路线、环境和部署记录
weights/           代表性训练权重
```

- [学习路线](docs/learning-roadmap.md)
- [学习过程记录](docs/learning-journal.md)
- [CIFAR-10 历史实验](docs/cifar10-experiments.md)
- [ResNet18 结构笔记](docs/resnet18-notes.md)
- [训练流程与技巧笔记](docs/training-notes.md)
- [环境说明](docs/environment.md) / [环境配置原始笔记](docs/environment-notes.md)
- [浏览器部署记录](docs/cifar10-web-demo.md)
- [MNIST 原理](docs/mnist-notes.md) / [踩坑记录](docs/mnist-troubleshooting.md)

## 下一步

1. 分析错误样本；标签平滑已随 100 轮实验启用，但与轮数、增强、AMP 同时改动，尚未做单变量消融。
2. 学习预训练模型的冻结与微调，核对输入预处理与可训练参数。
3. 做 CLIP 文本搜图：固定评测集、报告检索指标、分析失败案例。
4. 基础并行学习 Python、数据结构算法、线性代数与 Linux/Git。先做一个可解释、可复现的项目，再扩展新方向。

训练集更新参数，验证集决定方案；测试集用于方案确定后的最终评估。数据集、缓存和一般训练产物不提交，仅归档代表性权重及运行 Demo 必需的模型资产。
