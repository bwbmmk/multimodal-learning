# 多模态学习

从图像分类基础出发，逐步走向 ResNet、恶意软件图像分类与多模态学习的学习仓库。

> GitHub 仓库名使用英文 `multimodal-learning`；项目中文名是「多模态学习」。

这里记录的不只是最终代码，还包括数据预处理、训练/验证流程、过拟合分析、实验对比和踩坑过程。代码以容易读懂为目标，适合作为从零入门 PyTorch 的第一套完整练习。

## 当前进度

| 阶段 | 内容 | 状态 | 代表结果 |
| --- | --- | --- | --- |
| 01 | MNIST 手写数字识别 | 已完成 | 测试准确率 **99.26%** |
| 02 | CIFAR-10 彩色图像分类 | 已完成基础实验 | 测试准确率 **89.22%** |
| 02+ | CIFAR-10 正则化与更长训练 | 已完成 | 50 轮 6 层 CNN，最佳验证 **89.86%** |
| 03 | ResNet 与迁移学习 | 计划中 | 从 CIFAR-10 过渡到预训练模型 |
| 04 | MalImg 恶意软件图像分类 | 计划中 | 图像分类与类别不平衡实战 |
| 05 | 多模态学习 | 计划中 | 图像 + 文本/元数据融合 |

## 实验概览

### MNIST

- 3 层卷积 + 2 层全连接
- 训练/验证/测试：54,000 / 6,000 / 10,000
- `Adam(lr=0.001)` + `CrossEntropyLoss`
- 最佳验证准确率 **99.20%**
- 测试准确率 **99.26%**，测试损失 **0.0243**

### CIFAR-10

| 实验 | 关键变化 | 最佳验证准确率 | 测试准确率 |
| --- | --- | ---: | ---: |
| baseline | 4 层卷积，无数据增强/BN/Dropout | 75.28% | 74.85% |
| regularized | 数据增强 + BatchNorm + Dropout + AdamW | 82.54% | 81.85% |
| extended | 6 层卷积、50 轮、余弦退火 | **89.86%** | **89.22%** |

baseline 在第 20 轮出现明显过拟合：训练准确率 **97.84%**，验证准确率只有 **73.92%**，验证损失从约 **0.77** 上升到 **1.48**。加入正则化后，训练与验证差距大幅缩小，验证准确率提升约 **7.26 个百分点**。

extended 实验把训练轮数从 20 提到 50、把网络从 4 层加深到 6 层，测试准确率再提高 **7.37 个百分点**，错误率从 18.15% 降到 10.78%。剩余问题只是轻度过拟合：验证损失在第 29 轮后停在 `0.39~0.40`，训练损失继续降到 `0.12`，但最佳模型出现在第 46/50 轮，说明训练轮数设置基本合理。

## 学习路线

1. **图像分类基础**：张量形状、卷积、池化、训练五步、验证集与测试集的职责。
2. **泛化能力**：数据增强、BatchNorm、Dropout、权重衰减、学习率调度。
3. **ResNet 与迁移学习**：残差连接、预训练权重、冻结/解冻策略、微调。
4. **MalImg 实战**：灰度恶意软件图像、类别不平衡、按家族划分数据、Macro-F1 与混淆矩阵。
5. **多模态入门**：图像编码器 + 文本/元数据编码器、早期融合与晚期融合、对比学习。
6. **工程化**：固定随机种子、保存最佳模型、记录实验、拆分 `model.py` / `train.py` / `predict.py`。

完整说明见 [学习路线](docs/learning-roadmap.md)。

## 仓库结构

```text
multimodal-learning/
├── README.md
├── requirements.txt
├── docs/
│   ├── environment.md              # Windows + CUDA + PyTorch 环境配置
│   ├── learning-roadmap.md         # 从 CNN 到多模态的学习路线
│   ├── learning-journal.md         # 两个项目的推进过程与关键问答
│   ├── mnist-notes.md              # MNIST 阶段的原理笔记
│   ├── mnist-troubleshooting.md    # MNIST 踩坑与报错速查
│   └── cifar10-experiments.md      # CIFAR-10 实验记录与过拟合分析
├── projects/
│   ├── mnist/
│   │   ├── README.md
│   │   ├── train.py
│   │   └── assets/
│   ├── cifar10/
│   │   ├── README.md
│   │   └── train.py
├── weights/
│   ├── README.md
│   ├── mnist/
│   └── cifar10/
└── tools/
    └── check_env.py                # PyTorch / CUDA 环境自检
```

## 快速开始

```bash
git clone https://github.com/bwbmmk/multimodal-learning.git
cd multimodal-learning
pip install -r requirements.txt
```

运行环境自检：

```bash
python tools/check_env.py
```

运行项目：

```bash
python projects/mnist/train.py
python projects/cifar10/train.py
```

两个训练脚本里的数据集路径目前是绝对路径，换机器运行前请先修改脚本中的 `path` 变量。代表性模型权重保存在 `weights/`；训练过程中新生成的其他权重仍按临时产物处理。

## 文档索引

- [环境配置](docs/environment.md)
- [学习路线](docs/learning-roadmap.md)
- [学习过程记录](docs/learning-journal.md)
- [MNIST 原理笔记](docs/mnist-notes.md)
- [MNIST 踩坑与问答](docs/mnist-troubleshooting.md)
- [CIFAR-10 实验记录](docs/cifar10-experiments.md)

## 实验原则

- 训练集更新参数，验证集选择模型，测试集只在最后使用。
- 先建立可复现的 baseline，再每次只改变一个因素。
- 保存验证集表现最好的模型，而不是最后一轮模型。
- 关注验证/测试指标，不用追求接近 100% 的训练准确率。
- 权重文件、数据集和缓存不进入 Git；只提交代码、文档和必要的示意图。
