# 训练流程与技巧笔记

> 来源：`projects/resnet18/CIFAR-10_resnet18_train.py`（以及 `projects/mnist/train.py`、`projects/cifar10/train.py`）里的代码注释，按主题重新整理。代码保持原位，这里只归纳知识点。

## 1. 可复现性：随机数到底有几处

深度学习的随机性来自四个地方，缺一处结果就会变：

```python
random.seed(seed)                 # Python 内置随机（如果 transform 里用了）
np.random.seed(seed)              # NumPy（如果有 numpy 随机操作）
torch.manual_seed(seed)           # PyTorch CPU：模型初始化、shuffle、torch 系变换
torch.cuda.manual_seed_all(seed)  # 所有 GPU 上的随机
```

`torch.Generator().manual_seed(42)` 只固定了 `random_split` 的**划分**，不覆盖上面这些。

**多进程的种子**：`num_workers>0` 时，DataLoader 会为每个 worker 派生种子（`base_seed + worker_id`），torchvision 的增强（`RandomCrop`/`RandomHorizontalFlip`/`RandomErasing`）用的是 torch 的 RNG，所以会自动不同且可复现。但如果你在 transform 里用了 Python `random` 或 `numpy.random`，就需要 `worker_init_fn` 自己给子进程设种子：

```python
def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

**`cudnn.benchmark = True`**：让 cuDNN 在第一次遇到某个卷积形状时自动 benchmark 多种算法挑最快的。输入尺寸固定时收益稳定，但不同算法浮点累加顺序不同，**不能保证 bit 级可复现**。要严格复现得设 `benchmark=False` 并启用 `torch.use_deterministic_algorithms(True)`（会更慢，个别算子还会直接报错）。

## 2. 数据与增强

训练集和验证/测试集必须用**不同的 transform**：

```python
train_transform = Compose([
    RandomCrop(32, padding=4),      # 先 pad 到 40×40，再随机裁回 32×32
    RandomHorizontalFlip(),         # 50% 概率水平翻转
    ToTensor(),                     # PIL → Tensor，像素缩到 [0,1]
    Normalize(mean, std),           # 按通道做 (x-mean)/std
    RandomErasing(p=0.5),           # 随机擦掉一块（必须放在 ToTensor 之后）
])
test_transform = Compose([ToTensor(), Normalize(mean, std)])
```

要点：

- **验证集不能带随机增强**，否则验证 loss 会抖动，早停/选模的信号被噪声污染。
- `RandomErasing` 操作的是 Tensor，必须放在 `ToTensor()` 之后；放在 `Normalize()` 之后时，填充的 0 表示"归一化后为 0"的颜色，而不是黑色。
- CIFAR-10 的官方统计量是 `mean=(0.4914,0.4822,0.4465)`、`std=(0.2470,0.2435,0.2616)`；用真实值比手写 0.5 更标准，归一化后数据更接近 0 均值、单位方差。

## 3. Dataset、索引与 `random_split`

`Dataset` 的"协议"只有两个方法：`__len__` 和 `__getitem__`。`Subset`、`ConcatDataset` 都是这个协议的实现。

`random_split(dataset, [45000, 5000])` 返回的是 `Subset(dataset, indices)`，它**只记录索引，不复制数据**；`Subset.__getitem__(i)` 等价于 `self.dataset[self.indices[i]]`。

所以只需要加载一份不带 transform 的数据，再给训练/验证各套一个 transform：

```python
base = CIFAR10(path, train=True, transform=None, download=True)   # 返回 PIL
train_idx, val_idx = random_split(range(len(base)), [45000, 5000], generator=g)
train_set = TransformSubset(base, train_idx, train_transform)
val_set   = TransformSubset(base, val_idx,   test_transform)
```

```python
class TransformSubset(Dataset):
    def __init__(self, base, subset, transform):
        self.base, self.subset, self.transform = base, subset, transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, i):
        img, y = self.base[self.subset[i]]      # self.subset[i] 返回整数索引
        return self.transform(img), y
```

**为什么这个类必须定义在模块顶层**：Windows 上 DataLoader 用 spawn 启动子进程，dataset 要被 **pickle** 传给 worker；定义在函数里的类是"局部类"，pickle 找不到它，会报 `Can't pickle local object 'main.<locals>.TransformSubset'`。

## 4. DataLoader 的参数

```python
DataLoader(train_set, batch_size=64, shuffle=True,
           num_workers=4, pin_memory=True, persistent_workers=True,
           prefetch_factor=4, worker_init_fn=worker_init_fn)
```

| 参数 | 作用 | 注意 |
| --- | --- | --- |
| `num_workers` | 子进程数，并行做读图和增强 | 0 = 主进程同步加载；Windows spawn 启动开销大 |
| `shuffle` | 每个 epoch 打乱 | 训练开、验证/测试关 |
| `persistent_workers` | worker 跨 epoch 不销毁 | 仅在 `num_workers>0` 时有效 |
| `prefetch_factor` | 每 worker 预取几个 batch | 默认 2，仅在 `num_workers>0` 时有效 |
| `pin_memory` | 数据放锁页内存，加速 CPU→GPU 拷贝 | 配合 `.to(device, non_blocking=True)` 才是异步 |
| `drop_last` | 丢掉最后不满的 batch | 可选，让每步 batch 恒定 |

**代价**：spawn 时每个 worker 都会把 dataset 反序列化一份（CIFAR-10 的 numpy 数组约 150MB），`num_workers=4` 就是多占几百 MB。这是 Windows 上开多进程的正常开销。

验证/测试用更小的 batch 和更少的 worker（因为只前向、不反传），例如 batch 32、`num_workers=2`。

## 5. 优化器与正则

```python
optimizer = SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
```

- `net.parameters()` 必须在建 optimizer 的那一刻传进去；之后换模型对象，optimizer 不会自动跟着变。
- **momentum**：`v ← 0.9 × v + grad`，保留 90% 的旧速度，相当于给更新加惯性，能平滑噪声、加速收敛。
- **weight_decay（L2）**：`grad ← grad + 5e-4 × w`，每一步额外把权重往 0 拉一点。注意它进入梯度后**也会进入动量**，所以不能把带动量的 SGD 更新简单等同于"参数乘固定因子"。
- **label smoothing**：`CrossEntropyLoss(label_smoothing=0.1)`，目标从 one-hot 变成 `0.9 正确类 + 0.01 其余每类`，抑制过度自信，和权重衰减互补。副作用：**loss 的下限被抬高**（十类时约 0.4），不同设置间的 loss 不能直接比。

## 6. 学习率调度

```python
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
...
scheduler.step()      # 每个 epoch 调一次
```

公式：`lr_t = eta_min + (base_lr - eta_min) × (1 + cos(π·t / T_max)) / 2`。

- `T_max` 必须等于总 epoch 数。若 `T_max=50` 但实际训 100 轮，50 轮之后余弦相位继续转，**lr 会回升**，训练后期反而震荡。
- 每个 epoch 调一次 `step()`（PyTorch 1.1 之后不再传 epoch 参数）。
- 前期 lr 高、粗学；后期 lr 趋近 0、精修。轮数太少（例如 50）就等于精修时间不够。

## 7. 混合精度 AMP

```python
scaler = torch.amp.GradScaler('cuda')       # PyTorch 2.4+ 的写法
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    outputs = net(images)
    loss = criterion(outputs, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)                       # 参数不能省
scaler.update()
```

- fp16 最大约 65504，最小正规数约 `6.1e-5`；fp32 的 8 位指数换成 5 位后，**很小的梯度会下溢成 0**。
- `autocast` 不是把整个模型转 fp16：它按算子白名单自动选择——卷积/全连接/矩阵乘用 fp16（吃 Tensor Core），**BatchNorm、softmax、loss 保持 fp32**。
- `GradScaler` 三步：先把 loss 放大再 backward（避免下溢）→ `step` 时缩回原尺度再更新 → 遇到 inf/NaN 就跳过这一步并调小放大倍数。
- 如果要梯度裁剪，必须先 `scaler.unscale_(optimizer)` 再 `clip_grad_norm_`。
- bf16 的指数范围和 fp32 相同、不会下溢，可以不用 GradScaler；fp16 + scaler 是更通用的组合。

## 8. 推理模式：`inference_mode` 与 `no_grad`

- `torch.no_grad()`：不构建计算图，但 tensor 仍维护 version counter（用于 in-place 检查）和 view 元数据。
- `torch.inference_mode()`：连 version counter 和 view 追踪也关掉，产物标记为 inference tensor，更快、更省内存；代价是这些 tensor 不能再参与求导。

验证/测试循环用 `inference_mode` 就够；训练循环不能用。

## 9. 评估与选模

```python
model.eval()      # BN 用 running stats，Dropout 关闭
...
total_loss += loss.item() * images.size(0)   # 按样本数加权，最后除以总数
correct += (outputs.argmax(dim=1) == labels).sum().item()
```

- 评估前必须 `model.eval()`，训练前必须 `model.train()`。
- 用一个 epoch 的验证准确率决定是否保存 best；测试集**只在最后用一次**，否则测试集就变成了验证集，结果不再诚实。
- `best_val_acc` 是"挑"出来的，会略偏乐观；最终的测试成绩才是对外报告的数字。

## 10. checkpoint：存什么、怎么续训

`state_dict()` 里只有**参数 + BN 的 running stats**。动量、学习率进度都不在里面：

| 内容 | 在哪 |
| --- | --- |
| 模型参数、BN running_mean/var | `net.state_dict()` |
| SGD 动量 buffer | `optimizer.state_dict()` |
| 当前 lr、`last_epoch` | `scheduler.state_dict()` |
| 第几轮、最佳准确率 | 自己存 |

```python
checkpoint = {
    'epoch': epoch + 1,
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_val_acc': best_val_acc,
}
torch.save(checkpoint, path)
```

续训：

```python
ckpt = torch.load(path, map_location=device)
net.load_state_dict(ckpt['model_state_dict'])
optimizer.load_state_dict(ckpt['optimizer_state_dict'])
scheduler.load_state_dict(ckpt['scheduler_state_dict'])
start_epoch = ckpt['epoch']
```

注意：加载时**不能**把整个字典直接传给 `load_state_dict`（会报 unexpected keys），必须取 `['model_state_dict']`。

## 11. 常见报错对照

| 报错 | 原因 | 处理 |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'torch'` | 解释器是 conda base，不是 `pytorch_env` | 激活环境或改用 `envs\pytorch_env\python.exe` |
| `ModuleNotFoundError: No module named 'model'` | import 的文件名和实际文件不一致 | 改 import 或改文件名 |
| `IndentationError: unexpected indent` | Code Runner 运行选区，临时文件里只有缩进的函数体 | 跑完整脚本，不要对选区 Run Code |
| `Can't pickle local object` | dataset 类定义在函数内部 | 类移到模块顶层 |
| `TypeError: step() missing 1 required positional argument` | `GradScaler.step` 忘了传 optimizer | `scaler.step(optimizer)` |
| `Unexpected key(s) in state_dict` | 把 checkpoint 字典整体当 state_dict | 取 `ckpt['model_state_dict']` |
| 训练开始前长时间 0% | cudnn benchmark + 4 个 worker 反序列化数据集 | 等 30~60 秒，属正常热身 |
| 验证集准确率剧烈抖动 | lr 偏大 + batch 偏小，权重每轮移动大 | 后期 lr 降低会稳定，或调整 lr/batch |

相关：[ResNet18 结构笔记](resnet18-notes.md) · [CIFAR-10 实验记录](cifar10-experiments.md) · [MNIST 原理笔记](mnist-notes.md) · [踩坑记录](mnist-troubleshooting.md)
