# PyTorch 环境配置笔记

> 导航：[仓库首页](../README.md) · [学习路线](learning-roadmap.md) · [环境自检脚本](../tools/check_env.py)

> 平台：Windows ｜ Anaconda ｜ RTX 5060 Laptop ｜ VSCode
> 记录时间：2026-10-08 ~ 2026-10-12
> 目的：换电脑或重建环境时可直接照抄

---

## 一、最终成果

`test_gpu.py` 的自检输出，四项全绿即代表环境正常：

```
interpreter : C:\Users\<用户名>\anaconda3\envs\pytorch_env\python.exe
python      : 3.10.22
torch       : 2.11.0+cu128
cuda ready  : True
gpu         : NVIDIA GeForce RTX 5060 Laptop GPU
capability  : (12, 0)
cuda runtime: 12.8
```

其中 **`(12, 0)`** 是最关键的一行：它代表显卡算力为 `sm_120`，说明装的 PyTorch 里确实带了这张卡的 kernel。

---

## 二、先搞懂三个概念

### 2.1 PyTorch 是一个"包"

PyTorch 就是第三方 Python 库，和 `numpy`、`requests` 同类。特殊之处只有两点：**体积大**（GPU 版 2.8GB）、**能调用显卡**。

### 2.2 虚拟环境 = 给项目单独修一间房子

Python 默认把所有第三方包装进同一个全局目录，于是会出现：

- 项目 A 要 `torch 2.0`、项目 B 要 `torch 2.4` → 装在一起必然打架
- 手滑升级某个包，旧项目直接跑不起来

虚拟环境的做法是**复制一套独立的 Python + 独立的包目录**，各项目互不干扰。

### 2.3 决定成败的不是代码，是"哪个 python 在跑它"

```
你的代码 .py  →  由哪个 python.exe 运行  →  能 import 哪些包
```

`import torch` 失败，几乎永远是"用错了 python.exe"，而不是代码有问题。

**激活环境的本质**，就是把"谁运行它"切换到目标环境里的 python.exe。

---

## 三、本机环境信息

| 项目 | 值 |
|---|---|
| conda 版本 | 26.5.3 |
| Anaconda 路径 | `C:\Users\<用户名>\anaconda3` |
| 目标环境 | `pytorch_env`（`C:\Users\<用户名>\anaconda3\envs\pytorch_env`） |
| 环境 Python | 3.10.22 |
| PyTorch | 2.11.0+cu128（torchvision 0.20.1 系列 / torchaudio 配套） |
| 显卡 | NVIDIA GeForce RTX 5060 Laptop GPU（Blackwell，sm_120） |
| 显卡驱动 | 617.14（CUDA UMD 13.4） |
| 项目目录 | `"C:\Users\<用户名>\Desktop\pytorch" |
| 执行策略 | PowerShell `CurrentUser = RemoteSigned` |

---

## 四、安装步骤（可复用）

### 步骤 1：创建环境

```cmd
conda create -n pytorch_env python=3.10
```

> 若要装到 D 盘（省 C 盘空间），先配 `conda config --add envs_dirs D:\conda_envs` 再建。
> **路径务必纯英文，不要有中文和空格。**

### 步骤 2：安装 PyTorch（关键一步）

```cmd
conda activate pytorch_env

python -m pip uninstall -y torch torchvision torchaudio

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

两个要点，少一个都不行：

1. **必须是 `--index-url`，不能是 `--extra-index-url`**
   PyPI 上的 torch 默认是 **CPU 版**。用 `--index-url` 强制只从 PyTorch 官方源拉，才能拿到 GPU 版。
2. **先卸载再装**
   不卸载的话 pip 认为"已经有了"，会直接跳过。

> 下载量约 **2.8GB**（含 CUDA 运行时、cuDNN），慢是正常的，中途断了可重跑（pip 有缓存）。

### 步骤 3：验证

```cmd
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
```

期望输出：

```
2.11.0+cu128
True
NVIDIA GeForce RTX 5060 Laptop GPU
(12, 0)
```

---

## 五、为什么必须装 cu128 —— RTX 50 系的特殊性

**这是整个配置过程中最大的坑。**

RTX 50 系（5060/5070/5080/5090）是 **Blackwell 架构，算力 `sm_120`**。而 PyTorch 对它的支持是分水岭式的：

| PyTorch 版本 | 支持的最高算力 | 能否用 RTX 5060 |
|---|---|---|
| ≤ 2.6 | sm_90 | ❌ |
| 2.5.1（最初误装的版本） | sm_90 | ❌ |
| **≥ 2.7** | **sm_120** | ✅ |

**症状**：装旧版后 `torch.cuda.is_available()` 返回 `False`，或者报 `no kernel image is available for execution on the device`。

**结论**：RTX 50 系必须同时满足两条 —— **PyTorch ≥ 2.7 且 CUDA ≥ 12.8（cu128）**。

---

## 六、VSCode 配置

### 6.1 全局配置

`C:\Users\<用户名>\AppData\Roaming\Code\User\settings.json`

```json
"python.defaultInterpreterPath": "C:\\Users\\MR\\anaconda3\\envs\\pytorch_env\\python.exe"
```

> 原值指向 `C:\Program Files\Python310\python.exe` —— 一个**空的** Python（只有 pip 和 setuptools），必须改掉。

### 6.2 项目配置

`<仓库目录>\.vscode\settings.json`（项目级优先级更高，最稳）

```json
{
  "python.defaultInterpreterPath": "C:\\Users\\MR\\anaconda3\\envs\\pytorch_env\\python.exe",
  "python.terminal.activateEnvironment": true,
  "code-runner.executorMap": {
    "python": "C:\\Users\\MR\\anaconda3\\envs\\pytorch_env\\python.exe -u"
  }
}
```

**`code-runner.executorMap` 这个值有两个讲究：**

1. **路径不要加引号**（路径本身无空格时）。加了引号后命令会以引号开头，**PowerShell 会报「意外的标记」**，必须改用 `&` 调用运算符才行。
2. **用绝对路径而不是 `conda activate`**。这样不管当前终端停在哪个环境都能跑对，比激活更可靠。

### 6.3 两个运行按钮的区别

| 按钮 | 来源 | 是否遵守"选中的解释器" |
|---|---|---|
| `Run Python File` | VSCode 官方 Python 扩展 | ✅ 遵守 |
| `Run Code` | Code Runner 扩展 | ❌ 用 PATH 里的裸 `python`（需额外配置） |

鼠标悬停按钮可看到名字。**推荐用 `Run Python File`。**

---

## 七、踩坑记录

| # | 坑 | 症状 | 解法 |
|---|---|---|---|
| 1 | PyPI 默认发 CPU 版 | `cuda.is_available() = False`，版本号无 `+cu` 后缀 | 加 `--index-url https://download.pytorch.org/whl/cu128` |
| 2 | PyTorch < 2.7 不支持 sm_120 | RTX 50 系用不了 GPU | 升到 ≥ 2.7 并配 cu128 |
| 3 | 机器上多套 Python 并存 | 裸敲 `python` 命中错误的解释器 | 显式指定绝对路径，或先激活环境 |
| 4 | `conda activate` 报 `Run 'conda init' before 'conda activate'` | cmd / PowerShell 各自未初始化 | `conda init cmd.exe` ／ `conda init powershell` |
| 5 | PowerShell 执行策略为 `Restricted` | 配置文件不加载，红字 `禁止运行脚本` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| 6 | Code Runner 无视 VSCode 解释器 | 点运行报 `No module named 'torch'` | 配 `code-runner.executorMap` |
| 7 | 配置里给路径加了引号 | PowerShell 报 `意外的标记 -u` | 去掉引号（路径无空格时） |
| 8 | `auto_activate_base` 默认为 True | 新终端停在 `(base)`，而 base 里没有 torch | `conda config --set auto_activate_base false` |
| 9 | 中文 `print` 输出乱码 | UTF-8 字节被 GBK 解码 | 脚本改用纯 ASCII 输出 |

### 关于第 4/5 条的连锁关系

```
执行 conda init powershell
   → 写入 Documents\WindowsPowerShell\profile.ps1
   → 但执行策略 Restricted 会拒绝加载
   → 终端一启动就报红字，且 conda activate 依然不可用
   → 必须先改执行策略为 RemoteSigned，profile 才能生效
```

**注意**：`conda init` 对 cmd 和 PowerShell 是**分开的**，改了一个不影响另一个。
另外，`profile` 只在**会话启动时**加载，改完必须**重开终端**才生效。

---

## 八、日常使用方式

### 方式 1：VSCode 点运行（最推荐）

打开 `.py` 文件 → 点右上角 ▷（`Run Python File`）→ 完成。

**配置一次，永久生效**，不用激活、不用开命令行。

### 方式 2：终端手动运行

```powershell
conda activate pytorch_env
python -u your_script.py
```

提示符变为 `(pytorch_env)` 才说明切换成功。

或者不激活，直接用绝对路径：

```powershell
C:\Users\<用户名>\anaconda3\envs\pytorch_env\python.exe your_script.py
```

### 方式 3：Jupyter Notebook（适合学习和试代码）

```powershell
conda activate pytorch_env
python -m pip install jupyter
jupyter notebook
```

---

## 九、报错速查表

| 报错 | 原因 | 解法 |
|---|---|---|
| `No module named 'torch'` | 用错了 python.exe | 激活环境，或指定绝对路径 |
| `cuda.is_available() = False` | 装的是 CPU 版 | 用 `--index-url` 重装 cu128 版 |
| `no kernel image is available...` | PyTorch 版本太旧，无 sm_120 kernel | 升到 ≥ 2.7 |
| `Run 'conda init' before 'conda activate'` | 该 shell 未初始化 | `conda init <shell>` + 重开终端 |
| `因为在此系统上禁止运行脚本` | 执行策略 Restricted | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `意外的标记 -u` | PowerShell 里命令以引号开头 | 去掉引号或用 `&` 开头 |
| `ModuleNotFoundError` 但明明装过 | 终端停在 `(base)` 或其他环境 | `conda activate pytorch_env` |

---

## 十、附录：本机的多套 Python

配置过程中发现机器上共 **6 套** Python，这是"裸敲 `python` 指错"的根源：

| 路径 | 版本 | 说明 |
|---|---|---|
| `C:\Users\<用户名>\.workbuddy\binaries\python\versions\3.13.12` | 3.13.14 | WorkBuddy 自带，**勿动** |
| `C:\Program Files\Python310` | 3.10.5 | 空环境（VSCode 原指向它） |
| `C:\Users\<用户名>\AppData\Local\Programs\Python\Python314` | 3.14.7 | PATH 里 `python` 默认命中这个 |
| `C:\Users\<用户名>\AppData\Local\Programs\Python\Python38` | 3.8.5 | 旧版本 |
| `C:\msys64\ucrt64` | 3.12.12 | MSYS2 工具链的一部分，**勿动** |
| `C:\Users\<用户名>\anaconda3` | base 3.14.6 | conda 本体 |
| └ `envs\pytorch_env` | 3.10.22 | ✅ **PyTorch 装在这里** |

**教训**：多套解释器并存时，永远不要依赖裸 `python`，要么先激活环境，要么写绝对路径。

---

## 十一、一句话总结

> 装了 GPU 版 PyTorch 之后，**唯一要记住的事**是：让代码由 `C:\Users\<用户名>\anaconda3\envs\pytorch_env\python.exe` 来运行。
> VSCode 的配置已经把这件事实现了，日常只需**点一下运行按钮**。
