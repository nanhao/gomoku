# 五子棋 (Gomoku)

一个支持 AI 训练的五子棋游戏项目。

## 项目结构

```
gomoku/
├── cli/              # 命令行界面
├── gui/              # 图形界面
├── game/             # 游戏核心逻辑
├── ai/               # AI 模块
├── tests/            # 测试脚本
├── ai_data/          # AI 对战记录数据
└── train_data/       # 训练样本数据
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch
- tkinter（GUI 依赖）

### 安装依赖

```bash
pip install torch
```

---

## 脚本说明

### 1. 游戏入口

#### GUI 图形界面（推荐）

```bash
python gui/main.py
```

启动图形界面，支持三种模式：
- **人人对战 (PvP)** - 双人对战
- **人机对战 (PvE)** - 玩家 vs AI，可选 AI 难度
- **AI 内战 (AI vs AI)** - 观看 AI 自动对战

#### CLI 命令行界面

```bash
python cli/main.py
```

启动交互式命令行游戏。

**命令列表：**

| 命令 | 说明 |
|------|------|
| `new [名称]` | 新建游戏 |
| `put <x> <y>` | 落子（坐标范围 0-14） |
| `list` | 列出所有存档 |
| `print` | 显示当前棋盘 |
| `show <名称>` | 显示完整存档信息 |
| `short <名称>` | 显示简要存档信息 |
| `help` | 显示帮助 |
| `quit / exit` | 退出程序 |

---

### 2. AI 对战脚本

```bash
python ai/battle.py --ai1 <AI名称> --ai2 <AI名称> --games <局数>
```

**参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--ai1` | 第一方 AI 名称 | 必填 |
| `--ai2` | 第二方 AI 名称 | 必填 |
| `--games` | 对战局数 | 10 |

**可用 AI：**
- `random` - 随机落子
- `fc_v1` - 全连接神经网络模型

**示例：**

```bash
# random vs random，10 局
python ai/battle.py --ai1 random --ai2 random --games 10

# fc_v1 vs random，50 局
python ai/battle.py --ai1 fc_v1 --ai2 random --games 50
```

对局记录保存在 `ai_data/` 目录下。

---

### 3. 模型训练脚本

#### 单轮次训练

```bash
python ai/fc_v1/train.py --sample-dir <sample目录> [--epochs <轮数>]
```

使用已有的训练样本训练模型。

**示例：**

```bash
# 使用指定样本训练（默认 1000 轮）
python ai/fc_v1/train.py --sample-dir train_data/random_vs_random_1/1

# 自定义训练轮数
python ai/fc_v1/train.py --sample-dir train_data/random_vs_random_1/1 --epochs 20
```

#### 训练周期（自我对弈训练）

```bash
python ai/fc_v1/train.py --cycle [--rounds <轮数>] [--games <每轮局数>] [--rate <衰减率>]
```

让模型自我对弈并训练。

**参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--cycle` | 启用训练周期模式 | - |
| `--rounds` | 训练轮数 | 10 |
| `--games` | 每轮对局数 | 100 |
| `--epochs` | 每轮训练迭代数 | 1000 |
| `--rate` | 奖励衰减率 | 0.9 |
| `--fresh` | 从新模型开始（不加载已有模型） | - |

**示例：**

```bash
# 启动训练周期（默认参数）
python ai/fc_v1/train.py --cycle

# 自定义参数
python ai/fc_v1/train.py --cycle --rounds 50 --games 50

# 从新模型开始
python ai/fc_v1/train.py --cycle --fresh
```

模型保存在 `ai/fc_v1/fc_v1.pth`。

---

### 4. 样本检视脚本

```bash
python ai/fc_v1/inspect_main.py --sample <sample.json路径> [--model-path <模型路径>]
```

检视模型在指定样本上的表现，逐步输出模型预测与目标值的对比。

**参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--sample` | sample.json 文件路径 | 必填 |
| `--model-path` | 模型文件路径 | ai/fc_v1/fc_v1.pth |

**示例：**

```bash
python ai/fc_v1/inspect_main.py --sample train_data/fc_v1_cycle_1_4/1/sample.json
```

---

### 5. 测试脚本

```bash
python tests/run_tests.py
```

运行所有单元测试（无需 pytest）。

---

## 数据目录

| 目录 | 说明 |
|------|------|
| `game/save/` | 游戏存档 |
| `ai_data/` | AI 对战记录（record.json） |
| `train_data/` | 训练样本（sample.json） |

---

## AI 模型

### Random

随机选择空位落子，作为基准对照。

### FCv1

全连接神经网络模型，通过自我对弈进行强化学习训练。

- **输入**：226 维向量（225 位棋盘状态 + 1 位当前玩家）
- **输出**：225 维 reward 列表
- **训练方式**：自我对弈 + 奖励反向传播

---

## 许可证

MIT License
