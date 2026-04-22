# 五子棋 (Gomoku)

一个支持 AI 训练的五子棋游戏项目。

## 项目结构

```
gomoku/
├── cli/              # 命令行界面
├── gui/              # 图形界面
├── game/             # 游戏核心逻辑
├── app/              # 应用层（会话管理、GUI辅助）
├── ai/               # AI 模块
│   ├── ai_data/      # AI 对战记录数据
│   └── train_data/   # 训练样本数据
├── tests/            # 测试脚本
└── design/           # 设计文档
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

## 命令速查表

### 启动游戏

| 命令 | 说明 |
|------|------|
| `python gui/main.py` | 启动图形界面（推荐） |
| `python cli/main.py` | 启动命令行界面 |

### AI 对战

```bash
python ai/battle.py --ai1 <AI名称> --ai2 <AI名称> [--games <局数>]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--ai1` | 第一方 AI 名称 | 必填 |
| `--ai2` | 第二方 AI 名称 | 必填 |
| `--games` | 对战局数 | 10 |

**可用 AI：** `random` / `fc_v1` / `cnn_v1`

```bash
# 示例
python ai/battle.py --ai1 fc_v1 --ai2 random --games 50
```

### 模型训练

#### 单轮次训练（使用已有样本）

```bash
python ai/train_main.py --model <模型名称> --sample-dir <sample目录> [--epochs <轮数>]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型名称（fc_v1/cnn_v1） | `fc_v1` |
| `--sample-dir` | 训练样本目录 | 必填 |
| `--epochs` | 训练迭代数 | 1000 |

```bash
# 示例
python ai/train_main.py --model fc_v1 --sample-dir train_data/random_vs_random_1/1
```

#### 训练周期（自我对弈训练）

```bash
python ai/train_main.py --model <模型名称> --cycle [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型名称（fc_v1/cnn_v1） | fc_v1 |
| `--cycle` | 启用训练周期模式 | - |
| `--rounds` | 训练轮数 | 10 |
| `--games` | 每轮对局数 | 100 |
| `--epochs` | 每轮训练迭代数 | 1000 |
| `--rate` | 奖励衰减率 | 0.9 |
| `--fresh` | 从新模型开始 | - |
| `--batch-size` | 每批次样本数（启用 mini-batch） | 无（全量更新） |
| `--shuffle` | 每轮打乱数据顺序（mini-batch 有效） | False |

```bash
# 示例
python ai/train_main.py --model fc_v1 --cycle              # 默认参数（全量更新）
python ai/train_main.py --model cnn_v1 --cycle             # CNN 训练
python ai/train_main.py --model fc_v1 --cycle --rounds 50 --games 50  # 自定义
python ai/train_main.py --model fc_v1 --cycle --fresh      # 从新模型开始
python ai/train_main.py --model cnn_v1 --cycle --batch-size 32 --shuffle  # mini-batch 训练
```

### 样本检视

```bash
python ai/models/fc_v1/inspect_main.py --sample <sample.json路径> [--model-path <模型路径>]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--sample` | sample.json 文件路径 | 必填 |
| `--model-path` | 模型文件路径 | 默认加载 `ai/models/fc_v1/fc_v1.pth` |

### 运行测试

```bash
python tests/run_tests.py
```

> 覆盖全部 487 个测试（基础测试 212 + AI 测试 275）

---

## CLI 命令一览

启动 CLI 后可用命令：

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

## AI 模型

### Random

随机选择空位落子，作为基准对照。

### FCv1

全连接神经网络模型，通过自我对弈进行强化学习训练。

- **输入**：226 维向量（225 位棋盘状态 + 1 位当前玩家）
- **输出**：225 维概率分布（Softmax）
- **网络结构**：输入层(226) → 隐藏层1-5(各1000, ReLU) → 输出层(225) → Softmax
- **训练方式**：自我对弈 + 奖励反向传播 + 五连追溯奖励
- **模型文件**：`ai/models/fc_v1/fc_v1.pth`

### CNN v1

卷积神经网络模型，通过卷积层提取棋盘空间特征。

- **输入**：15×15×3 张量（黑棋通道、白棋通道、当前玩家通道）
- **输出**：225 维概率分布（Softmax）
- **网络结构**：卷积层(3→32→64→128) → 全连接层(28800→150→225) → Softmax
- **训练方式**：自我对弈 + 奖励反向传播 + 五连追溯奖励
- **模型文件**：`ai/models/cnn_v1/cnn.pth`

---

## 数据目录

| 目录 | 说明 |
|------|------|
| `game/save/` | 游戏存档 |
| `ai/ai_data/` | AI 对战记录（record.json） |
| `ai/train_data/` | 训练样本（sample.json） |

---

## 许可证

MIT License
