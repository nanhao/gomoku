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
│   ├── train_data/   # 训练样本数据
│   └── merge_samples/# 合并后的训练样本
├── tests/            # 测试脚本
├── design/           # 设计文档
└── temp/             # 临时分析脚本等
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch
- tkinter（GUI 依赖）

### 安装依赖

```bash
pip install torch matplotlib numpy
```

---

## 命令速查表

### 启动游戏

| 命令 | 说明 |
|------|------|
| `python gui/main.py` | 启动图形界面（推荐） |
| `python cli/main.py` | 启动命令行界面 |

GUI 支持三种游戏模式：
- **人人对战（PvP）**：两名玩家轮流落子
- **人机对战（PvE）**：玩家选择执黑或执白，对阵 AI
- **AI 内战（AI vs AI）**：选择双方 AI 和局数，自动对战并统计胜率

GUI 还支持：存档加载、AI 对战回放（含热力图对比）、游戏存档列表。

### AI 对战

```bash
python ai/battle.py --ai1 <AI名称> --ai2 <AI名称> [--games <局数>]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--ai1` | 第一方 AI 名称 | 必填 |
| `--ai2` | 第二方 AI 名称 | 必填 |
| `--games` | 对战局数 | 10 |

**可用 AI：** `random` / `fc_v1` / `cnn_v1` / `cnn_v2` / `cnn_v3`

```bash
# 示例
python ai/battle.py --ai1 fc_v1 --ai2 random --games 50
```

### 模型训练

#### 自我对弈生成数据（不训练）

```bash
python ai/train_main.py --model <模型名称> --self-play --games <对局数> [--rate <衰减率>] [--merge-output <合并文件路径>]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--self-play` | 仅生成训练数据（自我对弈+评价） | - |
| `--games` | 对局数 | 100 |
| `--rate` | 奖励衰减率 | 0.9 |
| `--merge-output` | 对局完成后自动合并样本到指定文件 | 无（不合并） |

```bash
# 示例
python ai/train_main.py --model cnn_v1 --self-play --games 100
python ai/train_main.py --model cnn_v1 --self-play --games 100 --merge-output ai/train_data/merge_cnn_v1.json
```

> **提示**：运行结束后会输出训练数据目录路径，可用 `--train-from-dir` 继续训练。

#### 从已有数据训练

```bash
python ai/train_main.py --model <模型名称> --train-from-dir <数据目录或文件> [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--train-from-dir` | 训练数据目录或 `.json` 文件路径 | 必填 |
| `--epochs` | 训练迭代数 | 1000 |
| `--batch-size` | 每批次样本数（启用 mini-batch） | 无（全量更新） |
| `--shuffle` | 每轮打乱数据顺序（mini-batch 有效） | False |
| `--val-ratio` | 验证集比例，0 表示不分割，推荐 0.2 | 0.0 |
| `--lr-scheduler` | 学习率调度：`step` / `plateau` | 无（固定学习率） |

> 指定目录时自动发现 `merge_*.json` 合并文件和子目录下的 `sample.json`；指定 `.json` 文件时直接加载。

```bash
# 示例：从目录训练
python ai/train_main.py --model cnn_v1 --train-from-dir ai/train_data/cnn_v1_self_play_1 --epochs 1000

python ai/train_main.py --model cnn_v2 --train-from-dir ai/merge_samples --epochs 1000 --batch-size 256 --shuffle
# 示例：从合并文件训练
python ai/train_main.py --model cnn_v1 --train-from-dir ai/merge_samples/merge_cnn_v1.json --epochs 1000

# 示例：mini-batch 训练
python ai/train_main.py --model cnn_v3 --train-from-dir ai/merge_samples/chong4.json --epochs 200 --batch-size 4096 --shuffle 
```

> **注意**：`--cycle`、`--self-play`、`--train-from-dir` 三个模式互斥，只能选其一。

#### 训练周期（自我对弈训练）

```bash
python ai/train_main.py --model <模型名称> --cycle [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型名称（fc_v1/cnn_v1/cnn_v2/cnn_v3） | fc_v1 |
| `--cycle` | 启用训练周期模式 | - |
| `--rounds` | 训练轮数 | 10 |
| `--games` | 每轮对局数 | 100 |
| `--epochs` | 每轮训练迭代数 | 1000 |
| `--rate` | 奖励衰减率 | 0.9 |
| `--fresh` | 从新模型开始 | - |
| `--batch-size` | 每批次样本数（启用 mini-batch） | 无（全量更新） |
| `--shuffle` | 每轮打乱数据顺序（mini-batch 有效） | False |
| `--val-ratio` | 验证集比例，0 表示不分割，推荐 0.2 | 0.0 |
| `--lr-scheduler` | 学习率调度：`step` / `plateau` | 无（固定学习率） |
| `--lr-step-size` | StepLR 衰减间隔（epoch） | 500 |
| `--lr-gamma` | StepLR 衰减系数 | 0.9 |
| `--lr-patience` | ReduceLROnPlateau 耐心值 | 100 |
| `--lr-factor` | ReduceLROnPlateau 衰减系数 | 0.5 |

> **注意**：使用 `--batch-size` 启用 mini-batch 训练时，系统会自动生成**密集 loss log**，
> 记录每个 batch 的 loss 和梯度大小，并在曲线图上以双 Y 轴形式叠加展示（Loss + Gradient Norm）。

```bash
# 示例
python ai/train_main.py --model fc_v1 --cycle              # 默认参数（全量更新）
python ai/train_main.py --model cnn_v1 --cycle             # CNN v1 训练
python ai/train_main.py --model cnn_v2 --cycle             # CNN v2 轻量化训练
python ai/train_main.py --model fc_v1 --cycle --rounds 50 --games 50  # 自定义
python ai/train_main.py --model fc_v1 --cycle --fresh      # 从新模型开始
python ai/train_main.py --model cnn_v1 --cycle --batch-size 32 --shuffle  # mini-batch 训练
python ai/train_main.py --model cnn_v2 --cycle --batch-size 32 --shuffle  # CNN v2 mini-batch 训练

# 使用学习率衰减（推荐后期训练使用）
python ai/train_main.py --model cnn_v1 --cycle --batch-size 32 --lr-scheduler step --lr-step-size 300
python ai/train_main.py --model cnn_v1 --cycle --batch-size 32 --lr-scheduler plateau --lr-patience 50
python ai/train_main.py --model cnn_v2 --cycle --batch-size 32 --lr-scheduler plateau --lr-patience 50

# 常用
python ai/train_main.py --model cnn_v2 --cycle --batch-size 128 --shuffle --rounds 1 --games 600 --epochs 2000
```

### 样本合并

将分散的 `sample.json` 合并为单个大文件，减少训练时的 I/O 开销：

```bash
python ai/merge_samples.py <目录1> [目录2] ... -o <输出文件路径>
```

| 参数 | 说明 |
|------|------|
| 位置参数 | 一个或多个包含 sample.json 的数据目录 |
| `-o` / `--output` | 合并后的输出文件路径（必需） |

```bash
# 示例：合并单个目录
python ai/merge_samples.py ai/train_data/cnn_v2_self_play_1 -o ai/merge_samples/merge_cnn_v2.json

# 示例：合并多个目录
python ai/merge_samples.py ai/train_data/cnn_v2_cycle_1_1 ai/train_data/cnn_v2_cycle_1_2 -o ai/merge_samples/merge_cnn_v2_all.json
```

### 样本检视

```bash
python ai/models/fc_v1/inspect_main.py --sample <sample.json路径> [--model-path <模型路径>]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--sample` | sample.json 文件路径 | 必填 |
| `--model-path` | 模型文件路径 | 默认加载 `ai/models/fc_v1/fc_v1.pth` |

### LR Range Test

学习率范围测试工具，用于确定最优学习率区间。每个 round 使用一个固定 lr，训练 `epochs` 个 step 后记录平均 loss，lr 从 `min_lr` 线性递增至 `max_lr`。每个 round 开始前重置模型到初始状态，确保 loss 只反映 lr 的影响。

```bash
python ai/lr_range_test.py --model <模型名称> --data <数据路径> [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型名称（cnn_v2） | 必填 |
| `--data` | 训练数据路径（`.json` 文件或目录） | 必填 |
| `--rounds` | lr 递增次数 | 50 |
| `--epochs` | 每个 round 内的训练 step 数 | 100 |
| `--min-lr` | 起始学习率 | 1e-7 |
| `--max-lr` | 终止学习率 | 1e-1 |
| `--output` | 输出图片路径 | `temp/lr_range_test.png` |

```bash
# 示例：对 CNN v2 执行 LR Range Test
python ai/lr_range_test.py --model cnn_v2 --data ai/merge_samples/merge_cnn_v1_100.json

# 示例：自定义 lr 范围和输出路径
python ai/lr_range_test.py --model cnn_v2 --data ai/merge_samples/merge_cnn_v1_100.json --min-lr 1e-6 --max-lr 1e-2 --output temp/my_lr_test.png
```

### 运行测试

项目使用 **pytest** 作为测试框架，支持以下运行方式：

```bash
# 方式1：直接调用 pytest（推荐）
python -m pytest tests/ -v -s

# 方式2：通过兼容性入口运行（已自动带 -s）
python tests/run_tests.py
```

> 覆盖全部 1076 个测试（自动发现，无需手工维护测试列表）
>
> 附加参数示例：
> - `python -m pytest tests/ -q` —— 静默模式
> - `python -m pytest tests/ai/ -v` —— 仅运行 AI 模块测试
> - `python -m pytest tests/ --tb=short` —— 简短错误追溯

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

### CNN v2 模型（轻量化）

- **网络结构**：卷积层(3→64→64→64) + BN → 全连接层(14400→64→225) → Softmax
- **参数量**：约 101 万（CNN v1 的 23%，计算量 18M MACs）
- **训练方式**：与 CNN v1 相同
- **模型文件**：`ai/models/cnn_v2/cnn_v2.pth`

### CNN v3 模型（纯卷积极简）

- **网络结构**：卷积层(3→32→32→32, 无BN) → 1×1卷积(32→1) → Softmax
- **参数量**：约 1.9 万（CNN v2 的 1.9%）
- **特点**：纯卷积，无 FC、无 BN、无 Dropout，1×1 Conv 替代 FC 层
- **训练方式**：与 CNN v1 相同
- **模型文件**：`ai/models/cnn_v3/cnn_v3.pth`

---

## 数据目录

| 目录 | 说明 |
|------|------|
| `game/save/` | 游戏存档 |
| `ai/ai_data/` | AI 对战记录（record.json） |
| `ai/ai_data/loss_logs/` | 训练 loss 日志和曲线图 |
| `ai/train_data/` | 训练样本（sample.json） |
| `ai/merge_samples/` | 合并后的训练样本大文件 |

---

## 许可证

MIT License
 