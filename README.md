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
├── tools/            # 验收与辅助工具
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
python ai/battle.py --ai1 <AI指定> --ai2 <AI指定> [--games <局数>]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--ai1` | 第一方 AI 指定（如 random, fc_v1, cnn_v3.1） | 必填 |
| `--ai2` | 第二方 AI 指定（如 random, fc_v1, cnn_v3.1） | 必填 |
| `--games` | 对战局数 | 10 |

**可用 AI：** `random` / `fc_v1` / `cnn_v1` / `cnn_v2` / `cnn_v3.1` / `cnn_v3.2` ...

> **多代模型说明**：`cnn_v3` 支持多代模型，使用 `模型名.代数` 格式指定（如 `cnn_v3.1`、`cnn_v3.2`），不同代数的模型权重存储在 `ai/models/cnn_v3/gen{i}/` 目录下。非多代模型（如 `fc_v1`、`cnn_v1`）直接使用模型名称。

```bash
# 示例
python ai/battle.py --ai1 fc_v1 --ai2 random --games 50
python ai/battle.py --ai1 cnn_v3.1 --ai2 cnn_v3.2 --games 50
```

### 模型训练

#### 生成训练样本（对弈 + sample_creator）

```bash
python ai/create_samples.py --model <模型名称> --output <输出文件路径> [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型指定（如 cnn_v3.1、fc_v1） | 必填 |
| `--output` | 输出 `.json` 文件路径 | 必填 |
| `--mode` | sample_creator 模式，可选 `winning_move` / `winner_path` | `winning_move` |
| `--games` | 对局数 | 100 |
| `--no-augment` | 禁用数据增强（默认启用 D4 群 8 变换；`winner_path` 模式下强制不增强，该开关无作用） | - |
| `--no-temp-sampling` | 关闭温度采样，全程 argmax 贪婪落子 | - |
| `--temperature` | 温度采样温度 T（越大越探索） | 1.0 |
| `--temp-moves` | 前 N 步启用温度采样，之后切回 argmax | 12 |
| `--temp-side` | 温度采样作用方：`black` / `white` / `both` / `random` | `both` |
| `--temp-seed` | 温度采样随机种子（便于复现） | None |

**模式说明**：
- `winning_move`：仅取每局最后一步（制胜那步）作为样本，默认启用 D4 8 变换增强。
- `winner_path`：取整局中**获胜方的每一步**作为样本，让模型拟合获胜方整条路径；由于每局已产出多条样本，**强制不做 D4 增强**以避免数据量失控。每步的 output 采用**软标签**（基于模型当步 softmax 输出，将落子位最小增强为严格 top1 后归一化），而非 one-hot，以保留模型分布、避免信号过于刚性。

> **对局级过滤**：当 `--temp-side` 为单方模式（`black`/`white`/`random`）时，仅保留采样方获胜的对局用于样本生成，贪婪方获胜的对局将被丢弃。`--temp-side both`（默认）和 `--no-temp-sampling` 模式下不做过滤。运行结束会输出过滤统计（保留/过滤对局数）。

```bash
# 示例
# winning_move 模式（默认）
python ai/create_samples.py --model cnn_v3.1 --output ai/train_data/win_samples.json
python ai/create_samples.py --model cnn_v3.1 --games 200 --output ai/train_data/win_samples.json
python ai/create_samples.py --model cnn_v3.1 --no-augment --output ai/train_data/win_samples.json
python ai/create_samples.py --model random  --output ai/merge_samples/winning/win_samples2.json --games 5000

# winner_path 模式：获胜方整条路径
python ai/create_samples.py --model cnn_v3.1 --mode winner_path --games 200 --output ai/train_data/winner_path_samples.json
```

> **提示**：运行结束后可用 `--train-from-dir` 加载生成的 `.json` 文件继续训练。

#### 样本可视化验收

基于 Flask 的 Web 工具，可视化验收 `create_samples.py` 生成的训练样本（热力图对比 + 对局浏览）。

```bash
# 启动验收工具
python -m tools.create_samples_tool.app
# 浏览器访问 http://127.0.0.1:5000
```

> 详细使用手册见 [tools/create_samples_tool/README.md](tools/create_samples_tool/README.md)。

#### 从已有数据训练

```bash
python ai/train_main.py --model <模型指定> --train-from-dir <数据目录或文件> [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--train-from-dir` | 训练数据目录或 `.json` 文件路径 | 必填 |
| `--epochs` | 训练迭代数 | 1000 |
| `--batch-size` | 每批次样本数（启用 mini-batch） | 无（全量更新） |
| `--shuffle` | 每轮打乱数据顺序（mini-batch 有效） | False |
| `--val-ratio` | 验证集比例，0 表示不分割，推荐 0.2 | 0.0 |
| `--lr-scheduler` | 学习率调度：`step` / `plateau` | 无（固定学习率） |

> 指定目录时递归加载其下所有 `.json` 文件（不限文件名前缀）；指定 `.json` 文件时直接加载。

```bash
# 示例：从目录训练
python ai/train_main.py --model cnn_v1 --train-from-dir ai/train_data/cnn_v1_self_play_1 --epochs 1000

python ai/train_main.py --model cnn_v2 --train-from-dir ai/merge_samples --epochs 1000 --batch-size 256 --shuffle
# 示例：从合并文件训练
python ai/train_main.py --model cnn_v1 --train-from-dir ai/merge_samples/merge_cnn_v1.json --epochs 1000

# 示例：mini-batch 训练
python ai/train_main.py --model cnn_v3.2 --fresh --train-from-dir ai/merge_samples/v2 --epochs 1000 --batch-size 4096 --shuffle --val-ratio 0.2
```

> **注意**：`--cycle` 与 `--train-from-dir` 两个模式互斥，只能选其一。

#### 训练周期（自我对弈训练）

```bash
python ai/train_main.py --model <模型指定> --cycle [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型指定（如 fc_v1、cnn_v3.2） | fc_v1 |
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
| `--no-temp-sampling` | 关闭温度采样，自对弈全程 argmax | - |
| `--temperature` | 温度采样温度 T | 1.0 |
| `--temp-moves` | 前 N 步启用温度采样 | 12 |
| `--temp-side` | 温度采样作用方：`black` / `white` / `both` / `random` | `both` |
| `--temp-seed` | 温度采样随机种子（便于复现） | None |

> **温度采样**：`--cycle` 模式**默认启用**温度采样——前 12 步按模型 softmax 分布以 T=1.0 随机落子，之后切回 argmax。目的是让自对弈数据具有开局多样性，避免陷入固定棋谱。加 `--no-temp-sampling` 可关闭。详见 [design/move_sampler_design.md](design/move_sampler_design.md)。

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

### 棋型样本生成

按指定棋型合成训练样本：每个样本包含一个精心构造的 15×15 棋盘，以及通过 `RewardCalculator + masked_softmax` 生成的 label。生成后自动用 Hit@N 等指标做 label 质量体检。

支持 6 种棋型：
- **进攻棋型**（己方落子形成威胁）：`chong4` / `huo4` / `huo3` / `huo2`
- **防守棋型**（对手即将形成威胁，必须抢占）：`defend_chong4` / `defend_huo3`

```bash
python -m ai.sample_generator --shape <棋型> --num-samples <数量> --output <输出路径> [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--shape` | 目标棋型（见上方 6 选一） | 必填 |
| `--num-samples` | 目标样本数 | 必填 |
| `--output` | 输出 `.json` 文件路径 | 必填 |
| `--seed` | 随机种子（相同 seed + 相同参数可复现一致批次） | 42 |
| `--max-extra` | 每样本最多干扰子数 | 4 |
| `--print-failures` | 验证未命中样本打印条数（0=不打印） | 5 |

```bash
# 示例：生成 10000 个冲四样本
python -m ai.sample_generator --shape chong4 --num-samples 10000 --output ai/merge_samples/chong4.json

# 示例：生成活三样本（干扰子更多）
python -m ai.sample_generator --shape huo3 --num-samples 5000 --max-extra 8 --output ai/merge_samples/huo3.json

# 示例：生成防守冲四样本
python -m ai.sample_generator --shape defend_chong4 --num-samples 5000 --output ai/merge_samples/defend_chong4.json
```

> **验证报告**：运行结束会打印 Hit@N / Hit@N+1 / Hit@N+2 命中率、`attack_prob_sum`（进攻位概率和）、`prob_gap`（进攻位与其他位的区分度）等指标，方便判断 label 质量是否达标。

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

### 样本可视化

将样本可视化为 PNG（棋盘 + 棋子 + attack 标记 + 热力图）。兼容新格式（`sample_generator` 输出，含 `_attack_positions`/`_shape`/`mask`）与旧 `winning_move` 格式：

```bash
python -m ai.sample_visualizer --sample <样本文件> [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--sample` | 样本文件路径（.json） | 必填 |
| `--count` | 随机抽样数量 | 6 |
| `--indices` | 指定样本索引（逗号分隔，如 `0,5,10`），优先于 `--count` | 无 |
| `--seed` | 抽样随机种子（相同 seed 可复现抽样） | 无（每次随机） |
| `--output` | 输出图片路径 | `temp/sample_visualizer.png` |
| `--no-heatmap` | 关闭热力图叠加（只画棋盘 + 棋子 + attack 标记） | 默认叠加 |
| `--heatmap-mode` | 归一化模式 `log` / `max` / `linear`（`log` 专为 softmax 极端分布设计） | `log` |
| `--heatmap-alpha` | 热力图透明度（0~1） | 0.75 |

> `--no-heatmap` 指定时，`--heatmap-mode` 和 `--heatmap-alpha` 被忽略。

```bash
# 示例：随机抽取 6 个样本可视化（默认叠加热力图）
python -m ai.sample_visualizer --sample ai/merge_samples/chong4.json

# 示例：指定索引 + 自定义输出
python -m ai.sample_visualizer --sample data.json --indices 0,5,10 --output temp/my_viz.png

# 示例：关闭热力图 + 固定种子复现
python -m ai.sample_visualizer --sample data.json --count 3 --no-heatmap --seed 42
```

> 热力图核心算法位于独立可复用的 `visualization/heatmap_color.py`（无 UI 依赖，默认白→黄→橙→红 4 色色板 + log 归一化），同时被 `ai/sample_visualizer.py`（matplotlib）和 `gui/components/heatmap.py`（tkinter 回放界面）共用。

### LR Range Test

学习率范围测试工具，用于确定最优学习率区间。每个 round 使用一个固定 lr，训练 `epochs` 个 step 后记录平均 loss，lr 从 `min_lr` 线性递增至 `max_lr`。每个 round 开始前重置模型到初始状态，确保 loss 只反映 lr 的影响。

```bash
python ai/lr_range_test.py --model <模型指定> --data <数据路径> [选项]
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

> 覆盖全部 1463 个测试（自动发现，无需手工维护测试列表）
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
- **模型文件**：`ai/models/cnn_v3/gen{i}/cnn_v3.pth`（多代模型，按代数存储）
- **多代支持**：使用 `cnn_v3.代数` 格式指定（如 `cnn_v3.1`），权重按代数隔离存储在 `gen{i}/` 子目录下

---

## 数据目录

| 目录 | 说明 |
|------|------|
| `game/save/` | 游戏存档 |
| `ai/ai_data/` | AI 对战记录（record.json） |
| `ai/ai_data/loss_logs/` | 训练 loss 日志和曲线图 |
| `ai/train_data/` | 训练样本（sample.json） |
| `ai/merge_samples/` | 合并后的训练样本大文件 |
| `tools/create_samples_tool/` | create_samples 验收工具 |

---

## 许可证

MIT License
 