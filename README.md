# 五子棋 (Gomoku)

一个支持 AI 训练的五子棋游戏项目。

## 项目结构

```
gomoku/
├── src/                  # 源代码
│   ├── cli/              # 命令行界面
│   ├── gui/              # 图形界面
│   ├── game/             # 游戏核心逻辑
│   ├── app/              # 应用层（会话管理、GUI辅助）
│   ├── ai/               # AI 模块
│   │   ├── ai_data/      # AI 对战记录数据
│   │   ├── train_data/   # 训练样本数据
│   │   └── merge_samples/# 合并后的训练样本
│   ├── visualization/    # 热力图等可视化算法（无 UI 依赖）
│   └── tools/            # 验收与辅助工具
├── tests/                # 测试脚本
├── cpp/                  # C++ MCTS 模块（高性能多线程搜索器）
│   ├── src/              # C++ 源代码
│   │   ├── board_state.* # 棋盘状态与胜负判定
│   │   ├── mcts_node.*   # MCTS 搜索树节点
│   │   ├── neural_net.*  # LibTorch GPU 异步推理引擎
│   │   ├── mcts_searcher.* # 多线程 MCTS 搜索器
│   │   └── bindings.cpp  # pybind11 Python 桥接
│   ├── tests/            # C++ 单元测试和集成测试
│   ├── python/           # Python 适配层
│   └── CMakeLists.txt    # CMake 构建配置
├── design/               # 设计文档
└── temp/                 # 临时分析脚本等
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
| `python src/gui/main.py` | 启动图形界面（推荐） |
| `python src/cli/main.py` | 启动命令行界面 |

GUI 支持三种游戏模式：
- **人人对战（PvP）**：两名玩家轮流落子
- **人机对战（PvE）**：玩家选择执黑或执白，对阵 AI
- **AI 内战（AI vs AI）**：选择双方 AI 和局数，自动对战并统计胜率

GUI 还支持：存档加载、AI 对战回放（含热力图对比）、游戏存档列表。

### AI 对战

```bash
python src/ai/battle.py --ai1 <AI指定> --ai2 <AI指定> [--games <局数>] [温度采样选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--ai1` | 第一方 AI 指定（如 random, fc_v1, cnn_v3.1） | 必填 |
| `--ai2` | 第二方 AI 指定（如 random, fc_v1, cnn_v3.1） | 必填 |
| `--games` | 对战局数 | 10 |
| `--no-temp-sampling` | 禁用开局阶段温度采样，全程走 argmax | —（默认启用） |
| `--temperature` | 温度值 T（必须 > 0），越大越随机，趋近 0 时等价 argmax | 1.0 |
| `--temp-moves` | 前 N 个全局步启用温度采样，之后切回 argmax | 12 |
| `--temp-side` | 温度采样作用方：`black` / `white` / `both` / `random` | `both` |
| `--temp-seed` | 温度采样随机种子（用于复现对战结果） | None（不可复现） |

**可用 AI：** `random` / `fc_v1` / `cnn_v1` / `cnn_v2` / `cnn_v3.1` / `cnn_v4.1` / `cnn_v5.1` / `cnn_v6.1` / `cnn_v7.1` / `cnn_v8.1` / `cnn_v9.1` / `cnn_v10.1` / `cnn_v11.1` ...

> **多代模型说明**：`cnn_v3` 支持多代模型，使用 `模型名.代数` 格式指定（如 `cnn_v3.1`、`cnn_v3.2`），不同代数的模型权重存储在 `src/ai/models/cnn_v3/gen{i}/` 目录下。非多代模型（如 `fc_v1`、`cnn_v1`）直接使用模型名称。

```bash
# 示例
python src/ai/battle.py --ai1 fc_v1 --ai2 random --games 50
python src/ai/battle.py --ai1 cnn_v3.1 --ai2 cnn_v3.2 --games 50

# 禁用温度采样，全程 argmax
python src/ai/battle.py --ai1 cnn_v3.1 --ai2 cnn_v3.2 --games 50 --no-temp-sampling

# 自定义温度值（更随机的对战）
python src/ai/battle.py --ai1 fc_v1 --ai2 cnn_v2 --games 20 --temperature 2.0

# 固定随机种子复现对战
python src/ai/battle.py --ai1 cnn_v3.1 --ai2 cnn_v3.2 --games 10 --temp-seed 42

# 组合使用
python src/ai/battle.py --ai1 cnn_v3.1 --ai2 cnn_v3.2 --games 100 --temperature 2 --temp-moves 2 --temp-side both
```

### 模型训练

#### 生成训练样本（对弈 + sample_creator）

```bash
python src/ai/create_samples.py --model <模型名称> --output <输出文件路径> [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型指定（如 cnn_v3.1、fc_v1） | 必填 |
| `--output` | 输出 `.json` 文件路径 | 必填 |
| `--games` | 对局数（与 `--sample_games` 互斥，两者都不传则默认 100） | 100 |
| `--sample_games` | 目标样本局数（与 `--games` 互斥）；持续对弈直到收集到足够样本局后停止 | None |
| `--no-augment` | 禁用数据增强（默认启用 D4 群 8 变换） | - |
| `--temperature` | 温度采样温度 T（越大越探索） | 1.0 |
| `--temp-moves` | 前 N 步启用温度采样，之后切回 argmax | 12 |
| `--temp-side` | 温度采样作用方：`black` / `white` / `both` / `random` | `random` |
| `--temp-seed` | 温度采样随机种子（便于复现） | None |

**模式说明**：
- `winner_path`（唯一模式）：取整局中**获胜方的每一步**作为样本，让模型拟合获胜方整条路径。每步的 output 采用**软标签**（基于模型当步 softmax 输出，将落子位最小增强为严格 top1 后归一化），而非 one-hot，以保留模型分布、避免信号过于刚性。默认启用 D4 增强（8 变换），可通过 `--no-augment` 禁用。
- **变异步过滤**（始终生效）：仅保留引擎实际落子不在模型 argmax 集合中的"变异步"，跳过模型已能准确预测的步，提高样本训练价值。

**运行模式**：
- **`--games` 模式**（默认）：固定对局数，打完即停。
- **`--sample_games` 模式**：以"样本局数量"为目标驱动，持续对弈直到收集到目标数量的有效样本局；适合需要精确控制样本量的场景。支持 Ctrl+C 中断并保留已收集数据。

> **过滤机制**：对局级过滤（`--temp-side` 为单方模式时仅保留采样方获胜对局）+ 步级过滤（变异步过滤始终生效）。运行结束会输出对局级和步级过滤统计。

```bash
# 示例
# 默认模式（winner_path）：取获胜方每一步，启用 D4 增强 + 变异步过滤
python src/ai/create_samples.py --model cnn_v3.1 --output src/ai/merge_samples/v4/path.json
python src/ai/create_samples.py --model cnn_v3.1 --games 200 --output src/ai/train_data/samples.json

# 禁用增强
python src/ai/create_samples.py --model cnn_v3.1 --no-augment --games 200 --output src/ai/train_data/samples.json

# sample_games 模式：以样本局数量为目标驱动
python src/ai/create_samples.py --model cnn_v3.1 --sample_games 50 --output src/ai/train_data/samples.json

python src/ai/create_samples.py --model cnn_v6.99 --sample_games 500 --output samples/6/v100/path.json --temp-moves 100
```

> **提示**：运行结束后可用 `--train-from-dir` 加载生成的 `.json` 文件继续训练。

#### 样本可视化验收

基于 Flask 的 Web 工具，可视化验收 `create_samples.py` 生成的训练样本（热力图对比 + 对局浏览）。

```bash
# 启动验收工具
python -m src.tools.create_samples_tool.app
# 浏览器访问 http://127.0.0.1:5000
```

> 详细使用手册见 [src/tools/create_samples_tool/README.md](src/tools/create_samples_tool/README.md)。

#### 从已有数据训练

```bash
python src/ai/train_main.py --model <模型指定> --train-from-dir <数据目录或文件> [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--train-from-dir` | 训练数据目录或 `.json` 文件路径 | 必填 |
| `--lr` | 初始学习率，必须为正数 | 0.001 |
| `--fresh` | 从新模型开始训练（不加载已保存权重） | False |
| `--epochs` | 训练迭代数 | 1000 |
| `--batch-size` | 每批次样本数（启用 mini-batch） | 无（全量更新） |
| `--shuffle` | 每轮打乱数据顺序（mini-batch 有效） | False |
| `--val-ratio` | 验证集比例，0 表示不分割，推荐 0.2 | 0.0 |
| `--lr-scheduler` | 学习率调度策略，可选 `step` 或 `plateau` | 无（固定学习率） |
| `--lr-step-size` | `step` 模式：每隔 N 个 epoch 衰减一次 | 500 |
| `--lr-gamma` | `step` 模式：衰减系数 γ，新 lr = 旧 lr × γ | 0.9 |
| `--lr-patience` | `plateau` 模式：验证 loss 不再下降后，等待 N 个 epoch 才衰减 | 100 |
| `--lr-factor` | `plateau` 模式：衰减系数，新 lr = 旧 lr × factor | 0.5 |

> **学习率调度策略说明**：
> - **`step`（StepLR）**：按固定间隔衰减。每 `--lr-step-size` 个 epoch，学习率乘以 `--lr-gamma`。公式：\( \text{lr}_n = \text{lr}_0 \times \gamma^{\lfloor n / s \rfloor} \)，其中 \( s \) 为 step_size，\( \gamma \) 为 gamma。
>   例：`--lr 0.01 --lr-scheduler step --lr-step-size 100 --lr-gamma 0.5`
>   → epoch 1~100: lr = 0.01
>   → epoch 101~200: lr = 0.005
>   → epoch 201~300: lr = 0.0025
> - **`plateau`（ReduceLROnPlateau）**：自适应衰减，监控验证 loss（`mode='min'`）。当 loss 连续 `--lr-patience` 个 epoch 未创新低时，学习率乘以 `--lr-factor`。适合 loss 进入平台期后自动降速以突破局部最优。
>   例：`--lr 0.01 --lr-scheduler plateau --lr-patience 100 --lr-factor 0.5`
>   → 若连续 100 个 epoch loss 不降，lr 从 0.01 降为 0.005，再等 100 epoch 仍不降则降为 0.0025，以此类推。

> 指定目录时递归加载其下所有 `.json` 文件（不限文件名前缀）；指定 `.json` 文件时直接加载。

```bash
# 示例：从目录训练
python src/ai/train_main.py --model cnn_v1 --train-from-dir src/ai/train_data/cnn_v1_self_play_1 --epochs 1000

python src/ai/train_main.py --model cnn_v2 --train-from-dir src/ai/merge_samples --epochs 1000 --batch-size 256 --shuffle
# 示例：从合并文件训练
python src/ai/train_main.py --model cnn_v1 --train-from-dir src/ai/merge_samples/merge_cnn_v1.json --epochs 1000

# 示例：mini-batch 训练
python src/ai/train_main.py --model cnn_v3.2 --fresh --train-from-dir src/ai/merge_samples/v2 --epochs 1000 --batch-size 4096 --shuffle --val-ratio 0.2
```

> **注意**：使用 `--batch-size` 启用 mini-batch 训练时，系统会自动生成**密集 loss log**，
> 记录每个 batch 的 loss 和梯度大小，并在曲线图上以双 Y 轴形式叠加展示（Loss + Gradient Norm）。

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
python -m ai.sample_generator --shape chong4 --num-samples 10000 --output src/ai/merge_samples/chong4.json

# 示例：生成活三样本（干扰子更多）
python -m ai.sample_generator --shape huo3 --num-samples 5000 --max-extra 8 --output src/ai/merge_samples/huo3.json

# 示例：生成防守冲四样本
python -m ai.sample_generator --shape defend_chong4 --num-samples 5000 --output src/ai/merge_samples/defend_chong4.json
```

> **验证报告**：运行结束会打印 Hit@N / Hit@N+1 / Hit@N+2 命中率、`attack_prob_sum`（进攻位概率和）、`prob_gap`（进攻位与其他位的区分度）等指标，方便判断 label 质量是否达标。

### 样本合并

将分散的 `sample.json` 合并为单个大文件，减少训练时的 I/O 开销：

```bash
python src/ai/merge_samples.py <目录1> [目录2] ... -o <输出文件路径>
```

| 参数 | 说明 |
|------|------|
| 位置参数 | 一个或多个包含 sample.json 的数据目录 |
| `-o` / `--output` | 合并后的输出文件路径（必需） |

```bash
# 示例：合并单个目录
python src/ai/merge_samples.py src/ai/train_data/cnn_v2_self_play_1 -o src/ai/merge_samples/merge_cnn_v2.json

# 示例：合并多个目录
python src/ai/merge_samples.py src/ai/train_data/cnn_v2_self_play_1 src/ai/train_data/cnn_v2_self_play_2 -o src/ai/merge_samples/merge_cnn_v2_all.json
```

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
python -m ai.sample_visualizer --sample src/ai/merge_samples/chong4.json

# 示例：指定索引 + 自定义输出
python -m ai.sample_visualizer --sample data.json --indices 0,5,10 --output temp/my_viz.png

# 示例：关闭热力图 + 固定种子复现
python -m ai.sample_visualizer --sample data.json --count 3 --no-heatmap --seed 42
```

> 热力图核心算法位于独立可复用的 `src/visualization/heatmap_color.py`（无 UI 依赖，默认白→黄→橙→红 4 色色板 + log 归一化），同时被 `ai/sample_visualizer.py`（matplotlib）和 `gui/components/heatmap.py`（tkinter 回放界面）共用。

### AlphaGo Zero MCTS 训练

基于 cnn_v7/v8 双头模型的强化学习训练体系，通过 MCTS 自对弈生成高质量样本，训练模型逼近 MCTS 决策。

#### MCTS 自我对弈

```bash
python src/ai/scripts/mcts_self_play.py --model <权重路径> --games <局数> [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型权重文件路径（.pth），自动解析模型名.代数 | 必填 |
| `--model-name` | 显式指定模型注册名.代数（如 cnn_v8.1），覆盖路径解析 | None |
| `--games` | 对弈局数 | 必填 |
| `--simulations` | 每步 MCTS 模拟次数 | 800 |
| `--c-puct` | PUCT 探索系数 | 2.0 |
| `--temp-steps` | 温度探索步数（前 N 步 τ=1.0） | 8 |
| `--temperature` | 温度参数 | 1.0 |
| `--dirichlet-epsilon` | 狄利克雷噪声比例 | 0.25 |
| `--dirichlet-alpha` | 狄利克雷分布参数 | 0.03 |
| `--output` | 输出目录名 | `{model}_mcts_{timestamp}` |
| `--device` | 推理设备 | cuda |
| `--workers` | 并行 worker 进程数（CPU 推理），1=单进程 | CPU 核数 |

```bash
# 示例：使用 cnn_v8 权重进行 100 局 MCTS 自对弈
python src/ai/scripts/mcts_self_play.py --model src/ai/models/cnn_v8/gen1/cnn_v8.pth --games 100

# 示例：自定义模拟次数和输出目录
python src/ai/scripts/mcts_self_play.py --model cnn_v7.pth --games 50 --simulations 400 --output my_samples

python src/ai/scripts/mcts_self_play.py --model src/ai/models/cnn_v7/gen4/cnn_v7.pth --games 4000 --simulations 800 --output samples/7/v4 --temp-steps 20

python src/ai/scripts/mcts_self_play.py --model src/ai/models/cnn_v8/gen1/cnn_v8.pth --games 100 --simulations 800 --output samples/8/v1 --temp-steps 50

# 多进程并行（4 workers，CPU 推理）
python src/ai/scripts/mcts_self_play.py --model src/ai/models/cnn_v8/gen1/cnn_v8.pth --games 100 --workers 4
```

#### MCTS 双模型对战

> 让两个不同代的 MCTS 模型对打，输出胜率统计。参数体系与 mcts_self_play 一致。

```bash
python src/ai/scripts/mcts_battle.py --model1 <注册名.代数> --model2 <注册名.代数> --games <局数> [选项]
```

**选项**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model1` | **必需** | 模型1 注册名.代数，如 cnn_v8.10 |
| `--model2` | **必需** | 模型2 注册名.代数，如 cnn_v8.11 |
| `--games` | **必需** | 对战局数 |
| `--simulations` | 800 | 每步 MCTS 模拟次数 |
| `--c-puct` | 2.0 | PUCT 探索系数 |
| `--temp-steps` | 8 | 温度探索步数 |
| `--temperature` | 1.0 | 温度参数 |
| `--workers` | CPU 核数 | 并行 worker 进程数 |
| `--device` | cuda | 推理设备（多进程强制 CPU） |

**示例**：

```bash
# 基本对战：cnn_v8 第10代 vs 第11代
python src/ai/scripts/mcts_battle.py --model1 cnn_v8.10 --model2 cnn_v8.11 --games 100

# 多进程并行
python src/ai/scripts/mcts_battle.py --model1 cnn_v8.10 --model2 cnn_v8.11 --games 200 --workers 4
```

#### MCTS C++ 双模型并行对战（设计 47）

> 使用 C++ MCTS 引擎在 GPU 上并行运行 cnn_v11 模型对战。所有对局在 C++ 层面用多线程并行，共享全局 MCTS 线程池和 GPU 推理引擎。

```bash
python src/ai/scripts/mcts_battle_cpp.py --ai1 <注册名.代数> --ai2 <注册名.代数> --games <局数> [选项]
```

**选项**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ai1` | **必需** | 模型1 registry spec，如 cnn_v11.1 |
| `--ai2` | **必需** | 模型2 registry spec，如 cnn_v11.2 |
| `--games` | **必需** | 对战局数（≥1） |
| `--simulations` | 800 | 每步 MCTS 模拟次数 |
| `--c-puct` | 2.0 | PUCT 探索系数 |
| `--temp-steps` | 8 | 温度探索步数 |
| `--temperature` | 1.0 | 温度参数 |
| `--dirichlet-epsilon` | 0.25 | 狄利克雷噪声比例 |
| `--dirichlet-alpha` | 0.03 | 狄利克雷分布参数 |
| `--virtual-loss` | 默认配置 | virtual loss 值 |
| `--workers` | 游戏数 | 游戏线程数（≤游戏数时轮询分发，默认每局一个线程） |

**示例**：

```bash
# 基本对战：cnn_v11 第1代 vs 第1代（自对弈）
python src/ai/scripts/mcts_battle_cpp.py --ai1 cnn_v11.1 --ai2 cnn_v11.1 --games 10

# 自定义 MCTS 参数
python src/ai/scripts/mcts_battle_cpp.py --ai1 cnn_v11.1 --ai2 cnn_v11.2 --games 50 --simulations 400 --temperature 0.5 --temp-steps 12

# 限制游戏线程数（4 个 worker 轮询分发 100 局）
python src/ai/scripts/mcts_battle_cpp.py --ai1 cnn_v11.1 --ai2 cnn_v11.2 --games 100 --workers 4
```

#### GPU 推理攒批配置

GPU 推理线程采用**动态区间攒批**机制，通过 `cpp/python/mcts_config.json` 配置：

```json
{
    "batch_lower_bound": 16,   // 队列达到此数量才开始取走（等待下限）
    "batch_upper_bound": 500   // 单次最多取走的请求数（上限）
}
```

- 推理线程每 1ms 检查一次队列，达到 `batch_lower_bound` 后取走 `min(queue_size, batch_upper_bound)` 条请求
- `batch_lower_bound` 越大 GPU 利用率越高但延迟也越大；`batch_upper_bound` 防止单次推理耗时过长
- 修改 JSON 后重新运行脚本即生效，无需重新编译

#### GPU 推理运行时监控

```python
from cpp.python.cpp_mcts_adapter import get_gpu_stats

stats = get_gpu_stats()  # 获取 GpuInferenceStats 快照
print(f"批次数: {stats.total_batches}")
print(f"请求数: {stats.total_requests}")
print(f"平均批次: {stats.avg_batch_size():.1f}")
print(f"GPU 等待比例: {stats.wait_ratio() * 100:.1f}%")  # 越低 GPU 利用率越高
print(f"队列峰值: {stats.queue_peak_size}")
```

推理线程每 100 个 batch 自动输出监控日志，示例：

```
[NeuralNet] batches=100 reqs=3200 avg_batch=32.0 wait_ratio=23.5%
    queue_peak=128 batch_range=[16,500]
```

> **性能诊断**：`wait_ratio` 低（如 10%）表示 GPU 利用率高、攒批充分；`wait_ratio` 高（如 80%）表示 GPU 在空等，瓶颈在 MCTS 生产端。`queue_peak` 持续偏高说明请求堆积，GPU 成为瓶颈。

#### MCTS 训练

> 支持所有双头模型（cnn_v7, cnn_v8, ...），根据权重路径自动选择对应的模型类和训练器。

```bash
python src/ai/scripts/mcts_train.py --model <权重路径> --train-from-dir <数据目录> --epochs <轮数> [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 初始权重文件路径（自动解析模型名.代数） | 必填 |
| `--model-name` | 显式指定模型注册名.代数（如 cnn_v8.1），覆盖路径解析 | None |
| `--train-from-dir` | 训练数据目录（含 sample.json） | 必填 |
| `--epochs` | 训练轮数 | 必填 |
| `--output` | 训练后权重保存路径 | `--model` 的值（覆盖） |
| `--batch-size` | 训练 batch size | 2048 |
| `--lr` | 初始学习率 | 0.001 |
| `--l2-reg` | AdamW weight_decay | 1e-4 |
| `--val-split` | 验证集比例 | 0.1 |
| `--no-shuffle` | 关闭训练数据 shuffle | — |
| `--device` | 训练设备 | cuda |

> **损失函数**：MSE(价值) + CrossEntropy(策略)，AdamW 优化器含 decoupled weight decay。
>
> **损失曲线图**：训练完成后自动生成双 Y 轴 PNG 图表（左轴 Loss，右轴 Gradient Norm），保存至 `mcts_loss_logs/` 目录。即使训练被 Ctrl+C 中断，已完成 epoch 的数据仍会生成图表。

```bash
# 示例：从 MCTS 自对弈样本训练
python src/ai/scripts/mcts_train.py --model src/ai/models/cnn_v8/gen1/cnn_v8.pth --train-from-dir samples/8/v1 --epochs 30 --output src/ai/models/cnn_v8/gen2/cnn_v8.pth --val-split 0.1

# 示例：指定输出路径和验证集比例
python src/ai/scripts/mcts_train.py --model src/ai/models/cnn_v7/gen1/cnn_v7.pth --train-from-dir samples/7/v2 --epochs 300 --output src/ai/models/cnn_v7/gen2/cnn_v7.pth --val-split 0.1
```

### 价值头训练（cnn_v10）

cnn_v10 是纯价值头模型（无策略头），训练仅使用 MSE 损失。自对弈通过"枚举-评估"方式生成样本，样本格式为 `{"state": [...], "z": int}`。

#### 价值头自对弈

```bash
python src/ai/scripts/value_self_play.py --model <权重路径> --games <局数> [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 模型权重文件路径（.pth） | 必填 |
| `--model-name` | 显式指定模型注册名.代数（如 cnn_v10.1） | 从路径解析 |
| `--games` | 对弈局数 | 必填 |
| `--output` | 输出目录名 | `{model}_value_{timestamp}` |
| `--device` | 推理设备 | cuda |
| `--workers` | 并行进程数，1=单进程 | CPU 核数 |
| `--temperature` | 温度值 | 1.0 |
| `--temp-steps` | 前 N 步启用温度采样 | 8 |
| `--temp-side` | 采样方：black/white/random/both | random |

```bash
# 示例
python src/ai/scripts/value_self_play.py --model src/ai/models/cnn_v10/gen1/cnn_v10.pth --games 100
python src/ai/scripts/value_self_play.py --model cnn_v10.pth --games 50 --output my_samples --workers 4
python src/ai/scripts/value_self_play.py --model src/ai/models/cnn_v10/gen1/cnn_v10.pth --games 100 --output samples/10/v1 --workers 1 --temp-steps 50 --temp-side both
```

#### 价值头训练

```bash
python src/ai/scripts/value_train.py --model <权重路径> --train-from-dir <数据目录> --epochs <轮数> [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` | 初始权重文件路径 | 必填 |
| `--model-name` | 显式指定模型注册名.代数 | 从路径解析 |
| `--train-from-dir` | 训练数据目录（可多个） | 必填 |
| `--epochs` | 训练轮数 | 必填 |
| `--output` | 输出权重路径 | 覆盖 --model |
| `--batch-size` | batch size | 2048 |
| `--lr` | 学习率 | 0.001 |
| `--l2-reg` | AdamW weight_decay | 1e-4 |
| `--val-split` | 验证集比例 | 0.1 |
| `--no-shuffle` | 关闭数据打乱 | — |
| `--device` | 训练设备 | cuda |

> **损失函数**：MSE(价值)，仅训练价值头。训练完成后自动生成 loss 曲线图保存至 `mcts_loss_logs/`。

```bash
# 示例
python src/ai/scripts/value_train.py --model src/ai/models/cnn_v10/gen1/cnn_v10.pth --train-from-dir samples/10/v1 --epochs 100 --output src/ai/models/cnn_v10/gen2/cnn_v10.pth
```

### 迭代进化训练

自动化「生成样本 → 训练 → 对战评估 → 条件晋升」闭环，使模型持续迭代进化：

```bash
python src/ai/iterative_train.py --ai1 <父代模型spec> --ai2 <子代模型spec> --sample_dir <样本目录> [--sample-games <样本局数>] [--epochs <训练轮数>]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--ai1` | 父代模型 spec（模型名.代数） | 必填，如 `cnn_v4.99` |
| `--ai2` | 子代模型 spec（模型名.代数） | 必填，如 `cnn_v4.100` |
| `--sample_dir` | 样本输出目录基路径 | 必填，如 `ai/merge_samples2` |
| `--sample-games` | 每轮目标样本局数 | 500 |
| `--epochs` | 每轮训练 epoch 数 | 300 |

每轮自动执行：
1. 复制 ai1 权重 → ai2（模型初始化）
2. 调用 `create_samples.py` 自对弈生成样本（默认 500 局，可通过 `--sample-games` 覆盖），输出到 `{sample_dir}/v{gen_of_ai2}/path.json`
3. 调用 `train_main.py` 训练 ai2（默认 300 epoch、batch 4096，epoch 数可通过 `--epochs` 覆盖）
4. 调用 `battle.py` 父子对战 100 局（temperature=2, temp_moves=2, temp_side=both）
5. **条件晋升**：ai2 胜率 > 55% 时晋升（mv ai2 → ai1），进入下一轮；否则丢弃训练结果重新开始

> 前置条件：`ai/models/{model_name}/gen{gen_of_ai1}/{model_name}.pth` 必须存在；不支持多实例并发（`.iterative_lock` 锁保护）。
> 意图 13：模型标识和样本目录通过 CLI 参数指定，不再硬编码。
> 意图 22：条件晋升（胜率 > 55%），`--sample-games` 和 `--epochs` 支持 CLI 覆盖。

```bash
# 示例：从 cnn_v4.99 开始迭代进化（使用默认参数）
python src/ai/iterative_train.py --ai1 cnn_v4.99 --ai2 cnn_v4.100 --sample_dir src/ai/merge_samples2

# 示例：自定义样本局数和训练轮数
python src/ai/iterative_train.py --ai1 cnn_v4.99 --ai2 cnn_v4.100 --sample_dir src/ai/merge_samples2 --sample-games 300 --epochs 200

python src/ai/iterative_train.py --ai1 cnn_v6.99 --ai2 cnn_v6.100 --sample_dir samples/6
```

### LR Range Test

学习率范围测试工具，用于确定最优学习率区间。每个 round 使用一个固定 lr，训练 `epochs` 个 step 后记录平均 loss，lr 从 `min_lr` 线性递增至 `max_lr`。每个 round 开始前重置模型到初始状态，确保 loss 只反映 lr 的影响。

```bash
python src/ai/lr_range_test.py --model <模型指定> --data <数据路径> [选项]
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
python src/ai/lr_range_test.py --model cnn_v2 --data src/ai/merge_samples/merge_cnn_v1_100.json

# 示例：自定义 lr 范围和输出路径
python src/ai/lr_range_test.py --model cnn_v2 --data src/ai/merge_samples/merge_cnn_v1_100.json --min-lr 1e-6 --max-lr 1e-2 --output temp/my_lr_test.png
```

### 运行测试

项目使用 **pytest** 作为测试框架，支持以下运行方式：

```bash
# 方式1：直接调用 pytest（推荐）
python -m pytest tests/ -v -s

# 方式2：通过兼容性入口运行（已自动带 -s）
python tests/run_tests.py
```

> 使用 pytest 自动发现机制覆盖所有测试。
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
- **训练方式**：自我对弈 + 棋型评估奖励 + KL 散度监督学习
- **模型文件**：`src/ai/models/fc_v1/fc_v1.pth`

### CNN v1

卷积神经网络模型，通过卷积层提取棋盘空间特征。

- **输入**：15×15×3 张量（黑棋通道、白棋通道、当前玩家通道）
- **输出**：225 维概率分布（Softmax）
- **网络结构**：卷积层(3→32→64→128) → 全连接层(28800→150→225) → Softmax
- **训练方式**：自我对弈 + 棋型评估奖励 + KL 散度监督学习
- **模型文件**：`src/ai/models/cnn_v1/cnn.pth`

### CNN v2 模型（轻量化）

- **网络结构**：卷积层(3→64→64→64) + BN → 全连接层(14400→64→225) → Softmax
- **参数量**：约 101 万（CNN v1 的 23%，计算量 18M MACs）
- **训练方式**：自我对弈 + 棋型评估奖励 + KL 散度监督学习
- **模型文件**：`src/ai/models/cnn_v2/cnn_v2.pth`

### CNN v3 模型（纯卷积极简）

- **网络结构**：卷积层(3→32→32→32, 无BN) → 1×1卷积(32→1) → Softmax
- **参数量**：约 1.9 万（CNN v2 的 1.9%）
- **特点**：纯卷积，无 FC、无 BN、无 Dropout，1×1 Conv 替代 FC 层
- **训练方式**：自我对弈 + 棋型评估奖励 + KL 散度监督学习
- **模型文件**：`src/ai/models/cnn_v3/gen{i}/cnn_v3.pth`（多代模型，按代数存储）
- **多代支持**：使用 `cnn_v3.代数` 格式指定（如 `cnn_v3.1`），权重按代数隔离存储在 `gen{i}/` 子目录下

### CNN v4 模型（纯卷积极简 + 1 隐藏层）

- **网络结构**：卷积层(3→32→32→32→32, 无BN) → 1×1卷积(32→1) → Softmax
- **参数量**：约 2.8 万
- **特点**：在 cnn_v3 基础上增加一层 32→32 卷积，纯卷积，无 FC、无 BN、无 Dropout
- **训练方式**：自我对弈 + 棋型评估奖励 + KL 散度监督学习
- **模型文件**：`src/ai/models/cnn_v4/gen{i}/cnn_v4.pth`（多代模型）

### CNN v5 模型（纯卷积极简 + 2 隐藏层）

- **网络结构**：卷积层(3→32→32→32→32→32, 无BN) → 1×1卷积(32→1) → Softmax
- **参数量**：约 3.8 万
- **特点**：在 cnn_v4 基础上再增加一层 32→32 卷积，纯卷积，无 FC、无 BN、无 Dropout
- **训练方式**：自我对弈 + 棋型评估奖励 + KL 散度监督学习
- **模型文件**：`src/ai/models/cnn_v5/gen{i}/cnn_v5.pth`（多代模型）

### CNN v6 模型（残差卷积 + BatchNorm）

- **网络结构**：卷积层(3→32→32→32→32→32, Conv→BN→ReLU) → 两个残差块 → 1×1卷积(32→1) → Softmax
- **参数量**：约 3.8 万
- **特点**：在 cnn_v5 基础上引入 BatchNorm 和残差连接，Conv2+Conv3、Conv4+Conv5 各组成一个残差块
- **训练方式**：自我对弈 + 棋型评估奖励 + KL 散度监督学习
- **模型文件**：`src/ai/models/cnn_v6/gen{i}/cnn_v6.pth`（多代模型）

### CNN v7 模型（双头残差卷积 + AlphaGo Zero）

- **输入**：15×15×3 张量（相对视角编码：我方棋子、对方棋子、我方颜色）
- **输出**：双头 —— 策略头 225 维 LogSoftmax + 价值头 Tanh [-1,1]
- **网络结构**：Conv1(3→32, BN, ReLU) → 两个残差块(32→32) → 1×1Conv(32→1) → Flatten → {LogSoftmax(策略), FC(225→1)→Tanh(价值)}
- **参数量**：约 4.0 万
- **特点**：在 cnn_v6 基础上增加价值头，采用 AlphaGo Zero 相对视角编码
- **训练方式**：AlphaGo Zero MCTS 自我对弈 + MSE(价值) + CrossEntropy(策略) + AdamW 正则化
- **模型文件**：`src/ai/models/cnn_v7/gen{i}/cnn_v7.pth`（多代模型）

### CNN v8 模型（双头残差卷积 + FC 隐藏层）

- **输入**：15×15×3 张量（相对视角编码：我方棋子、对方棋子、我方颜色）
- **输出**：双头 —— 策略头 225 维 LogSoftmax + 价值头 Tanh [-1,1]
- **网络结构**：在 cnn_v7 基础上为双头各添加一层 FC(225→64)→ReLU 隐藏层
- **参数量**：约 8.2 万（cnn_v7 的 2.1 倍）
- **特点**：共享特征层与 cnn_v7 一致，策略头 conv_out → Flatten → FC→ReLU→FC→LogSoftmax，价值头 value_conv_out → Flatten → FC→ReLU→FC→Tanh
- **训练方式**：AlphaGo Zero MCTS 自我对弈 + MSE(价值) + CrossEntropy(策略) + AdamW 正则化
- **模型文件**：`src/ai/models/cnn_v8/gen{i}/cnn_v8.pth`（多代模型）
- **结构图**：[cnn_v8_architecture.puml](src/ai/models/cnn_v8/cnn_v8_architecture.puml) / [cnn_v8_architecture.png](src/ai/models/cnn_v8/cnn_v8_architecture.png)

### C++ MCTS 模块（设计 46 + 47）

高性能 C++17 MCTS 搜索器，通过 pybind11 桥接为 Python 扩展模块，替代纯 Python 搜索逻辑。

- **核心收益**：消除 Python GIL 瓶颈，支持多线程并行 MCTS + GPU 异步批量推理；支持 N 局并行对战共享全局 GPU
- **技术栈**：C++17 / LibTorch / pybind11 / CMake
- **目标模型**：cnn_v11（TorchScript 导出）
- **新增（设计 47）**：`GameRunner` 并行对局管理器、`GameConfig` 对战配置、`CppMctsBattleAdapter` Python 适配器、`mcts_battle_cpp.py` CLI 脚本

#### 构建

> **一键构建（推荐）**：项目根目录下运行 `build_cpp.bat`，自动检测 LibTorch/pybind11、
> 配置 CMake、编译并复制 `.pyd` 到 `cpp/python/`。
> ```cmd
> build_cpp.bat              # 默认 Release 构建
> build_cpp.bat --debug      # Debug 构建
> build_cpp.bat --test       # 构建 + 运行 C++ 测试
> build_cpp.bat --clean      # 清理后重新构建
> ```

手动构建（备选）：

```bash
cd cpp
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build . --config Release
```

#### 使用

```python
from cpp.python.cpp_mcts_adapter import CppMctsAdapter
from ai.engine.policy_value import PolicyValueAiEngine

searcher = CppMctsAdapter(
    model_path="src/ai/models/cnn_v11/cnn_v11_torchscript.pt",
    num_simulations=800,
    num_threads=32,
    batch_size=32,
    use_gpu=True,
)
engine = PolicyValueAiEngine(searcher, temperature=1.0, temp_steps=8)
move = engine.get_move(board, current_player, strategy="sampling")
```

#### 运行测试

```bash
# C++ 单元测试
cd cpp/build && ctest --output-on-failure

# Python 集成测试
python -m pytest tests/integration/test_cpp_mcts_bridge.py -v
python -m pytest tests/integration/test_cpp_mcts_e2e.py -v
```

---

## 数据目录

| 目录 | 说明 |
|------|------|
| `src/game/save/` | 游戏存档 |
| `src/ai/ai_data/` | AI 对战记录（record.json） |
| `src/ai/ai_data/loss_logs/` | 训练 loss 日志和曲线图 |
| `mcts_loss_logs/` | MCTS 训练损失曲线图（PNG） |
| `src/ai/train_data/` | 训练样本（sample.json） |
| `src/ai/merge_samples/` | 合并后的训练样本大文件 |
| `src/tools/create_samples_tool/` | create_samples 验收工具 |

---

## 许可证

MIT License
 