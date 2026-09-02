# 五子棋 (Gomoku) 项目架构导览

## 项目概述

一个支持 AI 训练的五子棋游戏项目。核心能力：
- 多种游戏模式（PvP / PvE / AI 内战）
- CLI 和 GUI 双界面
- AlphaGo Zero 风格的 MCTS 强化学习训练体系
- 多代 CNN/FC 神经网络模型，支持迭代进化
- C++ MCTS 高性能推理引擎（多线程 + GPU 异步推理）

**技术栈**：Python 3.8+ / PyTorch / tkinter / numpy / matplotlib / Flask

**C++ MCTS 扩展**：C++17 / LibTorch / pybind11 / CMake — 高性能多线程 MCTS 搜索器（详见 [设计 46](../design/46/main_docs/overview.md)）

**C++ MCTS 并行对战**：C++ 多线程并行运行 N 局 cnn_v11 对战，共享全局线程池和 GPU（详见 [设计 47](../design/47/main_docs/overview.md)）

**C++ MCTS 自对弈样本缓冲区**：C++ 多线程自对弈持续产出训练样本，D4 增强 + FIFO 滑动窗口缓冲区管理（详见 [设计 49](../design/49/main_docs/overview.md)）

**CNN v11 训练闭环**：样本缓冲区驱动的增量训练 + C++ MCTS 对战评估晋升（详见 [设计 50](../design/50/main_docs/overview.md)）

**C++ MCTS 状态校验与落子合法性修复**：搜索树根一致性校验三分支（一致复用 / 差 1 步且已展开则 promote 复用 / 否则整树重建并计数）、落子空位过滤防御、树规模限幅（详见 [设计 59](../design/59/main_docs/overview.md)）

**JSON 动态配置**：C++/Python 共享 `cpp/python/mcts_config.json` 动态加载 MCTS 参数（模拟数、线程数、PUCT 等），修改无需重新编译

---

## 总体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户入口层                                │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │  src/gui/    │  │  src/cli/    │  │  src/ai/scripts/       │ │
│  │  tkinter GUI │  │  命令行界面   │  │  MCTS 自对弈/训练脚本   │ │
│  └──────┬───────┘  └──────┬───────┘  └───────────┬────────────┘ │
└─────────┼──────────────────┼──────────────────────┼──────────────┘
          │                  │                      │
┌─────────┼──────────────────┼──────────────────────┼──────────────┐
│         ▼                  ▼                      ▼              │
│                    应用层 (src/app/)                             │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │  session.py  │  │ gui_helper.py│  │     battle.py          │ │
│  │  游戏会话管理 │  │  GUI辅助工具  │  │    AI内战编排          │ │
│  └──────┬───────┘  └──────────────┘  └────────────────────────┘ │
└─────────┼───────────────────────────────────────────────────────┘
          │
┌─────────┼───────────────────────────────────────────────────────┐
│         ▼                 核心逻辑层                             │
│  ┌─────────────────────────┐  ┌───────────────────────────────┐ │
│  │      game / 游戏引擎     │  │       ai / AI 系统            │ │
│  │  game.py     棋盘+规则   │  │  registry.py     模型注册表    │ │
│  │  controller.py 接口协调  │  │  base_model.py   模型基类      │ │
│  │  save.py      存档管理   │  │  base_trainer.py 训练器基类    │ │
│  └─────────────────────────┘  │  engine/         AI引擎层       │ │
│                               │  mcts/           MCTS搜索       │ │
│                               │  models/         模型实现       │ │
│                               │  evaluator.py    传统评价器     │ │
│                               │  game_manager.py AI对战管理     │ │
│                               │  create_samples.py  样本生成   │ │
│                               └───────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 目录结构

```
src/
├── game/          # 游戏核心——棋盘规则、控制器、存档
│   ├── game.py       检查胜利/平局、落子逻辑
│   ├── controller.py 统一接口（协调 game + save）
│   └── save.py       存档读写
│
├── app/           # 应用层——薄的代理层，委托 game 和 ai
│   ├── session.py     PvP/PvE 会话（持有 controller + AI engine）
│   ├── gui_helper.py  GUI 辅助函数
│   ├── battle.py      AI 内战批处理编排
│   └── exceptions.py  模型加载异常
│
├── ai/            # AI 核心系统
│   ├── base_model.py      模型抽象基类（BaseModel → NeuralNetworkModel）
│   ├── base_trainer.py    训练器抽象基类（Trainer → NeuralNetworkTrainer）
│   ├── registry.py        统一模型注册表（懒加载 + 多代支持）
│   ├── constants.py       常量（路径、策略名）
│   ├── sample_buffer_manager.py  样本缓冲区管理器（FIFO+chunk管理）
│   ├── sample_buffer_reader.py   样本缓冲区读取器（随机抽样）
│   ├── checkpoint_evaluator.py   checkpoint MCTS 对战评估器
│   ├── cnn_v11_train_loop.py     CNN v11 训练-评估循环编排器
│   │
│   ├── engine/            AI 引擎层——统一落子接口
│   │   ├── ai_engine.py      AIEngine 抽象基类 + MoveDecision
│   │   ├── builder.py        工厂 builder + create_engine 智能路由
│   │   ├── single_policy.py  SinglePolicyAiEngine（单头模型：argmax/sampling）
│   │   ├── policy_value.py   PolicyValueAiEngine（双头模型：MCTS搜索）
│   │   └── value_only.py     ValueOnlyAiEngine（价值头模型：枚举-评估）
│   │
│   ├── mcts/              MCTS 搜索器
│   │   ├── searcher.py       PUCT选择·评估·回传·子树复用
│   │   └── search_logger.py  结构化日志（JSON，支持验收验证）
│   │
│   ├── models/            11 种模型实现（每种独立子目录）
│   │   ├── random/      纯随机
│   │   ├── fc_v1/       全连接（226→1000×5→225）
│   │   ├── cnn_v1/      卷积 v1（3→32→64→128）
│   │   ├── cnn_v2/      卷积 v2 轻量（3→64→64→64）
│   │   ├── cnn_v3/      极简纯卷积（3→32→32→32，~1.9万参）
│   │   ├── cnn_v4/      +1隐藏层（~2.8万参）
│   │   ├── cnn_v5/      +2隐藏层（~3.8万参）
│   │   ├── cnn_v6/      残差+BN（~3.8万参）
│   │   ├── cnn_v7/      双头残差（策略+价值，~4.0万参）★AlphaGo Zero
│   │   ├── cnn_v8/      双头残差+FC隐藏层（策略+价值，~8.2万参）★
│   │   ├── cnn_v9/      双头残差+48通道（策略+价值，~12.8万参）★
│   │   ├── cnn_v10/     价值头残差（仅价值，~10万参）★
│   │   │   ├── model.py       CNNV10Network + CNNV10Model
│   │   │   └── trainer.py     CNNV10Trainer（MSE 训练）
│   │   └── cnn_v11/     双头残差（策略+价值，~843万参）★
│   │       ├── model.py       CNNV11Network + CNNV11Model
│   │       └── trainer.py     CNNV11Trainer（MSE+CrossEntropy）
│   │
│   ├── game_manager.py    AI 对战管理（PvE + AI内战 + 温度采样编排）
│   ├── data_recorder.py   对战数据记录
│   ├── battle.py          AI 内战 CLI 入口
│   ├── create_samples.py  训练样本生成（自对弈 → sample.json）
│   ├── sample_creator.py  样本创建器核心逻辑
│   ├── self_play_generator.py  自对弈样本生成器
│   ├── sample_loader.py   样本加载器
│   ├── sample_generator.py  棋型样本合成
│   ├── sample_visualizer.py  样本可视化（matplotlib→PNG）
│   ├── evaluator.py       传统评价器（棋型评估→reward→softmax）
│   ├── reward_calculator.py  奖励计算
│   ├── shape_detector.py  棋型检测
│   ├── dual_head_trainer.py  cnn_v7 双头训练器（α-GZ 训练）
│   ├── train_main.py      通用训练入口
│   ├── merge_samples.py   样本合并
│   ├── iterative_train.py 迭代进化训练（生成→训练→对战→晋升闭环）
│   ├── lr_range_test.py   学习率范围测试
│   ├── move_sampler.py    温度采样逻辑
│   ├── probability_utils.py  概率工具（masked_softmax等）
│   ├── loss_logger.py     训练损失日志+图表
│   ├── runners.py         训练运行器
│   ├── game_filter.py     对局过滤
│   ├── keep_gpu_alive.py  GPU 保活
│   │
│   └── scripts/           独立脚本
│       ├── mcts_self_play.py     MCTS 自对弈（AlphaGo Zero）
│       ├── mcts_self_play_worker.py  MCTS 自对弈 Worker
│       ├── mcts_game_runner.py   MCTS 公用对局运行器
│       ├── mcts_config.py        MCTS 配置数据类
│       ├── mcts_battle.py        MCTS 双模型对战
│       ├── mcts_battle_worker.py MCTS 对战 Worker
│       ├── mcts_battle_cpp.py    MCTS C++ 双模型对战（GPU 并行）
│       ├── mcts_sample_buffer.py MCTS C++ 自对弈样本缓冲区（D4增强+FIFO，--model 可选）
│       ├── mcts_train.py         MCTS 训练
│       ├── value_self_play.py    价值头自对弈（cnn_v10）
│       ├── value_self_play_worker.py  价值头自对弈 Worker
│       ├── value_train.py        价值头训练（cnn_v10）
│       ├── mcts_train_cnn_v11.py  CNN v11 样本缓冲区训练（--model 可选）
│       └── mcts_train_cnn_v11_offpolicy.py  CNN v11 off-policy 监督训练（--model 可选）
│
├── gui/           # GUI 界面（tkinter）
│   ├── main.py         主窗口
│   ├── ai_battle.py    AI 内战界面
│   ├── display_updater.py  棋盘显示更新
│   └── components/     GUI 组件
│       ├── game_board.py     棋盘渲染
│       ├── mode_selector.py  模式选择
│       ├── heatmap.py        热力图叠加（tkinter）
│       ├── replay_viewer.py  对局回放查看
│       ├── replay_browser.py 回放浏览器
│       ├── load_game_dialog.py 存档加载
│       ├── ai_battle.py      AI 内战界面组件
│       └── constants.py      GUI 常量定义
│
├── cli/           # 命令行界面
│   └── main.py
│
├── config/        # 配置常量
│   └── mcts.py        MCTS 配置（日志、batch_size、virtual_loss）
│
├── visualization/ # 可视化算法（无 UI 依赖，被 GUI 和 AI 共用）
│   └── heatmap_color.py  热力图色彩映射
│
└── tools/         # 辅助工具
    └── create_samples_tool/  Flask Web 样本验收工具
        ├── app.py           Flask 应用入口
        ├── data_parser.py   样本数据解析
        └── executor.py      样本执行/验收逻辑
```

---

## 核心架构设计

### 1. 模型注册表 (`registry.py`)

统一管理所有 AI 模型的懒加载注册。通过 `get_model(spec)` 和 `get_trainer(spec)` 按需实例化，支持多代模型（`cnn_v3.1` 格式）。新增模型只需添加 RegistryEntry 即可。`get_highest_gen(model_name)` 扫描 gen 目录返回最高代数，供训练/采样脚本自动识别最新模型。

### 2. 引擎智能路由 (`engine/builder.py`)

`create_engine(ai_name)` 自动判定模型类型：
- **单头模型**（Policy Only）→ `SinglePolicyAiEngine`（直接 argmax/sampling）
- **双头模型**（Policy+Value，cnn_v7）→ `PolicyValueAiEngine`（内含 MCTS 搜索）

### 3. MCTS 搜索器 (`mcts/searcher.py`)

AlphaGo Zero 风格的蒙特卡洛树搜索，支持 Batch MCTS + Virtual Loss：
- **Selection**：PUCT 公式 \(a = \text{argmax}(Q_a + c_{\text{puct}} \cdot P_a \cdot \frac{\sqrt{\sum N_b}}{1 + N_a})\)，批次内施加 VL 临时惩罚（N+=3, W-=3）以多样化路径
- **Batch Evaluation**：每批收集 B=225 个叶节点（已退化为单批次处理，一步搜索 225 次后按 Q 值落子），合并为一次模型推理
- **Backup**：沿路径回传，每层价值取反；VL 撤销后执行真实回传
- **Dirichlet 噪声**：根节点施加 \(\alpha=0.03, \epsilon=0.25\)
- **子树复用**：`advance_root` 推进根节点复用搜索树

### C++ MCTS 引擎（`cpp/`）

基于 C++17/LibTorch 的高性能 MCTS 实现，通过 pybind11 桥接为 Python 扩展（详见 [设计 46](../design/46/main_docs/overview.md)、[设计 47](../design/47/main_docs/overview.md)、[设计 49](../design/49/main_docs/overview.md)、[设计 50](../design/50/main_docs/overview.md)、[设计 51](../design/51/main_docs/overview.md)、[设计 59](../design/59/main_docs/overview.md)）。

**核心模块**：

| 模块 | 文件 | 职责 |
|------|------|------|
| `BoardState` | `board_state.h/cpp` | 棋盘编码、胜负判定、相对视角编码 |
| `MctsNode` | `mcts_node.h/cpp` | 并发安全搜索树节点（PUCT/VL/Backup） |
| `NeuralNet` | `neural_net.h/cpp` | LibTorch 异步批量推理（全局单例、GPU 共享、双模型缓存） |
| `MctsThreadPool` | `thread_pool.h/cpp` | 全局工作线程池（支持批次屏障 + 任务队列双模式） |
| `MctsConfig` | `mcts_config.h/cpp` | 运行时配置（模块导入时从 mcts_config.json 加载到全局 `g_mcts_config`，参数含模拟数/线程数/PUCT/狄利克雷/攒批阈值/VL/GPU/树展开上限） |
| `MctsSearcher` | `mcts_searcher.h/cpp` | 多线程搜索编排（支持共享资源注入）、Dirichlet 噪声、子树复用、树根局面一致性校验与整树重建计数（design/59） |
| `GameRunner` | `game_runner.h/cpp` | 并行对局管理器（N 个 C++ 游戏线程共享全局资源）— 支持对战/自对弈双模式；落子前空位过滤防御（design/59） |
| `GameConfig` | `game_runner.h` | 对战配置结构体（MCTS + 温度参数） |
| `GameResult` | `game_runner.h` | 单局对战结果结构体（含训练样本字段） |
| `GameSample` | `game_sample.h` | AGZ 格式训练样本结构体 {state, pi, z} |
| `SampleAugmentor` | `game_sample.h` | D4 群 8 种棋盘对称变换工具类 |

**Python 桥接**：
- `bindings.cpp` — pybind11 定义，暴露 `cpp_mcts.MctsSearcher`、`GameRunner`、`GameConfig`、`GameSample`、`MctsResult`、`GpuInferenceStats`，以及线程池管理函数 `init_thread_pool` / `shutdown_thread_pool` / `thread_pool_size` / `get_gpu_stats`
- `cpp_mcts_adapter.py` — 单搜索适配器，接口兼容 Python `MCTSSearcher`
- `cpp_mcts_battle_adapter.py` — 并行对战适配器，包装 C++ `GameRunner` 为 Python 接口
- `cpp_sample_buffer_adapter.py` — 自对弈样本缓冲区适配器，包装 C++ `GameRunner`（自对弈模式）+ `SampleBufferManager`
- `cpp_battle_game_result.py` — `CppBattleGameResult` 数据类，将 C++ `GameResult` 转换为 Python dataclass
- `mcts_config.py` + `mcts_config.json` — Python 端配置入口，与 C++ 端共享同一 JSON（含 `init_global_thread_pool` 显式初始化全局线程池）

**线程模型**：
- 单搜索：N 个 MCTS 工作线程 + 1 个 GPU 推理线程（生产者-消费者模式）
- 并行对战：K 个 MCTS 工作线程（任务队列）+ N 个游戏线程 + 全局单例 NeuralNet（GPU 共享、双模型缓存）
  推理线程按 `model_id` 分组攒批，每组独立 GPU forward。
  `Game threads → MctsSearcher.search() → pool.submit() → predict(board, model_id) → GPU thread groups by model_id → batch inference`

### 4. 两类 AI 引擎

| 引擎 | 适用模型 | 决策方式 | 用途 |
|------|---------|---------|------|
| `SinglePolicyAiEngine` | random, fc_v1, cnn_v1~v6 | 模型 softmax → argmax/sampling | 传统 AI、PvE |
| `PolicyValueAiEngine` | cnn_v7, cnn_v8, cnn_v9, cnn_v11 | MCTS 搜索 → 温度采样 | AlphaGo Zero 训练 |
| `ValueOnlyAiEngine` | cnn_v10 | 枚举候选+模型评估+softmax → argmax/sampling | 价值头模型对战 |

### 5. 模型层次结构

```
BaseModel (ABC)
├── RandomModel                     # 不使用网络
└── NeuralNetworkModel              # PyTorch 模型基类
    ├── FCv1Model                   # 226→1000×5→225
    ├── CNNV1Model                  # 绝对视角: 黑/白/当前方 3通道
    ├── CNNV2Model                  # 轻量CNN
    ├── CNNV3Model / V4 / V5        # 纯卷积极简系列
    ├── CNNV6Model                  # 残差+BN
    ├── CNNV7Model                  # 双头: LogSoftmax(225) + Tanh(1) ★
    │   └── CNNV8Model              # 双头+FC隐藏层: LogSoftmax(225) + Tanh(1) ★
    │       └── CNNV9Model          # 双头+48通道: LogSoftmax(225) + Tanh(1) ★
    │           └── CNNV11Model     # 双头+256通道+7残差块: LogSoftmax(225) + Tanh(1) ★
    └── CNNV10Model                 # 价值头: Tanh(1) ★
```

输入编码差异：
- **CNN v1~v6**：绝对视角（通道=黑棋/白棋/当前方颜色指示）
- **CNN v7/v8/v9/v10/v11**：相对视角（通道=我方/对方/我方颜色），保证颜色互换不变性

### 6. 训练体系

```
传统训练流程（cnn_v1~v6）:
  self_play → record.json → evaluator(棋型评估) → sample.json → trainer(KL散度)

AlphaGo Zero 训练流程（cnn_v7/v8）:
  mcts_self_play → sample.json → dual_head_trainer(MSE+CrossEntropy)

迭代进化:
  create_samples → train → battle(胜率>55%) → 晋升 → 下一轮
```

---

## 关键数据流

```
GUI PvE: 用户点击 → GameSession.make_player_move()
         → GameController.make_move()
         → [need_ai_move] → GameSession.do_ai_move()
           → AIGameManager.ai_move() → engine.get_move() → 落子

AI内战: battle.py → AIGameManager.start_ai_vs_ai()
        → create_engine×2 → play_single_game()
        → 循环: engine.get_move() → controller.make_move()

MCTS训练: mcts_self_play.py → create_engine(cnn_v7)
          → play_single_game() → PolicyValueAiEngine
          → MCTSSearcher.search() × 800模拟/步
          → 生成 sample.json → mcts_train.py → dual_head_trainer

价值头训练: value_self_play.py → create_engine(cnn_v10)
          → play_single_game() → ValueOnlyAiEngine
          → 枚举-评估 → 生成 sample.json → value_train.py → CNNV10Trainer

MCTS对战: mcts_battle.py → create_engine(cnn_v7.10) × 2
          → MctsBattleWorker → MctsGameRunner.run
          → 输出双方胜率统计

C++ MCTS对战: mcts_battle_cpp.py → CppMctsBattleAdapter
              → C++ GameRunner（N 个游戏线程，共享线程池+GPU）
              → 输出双方胜率统计

C++ MCTS样本生成: mcts_sample_buffer.py → CppMctsSampleBufferAdapter
                  → C++ GameRunner（自对弈模式，D4增强）
                  → SampleBufferManager（FIFO 缓冲区，每10局chunk落盘）
                  → 输出 sample_buffer/ 目录下的 chunk JSON 文件

CNN v11 缓冲区训练: mcts_train_cnn_v11.py（--model 可选，默认最高代数）→ CnnV11TrainLoop
                  → SampleBufferReader（随机抽样）
                  → CNNV11Trainer._train_step（MSE+CrossEntropy）
                  → CheckpointEvaluator（C++ MCTS 对战评估）
                  → 胜率 ≥ 55% → 晋升 gen{N+1}
                  → 连续 3 次失败 → 停止

CNN v11 off-policy 训练: mcts_train_cnn_v11_offpolicy.py（--model 可选，默认最高代数）→ OffPolicyTrainLoop
                  → OffPolicyTrainer._load_samples（chunk JSON / sample.json，多 epoch 全量遍历）
                  → 基类 train_epochs（mini-batch + val_split + 每 epoch 打印含 grad_norm + 自动曲线图）
                  → 保存 gen{N+1}（pth + TorchScript）
```

---

## 测试结构

```
tests/
├── conftest.py                # 全局 fixtures
├── run_tests.py               # 测试运行入口
├── test_game.py 等            # 游戏核心测试（位于根目录，无独立 game/ 子目录）
├── ai/                        # AI 模块测试（含 unit/ + integration/ 子目录）
├── gui/                       # GUI 组件测试
├── app/                       # 应用层测试（含 edge/ 边界场景）
├── cli/                       # CLI 测试
├── unit/                      # 跨模块单元测试
├── integration/               # MCTS 集成测试
├── e2e/                       # 端到端测试
└── tools/                     # 工具测试
```

---

## 常见入口命令

| 场景 | 命令 |
|------|------|
| 启动 GUI | `python src/gui/main.py` |
| 启动 CLI | `python src/cli/main.py` |
| AI 内战 | `python src/ai/battle.py --ai1 cnn_v3.1 --ai2 cnn_v3.2 --games 50` |
| 生成样本 | `python src/ai/create_samples.py --model cnn_v7.1 --sample_games 500 --output samples/7/v1/path.json` |
| 训练模型 | `python src/ai/train_main.py --model cnn_v7.2 --fresh --train-from-dir data --epochs 300` |
| MCTS 自对弈 | `python src/ai/scripts/mcts_self_play.py --model cnn_v7.pth --games 100` |
| MCTS 自对弈（多进程） | `python src/ai/scripts/mcts_self_play.py --model cnn_v7.pth --games 100 --workers 4` |
| MCTS 双模型对战 | `python src/ai/scripts/mcts_battle.py --model1 cnn_v8.10 --model2 cnn_v8.11 --games 100` |
| MCTS 双模型对战（多进程） | `python src/ai/scripts/mcts_battle.py --model1 cnn_v8.10 --model2 cnn_v8.11 --games 100 --workers 4` |
| C++ MCTS 对战 | `python src/ai/scripts/mcts_battle_cpp.py --ai1 cnn_v11.1 --ai2 cnn_v11.2 --games 100` |
| C++ MCTS 样本生成 | `python src/ai/scripts/mcts_sample_buffer.py --model cnn_v11.1 --games 1000`（--model 可选，默认 cnn_v11 最高代数） |
| MCTS 训练 | `python src/ai/scripts/mcts_train.py --model cnn_v7.pth --train-from-dir samples --epochs 100` |
| 价值头自对弈 | `python src/ai/scripts/value_self_play.py --model cnn_v10.1 --games 100` |
| 价值头训练 | `python src/ai/scripts/value_train.py --model cnn_v10.1 --train-from-dir samples --epochs 100` |
| 迭代进化 | `python src/ai/iterative_train.py --ai1 cnn_v4.99 --ai2 cnn_v4.100 --sample_dir samples` |
| CNN v11 缓冲区训练 | `python src/ai/scripts/mcts_train_cnn_v11.py --model cnn_v11.1 --buffer-dir sample_buffer`（--model 可选，默认最高代数） |
| CNN v11 off-policy 训练 | `python src/ai/scripts/mcts_train_cnn_v11_offpolicy.py --model cnn_v11.1 --data-dir renjunet_samples`（--model 可选，默认最高代数） |
| 自对弈+训练循环 | `run_selfplay_train_loop.bat`（循环执行 采样→训练，任一步失败即停止） |
| 运行测试 | `python -m pytest tests/ -v -s` |
