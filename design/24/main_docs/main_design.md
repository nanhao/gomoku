# 设计文档-24：AlphaGo Zero MCTS 训练体系

> 意图 24 | 日期：2026-06-24 | 关联模型：cnn_v7

---

## 1. 概述

在 cnn_v7 双头模型基础上实现 AlphaGo Zero MCTS 强化学习训练体系：自对弈时用 MCTS 搜索提升落子质量，训练时让模型逼近 MCTS 的决策，形成策略迭代正循环。

**与旧体系关系**：新建文件，不修改 `create_samples.py`、`train_main.py`、`engine.py`、`iterative_train.py`。`battle.py` 保持通用不变。`game.py` 仅做向后兼容的纯函数提取（新增 `check_winner` + `is_board_full`，不修改现有类行为），属于基础设施优化。

**范围**：本意图实现自对弈脚本和训练脚本，二者独立运行。评估迭代闭环（新旧模型对战→胜者替换）属于后续意图。

> 📋 **验收方案**：见 [acceptance.md](acceptance.md)，定义 MCTS 算法、样本生成、训练规约对照的 AI 辅助验收流程。

> ⚠️ **模型迁移**：相对视角编码与绝对视角编码不兼容。使用本体系前必须用 `fresh=True` 创建全新模型，旧 `cnn_v7.*.pth` 权重的预测结果在此编码下无意义。

---

## 2. 目录结构（新建/修改文件）

```
src/ai/
├── mcts/
│   ├── __init__.py
│   ├── searcher.py              # 新增：MCTS 搜索器核心
│   └── decision_maker.py        # 新增：MCTS 落子决策器（替代 AIEngine 的决策职责）
├── models/cnn_v7/
│   ├── model.py                 # 修改：_prepare_input 改为相对视角编码
│   └── trainer.py               # 修改：_convert_inputs 改为相对视角编码，改继承 DualHeadTrainer
├── dual_head_trainer.py         # 新增：双头模型训练基类（MSE+CE loss，新体系专用）
├── base_model.py                # 新增：board_vector_to_cnn_tensor_relative()，保留 board_vector_to_cnn_tensor() 不动
└── scripts/
    ├── mcts_self_play.py        # 新增：MCTS 自我对弈脚本
    └── mcts_train.py            # 新增：MCTS 训练脚本

src/game/
└── game.py                      # 修改：提取 check_winner(board, last_move) -> int 为独立纯函数

src/config/
└── mcts.py                      # 新增：MCTS 日志配置（MCTS_LOG_ENABLED=False, MCTS_LOG_DIR="mcts_log"）
```

> **新旧体系分离**：`CNNV7Trainer` 从 `DualHeadTrainer`（新建）继承，不再使用旧 `BaseTrainer`。旧 `BaseTrainer`（KL 散度、单策略头训练）保留不动，继续服务 cnn_v1~v6。`board_vector_to_cnn_tensor`（绝对编码）不删除，cnn_v3~v6 仍依赖它。

---

## 3. 架构

```
mcts_self_play.py ──生成样本──▶ mcts_train.py
       │                              │
       ▼                              ▼
  MCTSMoveDecider               CNNV7Trainer (继承 DualHeadTrainer)
       │                              │
       ▼                              ▼
  MCTSSearcher ──调用模型──▶   CNNV7Model.predict() → (p, v)
  (选择→扩展→评估→回传)        l = (z-v)² - πᵀp_log
```

**类层次（新体系）**：

```
DualHeadTrainer                  # 新建：双头训练基类（MSE + CrossEntropy）
  └── CNNV7Trainer               # 修改：改继承 + 相对编码

BaseTrainer                      # 保留不动（旧体系，KL 散度 + 单策略头）
  └── FCv1Trainer ... CNNV6Trainer
```

- `MCTSMoveDecider` 不继承/不修改 `AIEngine`。棋盘规则复用 `GomokuGame` 的落子校验和终局判定（不在 MCTS 搜索循环内创建 `GomokuGame` 对象，而是在 Searcher 内部维护轻量棋盘副本）。
- 搜索使用 `predict()` 获取 softmax 概率（作为先验 P），训练使用 `forward()` 直接获取 `p_log`（LogSoftmax 原始输出）。
- `mcts_train.py` 创建 `CNNV7Model` → 创建 `CNNV7Trainer(model=model)` → 调用训练方法。训练循环由 `DualHeadTrainer` 提供，不经过旧 `BaseTrainer`。

---

## 4. 组件设计

### 4.1 相对视角编码

**位置**：`src/ai/models/cnn_v7/model.py` — 修改 `_prepare_input`

**新增**：`src/ai/base_model.py` — `board_vector_to_cnn_tensor_relative(board_vector: List[int]) -> Tensor`，从 226 维 board_vector 生成 3 通道相对视角张量。

**输入（3 通道，当前落子方视角）**：

| 通道 | 内容 | 取值 |
|------|------|------|
| 0 | 我方棋子 | 有子=1.0，空=0.0 |
| 1 | 对方棋子 | 有子=1.0，空=0.0 |
| 2 | 我方颜色 | 黑方=1.0，白方=0.0 |

### 4.2 MCTSSearcher

**位置**：`src/ai/mcts/searcher.py`

| 公共方法 | 参数 | 返回值 | 说明 |
|---------|------|--------|------|
| `search` | `state, num_simulations, log_key=None` | `(visit_counts: List[int], root_value: float, root_policy: List[float])` | 执行 MCTS 搜索，返回各动作原始访问计数、根节点平均价值、根节点先验策略分布（施加噪声前的模型原始 softmax 输出，225 维）。π 由调用方按需计算（训练目标用 τ=1.0，动作选择由 `decide()` 按温度决定） |
| `advance_root` | `action: int` | `bool` | 将 action 对应子节点提升为新根，丢弃兄弟子树。返回是否成功（子节点存在时成功）。若不存在则调用方应 `reset()` 后重新搜索 |

**搜索日志 JSON 格式**（构造时 `log_dir` 非空则启用，`search(log_key="...")` → `{log_dir}/{log_key}.json`）：

日志旨在支持 **可视化渲染** 和 **正确性验证**——拿到日志即可逐 simulation 重现并校验 MCTS 全流程。

```json
{
  "board": [0,0,1,...,2],       // 225 维棋盘（0=空,1=黑,2=白）
  "current_player": 1,          // 当前落子方
  "simulations": 800,           // 总模拟次数
  "hyperparams": {               // 本次搜索超参数
    "c_puct": 2.0,
    "dirichlet_epsilon": 0.25,
    "dirichlet_alpha": 0.03,
    "virtual_loss": 3
  },
  "root_noise": [                // 根节点狄利克雷噪声施加后的 P' 值（225维）
    0.005, 0.003, ..., 0.002
  ],
  "simulation_logs": [           // 每轮 simulation 一条记录
    {
      "sim_id": 0,
      "selection": [             // 选择路径（根→叶）
        {
          "state_hash": "a1b2c3",  // hashlib.md5(bytes(board)+bytes([player])).hexdigest()[:8]，确定性哈希，跨进程/跨运行可比对
          "current_player": 1,     // 该节点轮到谁落子（用于校验回传取反）
          "action": 112,           // 扁平索引 0-224
          "P": 0.05,               // 先验概率
          "Q": 0.30,               // 平均价值（从当前玩家视角）
          "N": 12,                 // 访问次数
          "sum_N_parent": 45,      // 父节点总访问次数（用于校验 U）
          "U": 0.08,               // PUCT 探索项
          "puct_score": 0.38       // Q + U 最终得分
        }
      ],
      "leaf": {
        "state_hash": "d4e5f6",
        "is_terminal": false,    // 是否终局
        "value": 0.45,           // 评估值（模型输出或规则判定）
        "policy": [              // 模型策略头输出的 log-probability（225维）
          -3.2, -4.1, ..., -2.8
        ]
      },
      "backup": [                // 回传路径（叶→根，与 selection 逆序）
        {
          "state_hash": "d4e5f6",  // 与 selection 中同节点 hash 一致
          "current_player": 2,     // 该节点视角（backup 值 v 按此视角取反后传给父节点）
          "action": 112,
          "N_before": 0, "N_after": 1,
          "W_before": 0.0, "W_after": 0.45,
          "Q_before": 0.0, "Q_after": 0.45
        }
      ]
    }
  ],
  "result": {
    "visit_counts": [12, 8, ..., 3],  // 各动作最终访问计数（225维）
    "root_value": 0.42,                // 根节点平均价值
    "pi": [0.10, 0.07, ..., 0.02]      // τ=1.0 归一化分布（225维）
  }
}
```

> **正确性校验要点**：
> - `U = c_puct × P × √sum_N_parent / (1 + N)` —— 可从 log 中取各字段验算
> - `Q_after = W_after / N_after` —— 回传后验算
> - `W_after` 在相邻层间符号翻转（玩家切换取反）
> - Virtual Loss 在 selection 前后 N 临时增加、backup 时已撤销

**内部设计**：
- 搜索树：`Dict[state_hash, Node]`。`state_hash = hashlib.md5(bytes(board) + bytes([player])).hexdigest()[:8]`。使用确定性哈希确保日志跨运行可比对（Python `hash()` 受 PYTHONHASHSEED 影响不稳定）。
- Node：`edges: Dict[action, {N, W, Q, P, child_hash}]`。`child_hash` 存储子节点 state_hash，用于子树复用（`advance_root` 沿此引用定位新根）。
- 终局判定：从 `GomokuGame` 提取五连判定逻辑为独立纯函数 `check_winner(board: List[int], last_move: int) -> int`，放到 `game/` 模块中。返回值：`0=无胜者, 1=黑胜, 2=白胜`，与 `GomokuGame.winner` 一致。**同时新增 `is_board_full(board: List[int]) -> bool`**，检测棋盘是否已满（225 个位置全部非空）。Searcher 在选择阶段判断终局时需同时检查：`check_winner(...) != 0`（有人胜）或 `is_board_full(...)`（平局），任一成立即停止 selection。平局时叶节点 `v = 0`，由规则判定，不调用模型。
- **Batch 推理 + Virtual Loss**：选择路径时对已选边施加临时惩罚（`W -= virtual_loss, N += virtual_loss`，默认 virtual_loss=3），使后续 simulation 分叉到不同叶节点；收集多个叶节点后 batch 评估，回传前撤销 Virtual Loss。batch 大小由 Virtual Loss 自然控制并发度，上限由 `mcts_batch_size`（默认 64）限制，防止极端情况单 batch 过大。

**关键超参数（构造时传入）**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `c_puct` | 2.0 | PUCT 探索系数 |
| `dirichlet_epsilon` | 0.25 | 狄利克雷噪声比例 |
| `dirichlet_alpha` | 0.03 | 狄利克雷分布参数 |
| `virtual_loss` | 3 | Virtual Loss 值 |
| `mcts_batch_size` | 64 | 单 batch 最大叶节点数 |

### 4.3 MCTSMoveDecider

**位置**：`src/ai/mcts/decision_maker.py`

| 公共方法 | 参数 | 返回值 | 说明 |
|---------|------|--------|------|
| `decide` | `board, current_player, log_key=None` | `(action: int, visit_counts: List[int], log_path: str|None, root_policy: List[float])` | 调用 `search()`，按温度选子，返回 action + 原始访问计数 + 日志路径 + 根节点原始策略 |
| `make_move` | `action: int` | — | 调用 `searcher.advance_root(action)`，复用子树 |

- `action` 为 0-224 扁平索引。坐标转换：`x = action // 15; y = action % 15`。
- `visit_counts`：MCTS 原始访问计数，调用方按需计算 π（τ=1.0 用于训练目标）。
- `root_policy`：根节点模型原始 softmax 输出（施加噪声前），供调用方写入 `record.json` 的 `output` 字段。
- `log_key`：日志标识，透传给 `search()`。**调用方（self_play.py）负责生成唯一 key**，如 `"game0_move12"`。
- `log_path`：日志文件完整路径。`Searcher` 从 `src/config/mcts.py` 读取 `MCTS_LOG_ENABLED` 和 `MCTS_LOG_DIR`；禁用时返回 `None`，启用时返回 `{MCTS_LOG_DIR}/{log_key}.json`。调用方写入 `record.json`。无需透传 `log_dir`。
- **子树复用**：`advance_root(action)` 沿边结构中的 `child_hash` 定位子节点，将其提升为新根，丢弃兄弟子树释放内存。若 action 对应子节点不存在（N=0），返回 `False`，调用方应 `reset()` 清空搜索树，下轮 `search()` 从新建根开始。

### 4.4 自我对弈脚本

**位置**：`src/ai/scripts/mcts_self_play.py`

#### 4.4.1 CLI 参数

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | 是 | — | 权重文件路径（加载此 .pth 做 MCTS 推理） |
| `--games` | 是 | — | 对弈局数 |
| `--simulations` | 否 | 800 | 每步 MCTS 模拟次数 |
| `--c-puct` | 否 | 2.0 | PUCT 探索系数 |
| `--temp-steps` | 否 | 8 | 温度探索步数（前 N 步 τ=1.0） |
| `--temperature` | 否 | 1.0 | 温度参数 |
| `--dirichlet-epsilon` | 否 | 0.25 | 狄利克雷噪声比例 |
| `--dirichlet-alpha` | 否 | 0.03 | 狄利克雷分布参数 |
| `--output` | 否 | `{model}_mcts_{timestamp}` | 输出目录名 |
| `--device` | 否 | cuda | 推理设备 |

#### 4.4.2 输出样本

写入 `{output_dir}/sample.json`：

```json
[
  {"state": [...], "pi": [...], "z": 1},
  {"state": [...], "pi": [...], "z": -1},
  ...
]
```

- 一个 `sample.json` 包含本局所有步的样本数组（每步一条，含终局步）。
- `state`：226 维 `board_vector`（前 225=棋盘按行展开，末位=当前落子方 1/2）。训练时通过 `board_vector_to_cnn_tensor_relative` 转为相对视角张量。
- `pi`：225 维 MCTS 归一化访问计数分布（τ=1.0）。
- `z`：对局结果，从当前落子方视角：赢=+1 / 输=-1 / 平=0。

**z 值翻转算法**：设最终获胜方为 `W`（1=黑/2=白/0=平），样本记录的落子方为 `P`（取自 state[225]）。若 `W == 0` → `z = 0`；若 `W == P` → `z = +1`；若 `W != P` 且 `W != 0` → `z = -1`。即对局结束后遍历所有样本，按此规则填入各样本的 `z` 值。

#### 4.4.3 MCTS 日志路径透传

```
self_play.py                            MCTSMoveDecider          MCTSSearcher
     │                                        │                       │
     │  log_key = f"game{i}_move{j}"          │                       │
     ├─ decide(board, player, log_key) ──────▶│                       │
     │                                        ├─ search(state, log_key)▶
     │                                        │   → 内部读 config/mcts.py
     │                                        │   → {MCTS_LOG_DIR}/{log_key}.json
     │                                        │◀─ (visit_counts, value) ─┤
     │◀─ (action, visit_counts, log_path) ────┤                       │
     │                                        │                       │
     │  写入 record.json:                                             │
     │  mcts_probs = normalize(visit_counts)                          │
     │  mcts_log   = log_path                                         │
```

- 每步 self_play.py 生成唯一 `log_key`（如 `"game0_move12"`），透传给 `decide()` → `search()`。
- `Searcher` 内部从 `src/config/mcts.py` 读取 `MCTS_LOG_ENABLED`（默认 `False`）和 `MCTS_LOG_DIR`（默认 `"mcts_log"`）。禁用时 `log_path` 返回 `None`。

#### 4.4.4 record.json 扩展与 DataRecorder 集成

**record.json 扩展**（每局独立，用于可视化）：

参照现有 `record.json` 结构，MCTS 对局的每条 `history` entry 额外包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `output` | `List[float; 225]` | 沿用现有字段，存储模型策略头原始输出（softmax 概率）。与 `mcts_probs` 语义不同：前者是模型直接 softmax 输出，后者是 MCTS 搜索后验分布 |
| `mcts_probs` | `List[float; 225]` | MCTS 搜索返回的 π 分布（访问计数归一化） |
| `mcts_log` | `str` | 对应 MCTS 搜索日志文件路径（如 `mcts_logs/game5_move12.json`） |

记录时机：每步 `decide()` 返回后，调用方从 `visit_counts` 计算 π（τ=1.0），写入 `mcts_probs`；`log_path` 写入 `mcts_log`。

**DataRecorder 集成**：给 `DataRecorder.record_move()` 增加可选参数 `mcts_probs=None, mcts_log=None`，保持向后兼容。MCTS 自对弈脚本调用时传入这两个字段。

### 4.5 训练脚本

**位置**：`src/ai/scripts/mcts_train.py`

#### 4.5.1 CLI 参数

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | 是 | — | 初始权重文件路径（以此为起点训练） |
| `--train-from-dir` | 是 | — | 训练数据目录（sample.json 所在目录） |
| `--epochs` | 是 | — | 训练轮数 |
| `--output` | 否 | `--model` 的值（覆盖） | 训练后权重保存路径。**注意**：默认覆盖原始权重文件，训练崩溃可能导致权重丢失。建议指定到新路径（如 `{model}_trained.pth`） |
| `--batch-size` | 否 | 2048 | 训练 batch size |
| `--lr` | 否 | 0.001 | 初始学习率 |
| `--l2-reg` | 否 | 1e-4 | AdamW weight_decay |
| `--val-split` | 否 | 0.1 | 验证集比例 |
| `--no-shuffle` | 否 | 关闭（默认 shuffle） | 关闭训练数据 shuffle |
| `--device` | 否 | cuda | 训练设备 |

#### 4.5.2 损失函数

\[
(p_{\log}, v) = f_\theta(s)
\]
\[
l = (z - v)^2 - \pi^\top p_{\log}
\]

训练时直接调用 `model.forward(input_tensor)` 获取 `(p_log, v)`，不使用 `model.predict()`（后者 exp 后输出概率，非 LogSoftmax 原始值）。

> ⚠️ **关键**：`p_log` 是 LogSoftmax 输出（值 ≤0），**严禁**再取 `log()`。
>
> **正确**：`loss_policy = -(pi * p_log).sum(dim=1).mean()`
>
> **L2 正则**：通过 `AdamW(params, weight_decay=1e-4)` 实现，**不在 loss 中显式加项**。

**日志输出**：每个 epoch 输出训练集 `value_loss`, `policy_loss`, `total_loss`；若 `--val-split > 0`，同步输出验证集指标。

#### 4.5.3 DualHeadTrainer

| 项 | 内容 |
|----|------|
| 位置 | `src/ai/dual_head_trainer.py` |
| 损失 | `loss_value = F.mse_loss(v, z)`，`loss_policy = -(π * p_log).sum(dim=1).mean()`，`total = value_loss + policy_loss` |
| 数据格式 | `(state: List[int; 226], π: List[float; 225], z: int)` — 加载自 sample.json |
| 输入转换 | `_convert_inputs(state) -> Tensor`，子类覆盖为相对编码 |
| 前向 | `p_log, v = model.forward(input_tensor)` — 直接取双头，不丢弃 value |
| 保存 | 训练完成后 `model.save(output_path)` |

> `DualHeadTrainer` 不继承 `BaseTrainer`，是完全独立的训练基类。`CNNV7Trainer` 改为继承 `DualHeadTrainer`，覆盖 `_convert_inputs` 使用相对编码。`mcts_train.py` 创建 `CNNV7Trainer` 实例并调用其训练方法即可。

---

## 5. 关键算法

> 本节为公式速查，实现细节见 §4.2（Searcher）、§4.3（MoveDecider）、§9（训练规约）。

### 5.1 PUCT 选择（详见 §4.2）

\[
a = \arg\max_a\left(Q(s,a) + c_{\text{puct}} P(s,a) \frac{\sqrt{\sum_b N(s,b)}}{1+N(s,a)}\right)
\]

### 5.2 扩展与评估（详见 §4.2）

叶节点：`(p, v) = model.predict(state)`，创建子边 `{N=0, W=0, Q=0, P=p}`。终局 v 由 `check_winner()` 规则判定。支持 batch 推理（多叶节点合并）。

### 5.3 回传取反（详见 §4.2）

\[
N \mathrel{+}= 1,\quad W \mathrel{+}= v,\quad Q = W/N,\quad v_{\text{parent}} = -v_{\text{child}}
\]

### 5.4 落子选择（详见 §4.3、§4.4）

\[
\pi(a|s_0) = \frac{N(s_0, a)^{1/\tau}}{\sum_b N(s_0, b)^{1/\tau}}
\]

根节点狄利克雷噪声：

\[
P'(s,a) = (1-\varepsilon)P(s,a) + \varepsilon \cdot \text{Dir}(\alpha)
\]

---

## 6. 交互流程图

| 图 | 文件 |
|----|------|
| MCTS 搜索四阶段 | [mcts_search_flow.puml](mcts_search_flow.puml) |
| 自我对弈主循环 | [self_play_flow.puml](self_play_flow.puml) |
| 训练流水线 | [training_flow.puml](training_flow.puml) |

---

## 7. 非功能设计决策

| 决策 | 内容 |
|------|------|
| 代码隔离 | 新建文件，不修改旧体系脚本（`create_samples.py`、`train_main.py` 等） |
| GPU 推理 | MCTS 内部 batch 推理（多叶节点合并为 batch，减少 GPU 调用次数） |
| 性能目标 | 同等条件对弈速度不超过旧体系 5 倍 |
| 兼容性 | `cnn_v7.predict()` 签名不变，但编码从绝对改为相对后语义已变。旧体系（battle.py / engine.py / create_samples.py）不应加载 MCTS 训练的相对编码权重使用，否则预测结果无意义。旧体系自身的 cnn_v1~v6 不受影响 |
| 正则化 | 采用 AdamW decoupled weight decay（`weight_decay=1e-4`）而非显式 L2 正则项。原因：①AdamW 解耦 weight decay 与自适应学习率，比 SGD+L2 在自适应优化器下更优；②`1e-4` 参照 AlphaGo Zero 论文惯例；③不在 loss 中显式加项，保持损失函数纯净。不考虑 Dropout——卷积网络已有足够归纳偏置 |

---

## 8. 术语

| 术语 | 定义 |
|------|------|
| MCTS | 蒙特卡洛树搜索 |
| PUCT | Predictor + UCT |
| π | MCTS 归一化访问计数分布（训练目标） |
| z | 对局结果（当前视角：赢+1 / 输-1 / 平0） |
| p_log | 策略头 LogSoftmax 输出（log-probability） |
| τ | 温度参数，控制落子随机性 |
| board_vector | 226 维向量（前 225=棋盘按行展开，末位=落子方） |

---

## 9. 训练过程规约

本节以精细粒度文档化训练单步逻辑，作为 `dual_head_trainer.py` 的对照规约。
代码以 `_train_step` 为唯一入口方法，核心步骤在入口方法及其直接调用链内完成。允许内部辅助方法，但审查时从入口方法出发即可追溯全部训练逻辑。

### 9.1 数据加载

| 步号 | 操作 | 输入/形状 | 输出/形状 | 说明 |
|------|------|-----------|-----------|------|
| L1 | 读取 sample.json | `str`（文件路径） | `samples: List[dict]` | 解析 JSON 数组 |
| L2 | 提取 state | `samples[i]["state"]` | `states: Tensor (N,226) float32` | 226 维 board_vector |
| L3 | 转换相对编码 | `states (N,226)` | `inputs: Tensor (N,3,15,15) float32` | 逐样本调用 `board_vector_to_cnn_tensor_relative` + `torch.cat` |
| L4 | 提取 π | `samples[i]["pi"]` | `pi_targets: Tensor (N,225) float32` | MCTS 归一化访问计数分布 |
| L5 | 提取 z | `samples[i]["z"]` | `z_targets: Tensor (N,1) float32` | 对局结果 ∈{-1,0,1}，reshape(-1,1) |
| L6 | shuffle | 上述 4 个 tensor | 同上（行序随机） | `torch.randperm`，`--no-shuffle` 时跳过 |
| L7 | 分割验证集 | `(N,*)` | `(M,*), (N-M,*)` | 按 `val_split` 比例，无验证集时全为训练集 |

### 9.2 单 Batch 训练步（核心）

**入口方法**：以下 T1-T12 通过 **入口方法** `_train_step(inputs, pi_targets, z_targets) -> Dict` 及其直接调用链完成。
Epoch 循环和 val 评估均以此方法为唯一调用入口。

**方法签名**：
```
_train_step(inputs: Tensor(B,3,15,15), pi_targets: Tensor(B,225), z_targets: Tensor(B,1))
  -> {"value_loss": float, "policy_loss": float, "total_loss": float, "grad_norms": Dict[str,float]}
```

| 步号 | 操作 | 张量形状 | 约束/验证 |
|------|------|----------|-----------|
| T1 | `self.optimizer.zero_grad()` | — | 清零梯度 |
| T2 | `p_log, v = self.model.network(inputs)` | `p_log (B,225)`, `v (B,1)` | `CNNV7Network.forward` 返回双头 |
| T3 | 校验 p_log 形状 | `p_log.shape == (B, 225)` | — |
| T4 | 校验 p_log 值域 | `p_log.max() <= 0` | **LogSoftmax 输出 ≤0，若 >0 则说明被二次取了 log** |
| T5 | 校验 v 形状 | `v.shape == (B, 1)` | — |
| T6 | 校验 v 值域 | `-1 <= v.min() <= v.max() <= 1` | **Tanh 输出 ∈[-1,1]** |
| T7 | `value_loss = F.mse_loss(v, z_targets)` | `标量 Tensor` | `z_targets` 值域 {-1,0,1} |
| T8 | `policy_loss = -(pi_targets * p_log).sum(dim=1).mean()` | `标量 Tensor` | **p_log 已是 LogSoftmax，绝不二次取 log** |
| T9 | `total_loss = value_loss + policy_loss` | `标量 Tensor` | 等权相加，无额外系数 |
| T10 | `total_loss.backward()` | — | 反向传播 |
| T11 | 逐参数梯度范数 | `{name: p.grad.norm().item() for name, p in self.model.network.named_parameters() if p.grad is not None}` | **value_fc.weight 梯度必须非零**，证明价值头参与训练 |
| T12 | `self.optimizer.step()` | — | AdamW, lr=0.001, weight_decay=1e-4 |

> **T4/T6 校验为 assert**：不满足时立即抛出 `AssertionError`，禁止静默通过。T11 的 value_fc 零梯度检查通过 `grad_norms` 返回值留给调用方/验证工具判断。

### 9.3 Epoch 循环

`train_epochs()` 方法以 `_train_step` 为唯一入口，循环体内所有训练张量操作均通过该入口完成。

| 步号 | 操作 | 说明 |
|------|------|------|
| E1 | `for epoch in range(epochs):` | — |
| E2 | shuffle：`indices = torch.randperm(N)` | 每 epoch 重新 randperm，行序同步打乱 |
| E3 | 按 `batch_size` 分 batch | 逐 batch 调用 `_train_step`。最后 batch 可能不足 batch_size |
| E4 | 聚合 epoch 均值：`sum(各 batch 返回的 loss) / n_batches` | 三个 loss 各自取 epoch 均值 |
| E5 | 若 val 集存在：`model.network.eval()` → 对 val 集逐 batch 执行 T2-T9（无 grad，无 backward） | 仅前向，不更新参数。完成后 `model.network.train()` |
| E6 | 输出日志 | `value_loss`, `policy_loss`, `total_loss`, 如有 val 同步输出 |

### 9.4 保存

| 步号 | 操作 |
|------|------|
| S1 | `torch.save(self.model.network.state_dict(), output_path)` |


