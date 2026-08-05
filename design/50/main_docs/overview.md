# cnn_v11 训练脚本（样本缓冲区驱动）

## 问题

现有训练流程（`dual_head_trainer.py`）从静态 `sample.json` 文件加载全部样本，按 epoch 循环训练。cnn_v11 需要一种新的训练范式：从**持续运行的样本缓冲区**中随机抽取 mini-batch 进行增量训练，每 N 步产生 checkpoint，通过 C++ MCTS 对战评估决定是否晋升。

## 核心决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 训练范式 | 随机 mini-batch（无 epoch 概念） | 缓冲区由独立进程持续写入，样本动态变化 |
| 损失函数 | `MSE(value) + CrossEntropy(policy)` | AlphaGo Zero 标准，复用 `DualHeadTrainer._train_step` |
| 正则化 | AdamW `weight_decay=1e-4` | 意图明确指定，替代 L2 正则项 |
| 评估引擎 | C++ MCTS (`CppMctsBattleAdapter`) | 复用设计 47 的高性能对战基础设施 |
| 评估温度 | `temp_steps=2`（前2步采样，之后贪婪） | 意图指定："前2步采样，之后贪婪" |
| Checkpoint 临时存储 | `temp/train_checkpoint/` | 仅晋升时才正式保存，避免中间产物污染模型目录 |
| 晋升保存 | `gen{N+1}/` 子目录 + TorchScript 导出 | 与现有多代模型体系一致 |
| 样本刷新 | 每次 checkpoint 评估前重新加载缓冲区 | 缓冲区由 `mcts_sample_buffer.py` 持续写入新样本 |
| 停止条件 | 晋升成功 OR 连续 3 次评估胜率 < 55% | 意图明确指定：晋升后脚本停止，由用户决定是否从新 best 重启 |

## 方案概要

```
mcts_train_cnn_v11.py --model cnn_v11.1 --buffer-dir sample_buffer

CnnV11TrainLoop
  ├── CNNV11Trainer (复用 DualHeadTrainer) — 执行 _train_step
  ├── SampleBufferReader — 从缓冲区随机抽取样本
  └── CheckpointEvaluator — 包装 CppMctsBattleAdapter 评估胜率

训练循环:
  while True:
      for i in range(50):  # 50 次 mini-batch
          samples = buffer_reader.random_sample(512)
          trainer._train_step(inputs, pi_targets, z_targets)
          log loss
      save checkpoint (pth + TorchScript)
      win_rate = evaluator.evaluate(checkpoint, best, games=50, temp_steps=2)
      if win_rate >= 55%:
          promote checkpoint → gen{N+1}, 停止
      else:
          consecutive_failures += 1
          if consecutive_failures >= 3: 停止
          继续训练
```

## 新增模块

### 1. `SampleBufferReader` (`src/ai/sample_buffer_reader.py`)

从 `sample_buffer/` 目录读取所有 chunk JSON 文件，提供随机抽样。

**公开接口：**

| 方法 | 说明 |
|------|------|
| `__init__(buffer_dir: str)` | 加载 manifest.json 和所有 chunk 到内存 |
| `random_sample(n: int) -> List[dict]` | 不放回随机抽取 n 条样本。样本不足 n 时返回全部并刷新缓冲区后重试 |
| `refresh() -> int` | 重新扫描缓冲区目录，加载新 chunk。返回新增样本数 |
| `total_samples() -> int` | 返回当前已加载样本总数 |

**样本格式**（与 `DualHeadTrainer._load_samples` 兼容）：
```json
{"state": [...226维...], "pi": [...225维概率分布...], "z": 1.0}
```

### 2. `CheckpointEvaluator` (`src/ai/checkpoint_evaluator.py`)

包装 `CppMctsBattleAdapter`，提供 checkpoint vs best 的对战评估。

**公开接口：**

| 方法 | 说明 |
|------|------|
| `__init__(mcts_config: MctsSelfPlayConfig)` | 初始化 MCTS 配置（simulations, c_puct 等） |
| `evaluate(checkpoint_ts_path: str, best_ts_path: str, num_games: int, temp_steps: int) -> float` | 执行对战，返回 checkpoint 的胜率（0.0~1.0）。双方交替执方，前 `temp_steps` 步温度采样 |

### 3. `CnnV11TrainLoop` (`src/ai/cnn_v11_train_loop.py`)

训练-评估循环编排器。持有 trainer、buffer reader、evaluator，执行完整的训练闭环。

**公开接口：**

| 方法 | 说明 |
|------|------|
| `__init__(model_spec: str, buffer_dir: str, batch_size: int, updates_per_checkpoint: int, eval_games: int, max_consecutive_failures: int, mcts_config: MctsSelfPlayConfig, lr: float, weight_decay: float)` | 加载初始模型（best），导出 best TorchScript，初始化各组件 |
| `run() -> TrainLoopResult` | 执行训练循环，返回结果 |

**`TrainLoopResult`**（dataclass）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `stopped_by` | `str` | 停止原因：`"promoted"` 或 `"max_failures"` |
| `promoted_to` | `Optional[str]` | 晋升后的 gen spec（如 `"cnn_v11.2"`），未晋升时为 None |
| `total_updates` | `int` | 总共执行的 mini-batch 次数 |
| `checkpoints_evaluated` | `int` | 评估过的 checkpoint 数量 |
| `best_win_rate` | `float` | 所有 checkpoint 中的最高胜率 |
| `loss_history` | `List[float]` | 每个 mini-batch 的 total_loss |

### 4. CLI 入口 (`src/ai/scripts/mcts_train_cnn_v11.py`)

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | 必填 | 初始模型 spec（如 `cnn_v11.1`），同时作为初始 best |
| `--buffer-dir` | str | `sample_buffer` | 样本缓冲区目录 |
| `--batch-size` | int | 512 | 每个 mini-batch 的样本数 |
| `--updates-per-checkpoint` | int | 50 | 多少步后触发 checkpoint 评估 |
| `--eval-games` | int | 50 | checkpoint 评估对战局数 |
| `--max-failures` | int | 3 | 连续评估失败上限 |
| `--simulations` | int | 800 | 评估时每步 MCTS 模拟次数 |
| `--c-puct` | float | 2.0 | PUCT 探索系数 |
| `--lr` | float | 0.001 | 学习率 |
| `--weight-decay` | float | 1e-4 | AdamW 权重衰减 |
| `--device` | str | `cuda` | 训练设备 |

## 模块变更

### 新增文件

| 文件 | 类 | 职责 |
|------|-----|------|
| `src/ai/sample_buffer_reader.py` | `SampleBufferReader` | 缓冲区样本读取与随机抽样 |
| `src/ai/checkpoint_evaluator.py` | `CheckpointEvaluator` | C++ MCTS 对战评估封装 |
| `src/ai/cnn_v11_train_loop.py` | `CnnV11TrainLoop` | 训练-评估循环编排 |
| `src/ai/scripts/mcts_train_cnn_v11.py` | — | CLI 入口脚本 |

### 不改动

- `CNNV11Trainer` / `DualHeadTrainer` — 复用 `_train_step`，无修改
- `CNNV11Model` / `CNNV11Network` — 无修改
- `CppMctsBattleAdapter` / `CppBattleGameResult` — 直接复用
- `MctsSelfPlayConfig` — 直接复用
- `SampleBufferManager` — 不修改（训练侧只读不写）
- `LossLogger` — 直接复用

## 关键流程

参见 [main_flow.puml](main_flow.puml)

## 类关系

参见 [class_diagram.puml](class_diagram.puml)

## 测试用例集

### SampleBufferReader

| ID | 场景 | 输入 | 预期 |
|----|------|------|------|
| TC-SBR-01 | 空缓冲区 | `buffer_dir` 下无 chunk 文件 | `total_samples() == 0`；`random_sample(10)` 返回空列表 |
| TC-SBR-02 | 单 chunk 抽样 | 1 个 chunk 含 100 条样本 | `random_sample(50)` 返回 50 条；格式含 `state`, `pi`, `z` 字段 |
| TC-SBR-03 | 请求数超过总量 | 总样本 30 条，`random_sample(100)` | 返回 30 条（全部），不报错 |
| TC-SBR-04 | 不放回抽样 | 总样本 100 条，连续两次 `random_sample(60)` | 第二次返回的样本不与第一次重复 |
| TC-SBR-05 | refresh 增量加载 | 初始 2 chunk 共 200 条，外部新增 1 chunk 含 80 条 | `refresh()` 返回 80；`total_samples()` 变为 280 |
| TC-SBR-06 | 损坏的 chunk 文件 | 一个 chunk JSON 格式损坏 | 跳过该文件，其他 chunk 正常加载 |
| TC-SBR-07 | manifest 与实际文件不一致 | manifest 记录 3 chunk 但磁盘只有 2 个 | 按实际存在的文件加载，缺失的静默跳过 |

### CheckpointEvaluator

| ID | 场景 | 输入 | 预期 |
|----|------|------|------|
| TC-CE-01 | 正常评估 | 两个有效 TorchScript 模型，50 局 | 返回 0.0~1.0 之间的胜率；双方执方均匀分布 |
| TC-CE-02 | 模型文件不存在 | checkpoint TS 文件路径无效 | 抛出 `FileNotFoundError` |
| TC-CE-03 | 不同 temp_steps | `temp_steps=2` vs `temp_steps=0` | temp_steps=2 时前 2 步有随机性；temp_steps=0 时全程贪婪 |
| TC-CE-04 | Ctrl+C 中断 | 评估过程中发送 SIGINT | 不抛异常，尽量返回已完成局数的结果或标记为中断 |

### CnnV11TrainLoop

| ID | 场景 | 输入 | 预期 |
|----|------|------|------|
| TC-TL-01 | 正常晋升 | 初始模型，checkpoint 胜率 ≥ 55% | `stopped_by == "promoted"`；新 gen 目录已创建 |
| TC-TL-02 | 连续失败停止 | 初始模型，3 次 checkpoint 胜率均 < 55% | `stopped_by == "max_failures"`；`checkpoints_evaluated == 3` |
| TC-TL-03 | 先失败后晋升 | 第 1 次失败，第 2 次晋升 | `stopped_by == "promoted"`；`checkpoints_evaluated == 2` |
| TC-TL-04 | 缓冲区样本不足 | 缓冲区仅 200 条，`batch_size=512` | `random_sample` 返回 200 条并触发 `refresh()`；若刷新后仍不足，使用全部样本训练 |
| TC-TL-05 | 初始模型不存在 | `--model cnn_v11.99` 无对应 gen 目录 | 抛出明确错误信息，脚本退出 |
| TC-TL-06 | 训练 loss 单调下降趋势 | 正常训练 50 步 | `loss_history` 整体呈下降趋势（允许局部波动） |
| TC-TL-07 | best TorchScript 缓存 | 同一 best 模型被多次评估 | 首次导出 TS 后缓存，后续评估直接复用 |
| TC-TL-08 | Ctrl+C 中断训练 | 训练过程中发送 SIGINT | 优雅停止，打印已完成的统计信息，不损坏现有模型 |
| TC-TL-09 | 设备回退 | `--device cuda` 但 CUDA 不可用 | 自动回退到 CPU，打印警告 |

### CLI 入口

| ID | 场景 | 输入 | 预期 |
|----|------|------|------|
| TC-CLI-01 | 必填参数缺失 | 不带 `--model` | 打印 usage，exit code ≠ 0 |
| TC-CLI-02 | 参数边界校验 | `--batch-size 0` | 打印错误，exit code ≠ 0 |
| TC-CLI-03 | 参数类型校验 | `--eval-games abc` | argparse 报错，exit code ≠ 0 |
| TC-CLI-04 | 默认值生效 | 仅传 `--model cnn_v11.1` | 其他参数使用文档声明的默认值 |

## 晋升后产物

晋升成功时，在 `src/ai/models/cnn_v11/gen{N+1}/` 下生成：

| 文件 | 说明 |
|------|------|
| `cnn_v11.pth` | PyTorch 权重文件 |
| `cnn_v11_torchscript.pt` | TorchScript 模型（供 C++ MCTS 使用） |

gen 编号规则：若初始模型为 `cnn_v11.N`，则晋升产物为 `gen{N+1}`。
