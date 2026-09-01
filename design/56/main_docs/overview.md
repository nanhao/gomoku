# cnn_v11 双代历史最强评估（intention 56）

## 问题

当前训练循环（`CnnV11TrainLoop`）每次 checkpoint 评估只与**当前代历史最强**（best，即训练起点 `gen N` 模型）对战一场，胜率 ≥ 55% 即晋升。这存在两个问题：

1. **对抗退化风险**：checkpoint 只需针对当前代模型优化即可晋升，可能在某一方面"过拟合"当前代而整体实力并未真正超越历史最强链。
2. **评估执方不公**：C++ `GameRunner` 双模型模式每局**随机分配执方**，五子棋黑方先手优势显著，引入 ±1~3% 胜率噪声，直接影响 55% 阈值判定。

## 核心决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 评估对手 | checkpoint vs best（gen N）+ vs 上一代历史最强（gen N-1）两场 | 意图明确：两代历史最强的胜率都 ≥ 55% 才晋升 |
| 上一代定位 | `gen{N-1}/cnn_v11_torchscript.pt`（磁盘持久化文件） | 上一代 best 已持久化在 gen 目录，直接引用无需复制 |
| 无上一代时 | 跳过 vs prev 评估（仅单场评估），打印提示 | gen1 是最早一代，不存在更早的历史最强（用户确认） |
| 局数分配 | 每场各 `eval_games` 局（总对局数 2×eval_games） | 保持单场统计精度不变（用户确认） |
| 胜率语义 | `best_win_rate = max(各轮 min(vs_best, vs_prev))` | 取两场较低者，晋升判断保守可靠（用户确认） |
| 执方公平性 | `GameConfig.alternate_colors` 标志（默认 false） | 仅评估器显式启用交替，其他调用方（mcts_battle_cpp.py 等）行为不变（用户确认） |

## 方案概要

```
训练循环（每轮）:
  for i in range(updates_per_checkpoint):
      mini-batch 训练
  save checkpoint (pth + TorchScript)
  win_rate_best  = evaluator.evaluate(checkpoint, best, eval_games, temp_steps=2)
  win_rate_prev  = evaluator.evaluate(checkpoint, prev_best, eval_games, temp_steps=2)  # 无上一代时跳过
  晋升条件: win_rate_best >= 55% AND (win_rate_prev is None OR win_rate_prev >= 55%)
```

## 新增模块

无新增模块。仅修改既有模块与 C++ 基础设施。

## 模块变更

### 1. `cpp/src/game_runner.h` — `GameConfig` 增加字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `alternate_colors` | bool | `false` | 双模型对战模式是否按局号奇偶交替执方。`true` 时偶数局 model1 执黑、奇数局执白；`false` 时保持随机分配（向后兼容） |

### 2. `cpp/src/game_runner.cpp` — `run_single_game` 执方逻辑

双模型模式（非自对弈）下：
- `config_.alternate_colors == true` → 按 `game_index % 2` 交替执方（与自对弈模式相同的分配逻辑）
- `false` → 保持现有随机分配

### 3. `cpp/src/bindings.cpp` — GameConfig 绑定

暴露 `alternate_colors` 读写字段。

### 4. `src/ai/checkpoint_evaluator.py` — 启用交替执方

`evaluate()` 构建 `cpp_config` 时设置 `alternate_colors = True`，保证评估执方公平。

### 5. `src/ai/cnn_v11_train_loop.py` — 双代评估

**`__init__` 新增：**
- 解析上一代 best 的 TorchScript 路径：`src/ai/models/cnn_v11/gen{N-1}/cnn_v11_torchscript.pt`
- 文件存在 → `self._prev_best_ts_path`；不存在（gen1 或代数不连续）→ `None`

**`run()` 评估部分：**
- 第 1 场：`win_rate_best = evaluator.evaluate(checkpoint_ts, best_ts, ...)`
- 第 2 场（prev 存在时）：`win_rate_prev = evaluator.evaluate(checkpoint_ts, prev_best_ts, ...)`
- 本轮有效胜率 `effective = min(win_rate_best, win_rate_prev)`（prev 不存在时即 win_rate_best）
- 晋升判定：`win_rate_best >= 0.55 and (win_rate_prev is None or win_rate_prev >= 0.55)`
- 失败计数：任一胜率 < 0.55 计一次失败
- `best_win_rate = max(best_win_rate, effective)`
- 打印两场胜率

**`TrainLoopResult` 新增字段：**

| 字段 | 类型 | 说明 |
|------|------|------|
| `win_rate_vs_best` | `Optional[float]` | 最后一次评估对 best 的胜率 |
| `win_rate_vs_prev` | `Optional[float]` | 最后一次评估对上一代 best 的胜率，无上一代时为 None |

### 6. `src/ai/scripts/mcts_train_cnn_v11.py` — CLI 打印

`[晋升规则]` 部分更新为双代评估说明（对 best 与上一代 best 各评估 `eval_games` 局，两场都 ≥ 55% 才晋升；无上一代时仅单场）。

## 关键流程

参见 [main_flow.puml](main_flow.puml)

## 测试用例集

### C++（`cpp/tests/unit/`）

| ID | 场景 | 输入 | 预期 |
|----|------|------|------|
| TC-CPP-01 | alternate_colors=false 时执方随机 | 双模型 10 局，GameConfig.alternate_colors=false | model1 执黑局数在 0~10 之间（随机，不强制交替） |
| TC-CPP-02 | alternate_colors=true 时执方交替 | 双模型 10 局，GameConfig.alternate_colors=true | model1 偶数局执黑、奇数局执白，严格交替 |

### Python 单元测试（`tests/ai/unit/test_cnn_v11_train_loop.py`）

| ID | 场景 | 输入 | 预期 |
|----|------|------|------|
| TC-TL-10 | 双代都通过晋升 | 从 cnn_v11.5 训练（prev=gen4 存在），两场胜率 0.58/0.57 | `stopped_by == "promoted"`；`win_rate_vs_best == 0.58`、`win_rate_vs_prev == 0.57`；evaluate 调用 2 次 |
| TC-TL-11 | vs prev 失败不晋升 | 从 cnn_v11.5 训练，胜率 0.60/0.45 | 不晋升，计失败；`best_win_rate == 0.45`（取较低者） |
| TC-TL-12 | 无上一代单场评估 | 从 cnn_v11.1 训练（gen0 不存在） | 仅调用 1 次 evaluate；`win_rate_vs_prev is None`；晋升判定只看单场 |
| TC-TL-13 | 双代判定失败重试 | 第 1 轮 vs prev 失败，第 2 轮双过 | 第 2 轮晋升，`checkpoints_evaluated == 2` |

### Python 集成测试

| ID | 场景 | 输入 | 预期 |
|----|------|------|------|
| TC-CE-05 | 评估执方交替 | 集成环境双模型评估 | GameRunner 结果中 model1 偶数局执黑、奇数局执白 |

## 晋升后产物

不变：晋升成功时在 `src/ai/models/cnn_v11/gen{N+1}/` 下生成 `cnn_v11.pth` 与 `cnn_v11_torchscript.pt`。

上一代（gen N-1）与当前代（gen N）的权重在晋升过程中**不被覆盖**（晋升写入 gen{N+1}），天然满足"保存历史最强和上一代历史最强"的要求。

## 变更记录（2026-09-01）

### 评估温度参数变更：temp_steps 以 CLI 配置为准（当前为 0，全程贪婪）

上文方案概要中 `evaluator.evaluate(..., temp_steps=2)` 为历史实现。2026-09-01 起：

- `CnnV11TrainLoop.run()` 两场评估的 `temp_steps` 改为 `self._mcts_config.temp_steps`，不再硬编码 2
- CLI `mcts_train_cnn_v11.py` 配置 `MctsSelfPlayConfig(temp_steps=0)`，即评估全程贪婪
- `CnnV11TrainLoop.__init__` 新增保存 `self._mcts_config` 属性

**影响：** 评估不再有前 2 步温度采样，全程贪婪。后续调整评估温度只需修改 CLI 中 `MctsSelfPlayConfig` 的 `temp_steps` 值。双代评估、交替执方、55% 晋升阈值等其余设计不变。
