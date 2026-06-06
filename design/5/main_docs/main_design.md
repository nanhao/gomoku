# 主设计文档：去掉五连追溯奖励机制和 Cycle 模式

> **行号说明**：本文档中所有行号基于变更前代码版本，实施时以实际代码为准。

| 字段 | 值 |
|------|-----|
| intention_id | 5 |
| PRD | requirements/5/PRD.md |
| 子文档 | interface_changes.md、test_design.md |

---

## 1. 设计概览

本次变更是纯删除操作，无新增功能。涉及两个独立但可并行执行的删除任务：

1. **FR-1**：删除五连追溯奖励（LineTracer 及 Evaluator 中的相关代码）
2. **FR-2**：删除 Cycle 模式（SelfPlayCycle 及 train_main/runners 中的相关代码）

交互图见 [cleanup_flow.puml](file://d:/projects/gomoku/design/5/main_docs/cleanup_flow.puml)。

---

## 2. 删除清单

### 2.1 整文件删除

| 文件 | 类别 | 说明 |
|------|------|------|
| `ai/line_tracer.py` | 源码 | LineTracer 模块（143 行） |
| `ai/self_play_cycle.py` | 源码 | SelfPlayCycle 模块（195 行） |
| `tests/ai/test_line_tracer.py` | 测试 | LineTracer 单元测试（229 行） |
| `tests/ai/test_self_play_cycle.py` | 测试 | SelfPlayCycle 编排测试（262 行） |
| `tests/ai/primary/test_evaluator_winning_line.py` | 测试 | 五连追溯奖励集成测试（464 行） |

### 2.2 模块内修改

详见 [interface_changes.md](file://d:/projects/gomoku/design/5/main_docs/interface_changes.md)。

---

## 3. 关键流程变更

### 3.1 Evaluator._compute_all_rewards() 简化

**现有流程**：
```
for each step:
  compute raw_rewards via RewardCalculator        ← 保留
  if step has line_bonus:
    blend raw_rewards[pos] with line_bonus         ← 删除
  apply masked_softmax                             ← 保留
```

**简化后流程**：
```
for each step:
  compute raw_rewards via RewardCalculator
  apply masked_softmax
```

说明：`winner` 和 `end_board` 参数均移除——简化后 `_compute_all_rewards` 内部不再使用它们（仅 `line_bonuses` 计算块使用 `winner` 和 `end_board`，已一并删除）。`winner` 的有效性判断仍在 `evaluate_record` 中完成（`winner == 0` 时抛 ValueError），但不再传递给 `_compute_all_rewards`。

删除内容：line_bonuses 的计算（第 153-160 行）、叠加逻辑（第 188-199 行）、`winner` 参数（第 132 行）、`end_board` 参数（第 133 行）、`evaluate_record` 中 `winner` 传参（第 118 行）、`evaluate_record` 中 `end_board` 提取（第 112 行）。

### 3.2 train_main.py CLI 简化

**现有模式互斥**：`--cycle` vs `--train-from-dir`（二选一）

**简化后**：仅 `--train-from-dir`，不指定时报错退出。

删除的 argparse 参数：`--cycle`、`--rounds`、`--games`、`--rate`、温度采样参数组（5 个）。

---

## 4. run_self_play_and_evaluate 函数签名变更

该函数是 SelfPlayGenerator 的核心，删除 SelfPlayCycle 后仍保留（SelfPlayGenerator 依赖它）。

**现有签名**：
```
run_self_play_and_evaluate(model, games, rate=0.9, ai_data_dir, train_data_dir, max_workers, name_prefix, project_root, temp_config)
```

**简化后签名**：
```
run_self_play_and_evaluate(model, games, ai_data_dir, train_data_dir, max_workers, name_prefix, project_root, temp_config)
```

删除 `rate` 参数。内部创建 Evaluator 时不再传 `decay_rate`。

同理，`SelfPlayGenerator.generate()` 删除 `rate` 参数。

---

## 5. CLI 破坏性变更与迁移指引

删除 `--cycle`、`--rounds`、`--games`、`--rate` 及温度采样参数组（5 个）对已有用户脚本是破坏性变更，运行时将报 SystemExit。

**CLI 报错消息变更**：

| 变更前 | 变更后 |
|--------|--------|
| "请指定 --cycle 或 --train-from-dir 参数" | "请指定 --train-from-dir 参数" |
| "以下参数不能同时使用: ..." | 不再需要（仅一个模式） |

**迁移方式**：

| 原用法 | 替代方案 |
|--------|----------|
| `--cycle`（自我对弈+评价+训练循环） | 先用 `create_samples.py` 或 `SelfPlayGenerator` 生成数据，再用 `--train-from-dir` 训练 |
| `--rounds N --games M` | `SelfPlayGenerator.generate(games=M)` 重复调用 N 次 |
| `--rate R` | 无替代，decay_rate 参数已删除 |
| 温度采样参数组 | `SelfPlayGenerator.generate(temp_config=TempConfig(...))` |

---

## 6. 验收策略

详见 [test_design.md](file://d:/projects/gomoku/design/5/main_docs/test_design.md)。

核心原则：
- 删除的模块对应的测试文件整文件删除
- 修改的模块对应的测试文件需适配更新
- 不新增测试用例，只删除和更新已有测试
- 全部测试通过作为最终验收门禁

验收用例编号与 PRD 对齐：AC-1（五连追溯移除）、AC-2（Cycle 模式移除）、AC-3（SelfPlayGenerator 不受影响）、AC-4（相关测试清理）、AC-5（全局引用清理）