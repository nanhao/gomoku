# PRD：去掉五连追溯奖励机制和 Cycle 模式

| 字段 | 值 |
|------|-----|
| intention_id | 5 |
| 版本 | v1.0 |
| 创建日期 | 2026-06-06 |
| 状态 | 待评审 |

---

## 1. 目标与背景

### 1.1 现状问题

**五连追溯奖励机制**（`LineTracer`）在 Evaluator 中为获胜方的五连落子步骤叠加额外衰减奖励。该机制存在以下问题：

- 五连追溯是一种"事后回溯"信号，与棋型评估奖励（活四、冲四等即时信号）在语义上不一致
- 融合策略经多次调整（max → 加法），仍存在信号冲突和裁剪溢出问题
- 棋型评估本身已经能识别活四、冲四等关键棋型，五连追溯 bonus 的增量价值有限

**Cycle 模式**（`SelfPlayCycle`）将自我对弈数据生成和模型训练耦合在一个闭环循环中。项目已实现了数据生成与训练的分离（`SelfPlayGenerator` + `--train-from-dir`），Cycle 模式成为冗余的耦合模式。

### 1.2 目标

1. 去掉五连追溯奖励机制，Evaluator 仅依赖棋型评估计算 reward
2. 去掉 Cycle 模式，保留 `SelfPlayGenerator`（数据生成）和 `--train-from-dir`（训练）两条独立路径
3. 清理所有仅服务于上述两个机制的参数、代码和测试

### 1.3 不做的事

- 不修改 Evaluator 的棋型评估逻辑（RewardCalculator 不变）
- 不修改 masked_softmax 转换逻辑
- 不修改 SampleCreator 的任何模式
- 不修改 SelfPlayGenerator 和 `--train-from-dir` 的任何逻辑
- 不引入新的奖励机制替代五连追溯

---

## 2. 范围与边界

### 2.1 功能范围

#### FR-1：去掉五连追溯奖励机制

| 项目 | 说明 |
|------|------|
| 删除模块 | `ai/line_tracer.py` 整个文件 |
| 修改模块 | `ai/evaluator.py`：去掉 LineTracer 依赖、BLEND_ALPHA 常量、line_bonuses 计算、奖励叠加逻辑 |
| 删除参数 | Evaluator 构造器的 `decay_rate` 参数 |
| 删除 CLI 参数 | `--rate`（仅服务于 LineTracer 的 decay_rate） |
| 保留不变 | RewardCalculator、probability_utils.masked_softmax |

#### FR-2：去掉 Cycle 模式

| 项目 | 说明 |
|------|------|
| 删除模块 | `ai/self_play_cycle.py` 整个文件 |
| 删除方法 | `BaseTrainer.train_cycle()` |
| 删除 CLI 参数 | `--cycle`、`--rounds`、`--games` |
| 删除 CLI 参数组 | 温度采样参数组（`--no-temp-sampling`、`--temperature`、`--temp-moves`、`--temp-side`、`--temp-seed`）仅服务于 `--cycle` 模式，一并移除 |
| 删除运行器 | `ai/runners.py` 中的 `run_cycle()` 函数 |
| 修改 CLI | `ai/train_main.py`：去掉 cycle 相关参数和路由，模式互斥逻辑改为仅 `--train-from-dir` |

#### FR-3：清理受影响的下游调用

| 项目 | 说明 |
|------|------|
| 修改模块 | `ai/self_play_generator.py`：去掉 `rate` 参数（仅传给 Evaluator 的 decay_rate） |
| 修改模块 | `ai/evaluator.py`：`_evaluate_record_worker` 顶层函数去掉 `decay_rate` 参数 |
| 修改模块 | `ai/base_trainer.py`：去掉 `train_cycle()` 方法 |

### 2.2 边界

- **不删除** `SelfPlayGenerator` 及其 CLI 入口（`create_samples.py`）
- **不删除** `--train-from-dir` 及 `run_train_from_dir()`
- **不删除** `RewardCalculator`、`ShapeDetector`、`probability_utils`
- **不删除** `DataRecorder`、`AIGameManager`、`AIEngine`
- **保留** `Evaluator` 类，简化后仍负责：读取 record → 棋型评估 reward → masked_softmax → 输出 sample.json

---

## 3. 功能说明

### 3.1 Evaluator 简化后的职责

去掉五连追溯后，Evaluator 的 `_compute_all_rewards()` 流程：

1. 读取 record.json（history, winner），end_board 字段仍存在于文件中但不再被读取
2. 调用 `_compute_all_rewards(history)`（仅 history 参数，winner 和 end_board 不再传递）
3. 遍历每个 step，获取棋盘状态和当前落子方
4. 对每个空位，用 RewardCalculator 计算 shape reward
5. 对 raw_rewards 应用 masked_softmax 转为概率分布
6. 输出 sample 列表（input, output, mask）

不再有 line_bonuses 的计算和叠加步骤。

### 3.2 CLI 模式简化

去掉 cycle 后，`train_main.py` 仅保留 `--train-from-dir` 模式：

```bash
# 从已有数据训练模型
python ai/train_main.py --model fc_v1 --train-from-dir ai/train_data/fc_v1_self_play_1 --epochs 1000

# mini-batch 训练
python ai/train_main.py --model fc_v1 --train-from-dir ai/train_data/fc_v1_self_play_1 --epochs 1000 --batch-size 32 --shuffle
```

不指定 `--train-from-dir` 时应报错退出（当前行为：不指定任何模式参数时报错）。

### 3.3 数据生成路径

数据生成通过 `create_samples.py`（已有）或 `SelfPlayGenerator`（已有），与本次变更无关。

---

## 4. 验收标准

### AC-1：五连追溯奖励机制已完全移除

| 编号 | 验收步骤 | 自动化方式 |
|------|---------|-----------|
| AC-1.1 | `ai/line_tracer.py` 文件不存在 | `test -f ai/line_tracer.py` 返回 False |
| AC-1.2 | Evaluator 不引用 LineTracer | `grep -r "LineTracer" ai/evaluator.py` 无匹配 |
| AC-1.3 | Evaluator 不引用 line_tracer 模块 | `grep -r "line_tracer" ai/evaluator.py` 无匹配 |
| AC-1.4 | Evaluator.BLEND_ALPHA 常量不存在 | `grep -r "BLEND_ALPHA" ai/evaluator.py` 无匹配 |
| AC-1.5 | `_compute_all_rewards()` 不含 line_bonuses 逻辑 | 代码审查 + 单元测试验证 |
| AC-1.6 | Evaluator 构造器不含 decay_rate 参数 | `grep -r "decay_rate" ai/evaluator.py` 无匹配 |
| AC-1.7 | Evaluator 仍能正确从 record.json 生成 sample.json | 单元测试：构造 record → 调用 evaluate_record → 验证 sample 结构 |
| AC-1.8 | 生成的 sample.json output 仅为棋型评估 + masked_softmax 的结果 | 单元测试：构造已知棋盘 → 验证 reward 值 |
| AC-1.9 | `--rate` CLI 参数不存在 | `python ai/train_main.py --rate 0.9` 报错 |

### AC-2：Cycle 模式已完全移除

| 编号 | 验收步骤 | 自动化方式 |
|------|---------|-----------|
| AC-2.1 | `ai/self_play_cycle.py` 文件不存在 | `test -f ai/self_play_cycle.py` 返回 False |
| AC-2.2 | `BaseTrainer.train_cycle()` 方法不存在 | `grep -r "train_cycle" ai/base_trainer.py` 无匹配 |
| AC-2.3 | `--cycle` CLI 参数不存在 | `python ai/train_main.py --cycle` 报错 |
| AC-2.4 | `--rounds` CLI 参数不存在 | `python ai/train_main.py --rounds 5` 报错 |
| AC-2.5 | `--games` CLI 参数不存在 | `python ai/train_main.py --games 10` 报错 |
| AC-2.6 | 温度采样参数组不存在 | `python ai/train_main.py --temperature 0.8` 报错 |
| AC-2.7 | `run_cycle()` 函数不存在 | `grep -r "run_cycle" ai/runners.py` 无匹配 |
| AC-2.8 | `train_main.py` 不引用 SelfPlayCycle | `grep -r "SelfPlayCycle" ai/train_main.py` 无匹配 |
| AC-2.9 | `--train-from-dir` 模式仍正常工作 | 单元测试：构造样本目录 → 调用 run_train_from_dir → 验证训练执行 |

### AC-3：SelfPlayGenerator 不受影响

| 编号 | 验收步骤 | 自动化方式 |
|------|---------|-----------|
| AC-3.1 | SelfPlayGenerator.generate() 不含 rate 参数 | `grep -r "rate" ai/self_play_generator.py` 在 generate 方法签名中无匹配 |
| AC-3.2 | SelfPlayGenerator 仍能正常生成数据 | 单元测试 |
| AC-3.3 | `create_samples.py` 不受影响 | 单元测试 |

### AC-4：相关测试已清理

| 编号 | 验收步骤 | 自动化方式 |
|------|---------|-----------|
| AC-4.1 | `tests/ai/test_line_tracer.py` 不存在 | `test -f tests/ai/test_line_tracer.py` 返回 False |
| AC-4.2 | `tests/ai/test_self_play_cycle.py` 不存在 | `test -f tests/ai/test_self_play_cycle.py` 返回 False |
| AC-4.3 | `tests/ai/primary/test_evaluator_winning_line.py` 不存在 | `test -f tests/ai/primary/test_evaluator_winning_line.py` 返回 False |
| AC-4.4 | 所有剩余测试通过 | `pytest tests/` 全部通过 |
| AC-4.5 | Evaluator 相关测试已更新（不含 line_bonuses 逻辑） | 代码审查 |
| AC-4.6 | train_main 相关测试已更新（不含 cycle 模式） | 代码审查 |

### AC-5：全局引用已清理

| 编号 | 验收步骤 | 自动化方式 |
|------|---------|-----------|
| AC-5.1 | 项目中无任何对 LineTracer 的引用 | `grep -r "LineTracer" ai/ tests/` 无匹配 |
| AC-5.2 | 项目中无任何对 SelfPlayCycle 的引用 | `grep -r "SelfPlayCycle" ai/ tests/` 无匹配 |
| AC-5.3 | 项目中无任何对 line_tracer 模块的 import | `grep -r "from ai.line_tracer" ai/ tests/` 无匹配 |
| AC-5.4 | 项目中无任何对 self_play_cycle 模块的 import | `grep -r "from ai.self_play_cycle" ai/ tests/` 无匹配 |
| AC-5.5 | 项目中无任何对 train_cycle 的引用（除设计文档/意图文件外） | `grep -r "train_cycle" ai/ tests/` 无匹配 |
| AC-5.6 | 项目中无任何对 decay_rate 的引用 | `grep -r "decay_rate" ai/` 无匹配（Evaluator/self_play_generator/runners/train_main/line_tracer 均已删除或清理） |

---

## 5. 术语定义

| 术语 | 定义 |
|------|------|
| 五连追溯奖励 | 当一方获胜时，追溯形成五连的 5 步落子，按衰减系数反向传播额外奖励的机制 |
| Cycle 模式 | `--cycle` CLI 模式，将自我对弈 + 评价 + 训练耦合在循环中的训练方式 |
| 棋型评估奖励 | 由 RewardCalculator 通过双向棋型检测计算的位置 reward（活四、冲四等） |
| masked_softmax | 将 raw_rewards 转为概率分布的转换函数，只考虑空位（mask=1） |