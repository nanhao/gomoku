# 验收方案

> 意图 24 | MCTS 训练体系

本文档定义 AI 辅助验收的执行命令和审查 prompt。
验收方为独立 AI，先按命令生成产物，再读取产物和设计文档进行验证。

所有文件路径以项目根目录 `d:\projects\gomoku` 为基准。

---

## 1. MCTS 算法正确性

### 背景

MCTS 搜索器对给定棋盘状态执行 `simulations` 次模拟，每次模拟经历 selection → expansion → evaluation → backup 四阶段。搜索结果写入结构化 JSON 日志。

### 前置

1. 修改 `src/config/mcts.py`：`MCTS_LOG_ENABLED = True`
2. 准备 cnn_v7 权重文件。**注意**：在训练脚本实现前，可用 `fresh=True` 创建随机初始化权重的模型（虽策略/价值头输出无实际意义，但足以验证 MCTS 算法的选择/扩展/回传流程的正确性）。验收方可自由选择使用随机权重还是已有训练权重。

### 执行

```bash
python -m ai.scripts.mcts_self_play --model <权重.pth> --games 1 --simulations 50 --output temp_verify_mcts
```

### 审查

请读取以下文件后开始审查：

- `{output}/mcts_logs/` 下所有 `.json` 日志文件
- `design/24/main_docs/main_design.md` §4.2（日志字段定义）和 §5（关键算法公式）

你需要验证 MCTS 搜索过程是否正确实现了 AlphaGo Zero 算法。这包括但不限于：PUCT 公式验算、backup 正确性、价值取反逻辑、Virtual Loss 的施加与撤销、访问计数一致性。

请基于日志数据和算法定义，自行判断哪些是关键验证点，并从日志中寻找任何可能暗示实现错误的异常模式。注意数值分布的合理性、搜索行为的合理性。

输出你对每条仿真和整体的判断，附上所有发现的不一致之处的具体数据。

---

## 2. 自对弈样本正确性

### 背景

`mcts_self_play.py` 使用 MCTS 进行自我对弈，每局生成 `sample.json`（样本数组）和 `record.json`（对局记录）。

### 前置

与第 1 节相同，需要已训练的 cnn_v7 权重。

### 执行

```bash
python -m ai.scripts.mcts_self_play --model <权重.pth> --games 2 --simulations 50 --output temp_verify_samples
```

### 审查

请读取 `{output}/` 目录下的所有输出文件，以及以下设计文档：

- `design/24/main_docs/main_design.md` §4.4（样本格式定义）

你需要验证自对弈脚本生成的样本在格式、语义和一致性上是否正确。考虑 sample.json 的 JSON 结构、每条样本的字段完整性、各字段的值域约束；以及同一局棋内 z 值随玩家切换的符号变化、π 分布的合理性、已落子位置是否被正确排除。

请自行判断该验证哪些方面，除了显而易见的格式检查外，注意棋局演进中样本之间的逻辑一致性。

输出逐文件的检查结果和总体结论。

---

## 3. 训练过程规约对照

### 背景

训练代码按照 `design/24/main_docs/main_design.md` §9 的规约实现。规约以步号标记（L1-L7 → T1-T12 → E1-E6 → S1），定义了数据加载、单 batch 训练步、epoch 循环和模型保存的完整流程。

### 审查

请读取以下两份材料：

- `design/24/main_docs/main_design.md` §9 "训练过程规约"
- `src/ai/dual_head_trainer.py`（训练实现代码）

你的任务是对照规约与实现，判断代码是否忠实地实现了规约中的每一步。

规约中的步骤编号（T4/T6 等）是审查的关键锚点，但你不需要局限于逐行比对——你应从代码整体逻辑出发，判断：

- 规约中的每个步骤在代码中是否有对应的实现
- 代码中是否存在违背规约的操作（如 p_log 被二次取 log）
- 张量形状和值域约束是否被 enforce（assert 或等效检查）
- optimizer 配置是否与规约一致
- 入口方法 `_train_step` 是否按预期作为训练的核心调用入口
- 是否存在绕过入口方法直接操作训练张量的路径
- 代码中有无设计规约未定义的、可能改变训练语义的额外操作

输出逐步骤的检查结果，标注任何不一致之处的代码行号，最后给出总体结论。
