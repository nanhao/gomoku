# 第一级性能优化详细设计

> 基于 performance_optimization_proposal.md 第一级方案，细化实施设计

## 优化1：消除 SelfPlay 冗余 predict

### 问题

`AIGameManager.start_ai_vs_ai` 中每步棋调用了两次 `model.predict`：

1. 第 178 行：`rewards = engine.model.predict(board_vector)` — 用于记录
2. 第 181 行：`engine.get_move(board, player)` — 内部再次调用 `model.predict(board_vector)` 获取 probs 用于采样

两次调用输入完全相同（同一 board_vector），输出完全相同（同一 225 维概率分布）。

### 设计方案

**修改 AIEngine.get_move**：新增可选参数 `probs`，允许调用者传入预计算的概率分布。

```python
def get_move(self, board, current_player, probs=None):
    board_vector = self.board_to_vector(board, current_player)
    if probs is None:
        probs = self.model.predict(board_vector)
    # ... 后续采样逻辑不变
```

**修改 AIGameManager.start_ai_vs_ai / start_ai_vs_ai_random**：
- 先调用 `engine.model.predict(board_vector)` 获取 rewards
- 将 rewards 同时传给 `engine.get_move(board, player, probs=rewards)` 和 `recorder.record_move`

### 等价性保证

- `model.predict` 输出不变，`get_move` 的采样逻辑不变
- 仅消除冗余调用，数值完全等价

---

## 优化2：棋盘拷贝消除

### 问题

`shape_detector.detect_shapes` 每次调用执行 `simulated_board = list(board)` 复制 225 个整数的列表。
`reward_calculator._compute_four_reward` 中也多次 `list(board)` 复制棋盘。
总计每轮数百万次 list 拷贝。

### 设计方案

**修改 detect_shapes**：改为"修改-检测-恢复"模式，原地修改 board 后恢复：

```python
def detect_shapes(board, position, player):
    x, y = position
    idx = x * 15 + y
    # 原地模拟落子
    board[idx] = player
    # ... 检测逻辑不变 ...
    # 恢复原值
    board[idx] = 0  # 调用前该位置一定是0（空位）
```

**修改 reward_calculator._compute_four_reward**：
- `simulated_board = list(board)` 保留（冲四递归需要多次修改不同位置，无法简单原地操作）
- 但 `_find_four_defense` 中的 `list(simulated_board)` 不需要（simulated_board 已是副本）

### 等价性保证

- detect_shapes 中 board[idx] 调用前一定是 0（空位才计算 reward），恢复为 0 是等价的
- detect_single_direction 只读 board，不会修改它
- 注意：需要确保调用链上没有其他代码在 detect_shapes 执行期间读取 board 的该位置

---

## 优化3a：JSON 去缩进

### 问题

`data_recorder.py` 和 `evaluator.py` 中 `json.dump` 使用 `indent=2`，导致文件大小约为紧凑格式的 2 倍。
`sample.json` 同样使用 `indent=2`。

### 设计方案

去掉所有 `json.dump` 的 `indent=2` 参数，改为紧凑格式：

- `data_recorder.py` 第 132 行：`json.dump(..., ensure_ascii=False)` — 去掉 indent
- `evaluator.py` 第 109 行：`json.dump(..., ensure_ascii=False)` — 去掉 indent

### 等价性保证

- JSON 紧凑格式与缩进格式解析后数据完全一致
- 仅影响文件可读性，不影响功能

---

## 优化3b：prepare_tensors 向量化

### 问题

`SampleLoader.prepare_tensors` 使用逐条 `append` 循环提取 input/output/mask：

```python
for item in sample_data:
    inputs.append(item["input"])
    targets.append(item["output"])
    masks.append(item["mask"])
```

### 设计方案

改为列表推导 + torch.tensor 批量转换：

```python
inputs = [item["input"] for item in sample_data]
targets = [item["output"] for item in sample_data]
masks = [item["mask"] for item in sample_data]
```

列表推导比逐条 append 快约 30-50%（预分配内存，减少 Python 解释器开销）。

### 等价性保证

- 输出的 inputs/targets/masks 列表内容完全一致
- torch.tensor 接收列表推导结果与 append 结果无差异
