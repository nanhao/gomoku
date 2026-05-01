# 引擎落子策略（argmax/sampling）设计文档

## 背景

需求来源：`requirements/key_points.md` 第 640-641 行（第三十七次追加）：

> 发现 ai 的输出概率已经很不错了，但是游戏引擎在选择落子的时候随机性太高了，需要修改。

当前实现（[ai/engine.py](file:///d:/projects/gomoku/ai/engine.py) L107-L108）一律使用 `random.choices` 对模型 Softmax 输出做加权采样：

```python
chosen_idx = random.choices(range(len(valid_positions)), weights=valid_probs, k=1)[0]
```

由此带来三类问题：
1. 模型已能给出合理的概率分布，但即便最优点概率为 0.3，仍有 70% 概率被其余 224 个格子瓜分，表现出明显的次优行为。
2. 人机对战（PvE）、评估、自对弈共用同一采样路径，三者对随机性的期望不同却被强行统一。
3. 没有任何贪婪/温度/top-k 开关，调优空间为零。

## 设计目标

1. 让 `AIEngine.get_move` 默认采用**贪婪选择（argmax）**，消除过度随机性。
2. 保留**参数化扩展位置**，以便未来单独实现更精细的 sampling 策略（温度、top-k 等）。
3. PvE、评估、自对弈三条调用路径**无需改动**即可直接受益（默认值变化即可）。
4. argmax 下若存在并列最大概率位置，从并列候选中随机选一个，避免固定第一个带来的路径退化。
5. 不改变 `AIEngine.get_move` 返回值签名、异常语义、预计算 `probs` 入参等既有行为。

## API 设计

### 修改：`AIEngine.get_move`

文件：`ai/engine.py`

```python
def get_move(
    self,
    board: List[List[int]],
    current_player: int,
    probs: List[float] = None,
    strategy: str = "argmax",
) -> Tuple[int, int]:
    """获取 AI 落子位置。

    Args:
        board: 15x15 棋盘。
        current_player: 当前落子方，1=黑，2=白。
        probs: 预计算的 225 维概率分布；为 None 则内部调用 model.predict。
        strategy: 落子策略。
            - "argmax"（默认）：在合法位置中选择概率最大的；存在并列最大值时
              从并列候选中随机挑一个。
            - "sampling"：按概率分布加权采样。**本次暂不实现**，调用时抛
              NotImplementedError，留待后续以独立模块提供（温度、top-k 等）。

    Returns:
        (x, y) 落子位置，0 <= x < 15, 0 <= y < 15。

    Raises:
        ValueError: 棋盘已满。
        NotImplementedError: 传入 strategy="sampling" 时。
        ValueError: 传入未知 strategy 时。
    """
```

### 不变的接口

- `AIGameManager.ai_move`（PvE）：继续调用 `engine.get_move(board, current_player)`，因默认值变为 argmax，无需改动。
- `AIGameManager._play_single_game`（自对弈 / 样本生成）：继续调用 `engine.get_move(board, player, probs=probs)`，同样默认 argmax。
- `board_to_vector`、`create_engine`、`AIEngine.__init__`：保持不变。

### 调用方显式指定策略的约定

- 所有生产调用点（PvE、评估、自对弈）**一律使用默认值**，不显式传 `strategy`。
- 若未来需要探索，应通过**新增**调用点或**新 API**显式传 `strategy="sampling"`，并届时补齐 sampling 实现。

## 主流程伪代码

### `AIEngine.get_move`

```python
def get_move(board, current_player, probs=None, strategy="argmax"):
    board_vector = board_to_vector(board, current_player)

    if probs is None:
        probs = self.model.predict(board_vector)

    # 收集所有空位及其概率
    valid_positions = []
    valid_probs = []
    for i in range(225):
        x, y = i // 15, i % 15
        if board[x][y] == 0:
            valid_positions.append((x, y))
            valid_probs.append(probs[i])

    if not valid_positions:
        raise ValueError("棋盘已满，无空位可落子")

    if strategy == "argmax":
        return _pick_argmax(valid_positions, valid_probs)
    elif strategy == "sampling":
        raise NotImplementedError(
            "sampling 策略尚未实现，未来将提供独立的采样模块。"
        )
    else:
        raise ValueError(f"未知落子策略: {strategy}")
```

### `_pick_argmax`（argmax 选择 + 并列随机打破）

```python
def _pick_argmax(valid_positions, valid_probs):
    # 找出最大概率值
    max_prob = max(valid_probs)

    # 收集所有达到最大概率的候选
    candidates = [
        pos for pos, p in zip(valid_positions, valid_probs)
        if p == max_prob
    ]

    # 并列时随机选一个；唯一最大值时 random.choice 等价于直接返回
    return random.choice(candidates)
```

说明：
- `max_prob` 可能为 0（模型把全部概率压在已落子位置上）。此时所有空位都并列，`random.choice` 从全部合法位置中均匀挑一个，行为合理。
- 不再需要既有实现里的 `total_prob <= 0` 兜底分支（该分支是为加权采样设计的，argmax 路径不需要）。

## 主流程测试用例

| # | 用例 | 输入 / 条件 | 断言 |
|---|------|-------------|------|
| 1 | argmax 默认策略 | 模型输出中某位置概率唯一最大 | 返回的就是该位置 |
| 2 | argmax 并列随机 | 两个合法位置概率相等且最大，其他更低 | 多次调用返回集合覆盖两个候选 |
| 3 | argmax 跳过已落子 | 概率最大的位置已落子，次大位置为空 | 返回次大位置 |
| 4 | argmax 全零兜底 | 所有空位概率为 0 | 从空位中均匀选一个，不抛异常 |
| 5 | PvE 调用默认 argmax | `game_manager.ai_move` 不传 strategy | 行为等价于 argmax |
| 6 | 自对弈调用默认 argmax | `_play_single_game` 不传 strategy | 行为等价于 argmax |
| 7 | sampling 抛 NotImplementedError | 显式传 `strategy="sampling"` | 抛 `NotImplementedError` |
| 8 | 未知策略抛 ValueError | 传 `strategy="foo"` | 抛 `ValueError` |
| 9 | 棋盘已满 | 所有位置均非 0 | 抛 `ValueError` |
| 10 | 预计算 probs 生效 | 传入 `probs` | 不调用 `model.predict`，行为与不传一致 |

### 测试用例示例（用例 2：并列随机）

```python
def test_argmax_ties_broken_randomly():
    board = [[0] * 15 for _ in range(15)]
    # 构造 probs：位置 (0,0) 与 (0,1) 并列最大，其余均小
    probs = [0.0] * 225
    probs[0] = 0.4
    probs[1] = 0.4
    probs[2] = 0.2

    engine = AIEngine(StubModel(probs))
    seen = set()
    for _ in range(200):
        seen.add(engine.get_move(board, 1))

    assert (0, 0) in seen
    assert (0, 1) in seen
    assert (0, 2) not in seen
```

### 测试用例示例（用例 7：sampling 未实现）

```python
def test_sampling_raises_not_implemented():
    board = [[0] * 15 for _ in range(15)]
    engine = AIEngine(StubModel([1.0 / 225] * 225))

    with pytest.raises(NotImplementedError):
        engine.get_move(board, 1, strategy="sampling")
```

## 涉及文件

### 修改文件

| 文件 | 变更说明 |
|------|----------|
| `ai/engine.py` | `get_move` 新增 `strategy` 参数，默认 `"argmax"`；新增内部 argmax 选择逻辑；`"sampling"` 抛 `NotImplementedError`；未知策略抛 `ValueError`。 |

### 无需修改文件

| 文件 | 说明 |
|------|------|
| `ai/game_manager.py` | PvE 与自对弈路径均使用默认参数，不显式传策略；默认改为 argmax 后直接受益。 |
| `ai/probability_utils.py` | Softmax 缩放因子不在本次变更范围，保持现状。 |
| 模型实现（`fc_v1`、`cnn_v1`、`cnn_v3` 等） | 模型输出语义不变，仅消费侧的选择策略变化。 |

### 测试文件

| 文件 | 变更说明 |
|------|----------|
| `tests/ai/test_engine.py`（或对应既有文件） | 新增用例 1-10；移除/调整依赖"加权采样"语义的旧用例，改为断言 argmax 行为。 |

### 未变更但需回归

- `tests/ai/test_game_manager.py`：确认 PvE、自对弈测试在默认 argmax 下仍通过。
- 所有评估/对战脚本（`ai/battle.py`、`ai/evaluator.py`）：确认调用路径未因默认值变化而回归失败。

## 未来扩展（非本次交付）

- 独立实现 sampling 策略，支持：
  - 温度 `temperature`（`p_i ** (1/T)` 再归一化）。
  - top-k 截断后再采样。
  - 可选的 `seed` 以便复现。
- sampling 实现建议作为独立函数或策略对象，而非塞进 `get_move` 内部分支，保持引擎职责单一。
