# MctsGameRunner 类设计

## 概述

从 `MctsSelfPlayWorker._play_one_game` 提取的公用的 MCTS 对局循环，封装棋盘管理、MCTS 搜索、交替落子、终局判定逻辑。通过回调机制让调用方注入差异化行为（如样本收集），实现 self-play 和 battle 两场景的代码复用。

## 文件

`src/ai/scripts/mcts_game_runner.py`

## 数据类: GameResult

```
@dataclass
class GameResult:
    winner: int          # 0=平局, 1=黑胜, 2=白胜
    move_count: int      # 本局总步数
```

## 类: MctsGameRunner

### 构造器

```
MctsGameRunner(config: MctsSelfPlayConfig)
```

- `config` — MCTS 超参数配置（复用 `MctsSelfPlayConfig`，各字段含义不变）

构造器本身不创建引擎，引擎由调用方在每局开始前创建并传入 `run` 方法。

### 公共方法

#### run

```
run(
    engine_black: PolicyValueAiEngine,
    engine_white: PolicyValueAiEngine,
    run_id: str,
    game_idx: int,
    on_move: Callable[[List[int], MoveDecision, int], None] | None = None,
) -> GameResult
```

- `engine_black` — 执黑引擎
- `engine_white` — 执白引擎
- `run_id` — 本次运行唯一标识
- `game_idx` — 局编号（用于 log_key 前缀）
- `on_move` — 每步回调，参数为 `(state: 226维棋盘, decision: MoveDecision, current_player: int)`。self-play 场景用于收集样本；battle 场景传 None

返回：`GameResult`，包含 winner 和 move_count。

**行为**：
1. 初始化 225 维空棋盘，current_player=1（黑先）
2. 循环直到终局：
   a. 根据 current_player 选择 `engine_black` 或 `engine_white`
   b. 构造 log_key：`"run{run_id}_game{game_idx}_search{step}"`
   c. 调用 `engine.get_move(board_2d, current_player, strategy=SAMPLING)` 执行 MCTS 搜索
   d. 调用 `on_move` 回调（若提供），传入当前 state、MoveDecision、current_player
   e. 落子到 board
   f. 调用 `check_winner(board, action)` 和 `is_board_full(board)` 判定终局
   g. 调用 `engine.after_move(x, y)` 推进搜索树
   h. 若未终局，切换 current_player
3. 返回 `GameResult`

**重要约束**：
- 每局必须传入全新创建的 engine 实例（MCTS 搜索树不可跨局复用）
- engine_black 和 engine_white 可以是同一实例（self-play）或不同实例（battle）
- 不负责模型加载、样本生成、结果统计——这些由调用方处理

## 与现有代码的关系

- 替代 `MctsSelfPlayWorker._play_one_game` 中的核心循环逻辑
- `MctsSelfPlayWorker._play_one_game` 重构为：创建 engine → 调用 `runner.run(engine, engine, ..., on_move=收集样本回调)` → z 值回填
- `MctsBattleWorker` 新建：创建两个 engine → 调用 `runner.run(engine_black, engine_white, ...)` → 返回 GameResult
- 不修改 `MCTSSearcher`、`PolicyValueAiEngine`、`AIEngineBuilder`
