# 接口变更

仅列出对外暴露的公共接口变更。

## AIEngine

### 构造函数

| | 签名 |
|---|---|
| **变更前** | `__init__(self, model: BaseModel)` |
| **变更后** | `__init__(self, model: BaseModel, temperature: float = 1.0, top_k: Optional[int] = None, rng: Optional[random.Random] = None)` |

新增参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temperature` | `float` | `1.0` | 温度值 T > 0，仅在 strategy="sampling" 时生效 |
| `top_k` | `Optional[int]` | `None` | 截断阈值，仅在 strategy="sampling" 时生效；None 表示不截断 |
| `rng` | `Optional[random.Random]` | `None` | 可选随机源；None 时使用 random 模块全局源 |

### get_move()

| | 签名 |
|---|---|
| **变更前** | `get_move(self, board: List[List[int]], current_player: int, strategy: str = "argmax", temperature: float = 1.0, top_k: Optional[int] = None, rng: Optional[random.Random] = None) -> MoveDecision` |
| **变更后** | `get_move(self, board: List[List[int]], current_player: int, strategy: str = "argmax") -> MoveDecision` |

删除参数：`temperature`、`top_k`、`rng`（改为构造时注入）。

## MCTSMoveDecider

### 构造函数

| | 签名 |
|---|---|
| **变更前** | `__init__(self, searcher, temperature: float = 1.0, temp_steps: int = 8)` |
| **变更后** | `__init__(self, searcher, temperature: float = 1.0, temp_steps: int = 8, log_key_prefix: Optional[str] = None)` |

新增参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `log_key_prefix` | `Optional[str]` | `None` | 日志键前缀，格式 `"run{run_id}_game{game_idx}"`；None 时不输出日志 |

### get_move()

| | 签名 |
|---|---|
| **变更前** | `get_move(self, state: List[int], log_key: Optional[str] = None) -> MoveDecision` |
| **变更后** | `get_move(self, board: List[List[int]], current_player: int, strategy: str = "argmax") -> MoveDecision` |

**删除**：
- `state` — 226 维向量（前 225=棋盘 + 末位=落子方），改为分开传递 `board` + `current_player`。
- `log_key` — 改为构造时通过 `log_key_prefix` 注入，get_move 内部自增 `_search_id` 拼接完整 log_key。

**新增**：
- `board`：`List[List[int]]` — 15×15 棋盘矩阵，0=空, 1=黑, 2=白。
- `current_player`：`int` — 当前落子方 (1=黑, 2=白)。
- `strategy`：`str` — 落子策略，与 AiEngine 一致。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `board` | `List[List[int]]` | — | 15×15 棋盘矩阵 |
| `current_player` | `int` | — | 当前落子方 (1=黑, 2=白) |
| `strategy` | `str` | `"argmax"` | 落子策略："argmax"=贪婪选择访问计数最大者；"sampling"=按温度从访问计数采样 |

### 内部行为变更

- `get_move()` 内部维护 `_search_id`（整型，从 0 递增）。
- 内部将 `board` + `current_player` 转换为 226 维向量后再传给 `searcher.search()`。
- 若 `log_key_prefix` 非 None，则拼接 `log_key = f"{log_key_prefix}_search{self._search_id}"` 并透传给 `searcher.search()`。
- `reset()` 重置 `_search_id = 0`。

## MCTSSearcher

无接口变更。

## 调用方适配要点

### play_single_game() (game_manager.py)

- 构造 `AIEngine` 时传入 `temperature`、`top_k`、`rng`。
- `engine.get_move()` 调用仅传 `board`、`player`、`strategy`。

### _run_single_game() (mcts_self_play.py)

- 构造 `MCTSMoveDecider` 时传入 `log_key_prefix=f"run{run_id}_game{game_idx}"`。
- `decider.get_move(board, current_player)` 调用传棋盘矩阵和当前玩家，不再传 `state` 和 `log_key`。
- 移除 `import uuid`。
