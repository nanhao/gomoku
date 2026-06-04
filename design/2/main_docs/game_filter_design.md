# 对局级样本过滤机制设计文档

## 背景

PRD（[requirements/2/PRD.md](file:///d:/projects/gomoku/requirements/2/PRD.md)）要求在单方温度采样模式下，仅保留采样方获胜的对局生成训练样本。当前 [SampleCreator](file:///d:/projects/gomoku/ai/sample_creator.py) 对所有非平局对局一视同仁，不具备按采样方过滤的能力。

## 设计目标

1. 新增 [ai/game_filter.py](file:///d:/projects/gomoku/ai/game_filter.py) 模块，提供 `should_keep_game` 判定函数
2. [DataRecorder.save_game_result](file:///d:/projects/gomoku/ai/data_recorder.py#L142) 扩展签名，持久化 `temp_side_this_game`
3. [AIGameManager._play_single_game](file:///d:/projects/gomoku/ai/game_manager.py#L124) 返回值携带 `temp_side_this_game`，并透传至 DataRecorder
4. [SampleCreator.create / create_from_round_dir](file:///d:/projects/gomoku/ai/sample_creator.py#L60) 新增 `record_filter` 参数
5. [create_samples.py](file:///d:/projects/gomoku/ai/create_samples.py) 构造并注入过滤函数，输出过滤统计

## 变更模块一览

| 模块 | 变更类型 | 说明 |
|------|---------|------|
| ai/game_filter.py | 新增 | 过滤判定函数 |
| ai/data_recorder.py | 修改 | save_game_result 新增参数；new_game 初始化 temp_side_this_game |
| ai/game_manager.py | 修改 | 新增 SingleGameResult(NamedTuple)；_play_single_game 返回值变更；透传 temp_side_this_game |
| ai/sample_creator.py | 修改 | create/create_from_round_dir 新增 record_filter 参数 |
| ai/evaluator.py | 修改 | evaluate_round 预留 record_filter 扩展点（当前未使用） |
| ai/create_samples.py | 修改 | 构造过滤函数、输出过滤统计 |

## API 设计

### 1. ai/game_filter.py（新增）

```python
def should_keep_game(record: dict) -> bool:
    """判定单局对弈是否应保留用于样本生成。

    规则：
    - temp_side_this_game 为 None / "both" → True（非单方采样，不过滤）
    - temp_side_this_game == "black" 且 winner == 1 → True（采样方黑胜）
    - temp_side_this_game == "white" 且 winner == 2 → True（采样方白胜）
    - 其余 → False（贪婪方胜或平局）

    Args:
        record: record.json 解析后的字典

    Returns:
        True → 保留该对局；False → 丢弃
    """
```

### 2. DataRecorder.save_game_result（修改）

```python
def save_game_result(
    self,
    game_id: int,
    winner: int,
    end_board: List[int],
    temp_side_this_game: str = None,   # ← 新增
):
    """保存游戏结果到文件

    Args:
        game_id: 游戏编号
        winner: 获胜方（1=黑胜, 2=白胜, 3=平局, 0=未结束）
        end_board: 最终棋盘状态（226 维向量）
        temp_side_this_game: 该局实际启用温度采样的一方。
            始终写入 record.json。取值：
            - "black" / "white"：单方采样
            - "both"：双方采样
            - None：未启用温度采样
    """
```

写入逻辑：始终将 `temp_side_this_game` 写入 `current_game_data`（含 None 值，JSON 中为 null）。

修订2：在 `new_game()` 中初始化 `temp_side_this_game` 为 None，与 `winner`（初始值 0）统一"先初始化后覆盖"策略：

```python
def new_game(self) -> int:
    self.current_game_id += 1
    self.current_game_data = {
        "history": [],
        "winner": 0,
        "temp_side_this_game": None,   # ← 新增初始化
        "end_board": []
    }
    return self.current_game_id
```

`save_game_result` 中覆盖写入：
```python
self.current_game_data["temp_side_this_game"] = temp_side_this_game
```

### 3. AIGameManager._play_single_game（修改）

返回值从 `int`（winner）变更为 `SingleGameResult(NamedTuple)`：

```python
from typing import NamedTuple, Optional

class SingleGameResult(NamedTuple):
    """单局对弈结果"""
    winner: int                    # 获胜方（1=黑, 2=白, 3=平局）
    temp_side_this_game: Optional[str]  # 该局实际采样方

@staticmethod
def _play_single_game(...) -> SingleGameResult:
    """执行单局 AI 内战

    Returns:
        SingleGameResult(winner, temp_side_this_game)
        - winner: 获胜方（1=黑, 2=白, 3=平局）
        - temp_side_this_game: 该局实际采样方
            - "black" / "white"：单方采样
            - "both"：双方采样
            - None：未启用温度采样
    """
```

关键改动点：
- 将 `temp_side_this_game`（已由 `resolve_temp_side` 解析）传入 `recorder.save_game_result`，始终传入（含 "both" 和 None）
- 返回值从 `return winner` 改为 `return SingleGameResult(winner, temp_side_this_game)`
- 使用 NamedTuple 替代裸 dict，消除 typo 风险（`result.winner` / `result.temp_side_this_game`）

### 4. AIGameManager.start_ai_vs_ai（修改）

结果列表每项新增 `temp_side_this_game` 键：

```python
# 变更前
results.append({"game": game_num, "winner": winner})

# 变更后
result = AIGameManager._play_single_game(...)
results.append({
    "game": game_num,
    "winner": result.winner,
    "temp_side_this_game": result.temp_side_this_game,
})
```

### 5. AIGameManager.start_ai_vs_ai_random（修改）

同上，适配 `_play_single_game` 返回值变更，结果列表新增 `temp_side_this_game` 键。

### 6. SampleCreator.create（修改）

```python
def create(
    self,
    record_files: List[str],
    record_filter: Callable[[dict], bool] = None,   # ← 新增
) -> List[Dict]:
    """从 record.json 文件列表生成 sample 列表

    Args:
        record_files: record.json 文件路径列表
        record_filter: 可选的记录过滤函数。接收 record 字典，
            返回 True 保留、False 跳过。None 表示不过滤。
    """
```

调用点：在已有的平局/空 history 跳过逻辑之后、mode 分派之前插入：

```python
# 外部过滤：调用方注入的自定义过滤逻辑
if record_filter is not None and not record_filter(record):
    continue
```

### 7. SampleCreator.create_from_round_dir（修改）

```python
def create_from_round_dir(
    self,
    round_dir: str,
    record_filter: Callable[[dict], bool] = None,   # ← 新增
) -> List[Dict]:
    # 透传 record_filter 到 self.create(...)
```

### 8. create_samples.py（修改）

```python
from ai.game_filter import should_keep_game

# ... 现有代码 ...

# 构造 record_filter
record_filter = should_keep_game

# 生成样本（传入 record_filter）
creator = SampleCreator(mode=args.mode, augment=not args.no_augment)
samples = creator.create_from_round_dir(round_dir, record_filter=record_filter)

# 过滤统计输出（修订1：复用 should_keep_game 消除重复逻辑和类型比较 bug）
filter_active = (
    temp_config is not None
    and temp_config.temp_side in ("black", "white", "random")
)
if filter_active:
    # 复用 should_keep_game 而非手写类型比较，
    # 避免 str("black") == int(1) 的类型比较 bug
    kept = sum(1 for r in results if should_keep_game(r))
    filtered_out = len(results) - kept
    print(f"\n样本过滤: 启用 (采样方获胜过滤)")
    print(f"  保留对局: {kept}")
    print(f"  过滤对局: {filtered_out}")
else:
    print(f"\n样本过滤: 未启用 (非单方采样模式)")
```

### 9. Evaluator.evaluate_round（修订3：预留扩展点）

```python
def evaluate_round(
    self,
    round_dir: str,
    max_workers: int = None,
    record_filter: Callable[[dict], bool] = None,   # ← 新增（默认 None）
) -> dict:
    """评价整个轮次的所有对局

    Args:
        round_dir: 轮次目录路径
        max_workers: 最大并行进程数
        record_filter: 可选的记录过滤函数（预留扩展点，当前未使用）
    """
```

当前版本 `record_filter` 参数仅声明，不影响现有行为。未来当训练管线需要
单方采样过滤时，可通过此参数启用，无需修改签名。

## 交互图

参见 [game_filter_sequence.puml](file:///d:/projects/gomoku/design/2/main_docs/game_filter_sequence.puml)

## record.json Schema 变更

| 字段 | 类型 | 说明 |
|------|------|------|
| temp_side_this_game | string or null | 始终写入。值为 "black"/"white"（单方采样）、"both"（双方采样）、null（未启用温度采样） |

向后兼容：已有 record.json 不含此字段，`should_keep_game` 中 `record.get("temp_side_this_game")` 返回 None，等同于 null 语义，行为与修改前一致。

## 影响范围分析

| 调用方 | 影响 | 兼容性 |
|--------|------|--------|
| create_samples.py → start_ai_vs_ai | 结果字典新增键，已有代码用 `r["winner"]` 不受影响 | 兼容 |
| create_samples.py → SampleCreator | 新增可选参数，默认不过滤 | 兼容 |
| train_main.py（cycle 模式） | 不传 record_filter，SampleCreator 行为不变 | 兼容 |
| train_main.py → Evaluator | evaluate_round 新增可选参数，默认 None 不影响行为 | 兼容 |
| start_ai_vs_ai_random | 适配 _play_single_game 返回值变更 | 需同步修改 |
| GUI AI 内战 | 不传 temp_config，save_game_result 写入 temp_side_this_game=null | 兼容（新增字段） |

## 测试用例

参见 [game_filter_test_cases.md](file:///d:/projects/gomoku/design/2/main_docs/game_filter_test_cases.md)