# GUI AI 对战温度采样与回放展示设计文档

对应需求：`requirements/key_points.md` 第四十一次追加。

## 背景

[自对弈温度采样](file:///d:/projects/gomoku/design/move_sampler_design.md) 已在 CLI 链路（`create_samples.py` / `train_main.py --cycle`）默认启用，但 **GUI AI 内战**链路（`gui/ai_battle.py` → `app/battle.py` → `AIGameManager.start_ai_vs_ai`）至今未传 `temp_config`，全程走 argmax。

同时，`DataRecorder.record_move()` 只记录 `input`、`output`（模型原始概率）、`position`，不区分 argmax / sampling，也不记录采样参数。`ReplayViewer` 仅展示 AI 热力图（原始概率分布）和 Sample 热力图，无法让用户了解每一步的落子策略及采样权重。

## 设计目标

1. GUI AI 内战默认启用温度采样，参数与 CLI 一致（`temperature=1.0, temp_moves=12, temp_side="both"`），**不可在 GUI 界面配置**。
2. `DataRecorder.record_move()` 扩展记录字段：`strategy` / `temperature` / `top_k` / `sampling_probs`。
3. `ReplayViewer` 新增 Sampling 热力图 + 信息面板策略/温度文本展示。
4. 旧 `record.json`（无温度字段）仍可正常回放，不报错、不降级。
5. PvE 模式不受影响，保持 argmax（`AIGameManager.ai_move()` 不传 `temp_config`）。

---

## 1. 核心变更

### 1.1 链路：GUI 对战 → 默认启用温度采样

```
gui/ai_battle.py
  │ run_ai_vs_ai_mode() 构造 TempConfig(temperature=1.0, temp_moves=12, temp_side="both")
  ▼
gui/components/ai_battle.py
  │ AIBattleProgressWindow.__init__() 新增 temp_config 参数
  │ _run_battle() → BattleSession(..., temp_config=temp_config)
  ▼
app/battle.py
  │ BattleSession.__init__() 新增 temp_config 参数
  │ BattleSession.run() → AIGameManager.start_ai_vs_ai(..., temp_config=self._temp_config)
  ▼
ai/game_manager.py
  │ _play_single_game(temp_config=...) — 已有逻辑，无需改动
  │ recorder.record_move() 时附加 strategy/temperature/sampling_probs（见 §1.2）
```

### 1.2 记录：`record.json` 扩展字段

`DataRecorder.record_move()` 签名变更：

```python
def record_move(
    self,
    game_id: int,
    input_vector: List[int],
    output_vector: List[float],
    position: Tuple[int, int],
    strategy: str = "argmax",                       # 新增
    temperature: Optional[float] = None,             # 新增
    top_k: Optional[int] = None,                     # 新增
    sampling_probs: Optional[List[float]] = None,    # 新增：225 维温度变换后概率
) -> None:
```

写入 `history` 的字段仅当非 None 时才附加，保持旧格式兼容：

```json
// sampling 步
{
    "input": [...],
    "output": [...],
    "position": [x, y],
    "strategy": "sampling",
    "temperature": 1.0,
    "top_k": null,
    "sampling_probs": [...]
}

// argmax 步（向后兼容：无 strategy 字段时回放端默认视为 argmax）
{
    "input": [...],
    "output": [...],
    "position": [x, y],
    "strategy": "argmax"
}
```

`AIGameManager._play_single_game()` 中 `use_temp=True` 时计算 `sampling_probs` 并传入：

```
# 伪代码
valid_probs = [probs[i] for all empty cells]
sampling_probs = [p ** (1/T) for p in valid_probs]  # 未归一化的权重
recorder.record_move(..., strategy="sampling", temperature=T, top_k=temp_config.top_k, sampling_probs=sampling_probs)
```

> **注意**：当前 `_play_single_game` 把 `probs` 传入 `engine.get_move(probs=probs)`，`valid_probs` 在 engine 内部构建。需在外部从 `board` 和 `probs` 重新构建 `valid_probs`，或提取公共函数。此处的 `sampling_probs` 仅用于记录/展示，不影响落子结果。

### 1.3 展示：`ReplayViewer` 新增 Sampling 热力图 + 策略信息

#### 布局调整（伪代码）

```
ReplayViewer._create_widgets()
  │
  ├─ 左侧：棋盘画布（不变）
  │
  └─ 右侧：信息面板
       ├─ AI 热力图 + Sample 热力图 + Sampling 热力图（新增，三列并排）
       ├─ 颜色标尺（不变）
       └─ info_text（新增策略/温度文本）
```

- Sampling 热力图标题为 "Sampling"，宽度与 AI/Sample 热力图一致（280×280），仅在 `move["strategy"] == "sampling"` 且有 `sampling_probs` 时绘制。
- 布局：AI / Sample / Sampling 三个热力图水平排列，若空间不足则 Sampling 换行。

#### `_draw_heatmaps()` 伪代码

```
def _draw_heatmaps(self, step, move):
    # 1. AI 热力图：使用 move["output"]（不变）
    draw(ai_heatmap, move["output"])

    # 2. Sample 热力图：使用 sample_data（不变）
    draw(sample_heatmap, sample_data[step-1]["output"])

    # 3. Sampling 热力图（新增）
    if move.get("strategy") == "sampling" and "sampling_probs" in move:
        draw(sampling_heatmap, move["sampling_probs"])
    else:
        sampling_heatmap.clear()
```

#### `_update_info_text()` 伪代码

```
def _update_info_text(self, step, move):
    # ... 现有信息 ...

    # 新增：温度采样信息
    strategy = move.get("strategy", "argmax")  # 缺省视为 argmax（兼容旧数据）
    info += f"\n--- 温度采样 ---\n"
    info += f"策略: {strategy}\n"
    if strategy == "sampling":
        info += f"温度: {move.get('temperature', 'N/A')}\n"
        info += f"Top-K: {move.get('top_k', 'None')}\n"
```

---

## 2. API 设计（接口变更）

### 2.1 `app/battle.py:BattleSession`

```python
class BattleSession:
    def __init__(
        self,
        ai1_name: str,
        ai2_name: str,
        num_games: int,
        ai_data_dir: str = None,
        train_data_dir: str = None,
        temp_config: Optional["TempConfig"] = None,   # 新增
    ):
        """...
        Args:
            temp_config: 温度采样配置。None 表示全程 argmax。
        """
```

### 2.2 `gui/components/ai_battle.py:AIBattleProgressWindow`

```python
class AIBattleProgressWindow:
    def __init__(
        self,
        num_games: int,
        ai1_name: str = "random",
        ai2_name: str = "random",
        temp_config: Optional["TempConfig"] = None,   # 新增
    ):
        """...
        Args:
            temp_config: 温度采样配置。None 表示全程 argmax。
        """
```

### 2.3 `gui/ai_battle.py:run_ai_vs_ai_mode`

```python
def run_ai_vs_ai_mode(
    num_games: int,
    ai1_name: str = "random",
    ai2_name: str = "random",
    temp_config: Optional["TempConfig"] = None,       # 新增
) -> None:
```

### 2.4 `gui/components/replay_viewer.py:ReplayViewer`

无公开接口变更。内部新增 `sampling_heatmap` 属性。

---

## 3. 主流程伪代码

### 3.1 GUI AI 对战启动流程

```
run_ai_vs_ai_mode(num_games, ai1_name, ai2_name):
    # 默认启用温度采样
    temp_config = TempConfig(temperature=1.0, temp_moves=12, temp_side="both")

    progress = AIBattleProgressWindow(num_games, ai1_name, ai2_name, temp_config=temp_config)

    def on_complete(recorder, should_replay):
        # ... 显示回放（不变）...

    progress.start_battle(on_complete)
```

### 3.2 BattleSession 执行对战流程

```
BattleSession.run(on_progress):
    engine1 = create_engine(self._ai1_name)
    engine2 = create_engine(self._ai2_name)

    recorder = DataRecorder(...)

    # temp_config 透传至 game_manager
    results = AIGameManager.start_ai_vs_ai(
        engine1, engine2, self._num_games, recorder,
        on_progress=on_progress,
        temp_config=self._temp_config,   # 新增
    )

    evaluator.evaluate_round(recorder.get_round_dir())
    return results
```

### 3.3 单局对弈记录温度信息流程

```
AIGameManager._play_single_game(engine_black, engine_white, recorder, game_num, temp_config):
    # ... 现有编排逻辑（不变）...

    while not controller.is_over():
        probs = engine.model.predict(board_vector)

        use_temp = should_sample(move_count, player, temp_config, temp_side_this_game)

        if use_temp:
            x, y = engine.get_move(board, player, probs=probs, strategy="sampling", ...)

            # 构建 valid_probs 用于记录采样权重
            valid_probs = [probs[i] for each empty cell on board]
            sampling_probs = [p ** (1/temp_config.temperature) for p in valid_probs]
            recorder.record_move(
                game_id, board_vector, probs, (x, y),
                strategy="sampling",
                temperature=temp_config.temperature,
                top_k=temp_config.top_k,
                sampling_probs=sampling_probs,
            )
        else:
            x, y = engine.get_move(board, player, probs=probs)
            recorder.record_move(
                game_id, board_vector, probs, (x, y),
                strategy="argmax",
            )

        controller.make_move(x, y)
        move_count += 1

    # ... 保存结果（不变）...
```

### 3.4 Replay 展示流程

```
ReplayViewer._update_display():
    step = self.current_step

    draw_board_grid()
    if step > 0:
        move = self.history[step - 1]

        # 绘制热力图
        draw_heatmaps(step, move)      # ← 扩展：增加 Sampling 热力图

        # 绘制棋子
        draw_pieces(input_vector, move)

        # 更新信息文本
        update_info_text(step, move)   # ← 扩展：增加策略/温度文本
    else:
        clear_heatmaps()               # ← 扩展：增加 Sampling 热力图清除
        clear_info_text()
```

---

## 4. 测试用例

### 4.1 GUI 对战温度采样启用验证

| # | 场景 | 输入 | 预期 |
|---|---|---|---|
| T1 | GUI AI 内战默认启用采样 | 启动 `run_ai_vs_ai_mode(1, "random", "random")` | `AIGameManager.start_ai_vs_ai` 收到 `temp_config` 非 None |
| T2 | temp_config 参数正确透传 | 自定义 `TempConfig(temperature=2.0, temp_moves=5)` | `_play_single_game` 收到的 temp_config 字段一致 |
| T3 | BattleSession 接收 temp_config | `BattleSession(..., temp_config=TempConfig())` | `self._temp_config` 正确存储 |
| T4 | BattleSession 透传给 game_manager | 调用 `run()` | `start_ai_vs_ai` 收到 `temp_config` |

### 4.2 record.json 扩展字段验证

| # | 场景 | 输入 | 预期 |
|---|---|---|---|
| T5 | sampling 步记录完整字段 | `use_temp=True`，`temperature=1.0`，`top_k=None` | history entry 含 `strategy="sampling"`, `temperature=1.0`, `top_k=None`, `sampling_probs` 长度 = 空位数 |
| T6 | argmax 步记录 strategy 字段 | `use_temp=False` | history entry 含 `strategy="argmax"`，无 `temperature`/`sampling_probs` |
| T7 | sampling_probs 数值正确 | `valid_probs=[0.5, 0.3, 0.2]`，`temperature=1.0` | `sampling_probs` 与 `valid_probs` 一致（T=1 不变换） |
| T8 | sampling_probs 数值正确（T≠1） | `valid_probs=[0.5, 0.3, 0.2]`，`temperature=2.0` | `sampling_probs` 为 `[sqrt(0.5), sqrt(0.3), sqrt(0.2)]` |
| T9 | temp_moves=0 时不记录采样 | `temp_config.temp_moves=0` | 全部步伐 history entry 均为 `strategy="argmax"` |

### 4.3 Replay 展示验证

| # | 场景 | 输入 | 预期 |
|---|---|---|---|
| T10 | Sampling 热力图绘制 | move 含 `strategy="sampling"` + `sampling_probs` | Sampling 热力图非空 |
| T11 | argmax 步不绘制 Sampling 热力图 | move 含 `strategy="argmax"` | Sampling 热力图清空 |
| T12 | 信息面板显示采样策略 | move 含 `strategy="sampling"` | info_text 显示 "策略: sampling" |
| T13 | 信息面板显示 argmax 策略 | move 含 `strategy="argmax"` | info_text 显示 "策略: argmax" |
| T14 | 信息面板显示温度值 | move 含 `strategy="sampling"` + `temperature=1.0` | info_text 显示 "温度: 1.0" |
| T15 | 信息面板显示 Top-K | move 含 `strategy="sampling"` + `top_k=None` | info_text 显示 "Top-K: None" |

### 4.4 向后兼容验证

| # | 场景 | 输入 | 预期 |
|---|---|---|---|
| T16 | 旧 record.json（无 strategy 字段） | 加载不含 `strategy` 的 record.json | 正常回放，Sampling 热力图留空，info_text 默认显示 "argmax" |
| T17 | 旧 record.json（无 sampling_probs） | 加载不含 `sampling_probs` 的 record.json | 正常回放，不崩溃 |
| T18 | PvE 模式不受影响 | 启动 PvE 对局 | AI 落子仍走 argmax，不记录温度字段 |

---

## 5. 变更文件清单

| 文件 | 变更类型 | 说明 |
|---|---|---|
| `gui/ai_battle.py` | 修改 | 构造默认 TempConfig 并透传 |
| `gui/components/ai_battle.py` | 修改 | `AIBattleProgressWindow` 新增 temp_config 参数 |
| `app/battle.py` | 修改 | `BattleSession` 新增 temp_config 参数 |
| `ai/data_recorder.py` | 修改 | `record_move` 扩展 strategy/temperature/top_k/sampling_probs |
| `ai/game_manager.py` | 修改 | `_play_single_game` 记录温度采样信息 |
| `gui/components/replay_viewer.py` | 修改 | 新增 Sampling 热力图 + 策略信息展示 |
| `design/gui_temp_sampling_replay_design.md` | 新增 | 本设计文档 |

---

## 6. 向后兼容策略

- `record.json` 新增字段均为可选的；`ReplayViewer` 读取时使用 `.get("strategy", "argmax")` 缺省兜底。
- `DataRecorder.record_move()` 新增参数均有默认值，旧调用方无需修改。
- `BattleSession.__init__()` 的 `temp_config` 默认为 `None`，不改动旧调用方行为。
- `AIBattleProgressWindow.__init__()` 的 `temp_config` 默认为 `None`，不改动旧调用方行为。
