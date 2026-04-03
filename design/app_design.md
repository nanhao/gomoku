# app 包设计文档

## 1. 概述

### 1.1 背景

`gui` 包中存在两类与 UI 无关的游戏流程管理逻辑，需要剥离：

**来自 `gui/main.py` 的 `GomokuGUI`**：
- 游戏模式判断（pvp / pve）
- 人机对战流程协调（判断是否轮到 AI、触发 AI 落子）

**来自 `gui/components/ai_battle.py` 的 `AIBattleProgressWindow`**：
- AI 内战的循环调度（多局对战循环）
- 对战完成后调用 `Evaluator` 生成训练样本

这些逻辑依赖 `ai` 包，无法下沉到 `game` 包（game 是底层包，不依赖其他包）。

另外，`gui/main.py` 和 `gui/components/ai_battle.py` 中各有一份 `_create_ai_engine()`，将 AI 名称映射到具体模型类。**此映射是 ai 包的职责**，应在 `ai` 包中实现，`app` 包只做调用代理。

### 1.2 解决方案

新增 `app` 包作为**应用层**，位于 `ai` 包和 `game` 包之上、`gui` 包之下。

`app` 包是薄的代理层：**不实现任何业务逻辑**，仅将来自 GUI 的调用转发给 `game` 包和 `ai` 包，并将结果组装成方便 GUI 消费的格式。

### 1.3 包依赖层级

```
┌─────────────────────────────────────────┐
│              gui 包                      │  上层（纯 UI）
└────────────────┬────────────────────────┘
                 │ 依赖
┌────────────────▼────────────────────────┐
│              app 包（新增）              │  应用层（薄代理）
│        GameSession, BattleSession       │
└──────┬───────────────────┬─────────────┘
       │ 依赖               │ 依赖
┌──────▼──────┐    ┌───────▼─────────────┐
│   ai 包     │    │     game 包          │  底层
│ AIEngine    │    │  GomokuController   │
│ AIGameMgr   │    │                     │
│ create_engine│   │                     │
└─────────────┘    └─────────────────────┘
```

---

## 2. ai 包变更：新增 `create_engine` 工厂函数

**文件**：`ai/__init__.py`（新增导出）及 `ai/engine.py`（新增函数）

**职责**：AI 名称 → 模型实例 → AIEngine 的映射，属于 ai 包职责，在 ai 包内实现。

**签名**：
```python
# ai/engine.py
def create_engine(ai_name: str) -> AIEngine
```

**映射规则**：

| ai_name | 对应模型 |
|---------|---------|
| `"random"` | `RandomModel` |
| `"fc_v1"` | `FCv1Model` |
| 其他（未知） | `RandomModel`（默认兜底） |

`app` 包调用此函数，不自己实现映射。

---

## 3. 目录结构

```
app/
├── __init__.py          # 导出 GameSession, BattleSession
├── session.py           # GameSession 类（人机对战会话）
└── battle.py            # BattleSession 类（AI 内战会话）
```

---

## 4. 类设计

### 4.1 `GameSession` 类

**文件**：`app/session.py`

**职责**：封装一局人机对战（pve）或人人对战（pvp）的会话状态。协调 `GomokuController` 与 `AIGameManager`，对 GUI 提供统一的落子接口。

**公共接口**：

| 方法 / 属性 | 签名 | 说明 |
|------------|------|------|
| 构造方法 | `__init__(self, mode: str, controller: GomokuController, ai_name: str = None, ai_player: int = None)` | 初始化会话；pve 模式时调用 `ai.create_engine` 并创建 `AIGameManager` |
| `make_player_move` | `(self, x: int, y: int) -> dict` | 玩家落子，返回含 `need_ai_move` 标志的结果 |
| `do_ai_move` | `(self) -> dict` | 执行 AI 落子，委托给 `AIGameManager.ai_move()` |
| `mode` | `@property -> str` | 游戏模式（`"pvp"` / `"pve"`） |
| `ai_player` | `@property -> int \| None` | AI 执棋方（1=黑, 2=白），pvp 模式返回 None |

**`make_player_move` 返回格式**：

```python
{
    "status": "success" | "error",
    "error": str,            # 仅 error 时存在
    "game_over": bool,       # 游戏是否结束
    "need_ai_move": bool,    # GUI 是否需要调度 AI 落子
    "winner": int            # 仅 game_over=True 时存在：1=黑, 2=白, 3=平局
}
```

**`do_ai_move` 返回格式**：

```python
{
    "status": "success" | "error",
    "game_over": bool,
    "winner": int            # 仅 game_over=True 时存在
}
```

---

### 4.2 `BattleSession` 类

**文件**：`app/battle.py`

**职责**：封装 AI 内战的完整流程——多局对战循环 + 完成后生成训练样本。将此逻辑从 `AIBattleProgressWindow` 中剥离，GUI 只负责进度展示和回调。

**公共接口**：

| 方法 | 签名 | 说明 |
|------|------|------|
| 构造方法 | `__init__(self, ai1_name: str, ai2_name: str, num_games: int, project_root: str)` | 初始化；内部创建两个 AIEngine 和 DataRecorder |
| `run` | `(self, on_progress: callable = None) -> list` | 执行全部对战，每完成一局调用 `on_progress(current, total)`；完成后自动调用 Evaluator 生成训练样本；返回对战结果列表 |
| `get_recorder` | `(self) -> DataRecorder` | 返回本次对战的 DataRecorder，GUI 可用于获取 round_dir 实现回放 |

**`run` 返回格式**（同 `AIGameManager.start_ai_vs_ai` 的返回值）：

```python
[
    {"game": 1, "winner": 1},
    {"game": 2, "winner": 2},
    ...
]
```

---

## 5. 主流程伪代码

### 5.1 `GameSession` 初始化

```
GameSession.__init__(mode, controller, ai_name, ai_player):
    self._mode = mode
    self._controller = controller
    self._ai_player = ai_player
    self._ai_manager = None

    if mode == "pve" and ai_player is not None:
        engine = ai.create_engine(ai_name)          # 委托 ai 包工厂
        self._ai_manager = AIGameManager(controller, engine, ai_player)
```

### 5.2 `GameSession.make_player_move`

```
make_player_move(x, y):
    result = controller.make_move(x, y)             # 委托 game 包

    if "error" in result:
        return {"status": "error", "error": result["error"]}

    game_over = controller.is_over()
    need_ai_move = (
        not game_over
        and mode == "pve"
        and controller.get_current_player() == ai_player
    )

    response = {"status": "success", "game_over": game_over, "need_ai_move": need_ai_move}
    if game_over:
        response["winner"] = controller.get_winner()
    return response
```

### 5.3 `GameSession.do_ai_move`

```
do_ai_move():
    if ai_manager is None:
        raise RuntimeError("pvp 模式无 AI")
    result = ai_manager.ai_move()                   # 委托 ai 包
    game_over = controller.is_over()
    response = {"status": result.get("status", "error"), "game_over": game_over}
    if game_over:
        response["winner"] = controller.get_winner()
    return response
```

### 5.4 `BattleSession.run`

```
BattleSession.run(on_progress):
    engine1 = ai.create_engine(ai1_name)            # 委托 ai 包工厂
    engine2 = ai.create_engine(ai2_name)
    recorder = DataRecorder(f"{ai1_name}_vs_{ai2_name}")

    results = AIGameManager.start_ai_vs_ai(         # 委托 ai 包
        engine1, engine2, num_games, recorder,
        on_progress=on_progress
    )

    evaluator = Evaluator(ai_data_dir, train_data_dir)  # 委托 ai 包
    evaluator.evaluate_round(recorder.get_round_dir())

    return results
```

> **注**：`AIGameManager.start_ai_vs_ai` 需新增 `on_progress` 回调参数（每完成一局调用），这是对 ai 包的最小扩展。

---

## 6. 主流程测试用例

### 6.1 `GameSession` 主流程测试（primary）

| 测试名称 | 场景 | 预期结果 |
|---------|------|---------|
| `test_pvp_make_move_success` | pvp 模式玩家落子合法位置 | status=success, need_ai_move=False |
| `test_pve_player_move_triggers_ai_flag` | pve 模式玩家（黑）落子后轮到 AI（白） | need_ai_move=True |
| `test_pve_player_move_no_ai_when_not_ai_turn` | pve 模式落子后未轮到 AI | need_ai_move=False |
| `test_do_ai_move_success` | pve 模式 AI 落子成功 | status=success |
| `test_game_over_returns_winner` | 落子导致游戏结束 | game_over=True, winner 正确 |

### 6.2 `GameSession` 边缘测试（edge）

| 测试名称 | 场景 | 预期结果 |
|---------|------|---------|
| `test_make_move_invalid_occupied` | 落子在已占用位置 | status=error |
| `test_make_move_out_of_bounds` | 落子坐标越界 | status=error |
| `test_do_ai_move_raises_in_pvp` | pvp 模式调用 do_ai_move | 抛出 RuntimeError |

### 6.3 `BattleSession` 主流程测试（primary）

| 测试名称 | 场景 | 预期结果 |
|---------|------|---------|
| `test_run_returns_results` | 执行 2 局对战 | 返回长度为 2 的结果列表 |
| `test_run_calls_on_progress` | 执行 3 局，on_progress 被调用 | on_progress 被调用 3 次，参数依次为 (1,3),(2,3),(3,3) |
| `test_run_generates_samples` | 执行对战后 | train_data 目录中生成 sample.json |
| `test_get_recorder_returns_recorder` | 执行 run 后调用 get_recorder | 返回有效的 DataRecorder 实例 |

---

## 7. GUI 层变更说明

### 7.1 `gui/main.py` 变更

**删除**：
- `_create_ai_engine()` 方法
- `self.ai_manager`
- `_on_board_click` 中手动判断 `game_mode == "pve"` 及 `get_current_player() == ai_player` 的逻辑

**新增**：
```python
# __init__ 中
from app import GameSession
self.session = GameSession(game_mode, self.controller, ai_model, ai_player)

# _on_board_click 中（纯 UI 调度逻辑）
result = self.session.make_player_move(row, col)
if result["status"] == "success":
    self._update_board()
    if result["game_over"]:
        self._show_game_over(result["winner"])
    elif result["need_ai_move"]:
        self.root.after(500, self._ai_move)

# _ai_move 中
result = self.session.do_ai_move()
self._update_board()
if result["game_over"]:
    self._show_game_over(result["winner"])
```

### 7.2 `gui/components/ai_battle.py` 变更

`AIBattleProgressWindow._run_battle()` 中的内战循环和 Evaluator 调用，替换为：
```python
from app import BattleSession

battle = BattleSession(ai1_name, ai2_name, num_games, project_root)
results = battle.run(on_progress=self._update_progress)  # _update_progress 是 UI 回调
self.recorder = battle.get_recorder()
```

删除 `_create_ai_engine()` 重复方法。

---

## 8. 涉及文件一览

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `ai/engine.py` | 修改 | 新增 `create_engine(ai_name)` 工厂函数 |
| `ai/__init__.py` | 修改 | 导出 `create_engine` |
| `app/__init__.py` | 新增 | 导出 GameSession, BattleSession |
| `app/session.py` | 新增 | GameSession 类 |
| `app/battle.py` | 新增 | BattleSession 类 |
| `gui/main.py` | 修改 | 用 GameSession 替换相关逻辑 |
| `gui/components/ai_battle.py` | 修改 | 用 BattleSession 替换内战循环，删除 _create_ai_engine |
| `ai/game_manager.py` | 修改 | start_ai_vs_ai 新增 on_progress 回调参数 |
| `tests/app/primary/test_session.py` | 新增 | GameSession 主流程测试 |
| `tests/app/edge/test_session.py` | 新增 | GameSession 边缘测试 |
| `tests/app/primary/test_battle.py` | 新增 | BattleSession 主流程测试 |
| `tests/gui/test_gomoku_gui.py` | 修改 | 适配 session 接口 |
| `tests/gui/test_ai_battle.py` | 修改 | 适配 BattleSession 接口 |
