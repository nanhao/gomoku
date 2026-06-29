# 调用方适配：AiGameManager 与 battle.py

## 1. `AiGameManager.play_single_game` — 引入 `after_move`

### 1.1 变更点

在落子循环中，每次 `controller.make_move(x, y)` 后调用 `engine.after_move(x, y)`：

```
变更前（落子后）:
  → controller.make_move(x, y)
  → move_count += 1

变更后（落子后）:
  → controller.make_move(x, y)
  → engine.after_move(x, y)
  → move_count += 1
```

### 1.2 位置

`src/ai/game_manager.py` — `play_single_game()` 静态方法，`controller.make_move(x, y)` 之后。

### 1.3 注意事项

- `after_move` 调用不改变 `move_count` 递增时机
- `SinglePolicyAiEngine` 的 `after_move` 是 no-op，调用无副作用
- `PolicyValueAiEngine` 的 `after_move` 可能抛 `RuntimeError`，上层调用方（如 `start_ai_vs_ai`）需决定是否捕获
- 每次落子仅当前方的引擎调用 `after_move`，**对手引擎的搜索树不推进**。双 `PolicyValueAiEngine` 对战时，每步只有一方享受搜索树复用的加速，另一方下步仍需从根搜索。这是树复用设计的固有特性，非缺陷

## 2. `ai/battle.py`（CLI）— 采用 `create_engine`

### 2.1 变更前

```python
# 直接调用 get_model + 手动构造
model1 = get_model(ai1_name)
model2 = get_model(ai2_name)
engine1 = SinglePolicyAiEngine(model1)
engine2 = SinglePolicyAiEngine(model2)
```

### 2.2 变更后

`run_battle` 不再创建引擎实例，改为将 ai 名称直接传给 `start_ai_vs_ai_random`：

```python
from ai.game_manager import AIGameManager

results = AIGameManager.start_ai_vs_ai_random(
    ai1_name, ai2_name, num_games, recorder, temp_config=temp_config)
```

> `create_engine` 调用发生在 `start_ai_vs_ai_random` 内部（每局一次）。调用方无需手动创建引擎或处理 GPU 迁移。

### 2.3 影响

- 移除对 `get_model`、`SinglePolicyAiEngine`、`ai.engine` 的直接引用
- `run_battle` 不再需要 `create_engine` 导入
- 当模型为双头模型（如 `cnn_v7.1`）时，`start_ai_vs_ai_random` 内部自动创建 `PolicyValueAiEngine`

## 3. `app/battle.py`（GUI BattleSession）— 适配新接口

### 3.1 变更点

`BattleSession.run()` 不再预先创建引擎，改为将 `ai1_name` / `ai2_name` 直接传给 `start_ai_vs_ai`：

```python
# 旧：先创建引擎再传入
engine1 = create_engine(self._ai1_name)
engine2 = create_engine(self._ai2_name)
results = AIGameManager.start_ai_vs_ai(engine1, engine2, ...)

# 新：直接传 ai 名称
results = AIGameManager.start_ai_vs_ai(self._ai1_name, self._ai2_name, ...)
```

- 移除 `create_engine` 导入（`start_ai_vs_ai` 内部调用）
- `ModelLoadError` 捕获逻辑上移至 `start_ai_vs_ai` 内部或由调用方在最外层包裹即可

## 4. `app/session.py`（GUI Session）— 无需变更

同 `app/battle.py`，已使用 `create_engine`。

## 5. `ai/scripts/mcts_self_play.py` — `after_move` 签名适配

将 `engine.after_move(action)` 改为 `engine.after_move(x, y)`，其中 `(x, y)` 来自 `decision.get_position()`。

## 6. 引擎实例不可跨局复用 — `start_ai_vs_ai` 接口变更

`PolicyValueAiEngine` 的 MCTS 搜索树每局结束后推进到终局节点，不可跨局复用。为确保所有引擎类型（含未来新增类型）安全复用，`start_ai_vs_ai` / `start_ai_vs_ai_random` 改为每局创建新引擎。

### 6.1 接口变更

```
AIGameManager.start_ai_vs_ai(ai1_name: str, ai2_name: str, num_games: int, recorder: DataRecorder, temp_config=None, on_progress=None) → List[dict]
AIGameManager.start_ai_vs_ai_random(ai1_name: str, ai2_name: str, num_games: int, recorder: DataRecorder, temp_config=None) → List[dict]
```

内部每局循环：
```
for game_num in range(1, num_games + 1):
    engine_black = create_engine(ai1_name if ai1_is_black else ai2_name)
    engine_white = create_engine(ai2_name if ai1_is_black else ai1_name)
    play_single_game(engine_black, engine_white, ...)
```

`play_single_game` 接口不变（仍接收 `AIEngine` 实例）。

### 6.2 调用方影响

| 调用方 | 变更 |
|--------|------|
| `app/battle.py` (`BattleSession.run`) | 移除 `create_engine` 调用，直接传 `ai1_name` / `ai2_name` |
| `ai/battle.py` (`run_battle`) | 移除 `get_model` + `SinglePolicyAiEngine`，直接传 `ai1_name` / `ai2_name` |
| `ai/scripts/mcts_self_play.py` | 保持不变（已每局手动创建） |

## 7. PvE 模式下的 `after_move`

`AIGameManager.ai_move()`（PvE 人机对战）当前不调用 `after_move`。本意图暂不修改此路径。若 PvE 使用 `PolicyValueAiEngine`，搜索树复用不会生效（同 §6 约束，PvE 场景下每局仅一局，影响有限）。
