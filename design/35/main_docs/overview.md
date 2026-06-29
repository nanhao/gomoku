# 设计文档：意图 35 — create_engine 智能路由 + after_move 签名重构

## 1. 概述

本设计包含四项协调变更：

| # | 变更 | 类型 | 影响范围 |
|---|------|------|----------|
| 1 | `after_move(action)` → `after_move(x, y)` | 重构（签名变更） | `AIEngine` 抽象基类、`SinglePolicyAiEngine`、`PolicyValueAiEngine`、所有调用方 |
| 2 | `create_engine(ai_name)` 智能路由 | 新功能 | `ai/engine/builder.py`（新增），`ai/engine/__init__.py`（移除旧实现） |
| 3 | `AiGameManager` / `battle.py` 采用新接口 | 适配 | `ai/game_manager.py`、`ai/battle.py`、`app/battle.py` |
| 4 | 引擎实例不可跨局复用 | 约束 | `AIGameManager.start_ai_vs_ai`、`start_ai_vs_ai_random` 改为每局新建引擎 |

## 2. 现状分析

### 2.1 `after_move` 当前签名

```
AIEngine.after_move(self, action: int)        # abstract
SinglePolicyAiEngine.after_move(self, action)  # no-op
PolicyValueAiEngine.after_move(self, action)   # → searcher.advance_root(action)
```

调用方将棋盘坐标转换为 `action = x * 15 + y`（平坦索引 0-224）后传入。这强制调用方了解内部编码方式。

### 2.2 `create_engine` 当前实现

位于 `ai/engine/__init__.py`，仅创建 `SinglePolicyAiEngine`：

```python
def create_engine(ai_name: str):
    return AIEngineBuilder.build_single_policy(ai_name)
```

不支持双头模型（如 cnn_v7），调用方需自行判断模型类型并手动创建 `PolicyValueAiEngine`。

### 2.3 `AiGameManager.play_single_game` 当前流程

- 接收 `engine_black`、`engine_white` 参数
- 落子循环中不调用 `after_move`（PolicyValueAiEngine 的搜索树复用未生效）

### 2.4 `ai/battle.py`（CLI）当前流程

- 使用 `get_model()` + `SinglePolicyAiEngine()` 直接构造引擎
- 未使用 `create_engine`

## 3. 设计决策

### 决策 1：`after_move` 使用 `(x, y)` 坐标

**理由**：`get_move` 返回 `MoveDecision` 含 `(x, y)`，调用方无需再转换为平坦索引。降低调用方的认知负担，接口更自然。

### 决策 2：`create_engine` 置于 `ai/engine/builder.py`

**理由**：与 `AIEngineBuilder` 同文件，集中管理引擎创建逻辑。`ai/engine/__init__.py` 中的旧 `create_engine` 改为从 `builder` 重新导出。

### 决策 3：引擎类型由模型预测输出类型决定

通过 `ModelPredict.type` 判断：
- `TYPE_POLICY_VALUE` → 创建 `PolicyValueAiEngine`
- `TYPE_POLICY_ONLY` → 创建 `SinglePolicyAiEngine`

### 决策 4：`create_engine` 内部处理 GPU 迁移

`create_engine` 对 `model.network` 非 None 的模型调用 `model.move_to_device()`，调用方无需手动处理。

### 决策 5：引擎实例不可跨局复用 — `start_ai_vs_ai` 改为每局创建引擎

`PolicyValueAiEngine` 的 MCTS 搜索树在局末推进到终局节点，不可跨局复用。

**方案**：`AIGameManager.start_ai_vs_ai` / `start_ai_vs_ai_random` 的接口从接收 `AIEngine` 实例改为接收 `ai_name: str`，内部每局调用 `create_engine(ai_name)` 创建新引擎。`play_single_game` 接口保持不变（仍接收 `AIEngine`）。

| 接口 | 旧签名 | 新签名 |
|------|--------|--------|
| `start_ai_vs_ai` | `(engine1, engine2, num_games, ...)` | `(ai1_name, ai2_name, num_games, ...)` |
| `start_ai_vs_ai_random` | `(engine1, engine2, num_games, ...)` | `(ai1_name, ai2_name, num_games, ...)` |

调用方（`app/battle.py`、`ai/battle.py`）不再预先调用 `create_engine`，直接传入 ai 名称字符串。详见 [caller_changes.md §6](caller_changes.md)。

## 4. 子文档索引

| 文档 | 内容 |
|------|------|
| [engine_interface.md](engine_interface.md) | `after_move` 签名变更 + `create_engine` 接口设计 |
| [caller_changes.md](caller_changes.md) | `AiGameManager` / `battle.py` 适配方案 |
| [test_cases.md](test_cases.md) | 完整测试用例集 |

## 5. 交互图索引

| 图 | 文件 |
|----|------|
| `create_engine` 智能路由时序 | [diagrams/engine_creation_flow.puml](diagrams/engine_creation_flow.puml) |
| 对局循环含 `after_move` 时序 | [diagrams/game_loop_after_move.puml](diagrams/game_loop_after_move.puml) |
| 类图与依赖关系 | [diagrams/class_diagram.puml](diagrams/class_diagram.puml) |
