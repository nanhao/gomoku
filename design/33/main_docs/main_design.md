# 意图 33：AiEngine / MCTSMoveDecider 接口重构

## 概述

将采样参数和日志标识的注入点从方法调用时提前到构造时，降低调用方的重复传参负担，
提升接口内聚性，并消除 UUID 依赖。

## 变更项

### 1. AiEngine：temperature / top_k / rng 移入构造函数

**现状**：`get_move(board, player, strategy, temperature, top_k, rng)` 每次调用传入全部采样参数。

**目标**：采样参数在构造时注入，`get_move()` 仅保留 `strategy` 控制切换 argmax / sampling。

**理由**：
- 一局对弈中 temperature / top_k / rng 全程不变，每次透传是冗余的。
- 构造时注入语义更清晰：引擎实例代表"用特定采样配置运行的 AI"。

### 2. MCTSMoveDecider：log_key 移入构造函数，取消 UUID

**现状**：
- `get_move(state, log_key)` 每次调用由外部拼接 `run{run_id}_game{game_idx}_step{uuid}`。
- `log_key` 透传给 `searcher.search()` → `logger.write_log()`。

**目标**：
- 构造函数新增 `log_key_prefix` 参数（格式 `"run{run_id}_game{game_idx}"`）。
- `get_move()` 内部自增 `_search_id`（从 0 起），拼接完整 `log_key = f"{prefix}_search{search_id}"`。
- 移除 UUID 依赖。

**理由**：
- 日志键应由决策器内部管理，调用方只需告知运行/局上下文。
- 自增 ID 比 UUID 更短、有序、可读，且不依赖外部库。

### 3. MCTSMoveDecider：get_move() 新增 strategy 参数

**现状**：`get_move()` 无 strategy 参数，始终按 `_get_temperature()` 决定贪婪/采样。

**目标**：新增 `strategy` 参数，语义与 `AiEngine.get_move()` 一致：
- `"argmax"`：强制贪婪选择（`_select_greedy`）。
- `"sampling"`：按温度采样（`_sample_action`，使用内部 tau 逻辑）。

**理由**：与 AiEngine 保持一致的策略控制接口，调用方可按步显式切换策略。

### 4. MCTSMoveDecider：get_move() 用 board + current_player 替代 state

**现状**：`get_move(state)` 接收 226 维向量（前 225=棋盘 + 末位=落子方）。

**目标**：`get_move(board, current_player)` 接收与 `AiEngine.get_move()` 相同的参数形式：
- `board: List[List[int]]` — 15×15 棋盘矩阵
- `current_player: int` — 当前落子方 (1=黑, 2=白)

**理由**：使 `MCTSMoveDecider.get_move()` 与 `AiEngine.get_move()` 接口完全一致，调用方无需区分两种引擎。内部由决策器自行完成 board → 226 维向量的转换。

## 影响范围

| 模块 | 变更性质 |
|------|---------|
| `ai/engine.py` — AIEngine | 构造函数新增参数；get_move() 删减参数 |
| `ai/mcts/decision_maker.py` — MCTSMoveDecider | 构造函数新增参数；get_move() 删 log_key/state、增 strategy/board/current_player |
| `ai/mcts/searcher.py` — MCTSSearcher | 无变更 |
| `ai/game_manager.py` — play_single_game | 调用方适配 |
| `ai/scripts/mcts_self_play.py` | 调用方适配，移除 uuid import |
| `tests/unit/test_mcts_decision_maker.py` | 测试适配 |
| `tests/integration/test_mcts_self_play.py` | 测试适配 |

## 关键交互图

- AiEngine 重构前后调用对比：见 [ai_engine_refactor_flow.puml](ai_engine_refactor_flow.puml)
- MCTSMoveDecider 重构前后调用对比：见 [mcts_decider_refactor_flow.puml](mcts_decider_refactor_flow.puml)

## 类与依赖关系图

见 [class_diagram.puml](class_diagram.puml)
