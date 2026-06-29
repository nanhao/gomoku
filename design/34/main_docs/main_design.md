# 意图 34：AIEngine 接口统一 · after_move 钩子 · reset 移除

## 概述

1. 将 `AIEngine` 和 `MCTSMoveDecider` 统一为一个类层次，共享 `get_move(board, current_player, strategy) → MoveDecision` 接口（意图 33 已使其签名一致）。
2. 添加抽象 `after_move(action)` 钩子，替代调用侧显式调用 `make_move()` / `reset()` 的决策逻辑。
3. **彻底删除** `reset()`：从 `MCTSSearcher`、`MCTSMoveDecider`（被替换）及所有调用侧移除。
4. 新增 `AIEngineBuilder` 工厂类，统一对外提供两种引擎的创建入口。
5. **目录重构**：创建 `ai/engine/` 子包，所有 engine 类迁入；删除旧的 `ai/engine.py` 和 `ai/mcts/decision_maker.py`。

## 动机

- **消除调用侧 workaround**：当前 `mcts_self_play.py` 在每步落子后需判断 `is_terminal` 以决定调用 `reset()` 还是 `make_move()`。由于意图 32 已确保终局节点在 MCTS 树中创建，`advance_root` 对终局落子同样可成功，无需区分。
- **reset() 无意义**：MCTS 树不跨局共享，每局创建新的 `MCTSSearcher` / 决策器实例，`reset()` 是死代码。
- **接口统一**：两个引擎 `get_move` 签名已一致，调用侧用同一接口即可，无需区分引擎类型。
- **统一创建入口**：通过 `AIEngineBuilder` 提供 `build_single_policy()` 和 `build_policy_value()` 两个工厂方法，替代分散的创建逻辑。
- **目录规范化**：`ai/engine.py` 单文件 → `ai/engine/` 子包，职责更清晰。

## 变更项

### 1. 目录重构

当前结构 → 新结构：

```
ai/engine.py                    →  删除
ai/mcts/decision_maker.py       →  删除
ai/builder.py                   →  移动内容至 ai/engine/builder.py

ai/engine/                      ← 新子包
├── __init__.py                  ← 对外统一导出
├── ai_engine.py                 ← AIEngine 抽象基类 + MoveDecision + board_to_vector
├── single_policy.py             ← SinglePolicyAiEngine（原 AIEngine 具体逻辑）
├── policy_value.py              ← PolicyValueAiEngine（原 MCTSMoveDecider 逻辑）
└── builder.py                   ← AIEngineBuilder 工厂类
```

### 2. AIEngine 重构为抽象基类

```
ai/engine/ai_engine.py:
  MoveDecision (dataclass-like)
    ├── get_position() → (x, y)
    ├── get_probs() → List[float] | None
    └── get_mcts_visit_counts() → List[int] | None

  AIEngine (抽象, ABC)
    ├── get_move(board, current_player, strategy) → MoveDecision    [抽象]
    ├── after_move(action)                                           [抽象]
    └── board_to_vector(board, current_player) → List[int]           [静态]

ai/engine/single_policy.py:
  SinglePolicyAiEngine(AIEngine)    ← 原 AIEngine 具体逻辑
    ├── __init__(model, temperature, top_k, rng)
    ├── get_move: model.predict + 策略选择
    ├── after_move: no-op
    └── _pick_argmax(…)                                            [静态]

ai/engine/policy_value.py:
  PolicyValueAiEngine(AIEngine)     ← 原 MCTSMoveDecider 逻辑
    ├── __init__(searcher, temperature, temp_steps, log_key_prefix)
    ├── get_move: MCTS 搜索 + 策略选择
    ├── after_move: searcher.advance_root(action)
    ├── _get_temperature() → float
    ├── _select_greedy(visit_counts, board) → int                  [静态]
    └── _sample_action(visit_counts, board, tau) → int             [静态]

ai/engine/builder.py:
  AIEngineBuilder                   ← 工厂类
    ├── build_single_policy(ai_name, …) → SinglePolicyAiEngine
    └── build_policy_value(model, …) → PolicyValueAiEngine
```

`AIEngine` 无 `__init__`，各子类自行管理构造参数和字段。`board_to_vector` 保留为 `AIEngine` 静态方法。

> **设计理由**：两个子类的构造参数完全不同（`SinglePolicyAiEngine` 接收 `model`，`PolicyValueAiEngine` 接收 `searcher`），无法在抽象基类中共享 `__init__`。各子类独立定义构造器。

### 3. AIEngineBuilder 工厂类

| 方法 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `build_single_policy(ai_name, temperature, top_k, rng)` | `ai_name: str` + 可选采样参数 | `SinglePolicyAiEngine` | 内部调用 `registry.get_model(ai_name)` 获取模型 |
| `build_policy_value(model, num_simulations, c_puct, dirichlet_epsilon, dirichlet_alpha, temperature, temp_steps, log_key_prefix, log_dir)` | `model: BaseModel` + MCTS 超参 | `PolicyValueAiEngine` | 内部创建 `MCTSSearcher`，再包装为 `PolicyValueAiEngine` |

**`create_engine` 兼容**：保留 `create_engine(ai_name)` 作为 `AIEngineBuilder.build_single_policy(ai_name)` 的薄封装，定义在 `ai/engine/__init__.py` 中对外导出。

### 4. after_move(action) 钩子

- `SinglePolicyAiEngine.after_move(action)`: 空实现（无子树管理需求）。
- `PolicyValueAiEngine.after_move(action)`: 调用 `searcher.advance_root(action)`，成功后再递增 `_step_count`（仅在成功时递增，失败则不变）。失败时抛 `RuntimeError`。

> **注意**：当前 `MCTSMoveDecider.make_move()` 是先 `_step_count += 1` 再 `advance_root`，失败时 `_step_count` 已被错误递增。`after_move` 修正为仅在 advance_root 成功后才递增，行为更安全。

### 5. 删除 reset()

| 位置 | 操作 |
|------|------|
| `MCTSSearcher.reset()` (searcher.py) | 删除方法 |
| `MCTSMoveDecider.reset()` (decision_maker.py) | 文件删除，方法随之移除 |
| `mcts_self_play.py` 中的 `decider.reset()` | 删除，改用 `after_move` |
| 测试中的 `reset()` 验证用例 | 更新或删除 |

### 6. MCTSMoveDecider 替换为 PolicyValueAiEngine

- `MCTSMoveDecider` 类重命名为 `PolicyValueAiEngine`，继承 `AIEngine`，**从 `ai/mcts/decision_maker.py` 移至 `ai/engine/policy_value.py`**。
- `MCTSMoveDecider.make_move()` 迁移为 `PolicyValueAiEngine.after_move()`。
- `MCTSMoveDecider.reset()` 删除。
- 原有 `_step_count`、`_search_id`、温度逻辑保留（不再需要 reset）。
- `get_move` 内部通过 `AIEngine.board_to_vector()` 做 board→state 转换。
- 文件 `ai/mcts/decision_maker.py` **删除**。

### 7. PolicyValueAiEngine.get_move() 棋盘已满行为

与 `SinglePolicyAiEngine` 保持一致：`get_move` 开始时检查棋盘是否已满（225 个位置全部非空），若满则抛出 `ValueError("棋盘已满，无空位可落子")`。

### 8. 调用侧简化

**mcts_self_play.py** 变更对比：

```
  # 创建
- searcher = MCTSSearcher(model, ...)
- decider = MCTSMoveDecider(searcher, ...)
+ from ai.engine.builder import AIEngineBuilder
+ engine = AIEngineBuilder.build_policy_value(model, num_simulations=..., ...)

  # 落子循环
  decision = engine.get_move(board, current_player, strategy)
  action = ...
  board[action] = current_player

+ engine.after_move(action)      # 无条件调用，替代谢子树/终局分支

  if is_terminal:
-     decider.reset()
      break
- decider.make_move(action)

  current_player = switch
```

### 9. PolicyValueAiEngine 跨局生命周期约束

`PolicyValueAiEngine` 不可跨局复用。其内部 `searcher` 的搜索树在局末会推进到终局根节点，后续局的 `get_move` 会从错误的根状态开始。

**规则**：每局必须创建新的 `MCTSSearcher` + `PolicyValueAiEngine` 实例。`AIEngineBuilder.build_policy_value()` 每次调用均创建全新实例，天然满足此规则。

**AIGameManager**：`start_ai_vs_ai()` / `start_ai_vs_ai_random()` 跨局复用 `engine1`/`engine2`。当前这些方法仅接收 `SinglePolicyAiEngine`（通过 `create_engine`），无问题。如果将来需要支持 `PolicyValueAiEngine`，调用方需自行按局重建引擎实例。本次不改变 `AIGameManager` 的引擎复用行为。

**AIGameManager 与 after_move**：`AIGameManager.ai_move()` 和 `play_single_game()` 仅调用 `engine.get_move()`，从未调用 `after_move`。因为 AIGameManager 当前只使用 `SinglePolicyAiEngine`（`after_move` 为 no-op），行为无影响。

## 影响范围

### 文件级变更

| 文件 | 变更 |
|------|------|
| `ai/engine.py` | **删除** |
| `ai/mcts/decision_maker.py` | **删除** |
| `ai/engine/__init__.py` | **新增**：统一导出 AIEngine, MoveDecision, SinglePolicyAiEngine, PolicyValueAiEngine, AIEngineBuilder, create_engine |
| `ai/engine/ai_engine.py` | **新增**：AIEngine 抽象基类 + MoveDecision + board_to_vector |
| `ai/engine/single_policy.py` | **新增**：SinglePolicyAiEngine（原 AIEngine 具体逻辑） |
| `ai/engine/policy_value.py` | **新增**：PolicyValueAiEngine（原 MCTSMoveDecider 逻辑） |
| `ai/engine/builder.py` | **新增**：AIEngineBuilder 工厂类 |
| `ai/mcts/searcher.py` | 删除 `reset()` 方法 |
| `ai/mcts/__init__.py` | 移除 `MCTSMoveDecider` 导出（仅保留 `MCTSSearcher`） |
| `ai/__init__.py` | 不变（`from .engine import AIEngine, create_engine`，`ai/engine` 从单文件变为包，`__init__.py` 同名导出） |

### 调用侧源码变更

| 文件 | 当前引用 | 变更 |
|------|---------|------|
| `ai/__init__.py` L8 | `from .engine import AIEngine, create_engine` | 不变（`ai/engine` 现在是包，`__init__.py` 内导出同名符号） |
| `ai/game_manager.py` L11 | `from ai.engine import AIEngine` | 不变 |
| `ai/game_manager.py` L237, L272 | `AIEngine.board_to_vector(...)` | 不变（静态方法仍在 AIEngine 上） |
| `ai/create_samples.py` L21 | `from ai.engine import AIEngine` | 不变 |
| `ai/create_samples.py` L228 | `engine = AIEngine(model)` | **改为** `engine = SinglePolicyAiEngine(model)`（AIEngine 变为抽象类不可直接实例化） |
| `ai/self_play_generator.py` L11 | `from ai.engine import AIEngine` | 不变 |
| `ai/self_play_generator.py` L75 | `engine = AIEngine(model)` | **改为** `engine = SinglePolicyAiEngine(model)` |
| `ai/battle.py` L32 | `from ai.engine import AIEngine` | 不变 |
| `ai/battle.py` L149-150 | `engine1 = AIEngine(model1)`<br>`engine2 = AIEngine(model2)` | **改为** `engine1 = SinglePolicyAiEngine(model1)`<br>`engine2 = SinglePolicyAiEngine(model2)` |
| `ai/scripts/mcts_self_play.py` L27 | `from ai.mcts.decision_maker import MCTSMoveDecider` | **改为** `from ai.engine.builder import AIEngineBuilder`（或 `from ai.engine import AIEngineBuilder`） |
| `ai/scripts/mcts_self_play.py` L160-172 | `searcher = MCTSSearcher(...)` + `decider = MCTSMoveDecider(...)` | **改为** `engine = AIEngineBuilder.build_policy_value(model, ...)` |
| `ai/scripts/mcts_self_play.py` L187 | `decider.get_move(...)` | **改为** `engine.get_move(...)` |
| `ai/scripts/mcts_self_play.py` L210 | `decider.reset()` | **删除** |
| `ai/scripts/mcts_self_play.py` L216 | `decider.make_move(action)` | **改为** `engine.after_move(action)`（移到 if is_terminal 之前，无条件调用） |

### 测试文件变更

| 文件 | 变更 |
|------|------|
| `tests/unit/test_mcts_decision_maker.py` | **删除**（测试对象 MCTSMoveDecider 已移除） |
| `tests/unit/test_mcts_decider_refactor.py` | **删除**（测试对象 MCTSMoveDecider 已移除） |
| `tests/unit/test_mcts_terminal_nodes.py` L279 | `MCTSMoveDecider` → `PolicyValueAiEngine` |
| `tests/integration/test_mcts_self_play.py` | `MCTSMoveDecider` → `PolicyValueAiEngine`（或通过 Builder 创建） |
| `tests/ai/test_engine.py` | `AIEngine` 实例化 → `SinglePolicyAiEngine` |
| `tests/ai/test_engine_no_fallback.py` | `create_engine` 行为不变 |
| `tests/ai/test_game_manager.py` | `AIEngine` 类型注解不变 |
| `tests/ai/test_game_manager_temp.py` | `AIEngine` 类型注解不变 |
| `tests/ai/test_game_manager_record.py` | `AIEngine` 类型注解不变 |
| `tests/ai/test_battle.py` | `AIEngine` → `SinglePolicyAiEngine` |
| `tests/ai/test_performance_level1.py` | `AIEngine` → `SinglePolicyAiEngine` |
| `tests/ai/test_move_decision.py` | `MoveDecision` 导入路径不变 |
| `tests/ai/unit/test_game_manager_unit.py` | `AIEngine` 类型注解不变 |
| `tests/unit/test_ai_engine_constructor.py` | `AIEngine` 实例化 → `SinglePolicyAiEngine` |
| `tests/unit/test_policy_value_engine.py` | **新增**：PolicyValueAiEngine 完整测试 |
| `tests/unit/test_single_policy_engine.py` | **新增**：SinglePolicyAiEngine 测试 |
| `tests/unit/test_engine_builder.py` | **新增**：AIEngineBuilder 测试 |

## 关键交互图

- 统一后落子流程：见 [flow.puml](flow.puml)
- 类与依赖关系：见 [class_diagram.puml](class_diagram.puml)

## 测试用例

见 [test_design.md](test_design.md)
