# 意图 34 测试用例

## T34-01：AIEngine 为抽象类，不可直接实例化

- 尝试 `AIEngine()` 直接构造 → 抛出 `TypeError`（抽象类不可实例化）。

## T34-02：AIEngine 声明抽象 get_move 接口

- `AIEngine` 类包含抽象方法 `get_move(board, current_player, strategy) → MoveDecision`。
- 签名参数：`board` 为 15×15 二维矩阵，`current_player` 为 1 或 2，`strategy` 默认 `"argmax"`。

## T34-03：AIEngine 声明抽象 after_move 接口

- `AIEngine` 类包含抽象方法 `after_move(action)`。
- `action` 为 0-224 的落子位置索引。

---

## T34-04：SinglePolicyAiEngine 继承 AIEngine

- `SinglePolicyAiEngine` 是 `AIEngine` 的子类。
- 可实例化，`isinstance(x, AIEngine)` 返回 `True`。

## T34-05：SinglePolicyAiEngine.get_move() 行为与原 AIEngine 一致

- **argmax 策略**：选择模型输出概率最大的合法空位，并列时随机打破。
- **sampling 策略**：委托 `temperature_sample`，使用构造时注入的 temperature / top_k / rng。
- **返回值**：`MoveDecision` 实例，`get_probs()` 返回 225 维概率分布。
- **棋盘已满**：抛 `ValueError`，消息含 "棋盘已满"。

## T34-06：SinglePolicyAiEngine.after_move(action) 为 no-op

- 调用 `after_move(任意 0-224 action)` → 不抛异常，无任何副作用。
- 重复调用同样无副作用。

---

## T34-07：PolicyValueAiEngine 继承 AIEngine

- `PolicyValueAiEngine` 是 `AIEngine` 的子类。
- 可实例化，`isinstance(x, AIEngine)` 返回 `True`。

## T34-08：PolicyValueAiEngine 构造函数接受 searcher

- 构造参数：`searcher`（`MCTSSearcher` 实例）、`temperature`（默认 1.0）、`temp_steps`（默认 8）、`log_key_prefix`（可选）。
- 内部初始化 `_step_count = 0`、`_search_id = 0`。

## T34-09：PolicyValueAiEngine.get_move() 通过 MCTS 搜索落子

- 调用 `get_move(board, current_player, strategy)` → 内部转换 board 为 226 维向量 → 调用 `searcher.search()` → 按 strategy 选择。
- **argmax**：强制选择访问计数最大的合法空位。多个空位访问计数并列最大时，选第一个遍历到的（确定性，无随机打破平局）。
- **sampling**：按 `_get_temperature()` 决定 tau，tau > 0 时从访问计数采样，tau = 0 时退化为贪婪。
- **返回值**：`MoveDecision` 实例，`get_mcts_visit_counts()` 返回 225 维访问计数。
- **strategy 非法**：抛 `ValueError`。

> **平局处理说明**：`PolicyValueAiEngine` argmax 的平局行为是确定性的（选第一个最大访问计数的空位），与 `SinglePolicyAiEngine._pick_argmax` 的随机打破平局不同。这是因为 MCTS 访问计数由搜索过程自然引入随机性（狄利克雷噪声），无需额外随机打破机制。

## T34-10：PolicyValueAiEngine.get_move() 自动生成 log_key

- 构造时传入 `log_key_prefix` → 每次 `get_move` 自动生成 `"{prefix}_search{_search_id}"` 并自增 `_search_id`。
- 构造时 `log_key_prefix=None` → 不传 log_key 给 searcher。

## T34-10a：PolicyValueAiEngine.get_move() 棋盘已满时抛 ValueError

- 棋盘 225 个位置全部非空 → `get_move` 抛出 `ValueError`，消息含 "棋盘已满"。
- 行为与 `SinglePolicyAiEngine.get_move()` 一致。

## T34-11：PolicyValueAiEngine.after_move(action) 调用 advance_root 并递增步数

- `after_move(action)` → 先调用 `searcher.advance_root(action)`。
- `advance_root` 返回 `True` → 成功，然后 `_step_count` 自增 1。
- `advance_root` 返回 `False` → 抛 `RuntimeError`（消息包含 action 值和 "advance_root failed"），`_step_count` 不变。
- 与当前 `make_move` 的顺序差异：`make_move` 是先 `_step_count += 1` 再 `advance_root`（失败时 _step_count 已被错误递增）；`after_move` 改为仅在成功后才递增。

## T34-12：PolicyValueAiEngine.after_move(action) 对终局落子成功

- 对必胜落子（如黑棋连五的最后一步）调用 `after_move(action)` → 不抛异常 → `advance_root` 返回 `True`（终局节点已在树中，意图 32）。
- `_step_count` 正确递增。

## T34-13：PolicyValueAiEngine.after_move(action) 对非终局落子成功

- 对普通落子调用 `after_move(action)` → 不抛异常 → `_step_count` 递增。

---

## T34-14：MCTSSearcher 无 reset() 方法

- `MCTSSearcher` 实例没有 `reset` 方法。
- 创建新的 `MCTSSearcher` 实例即获得干净的搜索树（`tree` 为空，`_root_hash` 为 `None`）。

## T34-15：MCTSSearcher 实例之间无状态泄漏

- 创建两个独立 `MCTSSearcher` 实例 → 各自搜索 → 树互不影响。

---

## T34-16：PolicyValueAiEngine 定义在 ai/engine/ 子包中

- `PolicyValueAiEngine` 定义在 `ai.engine.policy_value` 模块中。
- 继承自 `ai.engine.ai_engine.AIEngine`。
- 可通过 `ai.engine` 包导入（`__init__.py` 重新导出）。
- 原 `MCTSMoveDecider` 类不存在于任何模块中。
- `ai/mcts/decision_maker.py` 文件已删除。

## T34-16a：旧文件已删除

- `ai/engine.py` 文件已删除。
- `ai/mcts/decision_maker.py` 文件已删除。
- `ai/engine/` 目录存在（为子包），包含 `__init__.py`、`ai_engine.py`、`single_policy.py`、`policy_value.py`、`builder.py`。

---

## T34-17：AIEngineBuilder.build_single_policy 返回 SinglePolicyAiEngine

- `AIEngineBuilder.build_single_policy("random")` → 返回 `SinglePolicyAiEngine` 实例，`isinstance(x, AIEngine)` 为 `True`。
- `AIEngineBuilder.build_single_policy("cnn_v7")` 等同样正确返回。
- 支持可选采样参数 `temperature`、`top_k`、`rng`。

## T34-18：AIEngineBuilder.build_policy_value 返回 PolicyValueAiEngine

- `AIEngineBuilder.build_policy_value(model)` → 内部创建 `MCTSSearcher(model, ...)` → 包装为 `PolicyValueAiEngine` → 返回。
- 返回实例的 `isinstance(x, AIEngine)` 为 `True`。
- MCTS 超参（`num_simulations`、`c_puct`、`dirichlet_epsilon`、`dirichlet_alpha`）可自定义。
- 采样参数（`temperature`、`temp_steps`）可自定义。
- 每次调用创建全新 `MCTSSearcher` + `PolicyValueAiEngine` 实例（满足跨局不复用约束）。

## T34-19：create_engine 委托给 AIEngineBuilder

- `create_engine("random")` → 等价 `AIEngineBuilder.build_single_policy("random")` → 返回 `SinglePolicyAiEngine`。
- 行为与现有调用方完全兼容。

---

## T34-20：mcts_self_play.py 调用 after_move 无条件、无 reset

- 每步落子后调用 `engine.after_move(action)`，位于终局判定之前。
- 无 `is_terminal` 条件分支来决定是否调用 `after_move`。
- 无 `reset()` 调用。
- 使用 `AIEngineBuilder.build_policy_value()` 创建引擎。

## T34-21：create_engine 返回 SinglePolicyAiEngine

- `create_engine("random")` → 返回 `SinglePolicyAiEngine` 实例。
- `create_engine("cnn_v7")` 等同样返回 `SinglePolicyAiEngine`。

---

## T34-22：AIGameManager 通过 AIEngine 接口工作

- `AIGameManager.__init__` 参数类型为 `AIEngine`（抽象基类）。
- 传入 `SinglePolicyAiEngine` → PvE 流程正常。
- `play_single_game` 的 `engine_black`/`engine_white` 参数类型为 `AIEngine`。

## T34-23：单局自对弈通过 PolicyValueAiEngine 完成

- 创建 `MCTSSearcher` + `PolicyValueAiEngine` → 执行完整一局 → 每步正确调用 `get_move` 和 `after_move` → 生成样本列表。
- 终局正确 break，无异常。

## T34-24：PolicyValueAiEngine 不可跨局复用

- 创建 `PolicyValueAiEngine` 实例 → 完成一局终局（after_move 推进到终局根节点）→ 用同一实例开始新局 → `get_move` 的行为未定义（搜索树根节点为上一局终局状态）。
- **约束**：每局必须创建新 `MCTSSearcher` + `PolicyValueAiEngine`。

## T34-25：AIGameManager 仅使用 SinglePolicyAiEngine

- `create_engine()` 始终返回 `SinglePolicyAiEngine` 实例。
- `AIGameManager.play_single_game()` 仅调用 `engine.get_move()`，不调用 `after_move`（因 `SinglePolicyAiEngine.after_move` 为 no-op，行为无害）。
- `AIGameManager.start_ai_vs_ai()` 跨局复用 engine 实例，仅对 `SinglePolicyAiEngine` 安全。

---

## T34-26：原 AIEngine 直接构造点改为 SinglePolicyAiEngine

- `create_samples.py` L228 `engine = AIEngine(model)` → 现改为 `engine = SinglePolicyAiEngine(model)`。
- `self_play_generator.py` L75 `engine = AIEngine(model)` → 现改为 `engine = SinglePolicyAiEngine(model)`。
- `battle.py` L149-150 `AIEngine(model1)` / `AIEngine(model2)` → 现改为 `SinglePolicyAiEngine(model1)` / `SinglePolicyAiEngine(model2)`。

## T34-27：ai/mcts/__init__.py 不再导出 MCTSMoveDecider

- `from ai.mcts import MCTSMoveDecider` → `ImportError`。
- `from ai.mcts import MCTSSearcher` → 正常。

## T34-28：ai/__init__.py 仍可导入 AIEngine 和 create_engine

- `from ai import AIEngine, create_engine` → 正常（通过 `ai/engine/__init__.py` 导出）。
