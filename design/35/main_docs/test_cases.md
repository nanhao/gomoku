# 测试用例集

测试文件位置：`tests/unit/`，按测试主题拆分。

---

## T35-01 ~ T35-04：`after_move(x, y)` 签名变更

### T35-01 — `SinglePolicyAiEngine.after_move(x, y)` 正常调用

| 项 | 内容 |
|----|------|
| **前提** | 创建 `SinglePolicyAiEngine` 实例 |
| **操作** | 依次调用 `after_move(0, 0)`、`after_move(7, 7)`、`after_move(14, 14)` |
| **预期** | 所有调用不抛异常；之后 `get_move` 仍正常工作 |

### T35-02 — `PolicyValueAiEngine.after_move(x, y)` 成功推进

| 项 | 内容 |
|----|------|
| **前提** | 创建 `PolicyValueAiEngine`，对空棋盘执行 `get_move` |
| **操作** | 取 `get_move` 返回的 `(x, y)`，调用 `after_move(x, y)` |
| **预期** | 不抛异常；`engine._step_count == 1` |

### T35-03 — `PolicyValueAiEngine.after_move(x, y)` 无效坐标抛异常（含消息验证）

| 项 | 内容 |
|----|------|
| **前提** | 创建 `PolicyValueAiEngine`，对空棋盘执行 `get_move` |
| **操作** | 调用 `after_move(99, 99)`（坐标超出 0-14） |
| **预期** | 抛出 `RuntimeError`；异常消息 **同时** 包含 `(99, 99)` 和 `1584`（action=99*15+99） |

> 消息格式示例：`PolicyValueAiEngine.after_move((99, 99)) / action=1584: advance_root failed.`

### T35-04 — `PolicyValueAiEngine.after_move` 终局节点不抛异常

| 项 | 内容 |
|----|------|
| **前提** | 构造必胜棋局（如黑方 (7,0)-(7,3) 四子），`PolicyValueAiEngine` 执行 `get_move` 获取必胜落子 `(x, y)` |
| **操作** | 调用 `after_move(x, y)` |
| **预期** | 不抛异常（终局节点已在搜索树中） |

---

## T35-05 ~ T35-10：`create_engine` 智能路由

### T35-05 — 单头模型创建 `SinglePolicyAiEngine`

| 项 | 内容 |
|----|------|
| **前提** | 注册中心有 `random` 模型 |
| **操作** | `engine = create_engine("random")` |
| **预期** | 返回类型为 `SinglePolicyAiEngine`；`engine.model.name == "random"` |

### T35-06 — 双头模型创建 `PolicyValueAiEngine`

| 项 | 内容 |
|----|------|
| **前提** | 注册中心有双头模型（如 `cnn_v7.1`，需 gen-1 权重文件存在） |
| **操作** | `engine = create_engine("cnn_v7.1")` |
| **预期** | 返回类型为 `PolicyValueAiEngine`；`engine.searcher` 非 None |

> `cnn_v7` 在注册中心标记为 `supports_multi_gen=True`，必须带代数后缀（如 `cnn_v7.1`）。若 gen-1 权重不存在，此测试 skip。

### T35-07 — 未知模型名抛 `ValueError`

| 项 | 内容 |
|----|------|
| **前提** | 注册中心无 `nonexistent_ai` |
| **操作** | `create_engine("nonexistent_ai")` |
| **预期** | 抛出 `ValueError` |

### T35-08 — `create_engine` 返回的引擎可正常 `get_move`

| 项 | 内容 |
|----|------|
| **前提** | `create_engine("random")` 成功 |
| **操作** | 空棋盘调用 `engine.get_move(board, 1)` |
| **预期** | 返回 `MoveDecision`；`get_position()` 在 0-14 范围内 |

### T35-09 — `create_engine` 对双头模型的 GPU 迁移

| 项 | 内容 |
|----|------|
| **前提** | 创建 `create_engine("cnn_v7.1")`（CUDA 可用时），否则 skip |
| **操作** | 引擎创建后，检查 `engine.searcher.model.network` 所在设备 |
| **预期** | 若 CUDA 可用，network 在 GPU 上；否则在 CPU 上。`get_move` 正常返回，无设备错误 |

> 无 CUDA 环境时此测试 skip。

### T35-10 — `ai/engine/__init__.py` 正确导出 `create_engine`

| 项 | 内容 |
|----|------|
| **前提** | 无 |
| **操作** | `from ai.engine import create_engine` |
| **预期** | 导入成功，`callable(create_engine)` 为 `True` |

---

## T35-11 ~ T35-14：`AiGameManager` 调用 `after_move`

### T35-11 — `play_single_game` 对 `SinglePolicyAiEngine` 调用 `after_move`

| 项 | 内容 |
|----|------|
| **前提** | 两个 `SinglePolicyAiEngine(random)`、`DataRecorder` |
| **操作** | `AiGameManager.play_single_game(engine1, engine2, recorder, 1)` |
| **预期** | 对局正常完成，不抛异常；recorder 记录中有数据 |

### T35-12 — `play_single_game` 对 `PolicyValueAiEngine` 调用 `after_move`

| 项 | 内容 |
|----|------|
| **前提** | 两个 `PolicyValueAiEngine`（使用 mock searcher，num_simulations=50） |
| **操作** | `AiGameManager.play_single_game(engine1, engine2, recorder, 1)` |
| **预期** | 对局正常完成；每步落子后 `after_move` 被调用（验证 `_step_count` 递增） |

### T35-13 — `after_move` 在落子后、`move_count` 递增前调用

| 项 | 内容 |
|----|------|
| **前提** | Mock 引擎（记录调用顺序的 stub） |
| **操作** | 执行一局 `play_single_game` |
| **预期** | 调用顺序为：`get_move` → `make_move` → `after_move`（不改变 move_count 递增时机） |

### T35-14 — `after_move` 对 `PolicyValueAiEngine` 异常不静默吞没

| 项 | 内容 |
|----|------|
| **前提** | Mock searcher，`advance_root` 恒返回 False |
| **操作** | `PolicyValueAiEngine` 参与对局 |
| **预期** | `play_single_game` 内部首次调用 `after_move` 时传播 `RuntimeError` |

> 注：此测试验证 `after_move` 失败不会静默吞没。上层（如 `start_ai_vs_ai`）决定是否捕获。

---

## T35-15：`ai/battle.py`（CLI）适配新接口

### T35-15 — CLI `run_battle` 直接传 ai 名称

| 项 | 内容 |
|----|------|
| **前提** | 无 |
| **操作** | `run_battle("random", "random", 1)` |
| **预期** | 对局正常完成（无异常）；结果列表含 1 局记录（有 winner 字段）。`run_battle` 内部不直接引用 `get_model` / `SinglePolicyAiEngine` / `create_engine` |

---

## T35-16 ~ T35-18：补充用例

### T35-16 — `mcts_self_play.py` `after_move(x, y)` 签名适配

| 项 | 内容 |
|----|------|
| **前提** | 对 `PolicyValueAiEngine` 执行一次 `get_move` |
| **操作** | 取 `md = engine.get_move(board, 1)`，`x, y = md.get_position()`，调用 `engine.after_move(x, y)` |
| **预期** | 不抛异常；`_step_count == 1`（验证平替原 `after_move(action)` 调用） |

### T35-17 — `create_engine` 测试预测失败抛 `ValueError`

| 项 | 内容 |
|----|------|
| **前提** | Mock 一个 `predict()` 抛异常的模型，临时注册 |
| **操作** | `create_engine("failing_model")` |
| **预期** | 抛出 `ValueError`（非原始异常直接透传） |

### T35-18 — `SinglePolicyAiEngine.after_move` 无副作用验证

| 项 | 内容 |
|----|------|
| **前提** | 创建 `SinglePolicyAiEngine(random)` |
| **操作** | 调用 `after_move(0, 0)` 后，`get_move` 空棋盘两次 |
| **预期** | 两次 `get_move` 结果一致（`after_move` 不改变引擎内部状态） |

---

## T35-19 ~ T35-20：引擎实例不可跨局复用

### T35-19 — `start_ai_vs_ai` 每局创建新引擎（接收 ai 名称）

| 项 | 内容 |
|----|------|
| **前提** | `DataRecorder` 实例 |
| **操作** | `AIGameManager.start_ai_vs_ai("random", "random", 2, recorder)` |
| **预期** | 2 局正常完成；结果列表含 2 条记录；内部每局调用 `create_engine`（不复用实例） |

### T35-20 — `start_ai_vs_ai_random` 每局创建新引擎（接收 ai 名称）

| 项 | 内容 |
|----|------|
| **前提** | `DataRecorder` 实例 |
| **操作** | `AIGameManager.start_ai_vs_ai_random("random", "random", 2, recorder)` |
| **预期** | 2 局正常完成；引擎实例不跨局复用 |
