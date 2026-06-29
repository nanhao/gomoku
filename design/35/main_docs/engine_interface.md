# 引擎接口设计：after_move 重构 + create_engine 智能路由

## 1. `after_move` 接口变更

### 1.1 抽象基类 — `AIEngine`

```
abstract AIEngine.after_move(self, x: int, y: int)
```

| 维度 | 旧 | 新 |
|------|-----|-----|
| 参数 | `action: int` (0-224) | `x: int` (0-14), `y: int` (0-14) |
| 含义 | 平坦索引 | 棋盘坐标 |
| 异常 | 无约束（抽象） | 无约束（抽象） |

### 1.2 `SinglePolicyAiEngine`

```
SinglePolicyAiEngine.after_move(self, x: int, y: int) → None
```

无搜索树，空实现。参数签名变更即可。

### 1.3 `PolicyValueAiEngine`

```
PolicyValueAiEngine.after_move(self, x: int, y: int) → None
  Raises: RuntimeError（推进失败时）
```

内部实现：
1. 计算 `action = x * 15 + y`
2. 调用 `self.searcher.advance_root(action)`
3. 失败时抛出 `RuntimeError`，消息含 `(x, y)` 和 `action`
4. 成功后 `self._step_count += 1`

## 2. `create_engine` 接口设计

### 2.1 位置

`ai/engine/builder.py`

### 2.2 函数签名

```
create_engine(ai_name: str) → AIEngine
  Raises: ValueError（模型未知 / 多代未指定 / 测试预测失败）
```

### 2.3 智能路由逻辑

```
create_engine(ai_name)
  → model = registry.get_model(ai_name)
  → model.move_to_device()  （若 model.network 非 None 且 CUDA 可用）
  → result = model.predict([0]*226)  （空棋盘，黑方先行；仅检测 result.type）
  → type == TYPE_POLICY_VALUE → AIEngineBuilder.build_policy_value(model, **默认参数)
  → type == TYPE_POLICY_ONLY  → AIEngineBuilder.build_single_policy(ai_name)
```

> 测试预测使用空棋盘向量 `[0]*226`（226 维：225 空位 + 黑方先行）。任意合法 226 维向量均可，仅 `result.type` 用于路由判断。

`PolicyValueAiEngine` 创建的默认参数：

| 参数 | 默认值 |
|------|--------|
| num_simulations | 800 |
| c_puct | 2.0 |
| dirichlet_epsilon | 0.25 |
| dirichlet_alpha | 0.03 |
| temperature | 1.0 |
| temp_steps | 8 |
| log_key_prefix | None |
| log_dir | None |

> `log_dir=None` 表示不使用 MCTS 搜索日志。需要日志时，直接调用 `AIEngineBuilder.build_policy_value(...)` 并传入 `log_dir`。

> 注：`create_engine` 仅接受 `ai_name`，不接受额外参数。需要自定义 MCTS 参数时，直接调用 `AIEngineBuilder.build_policy_value(...)`。

### 2.4 GPU 迁移

`create_engine` 在 `registry.get_model(ai_name)` 之后、构造引擎之前，对 `model.network` 非 None 的模型调用 `model.move_to_device()`。该调用内部检测 CUDA 可用性，不可用时无副作用。调用方无需手动处理 GPU 迁移。

### 2.5 导出变更

`ai/engine/__init__.py` 中移除 `create_engine` 函数体，改为从 `builder` 重新导出：

```
from ai.engine.builder import create_engine
```

### 2.6 异常策略

- `registry.get_model` 失败 → 上抛 `ValueError`
- 测试预测失败（模型加载异常 / 设备错误） → 上抛 `ValueError`
- `build_policy_value` / `build_single_policy` 内部异常 → 透传
- 保持与现有 `create_engine` 一致的异常语义
