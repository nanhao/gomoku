# 设计文档-31：移除 MCTS 的 Batch 和 Virtual Loss 机制

> 意图 31 | 日期：2026-06-27 | 关联模块：src/ai/mcts/

---

## 1. 概述

移除当前 MCTS 搜索器中的 Batch 收集机制和 Virtual Loss 机制，将搜索流程简化为传统的单步 MCTS 循环：每轮 simulation 依次完成 selection → evaluation → backup。

### 动机

1. **Batch 未真正并行**：`_batch_evaluate()` 内部是逐个串行调用 `model.predict()`，不是批量 GPU 推理。batch 收集只增加了复杂度，未带来性能收益。
2. **VL 机制冗余**：Virtual Loss 的作用是让 batch 内的多个 selection 走不同路径。移除 batch 后，每轮 simulation 只选一个叶节点并立即 backup，下一轮自然走不同路径——VL 不再必要。
3. **简化代码**：移除后搜索流程变清晰，`searcher.py` 行数减少约 25%。

---

## 2. 当前状态（变更前）

### 2.1 搜索流程

```
while sim_id < total_sims:
    1. select_and_collect(sims - sim_id)  → 收集最多 batch_size 个叶节点
       ├─ 每步 PUCT 选择后 apply_virtual_loss
       └─ 返回 leaf_nodes 列表
    2. batch_evaluate(leaf_nodes)  → 逐个模型推理 + 扩展子节点
    3. for each leaf:
       backup(leaf)  → undo_virtual_loss → N+=1, W+=v, Q=W/N
```

### 2.2 涉及参数

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `virtual_loss` | 3 | 选择阶段施加的 N/W 惩罚量 |
| `mcts_batch_size` | 16 | 单轮最多收集的叶节点数 |

### 2.3 涉及方法

- `_select_and_collect()` — batch 收集 + VL 施加
- `_batch_evaluate()` — 逐个评估 + 扩展
- `_apply_virtual_loss()` — 施加 VL
- `_undo_virtual_loss()` — 撤销 VL

### 2.4 当前流程交互图

见原设计文档 [design/24/main_docs/mcts_search_flow.puml](../../24/main_docs/mcts_search_flow.puml)。

---

## 3. 目标状态（变更后）

### 3.1 搜索流程

```
while sim_id < total_sims:
    1. select()  → PUCT 下降直到叶节点
    2. evaluate(leaf)  → 模型推理 + 扩展子节点
    3. backup(leaf)  → 沿路径向上 N+=1, W+=v, Q=W/N
```

每轮 simulation 完整处理一个叶节点，全程无 VL 施加/撤销，无 batch 收集。

### 3.2 接口变更

#### MCTSSearcher.__init__

移除参数 `virtual_loss` 和 `mcts_batch_size`。

```
变更前：
def __init__(self, model, num_simulations=800, c_puct=2.0,
             dirichlet_epsilon=0.25, dirichlet_alpha=0.03,
             virtual_loss=3, mcts_batch_size=16, log_dir=None)

变更后：
def __init__(self, model, num_simulations=800, c_puct=2.0,
             dirichlet_epsilon=0.25, dirichlet_alpha=0.03, log_dir=None)
```

#### MCTSSearcher.search()

返回值、参数均不变。内部实现从三阶段 batch 循环改为单步 simulation 循环。

### 3.3 搜索流程交互图

见 [mcts_search_flow.puml](mcts_search_flow.puml)。

---

## 4. 变更范围

### 4.1 修改文件

| 文件 | 变更内容 |
|------|----------|
| `src/ai/mcts/searcher.py` | 移除 `virtual_loss`/`mcts_batch_size` 参数和属性；重写 `search()` 主循环为单步模式；删除 `_select_and_collect()`、`_batch_evaluate()`、`_apply_virtual_loss()`、`_undo_virtual_loss()`；新增简化版 `_select()`、`_evaluate()` |
| `src/ai/mcts/search_logger.py` | 移除 `__init__` 的 `virtual_loss` 参数和 `init_log_data` 中 hyperparams 的 `virtual_loss` 字段 |

### 4.2 不受影响的文件

| 文件 | 原因 |
|------|------|
| `src/ai/mcts/decision_maker.py` | 不引用 `virtual_loss`/`mcts_batch_size`，只调用 `searcher.search()` 和 `advance_root()` |
| `src/ai/scripts/mcts_self_play.py` | 创建 `MCTSSearcher` 时未传递 VL/batch 参数，使用默认值 |
| `src/config/mcts.py` | 不含 VL/batch 配置项 |
| `src/gui/`、`src/app/` | 不引用 `MCTSSearcher` |
| `src/ai/mcts/__init__.py` | 导出 `MCTSSearcher` 不变 |

### 4.3 测试文件变更

| 文件 | 变更 |
|------|------|
| `tests/unit/test_mcts_searcher.py` | 删除 `TestMCTSSearcherVirtualLoss` 测试类（含 2 个用例）；`TestMCTSSearcherBatch` 类改名为 `TestMCTSSearcherModelCall`，保留 `test_model_called_during_search`；新增验证搜索正确性的回归用例 |

---

## 5. 验收标准

### 5.1 功能验收

- [ ] `MCTSSearcher.__init__` 不再接受 `virtual_loss` 和 `mcts_batch_size` 参数
- [ ] `search()` 每轮 simulation 仅处理 1 个叶节点（通过日志或 mock 验证模型调用次数 = simulation 次数）
- [ ] 搜索结果（visit_counts, root_value, root_policy）与原实现一致（给定相同随机种子）
- [ ] `advance_root()` 子树复用不受影响
- [ ] `MCTSSearchLogger` 日志中不再包含 `virtual_loss` 字段
- [ ] `mcts_self_play.py` 正常运行

### 5.2 测试验收

- [ ] 全量测试通过（含新增回归用例）
- [ ] 无 `virtual_loss`、`mcts_batch_size`、`_apply_virtual_loss`、`_undo_virtual_loss`、`_select_and_collect`、`_batch_evaluate` 的死代码残留

### 5.3 代码质量

- [ ] `searcher.py` 不超过 500 行
- [ ] 无重复代码
- [ ] 有丰富注释

---

## 6. 设计文档更新

移除 batch+VL 后，以下设计文档需同步更新（后续意图处理）：

| 文档 | 更新内容 |
|------|----------|
| `design/24/main_docs/mcts_search_flow.puml` | 流程图需反映新的单步搜索流程 |
| `design/24/main_docs/main_design.md` | 移除 VL/batch 相关参数说明 |
