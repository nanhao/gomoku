# Batch MCTS 与 Virtual Loss 设计文档

## 1. 概述

将当前逐 simulation 串行执行的 MCTS 改为 **batch MCTS + virtual loss**，核心变化：
- **Batch MCTS**：每批次收集 B 条 simulation 的叶节点，合并为一批做一次模型推理，减少模型调用次数（800 sims→50 批模型调用），提升 GPU 利用率
- **Virtual Loss**：批次内 collection 阶段，每遍历一条边施加临时惩罚（N+=VL, W-=VL），迫使后续 selection 走不同路径，批次评估完后撤销 VL、执行真实回传
- **单线程**：Python 执行保持单线程，batch 指模型推理批量化

**关键参数**：batch_size=16, virtual_loss=3.0（均可配置）。

**向后兼容**：batch_size=1 且 virtual_loss=0 时等价于原逐 simulation 模式。

## 2. 受影响的模块

| 模块 | 文件 | 变更类型 |
|------|------|---------|
| MCTSSearcher | `src/ai/mcts/searcher.py` | 修改：batch 循环 + VL 机制 |
| CNNV7Model | `src/ai/models/cnn_v7/model.py` | 新增：`predict_batch()` |
| 配置常量 | `src/config/mcts.py` | 新增：batch_size/VL 默认值 |
| AIEngineBuilder | `src/ai/engine/builder.py` | 修改：透传新参数 |
| mcts_self_play | `src/ai/scripts/mcts_self_play.py` | 新增：CLI 参数 |
| MCTSSearchLogger | `src/ai/mcts/search_logger.py` | 无变更（per-sim 日志结构不变） |

## 3. 接口设计

### 3.1 MCTSSearcher（修改）

**新增构造参数**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `batch_size` | `int` | `16` | 每批收集 simulation 数。设为 1 时退化为逐 sim 模式 |
| `virtual_loss` | `float` | `3.0` | 虚拟损失值。设为 0 时禁用 VL |

**公开方法不变**：`search()`, `advance_root()` 签名和行为契约不变，仅内部实现改为 batch 循环。

### 3.2 CNNV7Model（新增方法）

**`predict_batch(board_vectors: List[List[int]]) -> List[ModelPredict]`**

- 输入：多个 226 维棋盘向量
- 输出：等长 `ModelPredict` 列表（含 `policy` 和 `value`）
- 内部利用 `CNNV7Network.forward()` 的 batch 输入能力（`(batch, 3, 15, 15)`）

> `predict_batch` 不放入 `BaseModel` 抽象接口，仅作为 `CNNV7Model`（双头模型）的公开方法。`MCTSSearcher` 构造时通过 `model` 参数接收的是具有此方法的实例。

### 3.3 配置常量（`src/config/mcts.py`）

新增：

| 常量 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `MCTS_BATCH_SIZE` | `int` | `16` | 默认 batch size |
| `MCTS_VIRTUAL_LOSS` | `float` | `3.0` | 默认 virtual loss 值 |

`MCTSSearcher` 构造未传参时读取这些常量。

### 3.4 AIEngineBuilder（修改）

`build_policy_value()` 新增两个透传参数：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `batch_size` | `int` | `16` | 透传给 MCTSSearcher |
| `virtual_loss` | `float` | `3.0` | 透传给 MCTSSearcher |

### 3.5 mcts_self_play CLI（新增参数）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-size` | `16` | 每批 simulation 数 |
| `--virtual-loss` | `3.0` | virtual loss 值 |

## 4. 关键流程

详见流程图 [batch_mcts_flow.puml](batch_mcts_flow.puml)。

核心算法伪码：

```
search(state, num_simulations):
    init_search(state)
    剩余 sims = num_simulations

    while 剩余 sims > 0:
        batch_leaves = []      # 待模型评估的非终局叶节点
        terminal_leaves = []   # 终局叶节点（无需模型）
        B = min(batch_size, 剩余 sims)

        # 阶段 1：收集 B 个叶节点（带 VL）
        for i in 1..B:
            leaf = _select_with_virtual_loss()
            if leaf.is_terminal:
                _revert_vl_on_path(leaf.path)
                terminal_leaves.append(leaf)
            else:
                batch_leaves.append(leaf)

        # 阶段 2：batch 模型评估（仅非终局节点）
        if batch_leaves 非空:
            results = model.predict_batch([l.board_vector for l in batch_leaves])

        # 阶段 3：撤销 VL + 真实回传
        for leaf in terminal_leaves:
            _create_terminal_node(leaf)
            _backup(leaf)
        for leaf, result in zip(batch_leaves, results):
            _revert_vl_on_path(leaf.path)
            leaf.value = result.value
            leaf.policy = result.policy
            _expand_leaf(leaf)
            _backup(leaf)

        剩余 sims -= B
```

**Virtual Loss 施加规则**：

- `_select_with_virtual_loss()` 遍历路径时，每条边施加：`N += virtual_loss`, `W -= virtual_loss`
- 这导致 Q'=(W-VL)/(N+VL)，PUCT 中的 Q 自然降低，批次内后续 selection 避免走同一条路径
- 撤销时对称恢复：`N -= virtual_loss`, `W += virtual_loss`

**重复叶节点处理**：

- VL 施加与撤销按每条 simulation 独立计数，天然对称
- 同一叶节点被收集两次，路径 VL 施加两次、撤销两次 → 数学上精确恢复
- **不去重**，省去重新收集的复杂度

**终局叶节点批次缩小**：

- 若 B=16 但收集到的终局叶节点数量接近 B，实际模型 batch 自动缩小
- 极端情况全为终局叶节点时，跳过模型调用

## 5. 类图与依赖关系

- 类图：[class_diagram.puml](class_diagram.puml)
- 依赖关系图：[dependency_diagram.puml](dependency_diagram.puml)

## 6. 测试用例

> 测试用例用文字与符号精确描述，不使用代码。测试数据使用随机初始化的 CNNV7Model，不依赖已训练的权重。

### 6.1 MCTSSearcher - Virtual Loss 基本行为

| 编号 | 场景 | Given | When | Then |
|------|------|-------|------|------|
| T1.1 | VL 施加 | 空棋盘，batch_size=2, VL=3.0, 模型随机权重 | 执行 2 次 `_select_with_virtual_loss()` | 每条被遍历边的 N 增加 3×被经过次数，W 减少相同值。对于被经过 n 次的边：N' = N_orig + 3n, W' = W_orig - 3n |
| T1.2 | VL 撤销 | T1.1 之后 | 对被收集叶节点的路径调用 `_revert_vl_on_path()` | 每条边的 N、W 恢复为施加前原始值 |
| T1.3 | VL 对称性 | 空棋盘，batch_size=1, VL=3.0, 执行一轮 search | 完成后检查树中所有边的 N、W 状态 | 没有任何残留 VL：所有统计量均为整数（未被 VL=3.0 部分抵消），且 search 返回值正常 |

### 6.2 MCTSSearcher - Batch 批处理

| 编号 | 场景 | Given | When | Then |
|------|------|-------|------|------|
| T2.1 | batch_size 整除总 sims | num_simulations=100, batch_size=16 | 执行 search() | 模型 `predict_batch()` 被调用 7 次（最后一次 batch=4） |
| T2.2 | batch_size=1 退化 | batch_size=1, VL=0 | 执行 search()，对比旧逐 sim 模式 | 搜索结果（visit_counts, root_value）完全相同 |
| T2.3 | 全终局批次 | 棋盘接近填满，剩余合法动作极少的终局状态 | 执行 search() | 终局叶节点正确走规则判定分支（不调模型），VL 正常撤销回传 |

### 6.3 MCTSSearcher - 批次内探索多样性

| 编号 | 场景 | Given | When | Then |
|------|------|-------|------|------|
| T3.1 | VL 阻止路径收敛 | 空棋盘，batch_size=8, VL=3.0, 模型随机权重 | 收集一整批 8 个叶节点 | 这 8 个叶节点的 state_hash 不相同（VL 迫使探索发散） |
| T3.2 | VL=0 路径收敛对比 | 空棋盘，batch_size=8, VL=0 | 收集一整批 8 个叶节点 | 至少存在重复 state_hash（无 VL 时批次内路径高度重合） |

### 6.4 MCTSSearcher - 重复叶节点

| 编号 | 场景 | Given | When | Then |
|------|------|-------|------|------|
| T4.1 | 重复叶节点 VL 正确撤销 | 极小棋盘（如 3×3），batch_size=4, VL=3.0；构造场景使同一叶节点被收集 2 次 | 批次完成后检查树状态 | 路径上每条边 N、W 净增量为真实 backup 贡献（无 VL 残留），重复叶节点路径上 VL 被施加 2 次、撤销 2 次 |

### 6.5 MCTSSearcher - 搜索正确性

| 编号 | 场景 | Given | When | Then |
|------|------|-------|------|------|
| T5.1 | batch 结果 vs 逐 sim 结果 | 同一棋盘状态，相同随机种子，VL=0 | 分别用 batch_size=16 和 batch_size=1 执行 800 sims | visit_counts 完全一致 |
| T5.2 | 与旧 search() 结果对比 | 同一棋盘状态，相同模型权重，batch_size=16, VL=3.0 | 对比新 old impl 的 MCTS 决策 | 落子位置不变（batch+VL 不改变 MCTS 数学正确性，仅改变模拟顺序和探索多样性） |
| T5.3 | 终局判定 | 棋盘已有五连的终局状态 | 执行 search() | 返回结果中的 root_value 符合终局价值，visit_counts 全 0（无合法动作） |
| T5.4 | 平局 | 棋盘满且无胜者 | 执行 search() | root_value=0，visit_counts 全 0 |

### 6.6 CNNV7Model - predict_batch

| 编号 | 场景 | Given | When | Then |
|------|------|-------|------|------|
| T6.1 | 单条等价性 | 随机权重 CNNV7Model | `predict(v)` 和 `predict_batch([v])` 同一输入 | policy 和 value 完全一致 |
| T6.2 | batch 多输入 | 随机权重 CNNV7Model, 16 个不同 226 维向量 | `predict_batch(vectors)` | 返回 16 个 ModelPredict，每个 policy 长度 225 且 sum(policy)≈1（概率分布），每个 value ∈ [-1, 1] |
| T6.3 | 空输入 | 随机权重 CNNV7Model | `predict_batch([])` | 返回空列表（不抛异常） |
| T6.4 | batch 性能 | 16 个不同输入 | 执行 predict_batch | 耗时显著小于 16 次单独 predict（至少 2× 加速） |

### 6.7 集成测试

| 编号 | 场景 | Given | When | Then |
|------|------|-------|------|------|
| T7.1 | 完整一局自对弈 | 随机权重 CNNV7Model, batch_size=16, VL=3.0, 800 sims/步 | 执行 `play_one_game()` 完整体验 | 对局正常结束（有胜者或平局），生成 samples 列表非空，每条 sample 含 state/pi/z |
| T7.2 | batch_size=N 完整 N 局 | 随机权重, 800 sims/步 | 执行 `run_self_play()` 3 局 | 3 局全部完成，sample.json 正确写入，summary.json 的 completed_games=3 |
| T7.3 | Ctrl+C 中断 | batch_size=16, 多局运行中 | 发送 SIGINT | 已完成的局数据保存至 sample.json，不会丢失 |
| T7.4 | 周期保存 | batch_size=16, VL=3.0, _SAVE_INTERVAL=50 | 运行 60 局自对弈 | 第 50 局末尾触发周期保存，第 60 局末尾触最终保存 |

### 6.8 配置

| 编号 | 场景 | Given | When | Then |
|------|------|-------|------|------|
| T8.1 | 配置常量生效 | MCTS_BATCH_SIZE=8, MCTS_VIRTUAL_LOSS=3.0 | 构造未传参的 MCTSSearcher | batch_size=8, virtual_loss=3.0 |
| T8.2 | 参数覆盖配置 | MCTS_BATCH_SIZE=8, 构造时传 batch_size=32 | 构造 MCTSSearcher | batch_size=32（构造参数优先） |
| T8.3 | CLI 参数传递 | `--batch-size 32 --virtual-loss 2.0` | 解析 CLI 参数 | batch_size=32, virtual_loss=2.0 |

## 7. 附图索引

| 图 | 文件 |
|---|------|
| Batch MCTS 关键流程活动图 | [batch_mcts_flow.puml](batch_mcts_flow.puml) |
| 类图 | [class_diagram.puml](class_diagram.puml) |
| 依赖关系图 | [dependency_diagram.puml](dependency_diagram.puml) |
