# 接口设计

本修复不改动任何**既有公开 API 的签名**（含 pybind11 暴露的 `search`/`advance_root`、GameRunner 构造与运行接口），仅改变 `MctsSearcher` 内部行为、扩展 GameRunner 一个私有方法的参数，并做三处**非破坏性新增**（`MctsNode::expand_with_policy`、`MctsNode::flip_value`、`MctsSearcher::get_rebuild_count`）。Python 侧零改动。

## 1. `MctsSearcher`（cpp/src/mcts_searcher.h）

**既有公开接口（签名不变，行为升级）：**

| 成员 | 签名 | 行为变化 |
|------|------|----------|
| `search(state)` | `MctsResult search(const std::vector<int>& state)` | 增加树根状态一致性校验：一致→复用；差 1 步且子边**已展开（非叶）**→内部推进后复用；其余（差多子 / 目标为未展开叶 / 无此分支 / 超展开上限）→整树销毁重建（对齐 Python root_hash 语义）。返回的 `visit_counts`/`root_value`/`root_policy` 均基于与 state 一致的树根 |
| `advance_root(action)` | `bool advance_root(int action)` | 成功（子节点存在且已展开）时同步更新内部树根状态记录并翻转新根价值视角；失败（返回 false）时状态不变，后续 `search` 自动走不一致路径自愈。内部实现收敛到私有 `advance_root_locked` |
| 构造器 ×2 | 不变 | 无变化（新状态默认"无树"） |

**新增公开只读接口（非破坏性，pybind 不暴露）：**

| 成员 | 签名 | 说明 |
|------|------|------|
| `get_rebuild_count` | `int get_rebuild_count() const` | 返回整树重建累计次数（P2-2：GameRunner 每局结束读取并打印需要公开读取通道，私有成员 + 外部打印不可行） |

**新增私有状态（对外不可见）：**

| 成员 | 类型 | 说明 |
|------|------|------|
| `root_board_` | `std::vector<int>` | 树根局面棋盘 225 维，与 `root_` 同生命周期 |
| `root_player_` | `int` | 树根局面当前落子方；无树时置 -1 作哨兵 |
| `rebuild_count_` | `std::atomic<int>` | 整树重建累计次数（初始 0） |
| `total_expands_` | `std::atomic<int>` | 树内累计展开节点数（每次 `expand_with_policy` 真正展开 +1；重建/新建清零），用于修复点 5 限幅 |
| `max_tree_expands_` | `int` | 展开上限（构造时从 `g_mcts_config.max_tree_expands` 读取，0=不限） |

**行为规格要点：**

1. 内部"差 1 步推进"（promote）与 `advance_root` 共用私有 `advance_root_locked(action)`（持锁上下文：get_child → detach → delete 旧根 → 换根 → 更新根状态 → `flip_value()`），公开 `advance_root` 与 `init_search` 内部均调用之，避免两份 detach/delete/换根逻辑漂移。
2. **promote 目标必须是已展开节点**（`!child->is_leaf()`，P0-1）：未展开叶节点（expand 创建了边但从未被选中、N=0、无子边）被提升为根后 `select_child` 返回 -1 → 模拟全部空转 → visit_counts 全 0 → 弃局。
3. 复用路径中 `root_policy_` 从新根节点重新读取（`root_->get_root_policy()`），保证与树根局面一致。
4. 重建路径 = 删除旧树后走全新建树逻辑（网络预测根策略 + 展开 + 噪声），并清零 `total_expands_`、`rebuild_count_++`。
5. 超限路径：`total_expands_ > max_tree_expands_` 时任何复用（①/②）改为整树重建。

## 2. `MctsNode`（cpp/src/mcts_node.h）

**既有公开接口：签名不变**（`set_root_policy`/`get_root_policy` 语义不变；`get_root_policy` 的"空则从子边 prior 回退"逻辑保留，但修复后各被展开节点均已持有自身策略，回退仅作兜底）。

**新增公开方法（非破坏性）：**

| 成员 | 签名 | 说明 |
|------|------|------|
| `expand_with_policy` | `bool expand_with_policy(const std::vector<float>& action_priors, const std::vector<int>& legal_actions, const std::vector<float>& full_policy)` | 在 `expand_lock_` 临界区内原子完成"展开 + 存储自身局面原始策略（噪声前 225 维）"。幂等：已展开返回 false 且不写策略；返回是否由本次调用真正展开。P1-1：禁止在 expand 返回后锁外 `set_root_policy`（并发 worker 双写 vector 属数据竞争 UB） |
| `flip_value` | `void flip_value()` | `q_sa_ = -q_sa_`（n_visited_ 不变）。P1-2：节点被提升为新根时调用，将历史 q 从父视角（边价值）翻转为自身视角（根节点价值口径）。调用时机无并发（搜索模拟结束后） |

**调用时机变化（mcts_searcher.cpp 内部）：**

- `run_one_simulation` 阶段 3（Expansion）改调 `expand_with_policy`：真正展开（返回 true）时 `total_expands_++` 并 `backup`。
- `advance_root_locked` 提升新根后调用 `flip_value()`。
- 既有 `expand` 保留（`init_search` 建根 + 单测使用，无竞争场景），不改签名。

## 3. `GameRunner`（cpp/src/game_runner.h）

**公开接口：无变化**（构造器、`start`/`wait_completion`/`get_results`、`resolve_colors` 均不变）。

**私有静态方法签名变化（头文件同步修改）：**

| 成员 | 原签名 | 新签名 |
|------|--------|--------|
| `sample_action` | `static int sample_action(const std::vector<int>& visit_counts, int step, const GameConfig& config)` | `static int sample_action(const std::vector<int>& visit_counts, const std::vector<int>& board, int step, const GameConfig& config)` |

**`sample_action` 行为规格：**

1. 首选候选：`visit_counts[i] > 0 && board[i] == 0`；
2. 兜底（P2-1，对齐 Python `_select_greedy_n`）：首选候选为空（搜索空转、全部空位 N=0）→ 候选退化为全部空位（含 N=0，按 N 取最大，全 0 时取首个空位）；
3. 仅当棋盘无空位（满盘）返回 -1。

**`run_single_game` 内部行为变化：**

- 调用处传入真实棋盘 `board`；
- 落子前防御：`action < 0`（仅满盘）→ 终止本局；`board[action] != 0`（理论不可达）→ 打印警告并从空位取 visit_counts 最大者兜底；
- `advance_root(action)` 调用保留（己方落子后立即推进，最大化后续复用）；失败时不影响对局，下次 search 自愈；
- 每局结束读取双方 searcher 的 `get_rebuild_count()` 打印一行诊断：`[GameRunner] game={i} rebuild: 黑={n1} 白={n2}（步数 {steps}）`。

## 4. pybind11 绑定（cpp/src/bindings.cpp）与 Python 侧

**暴露接口零改动。** `cpp_mcts.MctsSearcher`、`cpp_mcts.GameRunner` 等类与方法签名完全不变；`CppMctsAdapter`、`cpp_mcts_battle_adapter.py`、GUI builder 回退逻辑均无需修改——修复通过引擎行为自动生效。

**配置加载内部变化（不改绑定签名）：** `load_mcts_config` 新增一行 `g_mcts_config.max_tree_expands = get_int("max_tree_expands", 20000);`；`MctsConfig`（mcts_config.h）新增字段 `int max_tree_expands = 20000;`；`cpp/python/mcts_config.json` 增加同名键（缺省时 fallback 默认值）。
