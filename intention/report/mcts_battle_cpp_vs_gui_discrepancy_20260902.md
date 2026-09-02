# mcts_battle_cpp.py 五五开 vs GUI 强弱分明 —— 调查报告

**日期**: 2026-09-02
**现象来源**: `python src/ai/scripts/mcts_battle_cpp.py --ai1 cnn_v11.1 --ai2 cnn_v11.2 --games 50 --workers 10` → v11.1 胜 24 (48%) / v11.2 胜 26 (52%)

---

## 调研目的

命令行 C++ MCTS 双模型对战（cnn_v11.1 vs cnn_v11.2）结果接近五五开，但 GUI 实测（人机对战与 AI 内战）均显示 **cnn_v11.2 远强于 cnn_v11.1**。调查两处结果矛盾的原因，判定哪一侧可信。

---

## 结论（先行）

**命令行五五开是 C++ 引擎的搜索树复用缺陷所致，结果不可信；GUI 结果（v11.2 远强）才是真实水平。**

CLI 与 GUI 根本不在同一套引擎上对弈：

| | CLI（mcts_battle_cpp.py） | GUI（人机 / AI 内战） |
|---|---|---|
| 实际引擎 | C++ `GameRunner` + `MctsSearcher` | **回退到 Python MCTS**（MCTSSearcher，800 sims） |
| 搜索树状态校验 | ❌ 无（`root_ != nullptr` 即复用） | ✅ `root_hash` 校验，不符即整树重建 |
| 落子合法性 | ❌ `sample_action` 不检查空位，可覆盖棋子 | ✅ 只选真实空位 |

GUI 走 Python 的原因：`create_engine` → `_try_create_cpp_engine`（builder.py）尝试创建 `CppMctsAdapter`，其 `__init__` 内 `import cpp_mcts` 失败——GUI 进程 `sys.path` 只有项目根，不含 `cpp/python/`（`cpp_mcts.pyd` 所在目录）——被 `except ImportError` 捕获后回退 Python MCTS。

---

## 证据链

### 1. 权重与模型本身（排除错位与"假五五开"）

- gen1/gen2 的 `.pth ↔ .pt` 权重完全一致（0 差异），两代权重显著不同（无文件错位）。
- **cnn_v11.1（gen1）是随机权重模型**：policy 近均匀分布（熵 5.42 ≈ ln225）、value≈0；gen1.pth 与 `temp/init_cnn_v11_gen1_random_20260831.py` 同为 8/31 21:17 生成。cnn_v11.2（gen2，9/1 训练晋升）policy 尖锐、棋力正常。
- 因此正确搜索下 v11.1 毫无棋力（GUI 中每局 9~10 手被 v11.2 速胜），CLI 中 v11.1 却能赢 48% 只可能是 v11.2 被引擎缺陷拖垮。

### 2. C++ 根因（两处缺陷叠加）

**缺陷 A：`MctsSearcher::init_search` 子树复用不校验棋盘状态**（`cpp/src/mcts_searcher.cpp` 141-160 行）

```cpp
if (root_ != nullptr) {          // 只要树存在就直接复用
    ...add_dirichlet_noise(root_, legal_actions);
    return;                       // ← 不校验树根局面与传入 state 是否一致
}
```

Python 对照（`src/ai/mcts/searcher.py` 266-268 行）：`root_hash not in self.tree → self.tree.clear()` 整树重建。

**缺陷 B：`GameRunner::sample_action` 不检查落子位置是否为空**（`cpp/src/game_runner.cpp` 345-358 行）

```cpp
if (visit_counts[i] > 0) { ... }   // 只按访问数选，不查 board[i] == 0
```

Python 对照（`policy_value.py` `_select_greedy_n`）：过滤 `board[a] == 0`。

**触发场景**：`GameRunner::run_single_game` 每步只对"当前落子方"的 searcher 调 `advance_root`（game_runner.cpp 275-279 行），**对手落子从不推进己方树**。轮到己方时：
- 己方树根 = 自己上一手之后、对手落子**之前**的局面（且视角是"对手该走"）；
- 传入 `state` = 真实棋盘（含对手刚落子）+ 己方视角；
- 两者不一致，但缺陷 A 不校验 → 直接在滞后一手、视角错位的旧树拓扑上搜索；
- 旧树的高访问边含"对手刚落子位置"（旧局面中它是空的）→ 缺陷 B 使落子**覆盖对方棋子** → 棋局非法化、胜负失真；
- 两模型棋力全部崩塌 → 强弱被抹平为五五开。

**推理实证**：对同一 C++ `MctsSearcher`（gen2.pt）连续 `search` 不同局面（不推进树），非空棋盘输出与空棋盘完全相同（恒推荐已占的天元 0.9531）——旧树先验与拓扑完全锁定，状态未被感知。

### 3. 参数因素排除

CLI `--temp-steps 0`（与评估器一致的全程贪婪）× 20 局 → 仍 50/50。温度/采样参数不是原因。

### 4. 决定性对照实验（`temp/verify_dual_searcher_usage_20260902.py`）

用**完全相同的对局驱动逻辑**（交替落子、只推进己方树、400 sims）分别驱动两套引擎：

| 实验组 | 结果 | 说明 |
|---|---|---|
| C++ MctsSearcher × 2（跨局复用，落子过滤空位）12 局 | 6:6 | 缺陷 A 单独已致五五开 |
| C++ × 2（每局新建 + **不过滤空位** = GameRunner 精确复刻）12 局 | 7:5 | 完整复现 CLI 现象 |
| Python MCTSSearcher × 2（GUI 同款）12 局 | 0:12（v11.2 全胜） | 复现 GUI 现象（9~10 手速胜） |

两次 Python 组均 v11.2 12:0，C++ 组均约五五开——同一逻辑、唯一差异是引擎实现，结论确定。

---

## 影响面

1. `mcts_battle_cpp.py` 及 C++ `GameRunner` 双模型对战结果 **不可信**，此前基于它的任何强弱判断需复核。
2. **训练评估器 `checkpoint_evaluator.py` 同样走 `GameRunner`**：cnn_v11 训练中"checkpoint vs 上代"晋升评估的正确性存疑（随机权重的 gen1 在正确搜索下不可能与训练模型五五开，但 CLI 模式下会）。gen1 → gen2 的晋升是否基于失真评估，建议核查。
3. GUI 走 Python MCTS 回退（纯属 sys.path 巧合），当前 GUI 判断可信；若未来 GUI 能导入 `cpp_mcts`，将同样中招。

---

## 修复建议（尚未实施，待授权）

1. **核心**：`MctsSearcher::init_search` 复用前校验树根局面（board+player）与传入 `state` 一致，不一致则销毁整树重建（对齐 Python `tree.clear()` 语义）。GameRunner 交替用法下校验必然失败 → 每步重建 → 等价关闭子树复用（正确性优先，性能损失可接受）。
2. **防御**：`GameRunner::sample_action` 增加真实棋盘空位过滤；主循环落子前校验 `board[action] == 0`。
3. **验证**：修复后重跑 50 局 CLI，预期 v11.2 大胜；并补 C++/Python 搜索结果一致性用例防回归。

---

## 相关文件

- 验证脚本（`temp/`）：
  - `verify_gen_weights_20260902.py`（权重一致性：.pt ↔ .pth）
  - `compare_gen_policy_value_20260902.py`（gen1 随机权重判定：H≈5.42）
  - `verify_cpp_inference_20260902.py`（C++ 树复用不感知新局面的推理实证）
  - `verify_dual_searcher_usage_20260902.py`（C++ vs Python 同驱动逻辑对照实验）
  - `diag_py_game_20260902.py`（Python 组速胜棋谱诊断）
- 关键代码：`cpp/src/mcts_searcher.cpp`、`cpp/src/game_runner.cpp`、`cpp/src/board_state.cpp`、`src/ai/mcts/searcher.py`、`src/ai/engine/policy_value.py`、`src/ai/engine/builder.py`
