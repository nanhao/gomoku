# 测试用例集

用例编号前缀：`TC-SR`（MctsSearcher 状态一致性）、`TC-GR`（GameRunner 落子合法性）、`TC-E2E`（端到端验证）。

C++ 测试沿用现有自研 TEST 宏与 SKIP 机制（需 `NN_MODEL_PATH` 环境变量或默认模型路径，模型缺失时 SKIP）。建议**新建** `cpp/tests/unit/test_mcts_searcher_state_sync.cpp`（避免 test_mcts_searcher.cpp 超过行数上限）与 `cpp/tests/integration/test_game_runner_legal_moves.cpp`，并按现有 CMake 注册方式加入构建。

**通用断言约定（P3-1/P3-5，所有涉及 search 的用例统一遵守）：**

1. **每次 search 后 `visit_counts` 总和 == 本次模拟数**（能捕获 P0-1 叶根空转：空转时总和为 0）。旧用例 TC-SR-01 只断言"∈ [1, sims]"，修订后一律用强断言"== sims"。
2. 搜索返回的 `root_policy` 须满足前置断言：225 维均为有限值且总和 > 0（非全 0）。
3. 结果合法性：`visit_counts` 中棋盘已占位处必须为 0；argmax 必须是真实棋盘空位。

---

## 一、MctsSearcher 状态一致性（C++ 单元测试，需模型，sims=100）

### TC-SR-01 空棋盘冒烟（无回归）

- **步骤**：构造 `MctsSearcher(path, 100, 2.0, 0.25, 0.03, 4, 3.0, false)` → `search(空棋盘+黑)`。
- **断言**：不抛异常；`visit_counts` 总和 == 100；`root_policy` 非全 0；`root_value` 有限。

### TC-SR-02 交替推进序列——已展开分支复用与未展开分支重建双路径（核心用例，修复前必失败）

- **步骤**：模拟 GameRunner 中"黑方 searcher"的遭遇（每步状态比树根恰差对手 1 子），执行 5 轮：
  1. `search(s = 当前棋盘 + 黑)` → 记录 `rc0 = get_rebuild_count()` → argmax `a` 落黑于真实棋盘 → `advance_root(a)`；
  2. 白方落 `X` 于真实棋盘（每轮 X 取不同空位，在"热区"与"角落冷区"之间切换，以覆盖两条内部路径）；
  3. 再次 `search(s = 真实棋盘 + 黑)`，记录 `rc1 = get_rebuild_count()`。
- **断言**（每轮 search 后）：
  - `visit_counts` 总和 == 100（**若 promote 了未展开叶节点，此断言失败**——空转时总和为 0）；
  - 棋盘已占位（含对手刚落 X）处访问数 == 0；argmax 为真实空位；
  - `root_policy` 非全 0；`root_value` 有限。
- **路径覆盖断言**：5 轮中必须**同时出现** `rc1 == rc0`（走了一致/已展开 promote 复用）与 `rc1 == rc0 + 1`（走了重建）两种结果——若 `get_rebuild_count` 只增不减说明 promote 路径从未命中（P0-1 修复缺失的表现之一），测试失败。
- **修复前行为**：旧树复用不校验 → 占位处 N>0 且 argmax 命中已占位 → 断言失败。

### TC-SR-03 局面跳变整树重建

- **步骤**：`search(空棋盘,黑)` → 直接 `search(预置 8 子中盘局面, 白)`（与树根无继承关系）→ 再 `search(空棋盘,黑)`。
- **断言**：两次跳变均不抛异常；每次 `visit_counts` 总和 == 100；argmax 为空位；已占位处 N==0；`get_rebuild_count()` 在每次跳变后 +1。

### TC-SR-04 advance_root 失败自愈

- **步骤**：`search(s0)` → `advance_root(-1)` 返回 false → `advance_root(225)` 返回 false → 再 `search(s0)`（同状态）。
- **断言**：`advance_root` 失败后树状态未被破坏；同状态再 search 正常返回（走"一致复用"分支），`visit_counts` 总和 == 100；`get_rebuild_count()` 不增加。

### TC-SR-05 root_policy 有效性（推进复用后）

- **步骤**：TC-SR-02 中某轮 `rc1 == rc0`（复用）后的 search 返回值。
- **断言**：`root_policy` 225 维均为有限值、总和 > 0 且 **top-1 概率 > 0**（修复点 3 后应为新根自身局面的网络策略；与 Python predict 数值一致性由 TC-E2E-03 覆盖）。

---

## 二、GameRunner 落子合法性（C++ 集成测试，需模型）

### TC-GR-01 自对弈样本棋盘子数严格递增（核心用例，修复前必失败）

- **步骤**：单模型自对弈 `GameRunner(path, 1, config{sims=200, temp_steps=2, use_gpu=false}, 1, collect_samples=true)` → `start()` → `wait_completion()` → `get_results()[0]`。
- **断言**：
  - `winner ∈ {0, 1, 2}`，`move_count ≤ 225`；
  - `winner != 0` 时 `samples` 非空（平局时无样本生成，不判非空）；
  - **按 D4 增强分组断言（P2-4 修订）**：每条原始样本经 `SampleAugmentor::augment` 产生 8 条变体（恒等 + 7 变换，保棋子数），`samples` 长度为 8×步数。断言：`samples[8k .. 8k+7]` 每组 8 条的 `state` 前 225 维**棋子总数均 == k**（第 k 条原始样本是第 k 步落子前的局面，应含 k 子；覆盖落子会使某步棋子数不增 → 断言失败）。
- **修复前行为**：sample_action 覆盖对手子 → 某步后棋子数停滞 → 失败。

### TC-GR-02 双模型对战回归冒烟（含 rebuild 诊断输出观测）

- **步骤**：`GameRunner(path1, path2, 1, config{sims=100, temp_steps=0, use_gpu=false}, 1)`（可 path1==path2 同模型）→ 运行完成。
- **断言**：不抛异常；有结果；`winner ∈ {0,1,2}`；stdout 含 `[GameRunner] game=0 rebuild:` 诊断行。
- **附加观测（P2-3，仅同强度对局适用）**：同模型对弈时双方 `rebuild` 次数应显著小于步数；此观测不作硬性断言（避免强弱悬殊场景误报），仅打印供人工核查。

---

## 三、端到端验证（Python，temp/ 目录脚本 + CLI）

### TC-E2E-01 CLI 对战强弱反转（主验收）

- **命令**：`python src/ai/scripts/mcts_battle_cpp.py --ai1 cnn_v11.1 --ai2 cnn_v11.2 --games 50 --workers 10`
- **验收阈值**：cnn_v11.2 胜率 **≥ 70%**（修复前 52%）。v11.1 为随机权重模型，正确引擎下预期接近全胜；不足 70% 判定修复未生效或有回归。平局/弃局计入分母后仍须满足阈值。
- **附加观测（P2-3 修订，不作硬性断言）**：验收脚本按执方分别输出每局 rebuild 计数——**强方（v11.2）侧预期显著小于步数**；**弱方（v11.1 随机权重）侧允许接近步数**（其树浅层 N>0 边占比极低、强方着法命中未展开叶 → 几乎每步走重建分支，属正常）。不得用弱方侧数值判定回归。

### TC-E2E-02 C++/Python 对照实验复跑

- **步骤**：升级 `temp/verify_dual_searcher_usage_20260902.py`：C++ 组改为"每局新建 searcher + 落子前不做外部过滤"（直接复刻 GameRunner 语义），12 局。
- **断言**：C++ 组结果方向与 Python 组一致（v11.2 大胜，非五五开）；修复前该组为 7:5。每步搜索结果同时满足通用断言（总和 == sims、落子为空位）。

### TC-E2E-03 C++ root_policy 与 Python predict 一致性

- **步骤**：temp 脚本中，C++ searcher 推进若干步后 search 返回的 `root_policy`，与 Python `model.predict(同局面).policy` 比较。
- **前置断言（P3-1）**：C++ `root_policy` 非全 0（总和 > 0）——叶根空转场景会以全 0/回退策略通过"有限值"断言，必须显式排除。
- **断言**：top-1 动作一致且 top-5 重合 ≥ 3/5（同权重、同局面，允许 MCTS 噪声/浮点微小差异）。

### TC-E2E-04 回归：既有功能不受影响

- **步骤**：修复后运行 `build_cpp.bat`（或等效构建）→ CTest 全量通过（既有 TC-MS/TC-GR/TC-NN 等全部用例）→ `mcts_sample_buffer.py --games 2 --workers 2` 冒烟自对弈样本采集正常。
- **断言**：无既有用例失败；样本缓冲区脚本不崩溃且产物结构合法（样本 state 棋子数逐条递增）。
