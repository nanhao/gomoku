# design/59/main_docs 设计方案 Review 报告

**日期**: 2026-09-02
**Review 对象**: `design/59/main_docs/`（overview.md / interface.md / main_flow.puml / test_cases.md）
**问题依据**: `intention/report/mcts_battle_cpp_vs_gui_discrepancy_20260902.md`（intention_id=59）
**Review 方式**: 对照 `cpp/src/`（mcts_searcher.h/.cpp、mcts_node.h/.cpp、game_runner.h/.cpp、game_sample.h、board_state.h、bindings.cpp）与 Python 对照实现（searcher.py、policy_value.py、cpp_mcts_adapter.py）逐条核验设计规格
**结论**: **方案方向正确、根因定位与三分支修复思路成立，但按当前规格实现会导致验收用例（TC-E2E-01）系统性失败（P0），另含 1 个并发数据竞争（P1）与若干规格/测试缺口，需修订后实施**

---

## 一、总体评价（方案优点，均已对照源码验证）

1. **根因定位准确**。缺陷 A 实锤：[mcts_searcher.cpp](file:///d:/projects/gomoku/cpp/src/mcts_searcher.cpp) L148-160 `init_search` 仅凭 `root_ != nullptr` 复用，不校验局面；缺陷 B 实锤：[game_runner.cpp](file:///d:/projects/gomoku/cpp/src/game_runner.cpp) L345-364 `sample_action` 仅按 `visit_counts[i] > 0` 选动作。触发场景（`run_single_game` 每步只推进当前落子方自己的树，L276）描述与代码一致。
2. **Python 语义对照属实**：[searcher.py](file:///d:/projects/gomoku/src/ai/mcts/searcher.py) L266-268 root_hash 不在树即 `tree.clear()`；[policy_value.py](file:///d:/projects/gomoku/src/ai/engine/policy_value.py) L193-211 `_select_greedy_n` 过滤 `board[action]==0`。C++ 端所有经 GameRunner 的路径（CLI、checkpoint_evaluator、mcts_sample_buffer 自对弈）确实同受影响。
3. **"差 1 步内部推进"（promote）语义推导正确**。核验了交替对局时序：己方 `advance_root` 后树根局面 = "轮到对手"，下次轮到己方时传入 state 恰为"树根局面 + 对手刚落的 1 子 + 视角翻转"，`one_step_diff` 的判定（player 翻转 + 单点差异且为 prev_player 所落）与此精确匹配；promote 目标即"对手落子对应的己方树子边"，复用成立。比 Python"每步整树重建"更高效且语义等价（Python 因 root_hash 机制推进后的树从不被复用，等同每步重建，这是其正确而慢的原因）。
4. **修复点 3（展开节点保存自身原始策略）必要且正确**。核验 [mcts_node.cpp](file:///d:/projects/gomoku/cpp/src/mcts_node.cpp) L119-130：`get_root_policy()` 在 `root_policy_` 为空时回退到子节点 prior——该回退值是**噪声混合后**的 prior，且只覆盖已创建子边，语义不符"原始策略"；且 [mcts_searcher.cpp](file:///d:/projects/gomoku/cpp/src/mcts_searcher.cpp) L179 只在全新建树时写成员 `root_policy_`，推进/复用后陈旧。修复点 3 + `reuse_tree` 中重读新根策略解决了这两处。Python 侧 `_root_policy` 亦在 `_init_search` 与 `_expand_leaf` 两处存储（searcher.py L321 注释），设计与之对齐。
5. **接口兼容策略正确**。`search`/`advance_root` 经 pybind 暴露（bindings.cpp L168-177），签名不变则 Python 侧（CppMctsAdapter L109-120 透传）零改动；`sample_action` 为私有静态（game_runner.h L170），加参不破坏绑定。
6. **可观测性（rebuild_count_）方向正确**，能区分 promote/重建分支的实际占比。

---

## 二、P0 —— 阻断性问题（按现规格实现将导致验收失败）

### P0-1：promote 允许提升"未展开（N=0）叶节点" → 整次搜索空转 → 弃局

**位置**：overview.md「修复点 1」伪码 `if x >= 0 and root_->get_child(x) != nullptr: promote(x)`

**问题链**（对照源码逐环验证）：

1. [mcts_node.cpp](file:///d:/projects/gomoku/cpp/src/mcts_node.cpp) L59-64：`expand` **跳过 prior < FLT_EPSILON 的动作**——子边集合是"创建边"而非"全部空位"；L49-67 展开只发生在节点作为 selection 的 child 被访问到时。
2. 但即使边被创建，**未被任何模拟选中的边 N=0 且保持叶态**（N>0 ⟺ 被 backup 传播 ⟺ 已被展开，仅此方向的蕴含成立；N=0 的边存在且大量存在）。
3. `promote` 把这样的叶节点提升为根后，`reuse_tree` 中 `add_dirichlet_noise` 无子边可施（no-op）；随后 800 次 `run_one_simulation` 中 `select_child` 对空 children 返回 -1（mcts_node.cpp L22、mcts_searcher.cpp L284-288），**root 自身永远不在"到达叶则展开"分支内**（展开逻辑只作用于 selection 到的 child）→ 800 次模拟全部空转 → `visit_counts` 全 0。
4. `sample_action` 空候选返回 -1（修复点 2 防御）→ `run_single_game` break → **本局以 winner=0 提前弃局**。
5. **触发概率（关键）**：CLI 验证场景正是 v11.1（随机权重，policy 熵≈5.42≈ln225，先验均匀）vs v11.2。弱方树：深层节点（如黑第 1 手后的对手应手局面）被访问次数 ≈ 800/225 ≈ 3.5 次，其展开创建的 ~224 条子边中 N>0 的仅 2~3 条；强方真实落子命中 N=0 边的概率 ≈ **98%+**（边已创建 → `get_child != nullptr` 成立 → 走 promote 而非重建）。即弱方第 2 步起几乎必然触发叶根空转弃局。强方侧命中率也有几十个百分点（强先验尖锐，expand 只创建长尾以上边，弱方随机落子命中未访问边的概率高）。
6. **后果**：修复后 mcts_battle_cpp.py 的 v11.1 vs v11.2 对局将大量以平局/弃局结束（v11.2 执黑、执白局均可能），v11.2 胜率被 0:0 稀释 → **TC-E2E-01（v11.2 ≥ 70%）无法通过**，且弃局同样使"CLI 结果不可信"，修复失去意义。

**建议修复**：promote 条件收紧为"子节点存在**且已展开**"（`get_child(x) != nullptr && !child->is_leaf()`，N>0 亦等价），否则落入整树重建分支。重建的正确性已由"树根与 state 一致"保证（同 Python 每步全清语义），只是损失一次根 predict + 树复用，性能影响可接受。另见 P2-1 建议 sample_action 兜底对齐 Python，双保险。

**测试缺口**：现有 TC-SR-02 无法捕获此缺陷——叶根空转时 `visit_counts` 全 0，恰好满足"占位处访问数为 0""argmax 是空位"断言。需补充断言"每次 search 后 visit_counts 总和 == 本次模拟数（或 ≥1）"，并将"对手子落在已展开分支"与"落在未展开分支"两类路径显式用例化（TC-SR-02 注释虽声称覆盖两种内部路径，但按其步骤"对手子每次取不同空位"，并不能保证触发 promote 叶根路径）。

---

## 三、P1 —— 高优先级问题

### P1-1：修复点 3 存在并发数据竞争（set_root_policy 无锁双写）

**位置**：overview.md「修复点 3」"`child->expand(...)` 后追加 `child->set_root_policy(nn_result.policy)`"

**问题**：`run_one_simulation` 由线程池 worker 并发执行。两个 worker 可能同时判断同一 child 为叶（`is_leaf()` 检查在 [mcts_searcher.cpp](file:///d:/projects/gomoku/cpp/src/mcts_searcher.cpp) L292 无锁进行），双双进入 expand——`expand` 内部有 `expand_lock_` 幂等保护（mcts_node.cpp L49-53），但 **`set_root_policy` 在 expand 返回之后、锁外执行**，第二个线程会再次写入同一 child 的 `root_policy_` → `std::vector` 并发 `operator=` 数据竞争（UB）。虽然同模型两次 predict 数值近似相同使实际损害隐蔽，但属未定义行为，且后写者覆盖先写者，与"节点持有自身局面策略"的不变性在语义上依赖时序。

**建议修复**：将存储策略并入 expand 的锁内原子完成——例如在 `MctsNode` 增加 `expand_with_policy(priors, legal, policy)`（锁内 set），或让 `set_root_policy` 返回是否由本线程执行展开来决定写权限（仅在 `expand` 返回"本次真正展开"时 set）。注意这会使 MctsNode 公开接口增加一个方法，interface.md"接口不变"表述需同步调整（新增方法非破坏性，但与文档声明有出入，见 P2-2）。

### P1-2：promote/advance_root 后新根 root_value 视角符号错位（既有缺陷被修复放大为可观测）

**位置**：mcts_node.cpp L69-97 backup 的根特判语义 + promote/advance_root 提升节点

**问题**：核验 backup 数值链可得：**非根节点的 q_sa_ 以"父视角"存储（边价值），根节点的 q_sa_ 以"自身视角"存储（根特判取反）**。推进（promote/advance_root）把非根节点提升为根后，其历史 q 累积仍是旧根视角（= 新根对手视角，符号差一），而后续搜索对新根的 backup 按"根特判"以新根视角累积 → **两种口径混合进同一个运行均值**，`get_result().root_value` 数值失真/向 0 漂移。visit_counts（落子与 pi 的唯一依据）不受影响，故不影响对战胜负、样本与晋升评估，但 root_value 是对外输出（GUI 胜率展示、日志、CppMctsAdapter 若启用即暴露），且修复使 promote 首次成为高频正常路径，该偏差从"不可观测"变为"常态输出错误"。

**建议**：二选一——
1. 节点增存 W 累积量，提升为根时对 W 取反（q 随之翻转，口径一致）；或
2. 在设计中显式声明"推进后 root_value 口径偏差已知且仅影响展示"，评估后接受（若 root_value 无训练/决策用途，接受成本最低，但需写入文档避免后人踩坑）。
同时建议与 Python 对齐确认：Python 由于推进后的树从不被复用（root_hash 不符即 clear），此问题天然不存在，故"对齐 Python root_value 语义"需在推进场景另行定义。

### P1-3：修复点 3 内存开销的跨步累积被低估（"800 sims 树 ≈ 720KB"口径错误）

**位置**：overview.md 决策表 root_policy 语义行

**问题**：720KB 只是"单次 search 内展开节点数 ≤ sims"的瞬时上界（800 节点 × ~900B）。但 GameRunner 场景下树沿真实走法**跨步保留**（推进仅删兄弟子树，[mcts_searcher.cpp](file:///d:/projects/gomoku/cpp/src/mcts_searcher.cpp) L115-135 advance_root 只删旧根），每步 search 的 800 次模拟继续在新根下展开新叶 → 树节点跨步单调累积，225 步对局内树规模可能达数万~十余万节点（量级 10~100MB/树）。GameRunner 多 worker（checkpoint 评估固定 10 线程、自对弈 workers 可配）各持 2 棵存活树 → 总内存可能达 GB 级边缘，且修复点 3 使每展开节点固定 +900B（vector<float> 225）。

**建议**：设计补充——(a) 用 rebuild/promote 计数实测每局平均展开节点数与峰值树节点数（P0-1 修复后 promote 成功率可用 rebuild_count 推算）；(b) 若实测超标，可加"树节点数上限，超限整树重建"的限幅策略（正确性无损，仅损复用）。文档中 720KB 的表述应修正为"单次搜索瞬时上界"，避免误导。

---

## 四、P2 —— 中优先级问题

### P2-1：sample_action 空候选返回 -1 弃局，与 Python 兜底语义不一致

**位置**：修复点 2 规格"候选集为空时保持返回 -1（主循环 break 弃局）"

**问题**：Python `_select_greedy_n`（policy_value.py L205-211）候选 = **全部空位**（初始 best_n=-1，N=0 的空位同样入选），全空位 N=0 时兜底返回 0——Python 在"搜索空转"时仍能继续对局；C++ 修复点 2 候选 = N>0 空位，全 0 即弃局。两者防御语义不一致：C++ 把"引擎异常"直接升级为"放弃整局"。虽然 P0-1 修复后空候选理论上不可达，但作为防御纵深，建议与 Python 对齐：候选为空时退化为"空位中 N 最大者（含 0），再兜底固定位置（如中心）"，避免任何单点异常直接废掉整局样本/评估。

### P2-2：rebuild_count_ 可观测性与"不改变任何接口"声明自相矛盾

**位置**：interface.md「5. 可观测性」vs 全文"公开接口不变"

**问题**：`rebuild_count_` 设计为 MctsSearcher 私有成员（overview.md L169），而"GameRunner 每局结束打印"需要 GameRunner 读取 searcher 内部计数 → 必须新增公开 getter（或友元声明），与 interface.md"不改变任何接口""仅影响诊断输出"冲突；反过来若在 searcher 内部自打印，GameRunner 每局多次 search（每步 1 次）将刷屏，无法"每局汇总"。需明确：新增 `int get_rebuild_count() const`（非破坏性新增，pybind 不必暴露），并同步修改"接口零变化"的表述。

### P2-3：TC-E2E-01 附加观测"rebuild 计数应远小于步数"与验证场景自相矛盾

**位置**：test_cases.md TC-E2E-01

**问题**：该局是 v11.1（随机权重）vs v11.2——弱方先验均匀，其树每层几乎全部边 N=0（见 P0-1 分析），P0-1 修复后弱方每次 search 几乎必然走"整树重建"分支 → **rebuild ≈ 步数**，"远小于步数"的观测在弱方侧必然不成立，会被误判为回归。建议：该附加观测仅用于同模型自对弈/势均力敌场景（TC-GR-01 可顺带打印），或从 TC-E2E-01 验收标准中移除；同时在验收脚本中按"执方"分别输出 rebuild 计数（强方侧应远小于步数，弱方侧允许接近步数）。

### P2-4：TC-GR-01 断言未考虑 D4 增强（每条原始样本 8 变体）

**位置**：test_cases.md TC-GR-01

**问题**：[game_runner.cpp](file:///d:/projects/gomoku/cpp/src/game_runner.cpp) L314-320 每条原始样本经 `SampleAugmentor::augment`（game_sample.h L135-152，恒等 + 7 变换共 8 条）后 push，`GameResult.samples` 长度为 8×步数。文档断言"第 k 条样本含 k 子"按线性下标不成立（第 8 条是第 0 步的镜像变体，0 子 ≠ 8）。断言应改为：`samples[8k..8k+7]` 每组棋子数 == k（D4 变换保棋子数），或仅取 tid=0 变体（每组第 1 条）线性校验。另注意：samples 仅在 `winner != 0` 时生成（L286），"samples 非空"断言在罕见平局时会误报，建议断言改为"非平局时非空"。

---

## 五、P3 —— 低优先级/文档问题

1. **TC-SR-05 与 TC-E2E-03 缺"root_policy 非全 0/总和 > 0"前置断言**：P0-1 场景下叶根回退构建的全 0 policy 会通过"均为有限值"断言；E2E-03 与 Python predict 的 top-k 重合断言有概率抓出该问题但依赖随机性（flaky），建议显式断言。
2. **数据补偿范围表述可澄清**：overview「影响与后续」称"sample_buffer / renjunet_samples 混合产物可能含覆盖落子"——renjunet_samples 来自 RIF 棋谱转换，不经过 GameRunner，**不受本缺陷影响**；需补偿的仅是缺陷期间 C++ 自对弈采集进 sample_buffer 的样本（训练集剔除/重采，renjunet_samples 无需重转）。建议文字上区分两类来源。
3. **main_flow.puml 未覆盖防御兜底与 advance_root 失败自愈路径**：流程图为代表性场景（可接受），但"防御分支落子""advance_root 返回 false"两条自愈路径是修复点 2 的行为规格（test_cases 亦含 TC-SR-04），建议在图中补一条注释性分支或接受现状（二选一，至少保证 puml 与最终实现一致）。
4. **promote 与 advance_root 实现重复**：overview 伪码 promote"语义同 advance_root(x)"，建议实现时提取 `advance_root_locked(action)` 私有方法共用（持锁上下文），避免两份 detach/delete/换根逻辑漂移。
5. **TC-SR-01 冒烟断言"visit_counts 总和 ∈ [1,100]"已在文档中正确包含，但其它用例未沿用**（与此前 P0-1 测试缺口同源，实现时统一补"总和 == 模拟数"最强断言）。

---

## 六、测试用例集整体评价

- 结构（TC-SR/TC-GR/TC-E2E 分层）合理，C++ 新建两个测试文件并按现有 TEST 宏/SKIP 机制（NN_MODEL_PATH）注册的建议与仓库现状（test_mcts_searcher.cpp、test_mcts_searcher_injection.cpp 模式）一致。
- TC-SR-02 是能抓修复前缺陷的核心用例（旧树复用不校验 → 占位处 N>0 → 断言失败），设计意图正确。
- TC-GR-01"子数严格递增"是样本合法性最直接的观测指标，方向正确（覆盖落子必然导致某步后子数停滞）。
- TC-E2E 三件套（强弱反转 / 对照实验 / Python 一致性）覆盖了主验收、缺陷复现、语义对齐三个维度，但存在 P0-1 使主验收失效、P2-3 附加观测误报、E2E-03 flaky 三类风险，需按上述建议修订。
- 建议补充一个"未展开分支落子"的确定性用例（见 P0-1），保证 promote 叶根路径在实现层被显式回归。

---

## 七、结论与建议

方案整体架构正确（状态校验三分支 + 空位过滤 + 节点策略存储 + 零 Python 改动），与代码现状及 Python 语义吻合度高，具备可实施性。**但必须先行修订以下三点再进入实现**：

1. **P0-1（必须）**：promote 目标限定"已展开节点"，叶节点目标走重建；同步补"visit_counts 总和 == 模拟数"断言并新增确定性用例。
2. **P1-1（必须）**：set_root_policy 并入 expand 锁内（原子完成展开+策略存储），消除并发数据竞争。
3. **P1-2 / P1-3（强烈建议）**：明确 root_value 提升后的视角口径（翻转 W 或文档化接受）；修正树内存评估口径并加实测/限幅机制。

其余 P2/P3 项为规格完备性修正，建议在实现设计修订版时一并吸收（尤其 P2-2 接口声明矛盾、P2-3 验收观测条件、P2-4 断言分组），避免实现者按字面落地后产生误报或接口误解。
