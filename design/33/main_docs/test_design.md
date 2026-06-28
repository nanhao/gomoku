# 测试设计

## T1：AIEngine 构造函数新增参数

### T1-01：默认构造 — 采样参数为默认值
- 仅传 model 构造 AIEngine。
- 验证 get_move() 以 strategy="sampling" 调用时使用默认 temperature=1.0、top_k=None、rng=None。
- 预期：行为与重构前 get_move(board, player, strategy="sampling") 一致。

### T1-02：自定义 temperature 构造
- 构造 AIEngine(model, temperature=2.0)。
- get_move(strategy="sampling") 调用温度采样。
- 预期：采样使用 temperature=2.0，落子分布与 T=1.0 不同。

### T1-03：自定义 top_k 构造
- 构造 AIEngine(model, top_k=3)。
- get_move(strategy="sampling") 调用温度采样。
- 预期：仅在前 3 个最高概率位置中选择。

### T1-04：自定义 rng 构造 — 可复现
- 构造 AIEngine(model, rng=Random(42))。
- 同棋盘连续两次 get_move(strategy="sampling")。
- 预期：两次返回相同落子位置（可复现）。

### T1-05：get_move 不再接受 temperature 参数
- 构造后调用 get_move(board, player, temperature=2.0)。
- 预期：抛出 TypeError（不接受 temperature 关键字参数）。

### T1-06：argmax 策略不受 temperature 影响
- 构造 AIEngine(model, temperature=0.1)。
- get_move(strategy="argmax")。
- 预期：始终选择概率最大者（argmax 语义不变）。

## T2：AIEngine get_move() 删减参数 — 调用方适配

### T2-01：play_single_game 使用构造时注入的 temperature
- 创建 AIEngine 时传入 temperature=1.5。
- 对弈过程中 strategy="sampling" 的步使用该 temperature。
- 预期：与重构前 behavior 一致（采样分布相同）。

### T2-02：play_single_game argmax 分支不受影响
- 创建 AIEngine（无 temperature 参数）。
- 对弈过程中 strategy="argmax" 的步。
- 预期：与重构前 behavior 一致。

### T2-03：不同引擎实例独立持有各自采样参数
- engine1 = AIEngine(model, temperature=1.0)
- engine2 = AIEngine(model, temperature=2.0)
- 两引擎分别 get_move(strategy="sampling")。
- 预期：采样分布不同，互不干扰。

## T3：MCTSMoveDecider 构造函数新增 log_key_prefix

### T3-01：默认构造 — log_key_prefix=None 不输出日志
- 构造 MCTSMoveDecider(searcher)，不传 log_key_prefix。
- 调用 get_move(board, current_player)。
- 预期：searcher.search() 收到 log_key=None，不生成日志文件。

### T3-02：传入 log_key_prefix 时自动拼接 search_id
- 构造 MCTSMoveDecider(searcher, log_key_prefix="runA_game0")。
- 第一次 get_move(board, current_player)。
- 预期：searcher.search() 收到 log_key="runA_game0_search0"。

### T3-03：search_id 随每次 get_move 递增
- 同上构造，依次 3 次 get_move(board, current_player)。
- 预期：searcher.search() 依次收到 log_key="runA_game0_search0"、"runA_game0_search1"、"runA_game0_search2"。

### T3-04：reset() 重置 search_id 为 0
- 同上构造，调用 2 次 get_move，然后 reset()，再 1 次 get_move。
- 预期：第三次收到 log_key="runA_game0_search0"。

### T3-05：log_key_prefix 为 None 时 search_id 仍递增但不传 log_key
- 构造时不传 log_key_prefix。
- 调用 3 次 get_move，中间调用 make_move。
- 预期：不抛异常，searcher.search() 始终收到 log_key=None。

## T4：MCTSMoveDecider get_move() 删除 state/log_key、新增 board/current_player/strategy

### T4-01：get_move 不再接受 state 参数
- 调用 get_move(state=[0]*226)。
- 预期：抛出 TypeError（签名已改为 board, current_player）。

### T4-02：get_move 不再接受 log_key 参数
- 调用 get_move(board, current_player, log_key="test")。
- 预期：抛出 TypeError。

### T4-03：strategy="argmax"（默认）— 贪婪选择
- 创建带固定访问计数的 mock searcher，某动作计数显著高于其他。
- get_move(board, current_player)（默认 strategy="argmax"）。
- 预期：返回该最高计数动作。

### T4-04：strategy="sampling" — 温度采样
- 同上 mock searcher，temperature=1.0。
- get_move(board, current_player, strategy="sampling")。
- 预期：按访问计数比例采样（非确定性，但多次采样后高频动作出现比例更高）。

### T4-05：strategy="sampling" 且 temperature=0 — 退化为贪婪
- temperature=0.0 构造决策器。
- get_move(board, current_player, strategy="sampling")。
- 预期：返回最高访问计数动作（等价 argmax）。

### T4-06：strategy 非法值
- get_move(board, current_player, strategy="invalid")。
- 预期：抛出 ValueError。

### T4-07：make_move 后 search_id 正确递增且不影响 log_key 拼接
- 构造 MCTSMoveDecider(searcher, log_key_prefix="runB_game1")。
- get_move(board, current_player) → 期望 log_key="runB_game1_search0"
- make_move(action)
- get_move(board, current_player) → 期望 log_key="runB_game1_search1"
- 预期：search_id 不被 make_move 重置，仅在 reset() 时重置。

### T4-08：内部 board→state 转换正确
- 给定 board=15×15 矩阵，current_player=2。
- 调用 get_move(board, current_player)。
- 预期：searcher.search() 收到 226 维 state=[board展开, current_player]。

## T5：集成 — mcts_self_play 适配

### T5-01：移除 UUID 后日志文件名格式正确
- 运行一局 self_play（log_key_prefix="runX_gameY"）。
- 预期：生成的日志文件名为 `runX_gameY_search0.json`、`runX_gameY_search1.json` 等。

### T5-02：无 UUID import 残留
- 验证 mcts_self_play.py 不再包含 `import uuid`。
- 预期：无 uuid 引用。

### T5-03：多局 self_play 日志不冲突
- 运行 2 局 self_play（每局 reset 后 search_id 归零）。
- 预期：两局日志的 search_id 均从 0 开始，前缀不同（game0 vs game1）不冲突。

## T6：回归 — 现有功能不受影响

### T6-01：UI 人机对战（PvE）正常
- 启动 PvE 对局，AI 正常落子。
- 预期：AIEngine 使用默认采样参数，行为不变。

### T6-02：AI 内战（battle）正常
- 运行 AI vs AI 多局对战。
- 预期：使用构造时注入的 temperature/top_k/rng，结果与重构前一致。

### T6-03：MCTS 测试套件通过
- 运行现有 test_mcts_decision_maker.py 和 test_mcts_self_play.py（适配后）。
- 预期：全部通过。
