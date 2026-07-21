# 测试用例集

## 一、MctsGameRunner 单元测试

### TC1.1 run — 黑胜场景

- **前置**: 创建两个 mock PolicyValueAiEngine，黑方引擎返回必胜序列的落子，白方引擎返回任意合法落子
- **操作**: `runner.run(engine_black, engine_white, "test", 0)`
- **验证**: 返回 `GameResult(winner=1, move_count>0)`

### TC1.2 run — 白胜场景

- **前置**: 两个 mock engine，白方引擎返回必胜序列的落子
- **操作**: `runner.run(engine_black, engine_white, "test", 0)`
- **验证**: 返回 `GameResult(winner=2, move_count>0)`

### TC1.3 run — 平局场景

- **前置**: 两个 mock engine，返回填满棋盘但不产生五连的落子序列
- **操作**: `runner.run(engine_black, engine_white, "test", 0)`
- **验证**: 返回 `GameResult(winner=0, move_count=225)`

### TC1.4 run — on_move 回调调用次数

- **前置**: 两个 mock engine，控制对局在 N 步后结束
- **操作**: `runner.run(..., on_move=callback)`，callback 记录调用次数
- **验证**: callback 被调用 N 次，每次参数中 state 和 current_player 正确

### TC1.5 run — engine 选择正确

- **前置**: 两个 mock engine，各自记录自己被 `get_move` 调用的次数
- **操作**: `runner.run(engine_black, engine_white, "test", 0)` 执行完整一局
- **验证**: engine_black 的 get_move 仅在 current_player=1 时被调用；engine_white 仅在 current_player=2 时被调用

### TC1.6 run — after_move 调用

- **前置**: 两个 mock engine，各自记录 `after_move` 调用次数
- **操作**: `runner.run(engine_black, engine_white, "test", 0)`
- **验证**: `after_move` 调用次数 = `get_move` 调用次数 = 对局步数

### TC1.7 run — self-play 模式（同一实例）

- **前置**: 单个 mock engine 同时作为 engine_black 和 engine_white 传入
- **操作**: `runner.run(engine, engine, "test", 0)`
- **验证**: 对局正常完成，engine 的 get_move 被正确调用（黑方步+白方步）

---

## 二、MctsBattleWorker 单元测试

### TC2.1 构造器 — 正常加载两个模型

- **前置**: 存在两个有效的 cnn_v7.pth 权重文件
- **操作**: 创建 `MctsBattleWorker(model1_path, model1_name, model2_path, model2_name, device="cpu", config)`
- **验证**: worker 实例创建成功，内部两个模型均加载到 CPU；内部 MctsGameRunner 已创建

### TC2.2 构造器 — model1 文件不存在

- **前置**: model1_path 指向不存在的文件
- **操作**: 创建 worker
- **验证**: 抛出 `FileNotFoundError`

### TC2.3 run_games — 单局执行

- **前置**: worker 已创建（使用两个真实小模型）
- **操作**: `worker.run_games(game_indices=[0], run_id="test")`
- **验证**: 返回长度为 1 的列表；BattleGameResult.winner ∈ {0,1,2}；model1_color + model2_color = 3；model1_won 和 model2_won 只有一个为 True（或都为 False 平局）

### TC2.4 run_games — 多局执行

- **前置**: worker 已创建
- **操作**: `worker.run_games(game_indices=[0, 1, 2], run_id="test")`
- **验证**: 返回长度为 3 的列表；每项的 game_idx 覆盖 [0, 1, 2]

### TC2.5 run_games — 执方随机性

- **前置**: worker 已创建
- **操作**: `worker.run_games(game_indices=list(range(50)), run_id="test")`
- **验证**: model1 执黑次数 ≈ 25（在 15~35 之间，即二项分布 95% 置信区间内）

### TC2.6 run_games — engine 不复用

- **前置**: worker 已创建
- **操作**: worker 执行 2 局
- **验证**: 两局结果独立，第二局初始棋盘为空

### TC2.7 worker_entry — 子进程入口

- **前置**: 有效的模型路径和配置
- **操作**: 在独立进程中调用 `MctsBattleWorker.worker_entry(...)`
- **验证**: 返回 `List[BattleGameResult]`，结果正确

### TC2.8 run_games — 设备确认

- **操作**: worker 以 `device="cpu"` 创建并执行
- **验证**: 两个模型网络的所有参数张量在 CPU 上

---

## 三、mcts_battle.py 集成测试

### TC3.1 单进程 — 基本对战

- **操作**: `python mcts_battle.py --model1 model1.pth --model2 model2.pth --games 2`
- **验证**: 输出 2 局结果；最终统计显示总局数=2、双方胜率+平局率之和=100%

### TC3.2 多进程 — 并行对战

- **操作**: `python mcts_battle.py --model1 model1.pth --model2 model2.pth --games 4 --workers 2`
- **验证**: 输出 4 局结果；各 worker 分配的局数合理；无死锁无报错

### TC3.3 单 worker 与多 worker 结果数量一致

- **操作**: 分别运行 `--games=8 --workers=1` 和 `--games=8 --workers=2`
- **验证**: 两次运行均产生 8 个 BattleGameResult

### TC3.4 超参数透传

- **操作**: `--simulations=100 --c-puct=1.5`
- **验证**: worker 内部创建的 engine 使用 simulations=100、c_puct=1.5

### TC3.5 --workers 大于 --games

- **操作**: `--games=2 --workers=8`
- **验证**: 仅有 2 个 worker 被分配任务，其余 worker 不执行对局，不报错

### TC3.6 game_indices 分配无重叠无遗漏

- **操作**: `--games=10 --workers=3`
- **验证**: 3 个 worker 的 game_indices 合集 = {0,1,...,9}，无重叠

### TC3.7 Ctrl+C 中断

- **操作**: 启动大量对局，中途发送 SIGINT
- **验证**: 打印中断信息，输出已完成局的统计，无异常退出

### TC3.8 统计输出格式正确

- **操作**: `--games=10`
- **验证**: 最终统计部分包含"总局数"、"model1名 胜: N 局 (X.X%)"、"model2名 胜: N 局 (X.X%)"、"平局: N 局 (X.X%)"

---

## 四、回归测试 — MctsSelfPlayWorker

### TC4.1 重构后 self-play 功能不变

- **操作**: 运行 `mcts_self_play.py --model model.pth --games 2`
- **验证**: 输出 sample.json 格式与重构前完全一致；样本结构（state/pi/z）不变

### TC4.2 重构后 self-play 多进程不变

- **操作**: 运行 `mcts_self_play.py --model model.pth --games 4 --workers 2`
- **验证**: 正常完成，样本总数 = 4 局步数之和

### TC4.3 重构后现有 self-play 单元测试全通过

- **操作**: 运行 `tests/ai/` 下所有 self-play 相关测试
- **验证**: 全部通过

---

## 五、边界与异常测试

### TC5.1 games=0

- **操作**: `--games 0`
- **验证**: 打印错误信息并退出，不执行任何对局

### TC5.2 games 为负数

- **操作**: `--games -1`
- **验证**: argparse 拒绝或打印错误信息并退出

### TC5.3 两个模型相同

- **操作**: model1 和 model2 指向同一权重文件
- **验证**: 正常执行，统计结果中双方胜率应接近 50%

### TC5.4 workers=0

- **操作**: `--workers 0`
- **验证**: 自动 fallback 为 1（单进程）
