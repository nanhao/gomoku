# 测试用例集

## 一、CppMctsBattleAdapter 单元测试

### TC1.1 构造器 — 正常加载两个模型

- **前置**: 存在两个有效的 cnn_v11 registry spec（如 `cnn_v11.1`、`cnn_v11.2`），对应目录下存在 `cnn_v11_torchscript.pt`
- **操作**: 创建 `CppMctsBattleAdapter("cnn_v11.1", "cnn_v11.2", num_games=1, config)`
- **验证**: 实例创建成功；内部 C++ GameRunner 已创建；全局线程池已初始化

### TC1.2 构造器 — model1 TorchScript 文件不存在

- **前置**: `cnn_v11.1` 目录下无 `.pt` 文件
- **操作**: 创建 adapter
- **验证**: 抛出 `FileNotFoundError`，错误信息包含路径

### TC1.3 构造器 — registry spec 格式无效

- **前置**: 使用不存在的 spec（如 `cnn_v11.999`）
- **操作**: 创建 adapter
- **验证**: 抛出 `ValueError`

### TC1.4 start — 非阻塞行为

- **前置**: adapter 已创建（games=2）
- **操作**: 调用 `adapter.start()` 并立即记录时间，随后调用 `adapter.is_finished()`
- **验证**: `start()` 在极短时间内返回（< 0.1s）；`is_finished()` 可能为 `False`（对局仍在进行）

### TC1.5 get_progress — 进度递增

- **前置**: adapter 已创建（games=4），已调用 `start()`
- **操作**: 循环调用 `get_progress()` 直到 `is_finished()` 为 True，记录每次返回值
- **验证**: completed 从 0 单调递增到 4；total 始终为 4

### TC1.6 wait_completion — 阻塞行为

- **前置**: adapter 已创建（games=2），已调用 `start()`
- **操作**: 调用 `wait_completion()`，记录前后时间
- **验证**: 调用阻塞直到所有对局完成；调用后 `is_finished()` 为 True

### TC1.7 get_results — 结果数量正确

- **前置**: adapter 已完成（games=5），调用 `wait_completion()`
- **操作**: `results = adapter.get_results()`
- **验证**: `len(results) == 5`；每个 `CppBattleGameResult` 的 `game_index` 覆盖 0..4

### TC1.8 get_results — 字段合法性

- **前置**: adapter 已完成
- **操作**: 获取结果列表，遍历检查每项
- **验证**: 每项满足：`winner ∈ {0,1,2}`；`model1_color ∈ {1,2}`；`model2_color = 3 - model1_color`；`model1_won` 与 `model2_won` 最多一个为 True（平局时均为 False）；`move_count ≥ 5`（五子棋最少步数）

---

## 二、mcts_battle_cpp.py 集成测试

### TC2.1 基本对战 — 单局

- **操作**: `python mcts_battle_cpp.py --ai1 cnn_v11.1 --ai2 cnn_v11.2 --games 1`
- **验证**: 输出 1 局结果；最终统计显示总局数=1、胜率+平局率=100%

### TC2.2 基本对战 — 多局

- **操作**: `python mcts_battle_cpp.py --ai1 cnn_v11.1 --ai2 cnn_v11.2 --games 10`
- **验证**: 输出 10 局结果（每局一行）；统计格式正确

### TC2.3 同模型自对弈

- **操作**: `--ai1 cnn_v11.1 --ai2 cnn_v11.1 --games 20`
- **验证**: 正常完成；双方胜率应接近 50%（在 25%~75% 区间内）

### TC2.4 自定义超参数

- **操作**: `--simulations 200 --c-puct 1.5 --temperature 0.5 --temp-steps 12`
- **验证**: 对局正常完成；控制台输出显示 MCTS 配置与传入值一致

### TC2.5 进度打印

- **操作**: `--games 5`，观察终端输出
- **验证**: 运行过程中周期打印 `[进度] X/5`；完成时打印 `[完成] 共 5 局`

### TC2.6 Ctrl+C 中断

- **操作**: `--games 50`，启动后 5 秒内发送 SIGINT
- **验证**: 打印中断信息；等待进行中对局完成后打印已完局的统计；正常退出

### TC2.7 输出格式验证

- **操作**: `--games 4`
- **验证**: 每局结果行格式为 `[局 X/4] ai1(执方) vs ai2(执方) → 结果 (N步)`；最终统计包含总局数、双方胜率%、平局率%

### TC2.8 games=0（边界）

- **操作**: `--games 0`
- **验证**: 打印错误信息并退出，不创建 adapter

### TC2.9 games 为负数（边界）

- **操作**: `--games -1`
- **验证**: argparse 拒绝（`type=int` 自动校验正负由业务逻辑处理）或打印错误信息退出

---

## 三、C++ 单元测试

### TC3.1 MctsThreadPool — 任务队列提交与等待

- **前置**: 初始化线程池（4 个 worker）
- **操作**: 依次 `submit()` 8 个任务（每个任务 `sleep(10ms)` 后递增原子计数器）；`wait_all()`
- **验证**: 计数器最终为 8；总耗时远小于 80ms（证明并行执行）

### TC3.2 MctsThreadPool — 并发 submit

- **前置**: 初始化线程池（4 个 worker）
- **操作**: 4 个线程同时各自 `submit()` 100 个任务；各线程独立 `wait_all()`
- **验证**: 总计数器为 400；无死锁无数据竞争

### TC3.3 GameRunner — 单局对局

- **前置**: 创建 GameRunner（2 个有效模型，games=1）
- **操作**: `start()` → `wait_completion()` → `get_results()`
- **验证**: 返回 1 个 GameResult；`winner ∈ {0,1,2}`

### TC3.4 GameRunner — 多局并行

- **前置**: GameRunner（games=4）
- **操作**: `start()` → 轮询 `get_completed_count()` → `wait_completion()`
- **验证**: completed 从 0 递增到 4；最终 `get_results().size() == 4`

### TC3.5 GameRunner — 模型文件不存在

- **前置**: model1_path 指向不存在文件
- **操作**: 创建 GameRunner
- **验证**: 构造器抛出 `std::runtime_error`

### TC3.6 GameRunner — results 顺序独立于完成顺序

- **前置**: GameRunner（games=4）
- **操作**: 执行并获取结果
- **验证**: 结果的 `game_index` 覆盖 0..3（不要求按序，但必须全覆盖）

### TC3.7 GameRunner — 执方随机性

- **前置**: GameRunner（games=100）
- **操作**: 执行后统计 model1 执黑次数
- **验证**: model1 执黑次数在 38~62 之间（二项分布 95% 置信区间）

---

## 四、回归测试

### TC4.1 现有 mcts_battle.py 功能不变

- **操作**: 运行 `mcts_battle.py --model1 model.pth --model2 model.pth --games 2`
- **验证**: 与设计前行为一致，无回归

### TC4.2 现有 MctsSearcher 单搜索场景不变

- **操作**: 通过现有 `CppMctsAdapter` 创建搜索器并执行一次 `search()`
- **验证**: 搜索结果（visit_counts, root_value, root_policy）与改造前一致

### TC4.3 现有线程池单批次模式兼容

- **操作**: 使用改造后的 MctsThreadPool 执行一次 `run_parallel` 风格的批量任务
- **验证**: 功能正常，结果与改造前一致
