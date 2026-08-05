# 测试用例集

## 一、C++ SampleAugmentor 单元测试

### TC1.1 transform_index — 恒等变换

- **操作**: `transform_index(idx, 0)` 对 ∀ idx ∈ [0, 224]
- **验证**: 返回值 == idx

### TC1.2 transform_index — 互逆性

- **操作**: 对 ∀ tid ∈ {1,2,3,4,5,6,7}、∀ idx ∈ [0, 224]，执行 `t1 = transform_index(idx, tid)` 后再 `t2 = transform_index(t1, tid_reverse)`
  - 0 ↔ 0, 1 ↔ 3, 2 ↔ 2, 4 ↔ 4, 5 ↔ 5, 6 ↔ 6, 7 ↔ 7
- **验证**: t2 == idx

### TC1.3 transform_index — 棋盘角点映射

- **操作**: `transform_index(0, 1)`（左上角顺时针 90°）
- **验证**: 返回 14（右上角，即 (0,14) → (1,14) 不对，实际是 idx=0 → r=0,c=0 → tid1 → (c,14-r) = (0,14) → idx=14）

### TC1.4 transform_225 — 恒等变换保序

- **操作**: 构造 225 维向量 v = [0, 1, 2, ..., 224]；`result = transform_225(v, 0)`
- **验证**: result == v

### TC1.5 transform_225 — 旋转 180° 两次等于恒等

- **操作**: `v1 = transform_225(v, 2)`；`v2 = transform_225(v1, 2)`
- **验证**: v2 == v

### TC1.6 augment — 返回 8 条样本

- **前置**: 构造一条 GameSample（state 随机, pi 随机, z = 1.0）
- **操作**: `augmented = SampleAugmentor::augment(sample)`
- **验证**: `augmented.size() == 8`

### TC1.7 augment — z 值不变

- **操作**: 对 z = 1.0 和 z = -1.0 各构造一条样本并增强
- **验证**: 8 条变体的 z 值全部与原始一致

### TC1.8 augment — state[225] 不变

- **操作**: 设置 `state[225] = 2`，增强
- **验证**: 8 条变体的 `state[225]` 全部为 2

### TC1.9 augment — pi 总和不变

- **前置**: 构造 pi 使 sum = 1.0
- **操作**: 增强
- **验证**: 每条变体的 pi sum 仍在 0.999~1.001 范围内（浮点误差容忍）

---

## 二、C++ GameRunner 自对弈模式单元测试

### TC2.1 构造器 — 自对弈模式正常创建

- **前置**: 存在有效模型 TorchScript 文件
- **操作**: `GameRunner(model_path, num_games=1, config, num_workers=1, collect_samples=true)`
- **验证**: 构造成功；全局线程池和 NeuralNet 已初始化

### TC2.2 构造器 — 模型文件不存在

- **前置**: 指定不存在的模型路径
- **操作**: 创建 GameRunner
- **验证**: 抛出 `std::runtime_error`

### TC2.3 run_single_game — 非平局产生样本

- **前置**: GameRunner（games=1, collect_samples=true），执行 start + wait_completion
- **操作**: `result = get_results()[0]`
- **验证**: 若 `result.winner != 0`，则 `result.samples` 非空；每条样本 `state.size() == 226`、`pi.size() == 225`、`z ∈ {1.0, -1.0}`

### TC2.4 run_single_game — 平局不产生样本

- **前置**: 同上
- **操作**: 检查 winner == 0 的对局
- **验证**: `result.samples` 为空

### TC2.5 run_single_game — 样本 z 值语义

- **前置**: 执行一局 winner != 0 的对局
- **操作**: 遍历每条样本，检查 `state[225]`（玩家标识）与 `winner` 的关系
- **验证**: 若 `state[225] == winner`，则 `z == 1.0`；否则 `z == -1.0`

### TC2.6 run_single_game — D4 增强生效

- **前置**: 执行一局非平局对局
- **操作**: 计算原始步数 N = `move_count`
- **验证**: `samples.size() == 8 * N`（每步产生 8 条增强变体）

### TC2.7 执方交替分配

- **前置**: GameRunner（games=100, collect_samples=true）
- **操作**: 统计所有对局中 model1_color == 1（执黑）的次数
- **验证**: 执黑次数在 48~52 之间（100 局交替：game 0/2/4/... 执黑 → 50 次）

### TC2.8 多局并行 — 样本不丢失

- **前置**: GameRunner（games=10, num_workers=4, collect_samples=true）
- **操作**: 统计所有 `GameResult` 中样本总数
- **验证**: 非平局对局全部有样本产出；样本总数 ≥ 8 × 平均步数 × 非平局局数

### TC2.9 collect_samples=false — 向后兼容

- **前置**: GameRunner（games=2, collect_samples=false）
- **操作**: 获取结果
- **验证**: `result.samples` 全部为空；执方分配为随机模式（非交替）；其他字段与现有行为一致

---

## 三、Python SampleBufferManager 单元测试

### TC3.1 构造器 — 创建目录和 manifest

- **前置**: 临时目录不存在
- **操作**: `SampleBufferManager(temp_dir, max_size=100, chunk_games=2)`
- **验证**: `temp_dir/` 已创建；`temp_dir/manifest.json` 已写入（total_samples=0, chunks=[]）

### TC3.2 构造器 — 加载已有状态（断点续跑）

- **前置**: 手动写入 manifest.json 和两个 chunk 文件到 temp_dir
- **操作**: 创建 SampleBufferManager
- **验证**: `get_total_samples()` 等于两 chunk 样本数之和；`get_chunk_count() == 2`

### TC3.3 ingest — 写入首个 chunk

- **前置**: 空缓冲区
- **操作**: `ingest(1, 2, samples)` 其中 len(samples)=80
- **验证**: chunk 文件 `chunk_000001_000002.json` 已创建，内容含 80 条样本；manifest 更新 total_samples=80, chunks 含 1 项

### TC3.4 ingest — 未超上限不淘汰

- **前置**: 空缓冲区，max_size=200
- **操作**: ingest 第一个 chunk（80 样本），再 ingest 第二个 chunk（90 样本）
- **验证**: 两个 chunk 文件均存在；`get_total_samples() == 170`；淘汰计数为 0

### TC3.5 ingest — 超上限淘汰最旧 chunk

- **前置**: 空缓冲区，max_size=150；先 ingest chunk1（80 样本），chunk2（90 样本）→ 总数 170 > 150
- **操作**: ingest chunk2（90 样本）
- **验证**: chunk1 文件被删除；`get_total_samples() == 90`；manifest 仅含 chunk2；`ingest` 返回 80

### TC3.6 ingest — 超上限淘汰多个旧 chunk

- **前置**: 空缓冲区，max_size=100；先 ingest chunk1（80 样本），chunk2（70 样本），chunk3（60 样本）
- **操作**: ingest chunk3 → 总数 210
- **验证**: chunk1 和 chunk2 均被删除（210-80=130 > 100，再 130-70=60 < 100）；最终仅 chunk3 保留

### TC3.7 get_summary — 返回完整信息

- **前置**: 缓冲区含 2 个 chunk
- **操作**: `summary = get_summary()`
- **验证**: summary 含键 "total_samples", "chunk_count", "max_size", "chunks"；chunks 列表中每项含 "game_range", "sample_count", "file"

### TC3.8 ingest — 写入后 manifest 可被重新加载

- **前置**: 创建 manager1，ingest 一个 chunk 后释放
- **操作**: 创建 manager2（同目录）
- **验证**: manager2 的 `get_total_samples()` 与 manager1 一致

---

## 四、CppMctsSampleBufferAdapter 集成测试

### TC4.1 构造器 — 模型路径解析

- **前置**: 存在有效 registry spec（如 `cnn_v11.1`），对应 TorchScript 文件存在
- **操作**: 创建 adapter
- **验证**: 内部 C++ GameRunner 已创建；SampleBufferManager 已初始化

### TC4.2 构造器 — spec 无效

- **操作**: `CppMctsSampleBufferAdapter("invalid_model.1", ...)`
- **验证**: 抛出 `ValueError`

### TC4.3 start + progress — 非阻塞与进度递增

- **前置**: adapter（games=4）
- **操作**: `start()` → 循环 `get_progress()` 直到 `is_finished()`
- **验证**: start() 即刻返回；completed 从 0 单调递增到 4

### TC4.4 样本落盘 — 按 chunk 写入

- **前置**: adapter（games=25, chunk_games=10）
- **操作**: `start()` → `wait_completion()`
- **验证**: `sample_buffer/` 下存在 chunk_000001_000010.json、chunk_000011_000020.json、chunk_000021_000025.json；manifest.json 含 3 个 chunk

### TC4.5 wait_completion — 阻塞行为

- **前置**: adapter（games=2）
- **操作**: `start()` → `wait_completion()` → 记录耗时
- **验证**: 调用阻塞直到所有对局完成；调用后 `is_finished()` 为 True

### TC4.6 get_buffer_summary — 完成后摘要

- **前置**: adapter 已完成
- **操作**: `summary = get_buffer_summary()`
- **验证**: summary["total_samples"] > 0；summary["chunk_count"] > 0

### TC4.7 get_sample_stats — 实时统计

- **前置**: adapter 运行中
- **操作**: 循环调用 `get_sample_stats()` 观察变化
- **验证**: `buffer_samples` 随对局进行递增；当达到 chunk_games 整数倍时 `buffer_chunks` 递增

### TC4.8 Ctrl+C 中断 — 部分样本不丢失

- **前置**: adapter（games=50），start 后 10 秒终止
- **操作**: 发送 KeyboardInterrupt
- **验证**: 已完成的局产生的样本全部落盘（可能最后一个 chunk 不足 10 局）；manifest.json 反映实际落盘状态

---

## 五、CLI 集成测试

### TC5.1 基本运行 — 少量对局

- **操作**: `python mcts_sample_buffer.py --model cnn_v11.1 --games 5 --chunk-games 5`
- **验证**: 正常完成；输出进度和最终统计；`sample_buffer/` 含一个 chunk 文件

### TC5.2 基本运行 — 多 chunk

- **操作**: `--games 25 --chunk-games 10`
- **验证**: 3 个 chunk 文件（10 + 10 + 5）；最终统计正确

### TC5.3 缓冲区淘汰触发

- **操作**: `--games 100 --chunk-games 5 --buffer-max-size 5000`
- **验证**: 运行过程中有淘汰事件打印；最终缓冲区总样本数 < 5000

### TC5.4 自定义超参数

- **操作**: `--simulations 200 --c-puct 1.5 --temperature 0.5 --temp-steps 12`
- **验证**: 对局正常完成；超参数打印与传入值一致

### TC5.5 games=0（边界）

- **操作**: `--games 0`
- **验证**: 打印错误信息，退出码非零

### TC5.6 games 不是 chunk_games 整数倍

- **操作**: `--games 23 --chunk-games 10`
- **验证**: 最后一个 chunk 文件命名为 `chunk_000021_000023.json`，正常完成

### TC5.7 进度打印格式

- **操作**: `--games 11 --chunk-games 5`
- **验证**: 每完成一个 chunk 打印进度行，格式含对局进度、缓冲区样本数、chunk 数、淘汰数

### TC5.8 Ctrl+C 中断

- **操作**: `--games 100`，启动后短时间按 Ctrl+C
- **验证**: 打印中断信息；已完局样本已落盘到 chunk 文件；正常退出

### TC5.9 输出文件格式验证

- **操作**: 执行完对局后读取 chunk JSON 文件
- **验证**: 文件为合法 JSON 数组；每项含 "state"（226 维）、"pi"（225 维）、"z"（±1 或 0）；不存在 z=0 的样本

---

## 六、回归测试

### TC6.1 现有 mcts_battle_cpp.py 功能不变

- **操作**: 运行 `mcts_battle_cpp.py --ai1 cnn_v11.1 --ai2 cnn_v11.2 --games 2`
- **验证**: 行为与设计前完全一致，无回归

### TC6.2 现有 CppMctsBattleAdapter 功能不变

- **操作**: 通过 `CppMctsBattleAdapter` 执行 2 局对战
- **验证**: 结果正常；GameResult.samples 字段存在但为空（collect_samples 未启用）

### TC6.3 现有 MctsSearcher 单搜索不受影响

- **操作**: 通过现有 CppMctsAdapter 执行一次 search()
- **验证**: 结果与改造前一致
