# cnn_v11 自对弈样本缓冲区

## 问题

现有 `mcts_battle_cpp.py` 仅输出胜率统计，不产生训练样本。需要一个新脚本：让 cnn_v11 模型通过 C++ MCTS 自对弈持续产出训练样本，并以**滑动窗口缓冲区**（FIFO，最大 8 万条样本）管理落盘，供下游训练器消费。

## 核心决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 执行引擎 | C++ `GameRunner`（同 `mcts_battle_cpp.py`） | 复用现有 C++ 多线程 + GPU 异步推理架构 |
| 对弈模式 | 自对弈（同一模型执双方），每局交替执方 | 自对弈无需引入对手模型；交替执方保证黑/白视角样本均衡 |
| 样本生成 | **全部在 C++ 内完成**（含 D4 增强） | Python 仅封装和 I/O，符合"全程 C++"要求 |
| 样本格式 | AGZ 双头网络格式：`{state[226], pi[225], z}` | 与现有 `mcts_self_play.py` 产出的样本完全一致 |
| D4 增强 | C++ 层对每步样本执行 8 种对称变换 | 数据量 8× 提升；C++ 实现保证性能 |
| 缓冲区粒度 | 每 10 局为一个 chunk，chunk 作为进出缓冲区的原子单位 | 以局组而非单局为单位，简化文件管理 |
| 缓冲区上限 | 80000 条样本（按样本数累计，非按局数） | 当总样本数 ≥ 上限时移除最旧 chunk |
| 落盘格式 | 每 chunk 一个独立 JSON 文件，由 `manifest.json` 追踪 | 按 chunk 删除简单；支持外部工具直接读取 |
| 进度上报 | Python 主线程轮询 `get_progress()` | 与现有架构一致，简单可靠 |
| CLI 参数 | 与 `mcts_battle_cpp.py` 保持一致的 MCTS 超参 | 用户无需记忆两套参数名 |

## 方案概要

```
mcts_sample_buffer.py --model cnn_v11.1 --games 1000

Python 主线程
  → 解析 registry spec，定位 TorchScript 模型路径
  → 创建 CppMctsSampleBufferAdapter
      ├── 创建 C++ GameRunner（model vs model，交替执方）
      └── 创建 SampleBufferManager（管理 sample_buffer/ 目录）
  → GameRunner 启动 N 个 C++ 游戏线程
     每个游戏线程：
       创建 MctsSearcher（共享全局线程池 + GPU）
       运行对局循环：MCTS 搜索 → 记录(state, visit_counts) → 落子
       对局结束后回填 z → D4 增强 → 产出 GameSample[]
  → Python 主线程轮询进度，实时打印统计
  → 每完成 10 局：SampleBufferManager 落盘一个 chunk 文件
  → buffer 超上限时自动淘汰最旧 chunk
```

## 模块变更

### C++ 新增

- **`GameSample`** — 单条训练样本结构体（state + pi + z）
- **D4 增强函数** — 8 种 15×15 棋盘对称变换

### C++ 修改

- **`GameResult`** — 新增 `samples` 字段（`std::vector<GameSample>`）
- **`GameRunner`** — `run_single_game()` 增加样本采集 + D4 增强逻辑；执方分配改为交替模式（自对弈）
- **`bindings.cpp`** — 暴露 `GameSample` 和新 `GameResult` 字段到 Python

### Python 新增

- **`cpp/python/cpp_sample_buffer_adapter.py`** — `CppMctsSampleBufferAdapter` 类
- **`src/ai/sample_buffer_manager.py`** — `SampleBufferManager` 类
- **`src/ai/scripts/mcts_sample_buffer.py`** — CLI 入口脚本

### 不改动

- `PolicyValueAiEngine`、`AIEngineBuilder`、`MctsGameRunner` — 无变化
- `MctsThreadPool`、`NeuralNet`、`MctsSearcher`、`BoardState` — 接口不变
- `mcts_battle_cpp.py`、`CppMctsBattleAdapter` — 无变化

## 交互流程

参见 [main_flow.puml](main_flow.puml)

## 类图与依赖关系

参见 [class_diagram.puml](class_diagram.puml)

## 子设计文档

- [C++ 模块接口设计](cpp_modules.md)
- [Python 适配层与 CLI 脚本](python_adapter.md)

## 测试用例

参见 [test_cases.md](test_cases.md)
