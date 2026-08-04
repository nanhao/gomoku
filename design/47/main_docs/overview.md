# cnn_v11 C++ MCTS 并行对战

## 问题

现有 `mcts_battle.py` 使用 Python 多进程（ProcessPoolExecutor）并行对战，存在以下局限：
- 多进程模式下强制 CPU 推理，无法利用 GPU
- 每个进程独立加载模型，内存开销大
- 进程间无共享线程池，MCTS 工作线程资源无法复用
- 不适合 cnn_v11 + C++ MCTS + GPU 的高性能评估场景

需要一个新的对战脚本：在 **C++ 层面**用多线程并行运行 N 局 cnn_v11 对战，所有对局共享全局 MCTS 线程池和 GPU 推理引擎。

## 核心决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 并行层级 | C++ 线程运行对局，非 Python 线程 | Python GIL 不阻碍 C++ 代码；C++ 线程直接管理 MCTS 搜索器生命周期 |
| 线程池策略 | 全局单例任务队列模式 | 所有对局共享 K 个 MCTS 工作线程，按需竞争，避免空转 |
| GPU 推理 | 全局单例 NeuralNet | 所有 MctsSearcher 共享同一 GPU 推理引擎，攒批推理跨对局复用 |
| 模型指定 | registry spec（如 `cnn_v11.1`） | 脚本内部自动查找 TorchScript 模型文件 |
| 执方分配 | 每局随机（model1 随机执黑或执白） | 消除先手优势对胜率统计的影响 |
| 对局循环 | C++ 侧实现完整对局循环 | BoardState 已支持终局判定，无需 Python 参与对局逻辑 |
| 进度上报 | Python 主线程轮询 `get_progress()` | 简单可靠，无需回调跨越 C++/Python 边界 |
| 输出 | 仅终端打印（每局结果 + 最终统计） | 不需要棋谱和报告文件 |

## 方案概要

```
mcts_battle_cpp.py --ai1 cnn_v11.1 --ai2 cnn_v11.2 --games 100

Python 主线程
  → 解析 registry spec，定位两个 TorchScript 模型路径
  → 创建 CppMctsBattleAdapter → 创建 C++ GameRunner
  → GameRunner 启动 N 个 C++ 游戏线程
     每个游戏线程：
       创建 MctsSearcher（共享全局线程池 + GPU）
       运行对局循环（MCTS 搜索 → 温度采样 → 落子 → 终局判定）
       上报结果
  → Python 主线程轮询进度，打印实时状态
  → 所有对局完成后打印每局结果和双方胜率
```

## 模块变更

### C++ 新增

- **`GameRunner`** — 管理 N 个并行游戏线程，收集结果，上报进度
- **`GameConfig`** — 游戏配置结构体（MCTS 参数 + 温度参数）

### C++ 修改

- **`MctsThreadPool`** — 从"单批次屏障"模式改为"任务队列"模式，支持多游戏并发提交
- **`NeuralNet`** — 改为全局单例，支持多个 MctsSearcher 共享
- **`MctsSearcher`** — 构造器改为接受外部注入的线程池和 NeuralNet 引用，不再自行创建
- **`bindings.cpp`** — 暴露 GameRunner、GameConfig、GameResult 到 Python

### Python 新增

- **`cpp/python/cpp_mcts_battle_adapter.py`** — `CppMctsBattleAdapter` 类，包装 C++ GameRunner
- **`src/ai/scripts/mcts_battle_cpp.py`** — CLI 入口脚本

### 不改动

- `PolicyValueAiEngine`、`AIEngineBuilder`、`MctsGameRunner`、`mcts_battle.py` — 无变化
- `BoardState` — 接口不变，仅被 GameRunner 内部使用

## 交互流程

参见 [main_flow.puml](main_flow.puml)

## 线程模型

参见 [threading_flow.puml](threading_flow.puml)

## 类关系

参见 [class_diagram.puml](class_diagram.puml)

## 子设计文档

- [C++ 模块接口设计](cpp_modules.md)
- [Python 适配层与 CLI 脚本](python_adapter.md)

## 测试用例

参见 [test_cases.md](test_cases.md)
