# C++ MCTS 实现 — 总体设计概述

## 1. 设计目标

用 C++ 实现高性能 MCTS 搜索器，通过 pybind11 桥接为 Python 扩展模块，替代现有 `src/ai/mcts/searcher.py` 中的纯 Python 搜索逻辑。

核心收益：
- 多线程并行 MCTS 搜索（Python GIL 瓶颈消除）
- GPU 异步批量推理，最大化 GPU 利用率
- 与现有 Python 训练/编排代码无缝集成

## 2. 技术选型

| 层面 | 选型 | 说明 |
|------|------|------|
| 语言 | C++17 | 兼容 LibTorch 要求 |
| 深度学习 | **LibTorch** (PyTorch C++ API) | 加载 TorchScript 模型，GPU 推理 |
| Python 桥接 | **pybind11** | 头文件库，编译为 `.pyd` 扩展 |
| 构建系统 | **CMake** ≥ 3.18 | LibTorch + pybind11 标准搭配 |
| 模型格式 | **TorchScript** (`torch.jit.script`) | Python 端导出，C++ 端加载 |
| 目标模型 | **cnn_v11** | 256 通道 + 7 残差块 + 双头输出 |

## 3. 系统边界

```
┌─ Python 层（不变）─────────────────────────────────────┐
│  PolicyValueAiEngine  →  温度选择 / 子树复用编排        │
│  train.py / mcts_self_play.py  →  训练流程              │
│  src/game/  →  棋盘规则（仅训练/对弈编排时使用）         │
├────────────────────────────────────────────────────────┤
│  pybind11 桥接层（新增）                                │
│  cpp_mcts.MctsSearcher  ←─── 与 Python 侧接口一致       │
├─ C++ 层（新增）────────────────────────────────────────┤
│  MctsSearcher  多线程 MCTS 搜索                         │
│  MctsNode      搜索树节点（并发安全）                    │
│  NeuralNet     GPU 异步批量推理                          │
│  BoardState    棋盘状态与胜负判定                        │
└────────────────────────────────────────────────────────┘
```

C++ 层替代的 Python 模块：
- `src/ai/mcts/searcher.py` → C++ `MctsSearcher`
- `src/ai/models/cnn_v11/model.py` 的 `predict_batch()` → C++ `NeuralNet`
- `src/game/game.py` 的 `check_winner` / `is_board_full` → C++ `BoardState`

## 4. 线程模型

```
┌── MCTS 线程池 (N 线程) ──────────────────────────────┐
│                                                       │
│   每个线程循环执行一次 simulation：                     │
│   ┌─────────┐    ┌──────────┐    ┌─────────┐         │
│   │ Select  │ →  │ 投递局面  │ →  │ 等待GPU  │         │
│   │(PUCT+V-L)│    │ 到推理队列 │    │ 返回结果  │         │
│   └─────────┘    └──────────┘    └────┬────┘         │
│                                       │               │
│                  ┌─────────┐    ┌─────▼────┐         │
│                  │ Backup  │ ←  │  Expand   │         │
│                  │(回传更新)│    │ (展开子节点)│         │
│                  └─────────┘    └──────────┘         │
│                                                       │
└──────────────────────┬────────────────────────────────┘
                       │  request queue (std::promise/future)
┌──────────────────────▼────────────────────────────────┐
│  GPU 推理线程 (1 线程)                                  │
│                                                       │
│  循环：攒批 → GPU forward → 拆分结果 → 唤醒等待线程     │
└──────────────────────────────────────────────────────┘
```

- MCTS 线程在投递局面后阻塞等待（`future.get()`），GPU 线程返回后继续 expand + backup
- Virtual Loss 保证不同 MCTS 线程探索不同分支
- 共享搜索树使用 `std::atomic` + 细粒度 `std::mutex` 保证并发安全

## 5. 目录结构

```
cpp/
├── CMakeLists.txt           # CMake 构建配置
├── src/
│   ├── mcts_searcher.h      # MctsSearcher 公开接口
│   ├── mcts_searcher.cpp
│   ├── mcts_node.h          # MctsNode 公开接口
│   ├── mcts_node.cpp
│   ├── neural_net.h         # NeuralNet 公开接口
│   ├── neural_net.cpp
│   ├── board_state.h        # BoardState 公开接口
│   ├── board_state.cpp
│   └── bindings.cpp         # pybind11 绑定定义
├── tests/
│   ├── test_board_state.cpp
│   ├── test_mcts_node.cpp
│   ├── test_neural_net.cpp
│   └── test_mcts_searcher.cpp
└── python/
    └── cpp_mcts_adapter.py  # Python 端适配层（替代 searcher.py 的调用）
```

## 6. 设计文档索引

| 文档 | 内容 |
|------|------|
| [module_interfaces.md](module_interfaces.md) | C++ 模块对外接口规范 |
| [python_adapter.md](python_adapter.md) | pybind11 桥接 + Python 端适配 |
| [test_cases.md](test_cases.md) | 完整测试用例集 |

| 图 | 内容 |
|----|------|
| [architecture.puml](architecture.puml) | 类图与依赖关系 |
| [mcts_flow.puml](mcts_flow.puml) | MCTS 单次 simulation 流程 |
| [threading_flow.puml](threading_flow.puml) | 多线程 + GPU 推理协作流程 |
| [integration_flow.puml](integration_flow.puml) | Python → C++ 完整调用链路 |
