# MCTS 自对弈多进程提速方案

## 问题

`mcts_self_play.py` 单进程串行跑 N 局，每局每步 800 次 MCTS 模拟。以 100 局为例耗时约 18 分钟，训练迭代周期被严重拖慢。

## 核心决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 并行粒度 | 进程（非线程） | Python GIL 使 MCTS 树操作无法多线程并行 |
| 推理设备 | CPU | 避免多进程 CUDA 上下文争抢；cnn_v7 仅 ~4 万参数，CPU 推理足够快 |
| worker 分配 | 每个 worker 跑多局 | 减少进程创建开销和模型加载次数 |

## 方案概要

新增 `--workers` 参数（默认 CPU 核数）。主进程将总局数按 worker 均分，通过 `ProcessPoolExecutor` 分发给各子进程并行执行。每个 worker 进程内独立加载模型、串行跑多局，完成后返回样本列表。

```
单进程:  局1→局2→局3→...→局100           (100 局串行)

4 workers:
  worker0: 局1→局2→...→局25
  worker1: 局26→局27→...→局50           (4 路并行)
  worker2: 局51→局52→...→局75
  worker3: 局76→局77→...→局100
```

## 模块变更

### 新增

- **`src/ai/scripts/mcts_self_play_worker.py`** — `MctsSelfPlayWorker` 类：封装模型加载 + 多局执行逻辑，供子进程调用

### 修改

- **`src/ai/scripts/mcts_self_play.py`** — 主进程增加 `--workers` 参数，用 `ProcessPoolExecutor` 调度 worker，合并结果后写盘

### 不改动

- `MCTSSearcher`、`PolicyValueAiEngine`、`AIEngineBuilder`、模型类 —— 无变化
- `mcts_train.py` —— 样本格式不变，无需适配

## 交互流程

参见 [main_flow.puml](main_flow.puml)

## 类关系

参见 [class_diagram.puml](class_diagram.puml)

## Worker 类设计

参见 [worker_design.md](worker_design.md)

## 测试用例

参见 [test_cases.md](test_cases.md)
