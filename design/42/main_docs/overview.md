# MCTS 双模型对战脚本

## 问题

当前缺少用 MCTS 搜素参数（可配 simulations/c_puct/temperature 等）让两个双头模型对打并统计胜率的能力。现有 `battle.py` 使用 `create_engine` 智能路由，对 cnn_v7/v8 虽然创建 PolicyValueAiEngine，但 MCTS 参数固定为内部默认值，无法在命令行调整。

## 核心决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 代码复用策略 | 提取 MctsGameRunner 公共类 | `_play_one_game` 对局循环与 `mcts_self_play` 共享，消除重复 |
| 模型指定方式 | `--model1/--model2` 权重路径 + 可选 `--model1-name/--model2-name` | 与 `mcts_self_play` 完全一致 |
| 执方分配 | 每局随机 | 消除先手优势对胜率统计的影响 |
| MCTS 参数 | 复用 `mcts_self_play` 全部 CLI 参数 | 对战场景同样需要灵活的搜索配置 |
| 多进程 | 支持 `--workers`，与 `mcts_self_play` 一致 | 大量对局需要并行提速 |
| 数据记录 | 不使用 DataRecorder | 避免磁盘 I/O 和多进程文件冲突 |

## 方案概要

```
mcts_battle.py --model1 cnn_v8.10.pth --model2 cnn_v8.11.pth --games 100 --workers 4

主进程 → 均分 game_indices → ProcessPoolExecutor
  worker0: 局1~25 (每局随机执方)
  worker1: 局26~50
  worker2: 局51~75
  worker3: 局76~100
→ 收集结果 → 打印统计（双方胜率、平局率）
```

## 模块变更

### 新增

- **`src/ai/scripts/mcts_game_runner.py`** — `MctsGameRunner` 类：封装公用的对局循环（棋盘管理 + MCTS 搜索 + 落子 + 终局判定），供 self-play 和 battle 共用
- **`src/ai/scripts/mcts_battle_worker.py`** — `MctsBattleWorker` 类：加载两个模型，执行多局对战并返回结果
- **`src/ai/scripts/mcts_battle.py`** — CLI 入口脚本

### 修改

- **`src/ai/scripts/mcts_self_play_worker.py`** — `MctsSelfPlayWorker._play_one_game` 重构为使用 `MctsGameRunner`，保持对外接口不变

### 不改动

- `MCTSSearcher`、`PolicyValueAiEngine`、`AIEngineBuilder`、模型类 — 无变化
- `mcts_self_play.py` — 接口不变，仅内部委托关系变化

## 交互流程

参见 [main_flow.puml](main_flow.puml)

## 类关系

参见 [class_diagram.puml](class_diagram.puml)

## 子设计文档

- [MctsGameRunner 类设计](mcts_game_runner_design.md)
- [MctsBattleWorker 类设计](mcts_battle_worker_design.md)
- [mcts_battle.py 入口脚本设计](mcts_battle_design.md)

## 测试用例

参见 [test_cases.md](test_cases.md)
