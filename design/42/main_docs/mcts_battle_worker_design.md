# MctsBattleWorker 类设计

## 概述

封装双模型加载 + 多局对战执行逻辑，供 `ProcessPoolExecutor` 的子进程调用。每个 worker 实例加载两个模型各一次，可串行执行多局对战。每局随机分配 model1/model2 的执方。

## 文件

`src/ai/scripts/mcts_battle_worker.py`

## 类: MctsBattleWorker

### 构造器

```
MctsBattleWorker(
    model1_path: str,
    model1_name: str | None,
    model2_path: str,
    model2_name: str | None,
    device: str,
    config: MctsSelfPlayConfig,
)
```

- `model1_path` / `model2_path` — 双方权重文件路径
- `model1_name` / `model2_name` — 模型注册名.代数（如 "cnn_v8.10"），None 时从路径自动解析
- `device` — 推理设备，多进程方案中固定为 "cpu"
- `config` — MCTS 超参数配置（复用 `MctsSelfPlayConfig`）

行为：
1. 通过 registry 加载两个模型（路径解析逻辑与 `MctsSelfPlayWorker._load_model` 一致）
2. 将两个模型移到指定设备
3. 创建内部 `MctsGameRunner` 实例用于执行对局

### 公共方法

#### run_games

```
run_games(
    game_indices: List[int],
    run_id: str,
) -> List[BattleGameResult]
```

- `game_indices` — 要执行的对局编号列表
- `run_id` — 本次运行唯一标识（时间戳后缀）

返回：`BattleGameResult` 列表，每项包含单局结果。

行为：对 `game_indices` 中的每个编号：
1. 随机决定 model1 执黑还是执白（model2 执相反方）
2. 每局创建全新的 `PolicyValueAiEngine` 实例（两个），分别绑定对应模型
3. 调用 `MctsGameRunner.run(engine_black, engine_white, ...)` 执行对局
4. 收集 `GameResult` 并包装为 `BattleGameResult`

#### worker_entry（静态方法）

```
@staticmethod
worker_entry(
    model1_path: str,
    model1_name: str | None,
    model2_path: str,
    model2_name: str | None,
    device: str,
    config: MctsSelfPlayConfig,
    game_indices: List[int],
    run_id: str,
) -> List[BattleGameResult]
```

ProcessPoolExecutor 子进程入口。在子进程中创建 `MctsBattleWorker` 并调用 `run_games`。

## 数据类: BattleGameResult

```
@dataclass
class BattleGameResult:
    game_idx: int           # 局编号
    winner: int             # 0=平局, 1=黑胜, 2=白胜
    model1_color: int       # model1 执方 (1=黑, 2=白)
    model2_color: int       # model2 执方 (1=黑, 2=白)
    model1_won: bool        # model1 是否获胜
    model2_won: bool        # model2 是否获胜
    move_count: int         # 本局总步数
```

## 与现有代码的关系

- 模型加载逻辑复用 `MctsSelfPlayWorker._load_model`（提取为共享静态方法或独立工具函数）
- 对局循环委托给 `MctsGameRunner.run`
- 不修改 `MCTSSearcher`、`PolicyValueAiEngine`、`AIEngineBuilder`
