# MctsSelfPlayWorker 类设计

## 概述

封装模型加载 + 多局执行逻辑，供 `ProcessPoolExecutor` 的子进程调用。每个 worker 实例加载一次模型，可串行执行多局对弈。

## 文件

`src/ai/scripts/mcts_self_play_worker.py`

## 类: MctsSelfPlayWorker

### 构造器

```
MctsSelfPlayWorker(
    model_path: str,
    model_name: str | None,
    device: str,
    config: MctsSelfPlayConfig,
)
```

- `model_path` — 权重文件路径
- `model_name` — 模型注册名.代数（如 "cnn_v8.1"），None 时从路径自动解析
- `device` — 推理设备，多进程方案中固定为 "cpu"
- `config` — 超参数配置（见下方 MctsSelfPlayConfig）

行为：
1. 通过 registry 加载模型（路径解析逻辑与当前 `run_self_play` 一致）
2. 将模型移到指定设备

### 公共方法

#### run_games

```
run_games(
    game_indices: List[int],
    run_id: str,
) -> List[dict]
```

- `game_indices` — 要执行的对局编号列表
- `run_id` — 本次运行唯一标识（时间戳后缀）

返回：所有对局产生的样本列表，每项为 `{"state": [...], "pi": [...], "z": int}`。

行为：对 `game_indices` 中的每个编号，依次调用内部对局逻辑（与当前 `play_one_game` 等价）：创建 PolicyValueAiEngine → 循环落子 → 记录样本 → 终局回填 z 值。每局使用全新 engine 实例（模型复用），确保 MCTS 搜索树不跨局污染。

---

## 数据类: MctsSelfPlayConfig

```python
@dataclass
class MctsSelfPlayConfig:
    simulations: int          # 每步模拟次数，默认 800
    c_puct: float             # PUCT 探索系数，默认 2.0
    temp_steps: int           # 温度探索步数，默认 8
    temperature: float        # 温度参数，默认 1.0
    dirichlet_epsilon: float  # Dirichlet 噪声比例，默认 0.25
    dirichlet_alpha: float    # Dirichlet 分布参数，默认 0.03
    batch_size: int           # 每批 simulation 数，默认 225
    virtual_loss: float       # Virtual Loss 值，默认 3.0
    use_prior: bool           # 是否用模型先验概率，默认 True
```

替代当前 `hyperparams` 字典，符合"结构化数据用 dataclass"规范。

---

## 与现有代码的关系

- 替代 `play_one_game()` 函数：将模型加载提升到构造器，执行逻辑封装为方法
- 每局内创建 engine 的流程完全复用 `AIEngineBuilder.build_policy_value()`
- 不修改 `MCTSSearcher`、`PolicyValueAiEngine`、`AIEngineBuilder`
