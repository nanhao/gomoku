# 接口变更详细设计

---

## 1. 变更模块：ai/battle.py

### 1.1 函数：run_battle()

#### 签名变更

**变更前**：

```
run_battle(ai1_name: str, ai2_name: str, num_games: int)
```

**变更后**：

```
run_battle(ai1_name: str, ai2_name: str, num_games: int,
           temp_config: Optional[TempConfig] = None)
```

#### 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `temp_config` | `Optional[TempConfig]` | `None` | 温度采样配置；`None` 时透传给 `AIGameManager`，底层解释为全程 argmax；默认启用温度采样由 `main()` 保证 |

#### 行为变更

- 调用 `AIGameManager.start_ai_vs_ai_random()` 时，直接透传 `temp_config` 参数
- `run_battle()` 自身不做 `None → 默认 TempConfig` 的语义转换
- 默认启用温度采样的职责在 `main()` 中完成（参见 [main_design.md §2.3](main_design.md#23-main-cli-变更)）

#### 兼容性

- CLI 调用不传任何温度参数：行为变化——`main()` 自动构造默认 TempConfig 传入 `run_battle()`，由全程 argmax 变为默认启用温度采样
- Python API 直接调用 `run_battle("random", "random", 10)` 不传 `temp_config`：行为不变——`temp_config` 默认 `None`，透传给 `AIGameManager`，全程 argmax（与改动前一致）
- 显式传 `temp_config=None` 的调用：透传 `None` 给 `AIGameManager`，等同于全程 argmax
- 若调用方需要全程 argmax，传 `temp_config=None` 即可

> **设计决策说明**：`temp_config=None` 的语义在 `run_battle()` 层与 `AIGameManager` 层保持一致（`None` = 不启用温度采样）。
> CLI 层面的"默认启用"通过在 `main()` 中构造默认 `TempConfig` 实现，而非在 `run_battle()` 中做语义转换。
> 这样避免了 `--no-temp-sampling` 标志与 `temp_config=None` 之间的语义冲突。
> 详细论证参见 [main_design.md §2.2](main_design.md#22-run_battle-签名变更)。

---

### 1.2 函数：main()

无签名变更。内部新增参数解析与 TempConfig 构造逻辑。

#### 新增 CLI 参数（5 个）

| 长参数 | argparse 类型 | 默认值 | choices | 说明 |
|--------|--------------|--------|---------|------|
| `--no-temp-sampling` | `store_true` | `False` | — | 禁用温度采样 |
| `--temperature` | `float` | `1.0` | — | 温度值 T > 0 |
| `--temp-moves` | `int` | `12` | — | 前 N 步温度采样，≥ 0 |
| `--temp-side` | `str` | `TEMP_SIDE_BOTH` | `list(SUPPORTED_TEMP_SIDES)` | 启用采样的一方（默认 `both`） |
| `--temp-seed` | `int` | `None` | — | 随机种子 |

#### 参数解析后处理

> 完整逻辑见 [main_design.md §2.3](main_design.md#23-main-cli-变更)。此处列出关键步骤：

```
# 1. 构造 TempConfig（在 try 块内，确保 ValueError 被捕获）
if args.no_temp_sampling:
    temp_config = None
else:
    temp_config = TempConfig(
        temperature=args.temperature,
        temp_moves=args.temp_moves,
        temp_side=args.temp_side,
        top_k=None,
        seed=args.temp_seed,
    )

# 2. 传入 run_battle（透传，不做语义转换）
run_battle(args.ai1, args.ai2, args.games, temp_config=temp_config)
```

#### 新增导入

> 在 `ai/battle.py` 中新增（与 `create_samples.py` 对齐）：

```
from ai.move_sampler import TempConfig, SUPPORTED_TEMP_SIDES, TEMP_SIDE_BOTH
```

- `TempConfig`：构造温度采样配置
- `TEMP_SIDE_BOTH`：作为 `--temp-side` 的默认值
- `SUPPORTED_TEMP_SIDES`：作为 `--temp-side` 的 `choices` 白名单

---

## 2. 未变更模块

以下模块无需修改：

| 模块 | 原因 |
|------|------|
| `ai/move_sampler.py` | `TempConfig` 数据类和 `SUPPORTED_TEMP_SIDES` 已存在且完整 |
| `ai/game_manager.py` | `start_ai_vs_ai_random()` 已支持 `temp_config` 参数 |
| `ai/engine.py` | `get_move()` 已支持 `strategy`/`temperature`/`top_k`/`rng` 参数 |
| `app/battle.py` | `BattleSession` 已支持 `temp_config` 参数 |
| `gui/ai_battle.py` | 已硬编码启用温度采样，独立于本次变更 |
| `ai/create_samples.py` | 独立脚本，无需修改 |
